#!/usr/bin/env python3
"""Certify packaged bytes on a trusted, dedicated GPU worker. No rebuilds/skips."""

import argparse
import contextlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from release import verify_candidate
from release_lib import ROOT, extract_archive, install_signal_handlers, policy, probe_binary, read_json, reference_agreement, require, sha256, write_json

CORPUS = [
    "The quick brown fox jumps over the lazy dog.", "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    "  leading and trailing spaces  ", "Multiple\n\nnewlines\tand tabs.", "3.14159 2,718,281 1e-9 0xDEADBEEF",
    "Émile Zola — naïve café façade.", "日本語 中文 한국어", "🚀👍🏽 ✓ ✗", "Привет, как дела?",
    "https://example.com/path?query=value&x=1#frag", "Don't, won't, shouldn't've, O'Reilly.", "mixedCase snake_case ALL_CAPS",
]
PROMPTS = ["Write a long story about a lighthouse keeper.", "Explain how a refrigerator works.", "List three uses of copper."]


def output(command, env=None, timeout=180):
    result = subprocess.run(command, cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=timeout, env=env)
    require(result.returncode == 0, "validation command failed: " + Path(command[0]).name)
    return result.stdout + result.stderr


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextlib.contextmanager
def process(command, log, cwd, env):
    """Own the entire subprocess group, including timeout/error/cancel paths."""
    with Path(log).open("wb") as stream:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield child
        finally:
            # Kill only this invocation's process group, even if its leader died.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)


def request(port, path, body=None, timeout=120):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=payload,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def json_request(port, path, body=None):
    with request(port, path, body) as response:
        return json.load(response)


def ready(child, port):
    deadline = time.monotonic() + 360
    while time.monotonic() < deadline:
        require(child.poll() is None, "server exited before becoming healthy")
        try:
            json_request(port, "/health")
            return
        except (OSError, ValueError):
            time.sleep(0.5)
    raise ValueError("server health timeout")


def gpu_evidence(text, family):
    """Check the selected device, never accept an arbitrary runner label."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    require("[FAIL]" not in clean and "READY" in clean, "packaged diagnostics failed")
    require(not re.search(r"llvmpipe|lavapipe|software rasterizer|out of range", clean, re.I), "software/wrong GPU selected")
    if family.startswith("apple"):
        require(re.search(r"GPU family\s*:\s*" + re.escape(family) + r"\b", clean), "Metal GPU family mismatch")
        return family
    if family == "gfx1201":
        require(re.search(r"Compute target\s*:\s*gfx1201\b", clean), "ROCm compute target mismatch")
        match = re.search(r"^[ \t]*Device:[ \t]*(.+?)[ \t]+\[gfx1201\][ \t]+\[OK\][ \t]*$", clean, re.M)
    else:
        tuning = {"rdna3": "amd_rdna3", "rdna4": "amd_rdna4", "strix-halo": "amd_rdna4_apu", "intel-xe2": "intel_arc_xe2"}[family]
        require(re.search(r"GPU tuning\s*:\s*" + tuning + r"\s*\|", clean), "Vulkan GPU family mismatch")
        match = re.search(r"^\s*Selected GPU:\s*#\d+\s+(.*?)(?:\s+\(auto\))?\s*$", clean, re.M)
    require(match is not None, "missing selected GPU identity")
    name = match.group(1)
    require(re.fullmatch(r"[A-Za-z0-9 ()_.,+/-]{1,200}", name) and re.search(r"AMD|Radeon|Intel", name, re.I), "unrecognized public GPU name")
    return name


def token_ids(text):
    for line in reversed(text.splitlines()):
        try:
            ids = json.loads(line)
            if isinstance(ids, list) and ids and all(type(n) is int and n >= 0 for n in ids):
                return ids
        except ValueError:
            pass
    raise ValueError("tokenizer did not return token IDs")


def hardware_evidence(diagnostics, family, device):
    gpu = gpu_evidence(diagnostics, family)
    if family.startswith("apple"):
        gpu = output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        require(re.fullmatch(r"Apple M[1-5](?: Pro| Max| Ultra)?", gpu), "unrecognized Apple chip")
        os_version = output(["sw_vers", "-productVersion"]).strip()
        if family == "apple7":
            require(os_version.split(".")[0] == "15", "Apple7 worker must exercise the macOS 15 minimum")
        return {"gpu": gpu, "os": "macOS " + os_version, "driver": "system Metal / macOS " + os_version}
    if family == "gfx1201":
        require(f"(index {device})" in diagnostics, "ROCm selected a different device")
        runtime = Path("/opt/rocm/.info/version").read_text().strip()
        require(runtime.startswith("7.2.4"), "ROCm worker must use the pinned 7.2.4 runtime")
        return {"gpu": gpu, "os": platform.system() + " " + platform.release(), "driver": "ROCm " + runtime}
    require(re.search(r"Selected GPU:\s*#" + str(device) + r"\s", diagnostics), "Vulkan selected a different device")
    summary = output(["vulkaninfo", "--summary"])
    blocks = re.split(r"GPU(\d+):", summary)
    block = next((blocks[i + 1] for i in range(1, len(blocks), 2) if int(blocks[i]) == device), "")
    fields = re.findall(r"^\s*(?:driverName|driverInfo|apiVersion)\s*=\s*(.*?)\s*$", block, re.M)
    require(len(fields) >= 3 and all(0 < len(value) < 256 for value in fields), "missing Vulkan driver evidence")
    return {"gpu": gpu, "os": platform.system() + " " + platform.release(), "driver": " / ".join(fields)}


def kernel_reference(tree, reference, checkout, device, work, env):
    executable = work / "validate-shared-cache"
    output(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(ROOT / "tools/validate_dmmv_shared_cache.c"),
            "-I" + str(checkout / "ggml/include"), "-L" + str(reference), "-Wl,-rpath," + str(reference),
            "-lggml-base", "-lvulkan", "-lm", "-o", str(executable)])
    log = work / "kernel-reference.log"
    with process([str(executable)], log, work, {**env, "ZINC_SHADER_DIR": str(tree / "share/zinc/shaders"), "ZINC_GPU": str(device)}) as child:
        require(child.wait(timeout=300) == 0, "packaged kernel reference validation failed")
    require("PASS: 84 cache-boundary/reference/offset/stride/guard cases" in log.read_text(), "kernel reference suite was incomplete")


def tokenizer_parity(tree, reference, model, work, env):
    for index, text in enumerate(CORPUS):
        prompt = work / f"corpus-{index}.txt"
        prompt.write_text(text)
        zinc = token_ids(output([str(tree / "bin/zinc-tokenize"), str(model), str(prompt)], env))
        llama = token_ids(output([str(reference / "llama-tokenize"), "-m", str(model), "-f", str(prompt),
                                 "--ids", "--no-escape", "--log-disable"], env))
        require(zinc == llama, f"tokenizer parity failed on corpus case {index}")


def cli_generated_ids(text):
    # All three backends expose generated IDs in --debug output. Vulkan's
    # ZINC_LOG_TOPK hook is backend-specific and cannot certify Metal or ROCm.
    tokens = re.search(r"(?:Output|Generated) tokens \((\d+)\):\s*(?:first20=)?\{([\d,\s]*)\}", text)
    require(tokens is not None, "CLI generated token IDs are missing")
    actual = [int(value) for value in re.findall(r"\d+", tokens.group(2))]
    count = int(tokens.group(1))
    require(0 < count <= 20 and len(actual) == count, "CLI generated token IDs are incomplete")
    return actual


def reference_generated_ids(expected):
    probabilities = expected.get("completion_probabilities", [])
    require(isinstance(probabilities, list) and probabilities and
            all(isinstance(item, dict) and type(item.get("id")) is int and item["id"] >= 0 for item in probabilities),
            "reference generated token IDs are missing")
    return [item["id"] for item in probabilities]


def cli_text_parity(text, content):
    # Include whitespace and every line; a matching first word is insufficient.
    output_text = re.search(r"^info\(zinc\): Output (?:text:|\(\d+ tokens\):) ", text, re.M)
    require(output_text is not None and isinstance(content, str) and content and
            text[output_text.end():].startswith(content + "\n"), "full CLI reference continuation differed")


def cli_reference_parity(text, expected):
    require(cli_generated_ids(text) == reference_generated_ids(expected), "CLI reference token IDs differed")
    cli_text_parity(text, expected.get("content"))


def reference_near_tie(actual, expected):
    wanted = reference_generated_ids(expected)
    require(len(actual) == len(wanted), "CLI reference token count differed")
    position = next((i for i, (a, b) in enumerate(zip(actual, wanted)) if a != b), None)
    require(position is not None, "missing reference choice difference")
    top = expected["completion_probabilities"][position].get("top_logprobs", [])
    require(isinstance(top, list) and len(top) >= 2 and all(isinstance(item, dict) for item in top[:2]) and
            all(type(item.get("id")) is int for item in top[:2]) and
            top[0].get("id") == wanted[position] and top[1].get("id") == actual[position] and
            all(type(item.get("logprob")) in (int, float) for item in top[:2]), "reference choice is not a near-tied second choice")
    gap = top[0]["logprob"] - top[1]["logprob"]
    tie = {"position": position, "chosen_id": actual[position], "reference_id": wanted[position], "rank": 2, "logprob_gap": gap}
    # Validate the individual case before starting another reference process.
    limits = policy()["reference"]["agreement"]
    require(limits["max_near_ties_per_case"] >= 1 and 0 <= gap <= limits["max_logprob_gap"], "reference choice exceeds near-tie policy")
    return tie


def qualify_reference_branch(port, tree, model, prompt, expected, actual, text, tie, work, env):
    ids = json_request(port, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True}).get("tokens")
    require(isinstance(ids, list) and ids and all(type(value) is int and value >= 0 for value in ids), "missing reference prompt tokens")
    prompt_file = work / "reference-branch-prompt.txt"
    prompt_file.write_text(prompt)
    native = token_ids(output([str(tree / "bin/zinc-tokenize"), str(model), str(prompt_file)], env))
    require(native == ids, "rendered prompt tokenization differed from reference")
    # Teacher-force the close choice, then demand the entire remaining suffix.
    # An unrelated later error cannot be excused by the earlier near tie.
    start = tie["position"] + 1
    if start < len(actual):
        tail = json_request(port, "/completion", {"prompt": ids + actual[:start], "n_predict": len(actual) - start,
                                                  "temperature": 0, "n_probs": 2, "cache_prompt": False})
        require(reference_generated_ids(tail) == actual[start:], "reference continuation after near tie differed")
    content = json_request(port, "/detokenize", {"tokens": actual}).get("content")
    cli_text_parity(text, content)
    expected["validated_content"] = content


def inference_parity(tree, reference, model, context, device, llama_args, work, env):
    port = free_port()
    samples = []
    generation = policy()["reference"]["generation"]
    # A batched quantized reference can choose different greedy tokens from its
    # own serial path. Freeze the oracle's arithmetic and report these settings.
    def reference_command(port):
        return [str(reference / "llama-server"), "-m", str(model), *llama_args,
               "--batch-size", str(generation["batch_size"]), "--ubatch-size", str(generation["micro_batch"]),
               "--parallel", str(generation["parallel"]), "--cache-type-k", generation["kv_cache"],
               "--cache-type-v", generation["kv_cache"], "--flash-attn", "on" if generation["flash_attention"] else "off",
               "--host", "127.0.0.1", "--port", str(port), "-c", str(context)]
    limit = policy()["reference"]["agreement"]["tokens_per_case"]
    with process(reference_command(port), work / "reference.log", work, env) as child:
        ready(child, port)
        for prompt in PROMPTS:
            rendered = json_request(port, "/apply-template", {"messages": [{"role": "user", "content": prompt}],
                                                              "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
            expected = json_request(port, "/completion", {"prompt": rendered, "n_predict": limit, "temperature": 0,
                                                          "n_probs": 2, "cache_prompt": False})
            samples.append((rendered, expected))
    write_json(work / "reference-samples.json", [{"prompt": prompt, "expected": expected} for prompt, expected in samples])
    # Reference is fully stopped before starting any ZINC GPU process.
    pending = []
    for index, (prompt, expected) in enumerate(samples):
        log = work / f"continuation-{index}.log"
        command = [str(tree / "bin/zinc"), "-m", str(model), "-d", str(device), "--raw", "--debug", "--prompt", prompt, "-n", str(limit), "-c", str(context)]
        with process(command, log, work, env) as child:
            require(child.wait(timeout=360) == 0, "packaged CLI inference failed")
        text = log.read_text(errors="replace")
        actual = cli_generated_ids(text)
        ties = []
        if actual == reference_generated_ids(expected):
            cli_reference_parity(text, expected)
        else:
            tie = reference_near_tie(actual, expected)
            ties.append(tie)
            pending.append((prompt, expected, actual, text, tie))
        expected["comparison"] = {"tokens": len(actual), "near_ties": ties}
    reference_agreement([expected["comparison"] for _, expected in samples], expected_cases=len(PROMPTS))
    if pending:
        port = free_port()
        with process(reference_command(port), work / "reference-near-ties.log", work, env) as child:
            ready(child, port)
            for prompt, expected, actual, text, tie in pending:
                qualify_reference_branch(port, tree, model, prompt, expected, actual, text, tie, work, env)
    write_json(work / "reference-samples.json", [{"prompt": prompt, "expected": expected} for prompt, expected in samples])
    return samples


def reference_http_parity(port, body, samples, work=None):
    # Check the HTTP path independently, including every line and whitespace.
    for index, (prompt, expected) in enumerate(samples):
        actual = json_request(port, "/v1/completions", {**body, "prompt": prompt,
                                                      "max_tokens": policy()["reference"]["agreement"]["tokens_per_case"], "stream": False})
        if work is not None:
            write_json(work / f"http-reference-{index}.json", actual)
        choices = actual.get("choices", [])
        require("validated_content" not in expected or expected.get("comparison", {}).get("near_ties"), "unqualified alternate reference content")
        accepted = [expected.get("content")]
        if "validated_content" in expected:
            accepted.append(expected["validated_content"])
        require(choices and choices[0].get("text") and choices[0]["text"] in accepted,
                "full HTTP reference continuation differed")


def unicode_chat_checks(port, model_id):
    """Require real UTF-8 and matching JSON/SSE, including split code points."""
    for budget in (1, 2, 3, 32):
        body = {"model": model_id, "messages": [{"role": "user", "content": "Reply exactly with café 東京 🚀."}],
                "max_tokens": budget, "temperature": 0, "enable_thinking": False}
        result = json_request(port, "/v1/chat/completions", {**body, "stream": False})
        content = result["choices"][0]["message"].get("content") or ""
        content.encode("utf-8", errors="strict")
        with request(port, "/v1/chat/completions", {**body, "stream": True}) as response:
            payloads = [line[6:] for line in response.read().decode("utf-8", errors="strict").splitlines() if line.startswith("data: ")]
        require(payloads and payloads[-1] == "[DONE]", "Unicode stream did not terminate")
        events = [json.loads(payload) for payload in payloads[:-1]]
        require(not any(event.get("error") for event in events), "Unicode stream returned an error")
        choices = [choice for event in events for choice in event.get("choices", [])]
        streamed = "".join(choice.get("delta", {}).get("content") or "" for choice in choices)
        finishes = [choice["finish_reason"] for choice in choices if choice.get("finish_reason")]
        require(streamed == content and "\ufffd" not in content, "Unicode JSON/SSE text differed or was corrupted")
        require(finishes == [result["choices"][0]["finish_reason"]], "Unicode JSON/SSE finish differed")
        require(any(event.get("usage", {}).get("prompt_tokens", 0) > 0 for event in events), "Unicode stream usage is missing")
        if budget == 32:
            require(all(word in content for word in ("café", "東京", "🚀")), "Unicode echo was incomplete")


def api_checks(tree, model, context, device, work, env, browser_chat, reference_samples):
    port = free_port()
    command = [str(tree / "bin/zinc"), "-m", str(model), "-d", str(device), "--port", str(port), "-c", str(context), "--parallel", "1"]
    with process(command, work / "server.log", work, env) as child:
        ready(child, port)
        available = json_request(port, "/v1/models").get("data")
        model_id = active_model_id(available)
        if browser_chat:
            with request(port, "/") as response:
                require("text/html" in response.headers.get("Content-Type", "") and "<html" in response.read().decode().lower(), "built-in chat page is missing")
        body = {"model": model_id, "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0, "stream": False}
        one = json_request(port, "/v1/completions", body)
        two = json_request(port, "/v1/completions", body)
        require(one.get("choices") and one["choices"][0].get("text") and one["choices"][0]["text"] == two["choices"][0]["text"], "raw completion is empty/non-deterministic")
        require(one.get("usage", {}).get("completion_tokens", 0) > 0, "completion usage is missing")
        reference_http_parity(port, body, reference_samples, work)
        chat = {"model": model_id, "messages": [{"role": "user", "content": "List three uses of copper."}], "max_tokens": 32, "temperature": 0, "stream": True, "enable_thinking": False}
        with request(port, "/v1/chat/completions", chat) as response:
            lines = response.read().decode().splitlines()
        payloads = [line[6:] for line in lines if line.startswith("data: ")]
        require(payloads and payloads[-1] == "[DONE]" and any(json.loads(p).get("choices") for p in payloads[:-1]), "chat SSE contract failed")
        unicode_chat_checks(port, model_id)
        # Disconnect an active stream; then prove the single-slot server recovers.
        with request(port, "/v1/completions", {**body, "stream": True, "max_tokens": 256}) as response:
            require(response.readline(), "stream did not start")
        require(json_request(port, "/v1/completions", body).get("choices"), "server did not recover after stream cancellation")
        # A healthy slot can still retain the wrong SSM reduction/state. Recheck
        # complete frozen continuations, not just a nonempty recovery response.
        after_cancel = work / "reference-after-cancellation"
        after_cancel.mkdir(exist_ok=True)
        reference_http_parity(port, body, reference_samples, after_cancel)
        prompt_file = work / "context.txt"
        low, high, near_prompt, count = 1, context * 2, "", 0
        while low <= high:
            middle = (low + high) // 2
            text = " copper" * middle
            prompt_file.write_text(text)
            size = len(token_ids(output([str(tree / "bin/zinc-tokenize"), str(model), str(prompt_file)], env)))
            if size <= context - 64:
                near_prompt, count, low = text, size, middle + 1
            else:
                high = middle - 1
        require(count >= context - 96, "failed to construct an exact context-boundary prompt")
        near = json_request(port, "/v1/completions", {**body, "prompt": near_prompt, "max_tokens": 8})
        require(near.get("choices") and near.get("usage", {}).get("prompt_tokens", 0) == count, "advertised context boundary was not exercised")
        # Malformed and oversized requests must fail cleanly without poisoning
        # the slot used by the next valid request.
        malformed = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=b'{"prompt":', headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(malformed, timeout=30):
                raise ValueError("malformed JSON was accepted")
        except urllib.error.HTTPError as error:
            require(error.code == 400, "malformed JSON did not return a client error")
        try:
            json_request(port, "/v1/completions", {**body, "prompt": near_prompt * 2})
            raise ValueError("oversized context was accepted")
        except urllib.error.HTTPError as error:
            require(error.code == 400, "oversized context did not return a client error")
        require(json_request(port, "/v1/completions", body).get("choices"), "server did not recover from invalid requests")
        # Long prefill and rejected inputs must not change the short-prompt
        # arithmetic on the reused slot (the MoE "fridge"/"box" regression).
        after_context = work / "reference-after-context"
        after_context.mkdir(exist_ok=True)
        reference_http_parity(port, body, reference_samples, after_context)


def active_model_id(available):
    require(isinstance(available, list) and available, "model listing failed")
    active = [model for model in available if model.get("active") is True]
    if not active and len(available) == 1 and "active" not in available[0]:
        active = available  # ROCm advertises only the loaded model.
    require(len(active) == 1 and isinstance(active[0].get("id"), str) and active[0]["id"], "loaded model identity is ambiguous")
    return active[0]["id"]


def worker_for_profile(base, identifier):
    # Vulkan and HIP can enumerate the same physical GPU at different indices
    # and use different reference builds. Keep their host reservation shared.
    overrides = base.get("profiles", {}).get(identifier, {})
    allowed = {"device", "reference_args", "reference_bin", "reference_source", "models"}
    require(isinstance(overrides, dict) and set(overrides).issubset(allowed), "unsupported worker override; the shared reservation cannot vary by backend")
    result = {**base, **overrides}
    if "models" in overrides:
        result["models"] = {**base.get("models", {}), **overrides["models"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = verify_candidate(args.directory)
    config_path = os.environ.get("ZINC_RELEASE_WORKER_CONFIG")
    require(config_path, "worker must set ZINC_RELEASE_WORKER_CONFIG to its private configuration file")
    require(ROOT not in Path(config_path).expanduser().resolve().parents, "private worker configuration must be outside the checkout")
    config = read_json(config_path)
    profiles = [p for p in policy()["gpu_profiles"] if p["id"] == args.profile]
    require(len(profiles) == 1, "unknown GPU profile")
    profile = profiles[0]
    config = worker_for_profile(config, profile["id"])
    artifact = next(a for a in data["artifacts"] if a["target"] == profile["target"])
    device = config.get("device")
    require(type(device) is int and device >= 0, "worker must configure a device index")
    reference, checkout = Path(config["reference_bin"]).resolve(), Path(config["reference_source"]).resolve()
    commit = policy()["reference"]["commit"]
    require(output(["git", "-C", str(checkout), "rev-parse", "HEAD"]).strip() == commit, "reference checkout is not pinned")
    require(not output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"]).strip(), "reference checkout is dirty")
    require(commit[:7] in output([str(reference / "llama-server"), "--version"]), "reference executable does not match pinned source")
    llama_args = config.get("reference_args")
    require(isinstance(llama_args, list) and llama_args and all(isinstance(a, str) for a in llama_args), "worker must explicitly configure reference GPU flags")
    lock = Path(config["lock_directory"]).expanduser().resolve()
    require(lock.parent.is_dir(), "worker lock parent must exist")
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError("GPU worker is reserved; refusing to interrupt another run")
    try:
        processes = output(["ps", "-eo", "comm="])
        require(not any(Path(line.strip()).name in ("zinc", "llama-server", "llama-cli") for line in processes.splitlines()), "GPU worker is busy")
        with tempfile.TemporaryDirectory(prefix="zinc-packaged-gpu-") as directory:
            work = Path(directory)
            tree = extract_archive(Path(args.directory) / artifact["filename"], work / "unpack", data["version"], profile["target"], data["commit"])
            print("Checking packaged identity, selected GPU and runtime", flush=True)
            probe_binary(tree, data["version"], profile["target"], data["commit"])
            env = {k: v for k, v in os.environ.items() if not k.startswith("ZINC_")}
            # Runtime assets must resolve from the installed archive itself.
            diagnostics = output([str(tree / "bin/zinc"), "--check", "-d", str(device)], env, timeout=360)
            hardware = hardware_evidence(diagnostics, profile["family"], device)
            if profile["family"].startswith("apple"):
                for executable in (tree / "bin/zinc", tree / "bin/zinc-tokenize"):
                    output(["codesign", "--verify", "--strict", str(executable)])
                    output(["spctl", "--assess", "--type", "execute", str(executable)], timeout=120)
            if profile["target"].endswith("vulkan"):
                print("Running 84 packaged shader/reference cases", flush=True)
                kernel_reference(tree, reference, checkout, device, work, env)
            models = []
            for identifier in profile["models"]:
                print(identifier + ": verifying pinned bytes and starting packaged tests", flush=True)
                expected = next(m for m in policy()["models"] if m["id"] == identifier)
                model = Path(config["models"][identifier]).expanduser().resolve()
                require(model.is_file() and model.stat().st_size == expected["size_bytes"] and sha256(model) == expected["sha256"], "model bytes differ from pinned policy: " + identifier)
                if profile["family"].startswith("apple"):
                    require(model == Path.home() / "Library/Caches/zinc/models/models" / identifier / "model.gguf", "Metal workers must use the managed model cache")
                tokenizer_parity(tree, reference, model, work, env)
                print(identifier + ": tokenizer matched; checking reference continuations", flush=True)
                samples = inference_parity(tree, reference, model, expected["context"], device, llama_args, work, env)
                agreement = reference_agreement([sample["comparison"] for _, sample in samples])
                print(identifier + f": reference choices {agreement['exact_tokens']}/{agreement['total_tokens']} exact, "
                      f"{agreement['near_tie_tokens']} qualified near ties; checking HTTP, cancellation and context", flush=True)
                browser_chat = next(t for t in policy()["targets"] if t["id"] == profile["target"])["interfaces"]["browser_chat"]
                api_checks(tree, model, expected["context"], device, work, env, browser_chat, samples)
                models.append({"id": identifier, "sha256": expected["sha256"], "context": expected["context"], "passed": True,
                               "reference_agreement": agreement})
                print(identifier + ": packaged tokenizer, reference continuation and HTTP checks passed", flush=True)
            report = {"schema_version": 1, "profile": profile["id"], "target": profile["target"], "family": profile["family"],
                      **{key: data[key] for key in ("version", "commit", "policy_sha256")},
                      "archive_sha256": sha256(Path(args.directory) / artifact["filename"]), "binary_sha256": sha256(tree / "bin/zinc"),
                      "hardware": hardware, "models": models,
                      "options": {"parallel": 1, "kv_cache": "f32", "mtp": "backend-default", "greedy": True},
                      "reference": {"commit": commit, "generation": policy()["reference"]["generation"],
                                    "server_sha256": sha256(reference / "llama-server"), "tokenizer_sha256": sha256(reference / "llama-tokenize")},
                      "checks": {name: True for name in ("package", "identity", "gpu", "model_hash", "tokenizer_parity", "inference_parity", "http", "cleanup")}}
            if profile["target"].endswith("vulkan"):
                report["checks"]["kernel_reference"] = True
        # Reports contain no machine identifiers, private paths, ports or URLs.
        write_json(args.output, report)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    install_signal_handlers()
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        # Worker paths and subprocess arguments remain private.
        detail = str(error) if type(error) is ValueError else type(error).__name__ + ": inspect the worker configuration and last reported stage"
        sys.exit("packaged GPU validation failed: " + detail)
