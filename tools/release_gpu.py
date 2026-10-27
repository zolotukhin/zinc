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
from release_lib import ROOT, extract_archive, install_signal_handlers, policy, probe_binary, read_json, require, sha256, write_json

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
        match = re.search(r"^\s*Device:\s*(.*?)\s*\[OK\]", clean, re.M)
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


def inference_parity(tree, reference, model, context, device, llama_args, work, env):
    port = free_port()
    samples = []
    command = [str(reference / "llama-server"), "-m", str(model), *llama_args, "--host", "127.0.0.1", "--port", str(port), "-c", str(context)]
    with process(command, work / "reference.log", work, env) as child:
        ready(child, port)
        for prompt in PROMPTS:
            rendered = json_request(port, "/apply-template", {"messages": [{"role": "user", "content": prompt}],
                                                              "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
            expected = json_request(port, "/completion", {"prompt": rendered, "n_predict": 16, "temperature": 0,
                                                          "n_probs": 2, "cache_prompt": False})
            samples.append((rendered, expected))
    # Reference is fully stopped before starting any ZINC GPU process.
    for index, (prompt, expected) in enumerate(samples):
        log = work / f"continuation-{index}.log"
        command = [str(tree / "bin/zinc"), "-m", str(model), "-d", str(device), "--raw", "--prompt", prompt, "-n", "16", "-c", str(context)]
        with process(command, log, work, {**env, "ZINC_LOG_TOPK": "1"}) as child:
            require(child.wait(timeout=360) == 0, "packaged CLI inference failed")
        actual = log.read_text(errors="replace")
        first = re.search(r"topk:.*?top=(\d+)", actual)
        probabilities = expected.get("completion_probabilities", [])
        require(first and probabilities and int(first.group(1)) == probabilities[0].get("id"), "first-token reference parity failed")
        text = re.search(r"Output (?:text:|\(\d+ tokens\):) ([^\n]+)", actual)
        wanted = expected.get("content", "").splitlines()
        require(text and wanted, "missing reference/CLI output")
        a, b = text.group(1).replace(" ", ""), wanted[0].replace(" ", "")
        require(a and b and (a.startswith(b) or b.startswith(a)), "continuation reference parity failed")


def api_checks(tree, model, context, device, work, env, browser_chat):
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
        chat = {"model": model_id, "messages": [{"role": "user", "content": "List three uses of copper."}], "max_tokens": 32, "temperature": 0, "stream": True, "enable_thinking": False}
        with request(port, "/v1/chat/completions", chat) as response:
            lines = response.read().decode().splitlines()
        payloads = [line[6:] for line in lines if line.startswith("data: ")]
        require(payloads and payloads[-1] == "[DONE]" and any(json.loads(p).get("choices") for p in payloads[:-1]), "chat SSE contract failed")
        # Disconnect an active stream; then prove the single-slot server recovers.
        with request(port, "/v1/completions", {**body, "stream": True, "max_tokens": 256}) as response:
            require(response.readline(), "stream did not start")
        require(json_request(port, "/v1/completions", body).get("choices"), "server did not recover after stream cancellation")
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
                inference_parity(tree, reference, model, expected["context"], device, llama_args, work, env)
                print(identifier + ": continuations matched; checking HTTP, cancellation and context", flush=True)
                browser_chat = next(t for t in policy()["targets"] if t["id"] == profile["target"])["interfaces"]["browser_chat"]
                api_checks(tree, model, expected["context"], device, work, env, browser_chat)
                models.append({"id": identifier, "sha256": expected["sha256"], "context": expected["context"], "passed": True})
                print(identifier + ": packaged tokenizer, reference continuation and HTTP checks passed", flush=True)
            report = {"schema_version": 1, "profile": profile["id"], "target": profile["target"], "family": profile["family"],
                      **{key: data[key] for key in ("version", "commit", "policy_sha256")},
                      "archive_sha256": sha256(Path(args.directory) / artifact["filename"]), "binary_sha256": sha256(tree / "bin/zinc"),
                      "hardware": hardware, "models": models,
                      "options": {"parallel": 1, "kv_cache": "f32", "mtp": "backend-default", "greedy": True},
                      "reference": {"commit": commit, "server_sha256": sha256(reference / "llama-server"), "tokenizer_sha256": sha256(reference / "llama-tokenize")},
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
