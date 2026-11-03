#!/usr/bin/env python3
"""Revalidate Gemma defaults, then measure matched ROCm or Vulkan servers on a GPU worker."""
import argparse
import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import release_gpu as gpu
from benchmark_matched_mtp import idle, interrupted, measure, run_json, text_hash
from release_lib import install_signal_handlers, policy, reference_agreement, sha256, worker_reservation, write_json

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "gemma4-26b-a4b-q4k-m"
HARNESS_FILES = (
    "tools/benchmark_gemma_rocm.py", "tools/gemma_rocm_workloads.mjs",
    "tools/benchmark_matched_mtp.py", "tools/parse_mtp_sample.mjs",
    "tools/performance_suite.mjs", "tools/benchmark_speculation.mjs",
    "tools/release_gpu.py", "tools/release_lib.py", "config/benchmark-reference.json",
    "config/release/targets.json", "tools/benchmark_reference.mjs",
)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def reference_identity(directory, source, env, pin):
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == pin, "reference source is not pinned"
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip(), "reference source is modified"
    version = subprocess.check_output([str(directory / "llama-server"), "--version"], env=gpu.reference_environment(directory, env), stderr=subprocess.STDOUT, text=True)
    assert pin[:7] in version, "reference executable differs from source"
    return {"commit": pin, "version": version.strip(), **gpu.reference_artifacts(directory)}


def correctness_prompts(model_pin):
    """Every model keeps short probes; sliding-window models also cross the mask."""
    window = model_pin.get("sliding_window")
    return gpu.PROMPTS + ([
        "The following is reference filler.\n" + "word " * (window + 76)
        + "\nIgnore the filler and answer this request: " + prompt for prompt in gpu.PROMPTS
    ] if window else [])


def correctness(args, work, env, model_pin):
    """Reuse release generation and agreement limits with the recorded oracle."""
    window = model_pin.get("sliding_window")
    context = getattr(args, "context", 4096)
    prompts = correctness_prompts(model_pin)
    oracle_env = {**env, "RADV_PERFTEST": "coop_matrix"}
    oracle_args = ["--device", args.oracle_device, "-ngl", "999", "-t", "4", "-tb", "4", "--spec-type", "none", "--no-cache-prompt"]
    device = getattr(args, "zinc_device", 0)
    print("Correctness: " + str(len(prompts)) + " complete CLI continuations" + (", including sliding-window boundaries" if window else ""), flush=True)
    samples = gpu.inference_parity(args.zinc_tree, args.oracle, args.model,
                                   context, device, oracle_args, work, oracle_env, prompts=prompts)
    cases = []
    for index, (rendered, expected) in enumerate(samples):
        prompt_file = work / f"prompt-{index}.txt"
        prompt_file.write_text(rendered)
        count = len(gpu.token_ids(gpu.output([str(args.zinc_tree / "bin/zinc-tokenize"), str(args.model), str(prompt_file)], env)))
        if index >= len(gpu.PROMPTS):
            assert count > window and count + 16 <= context, "extended prompt did not cross the sliding window"
        actual = gpu.cli_generated_ids((work / f"continuation-{index}.log").read_text())
        cases.append({"prompt": rendered, "prompt_tokens": count,
                      "reference_ids": gpu.reference_generated_ids(expected), "zinc_ids": actual,
                      "reference_text": expected["content"], "validated_text": expected.get("validated_content"),
                      "comparison": expected["comparison"]})
    print("Correctness: complete HTTP continuations, cancellation and context/error recovery", flush=True)
    check_prefill = getattr(args, "backend", "vulkan") == "vulkan" and model_pin["id"] == "qwen36-35b-a3b-q4k-xl"
    prefill = gpu.api_checks(args.zinc_tree, args.model, context, device, work, env, False, samples, check_prefill)
    return {"passed": True, "oracle_backend": "Vulkan", "generation": policy()["reference"]["generation"],
            "agreement_policy": policy()["reference"]["agreement"],
            "reference_agreement": reference_agreement([expected["comparison"] for _, expected in samples], expected_cases=len(prompts)),
            "cases": cases, "http_full_continuations": 3 * len(samples),
            **({"prefill_cancellation": prefill} if check_prefill else {}),
            "http_after_cancellation": True, "http_after_context_error": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zinc-tree", type=Path, required=True)
    parser.add_argument("--zinc-source", type=Path, required=True)
    parser.add_argument("--llama-bin", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True, help="Vulkan reference bin directory")
    parser.add_argument("--oracle-source", type=Path, help="Separate source for the pinned release correctness reference")
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", choices=(MODEL_ID, "gemma4-31b-q4k-m"), default=MODEL_ID)
    parser.add_argument("--backend", choices=("rocm", "vulkan"), default="rocm")
    parser.add_argument("--target", choices=("rdna", "intel"), default="rdna")
    parser.add_argument("--device", type=int, required=True, help="Physical ROCr ordinal or Vulkan index")
    parser.add_argument("--oracle-device", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reference-selected-at", help="Reuse the pin verified before this campaign's builds (at most two hours old)")
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or min(args.rounds, args.runs, args.warmup) < 1:
        parser.error("run on the remote Linux GPU worker with positive repetition counts")
    if args.backend == "rocm" and (args.target != "rdna" or args.model_id != MODEL_ID):
        parser.error("the ROCm protocol currently covers RDNA Gemma MoE only")
    if args.backend == "vulkan" and args.oracle_device != f"Vulkan{args.device}":
        parser.error("the Vulkan reference must select the same physical device")
    args.zinc_device = 0 if args.backend == "rocm" else args.device
    args.oracle_source = args.oracle_source or args.llama_source
    kv = "f32" if args.backend == "rocm" else "f16"
    for name in ("zinc_tree", "zinc_source", "llama_bin", "oracle", "oracle_source", "llama_source", "model", "reservation", "output"):
        setattr(args, name, getattr(args, name).resolve())
    install_signal_handlers()
    signal.signal(signal.SIGHUP, interrupted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": f"zinc-gemma-{args.backend}-v1", "started_at": now(), "passed": False,
              "target": "rdna-rocm" if args.backend == "rocm" else args.target,
              "release_qualification": False, "samples": [],
              "settings": {"backend": args.backend, "context": 4096, "slots": 1, "kv_cache": kv,
                           "mtp": "off", "prefill": "default-serial" if args.backend == "rocm" else "backend-default", "prompt_cache": False,
                           "rounds": args.rounds, "runs": args.runs, "warmup": args.warmup,
                           "temperature": 0, "top_p": 1, "seed": 0}}
    def save():
        write_json(args.output, report)
    save()
    try:
        env = gpu.packaged_environment()
        env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", ZINC_MTP="0",
                   ZINC_KV_CACHE=kv, ZINC_KV_F16="0" if kv == "f32" else "1", ZINC_KV_Q8="0", ZINC_PROMPT_CACHE="0")
        if args.backend == "rocm":
            env["ROCR_VISIBLE_DEVICES"] = str(args.device)
        else:
            env["RADV_PERFTEST"] = "coop_matrix"
        selected = json.loads((ROOT / "config/benchmark-reference.json").read_text())
        pin = selected["commit"]
        latest = subprocess.check_output(["bun", str(ROOT / "tools/benchmark_reference.mjs"), "head"], text=True, timeout=60).strip()
        if args.reference_selected_at:
            assert args.reference_selected_at == selected["checked_at_utc"], "campaign selection differs from the recorded pin"
            age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(args.reference_selected_at.replace("Z", "+00:00"))
            assert datetime.timedelta(0) <= age <= datetime.timedelta(hours=2), "select a fresh reference before building the campaign"
            checked_at, scope = args.reference_selected_at, "campaign-before-build"
        else:
            assert latest == pin, "the performance reference is stale"
            checked_at, scope = now(), "run-start"
        report["latest_reference_check"] = {"commit": pin, "checked_at": checked_at, "scope": scope, "observed_head_at_run": latest}
        report["reference"] = reference_identity(args.llama_bin, args.llama_source, env, pin)
        oracle_pin = pin if args.oracle_source == args.llama_source else policy()["reference"]["commit"]
        report["oracle"] = reference_identity(args.oracle, args.oracle_source, env, oracle_pin)
        assert subprocess.run(["git", "-C", str(args.zinc_source), "diff", "--quiet", "HEAD", "--", "src", "build.zig", "build.zig.zon"]).returncode == 0, "ZINC native source is modified"
        zinc = args.zinc_tree / "bin/zinc"
        report["zinc"] = {"commit": subprocess.check_output(["git", "-C", str(args.zinc_source), "rev-parse", "HEAD"], text=True).strip(), "sha256": sha256(zinc)}
        native_version = gpu.output([str(zinc), "--version"], env).strip()
        assert report["zinc"]["commit"][:7] in native_version, "ZINC executable differs from its source"
        report["zinc"]["version"] = native_version
        model_pin = next(model for model in policy()["models"] if model["id"] == args.model_id)
        assert args.model.stat().st_size == model_pin["size_bytes"] and sha256(args.model) == model_pin["sha256"], "GGUF differs from pinned model"
        report["model"] = {key: model_pin[key] for key in ("id", "sha256", "size_bytes", "sliding_window")}
        report["harness_sha256"] = {name: sha256(ROOT / name) for name in HARNESS_FILES}
        scenarios = run_json(["bun", str(ROOT / "tools/gemma_rocm_workloads.mjs"), args.model_id])
        report["scenarios"] = [{key: value for key, value in row.items() if key != "body"} | {"request_sha256": text_hash(json.dumps(row["body"], sort_keys=True))} for row in scenarios]
        save()
        with worker_reservation(args.reservation):
            idle()
            work = Path(tempfile.mkdtemp(prefix="gemma-rocm-work-", dir=args.output.parent))
            if args.backend == "vulkan":
                family = "rdna4" if args.target == "rdna" else "intel-xe2"
                diagnostics = gpu.output([str(zinc), "--check", "-d", str(args.device)], env, timeout=360)
                report["hardware"] = gpu.hardware_evidence(diagnostics, family, args.device)
                print("Correctness: 376 shader/reference cases", flush=True)
                gpu.kernel_reference(args.zinc_tree, args.oracle, args.oracle_source, args.device, work, env)
                report["kernel_reference"] = {"passed": True, "cases": 376}
            print("Correctness: twelve tokenizer corpus cases", flush=True)
            gpu.tokenizer_parity(args.zinc_tree, args.oracle, args.model, work, env)
            report["correctness"] = {**correctness(args, work, env, model_pin), "tokenizer_cases": len(gpu.CORPUS)}
            save()
            identities, counts = {}, {}
            number = 0
            for round_index in range(args.rounds):
                for engine in (["zinc", "llama"] if round_index % 2 == 0 else ["llama", "zinc"]):
                    idle()
                    port = gpu.free_port()
                    if engine == "zinc":
                        command = [str(zinc), "-m", str(args.model), "-d", str(args.zinc_device), "--context", "4096", "--parallel", "1", "--port", str(port)]
                        active_env = env
                    else:
                        command = [str(args.llama_bin / "llama-server"), "-m", str(args.model), "--device", "ROCm0" if args.backend == "rocm" else args.oracle_device, "-ngl", "999",
                                   "--host", "127.0.0.1", "--port", str(port), "-c", "4096", "--parallel", "1", "-ctk", kv, "-ctv", kv,
                                   "-b", "4096", "-ub", "1024", "--flash-attn", "on", "--spec-type", "none", "--no-cache-prompt", "-t", "4", "-tb", "4"]
                        active_env = gpu.reference_environment(args.llama_bin, env)
                    print(f"Benchmark round {round_index + 1}/{args.rounds}: {engine}, MTP off", flush=True)
                    log = work / f"benchmark-{round_index}-{engine}.log"
                    with gpu.process(command, log, args.zinc_source, active_env) as child:
                        gpu.ready(child, port)
                        for scenario in scenarios:
                            for iteration in range(args.warmup + args.runs):
                                number += 1
                                sample = measure(engine, "off", child, port, log, scenario, work, number)
                                key = (engine, scenario["id"])
                                assert identities.setdefault(key, sample["content_sha256"]) == sample["content_sha256"], "repeated outputs differ"
                                observed = counts.setdefault(scenario["id"], [])
                                observed.append(sample["promptTokens"])
                                assert max(observed) - min(observed) <= 1, "prompt token counts differ"
                                report["samples"].append({"engine": engine, "mtp": "off", "round": round_index + 1,
                                                          "scenario": scenario["id"], "warmup": iteration < args.warmup, **sample})
                                save()
                    idle()
            assert reference_identity(args.llama_bin, args.llama_source, env, pin) == report["reference"], "baseline bytes changed"
            assert reference_identity(args.oracle, args.oracle_source, env, oracle_pin) == report["oracle"], "oracle bytes changed"
            assert sha256(zinc) == report["zinc"]["sha256"], "ZINC bytes changed"
            report.update(passed=True, cleanup="no GPU process remains")
    except BaseException as error:
        report["error"] = str(error) if isinstance(error, (ValueError, AssertionError)) else type(error).__name__
        raise
    finally:
        report["completed_at"] = now()
        save()


if __name__ == "__main__":
    main()
