#!/usr/bin/env python3
"""Run on Intel Linux: correct native Vulkan inference, then matched server metrics."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time

import release_gpu as gpu
from benchmark_gemma_rocm import correctness, now, reference_identity
from benchmark_matched_mtp import idle, interrupted, measure, run_json, text_hash
from benchmark_metrics import MemoryTracker, stream_probe
from release_lib import ROOT, install_signal_handlers, policy, require, sha256, worker_reservation, write_json

HARNESS = ("tools/benchmark_intel.py", "tools/benchmark_metrics.py", "tools/intel_workloads.mjs",
           "tools/benchmark_gemma_rocm.py", "tools/benchmark_matched_mtp.py", "tools/parse_mtp_sample.mjs",
           "tools/performance_suite.mjs", "tools/benchmark_speculation.mjs", "tools/benchmark_reference.mjs",
           "tools/release_gpu.py", "tools/release_lib.py", "config/benchmark-reference.json", "config/release/targets.json")
HARNESS += ("tools/validate_dmmv_shared_cache.c", "tools/validate_gemma_router.c", "tools/validate_attention_window.c")


def native_gpu_layers(log):
    text = re.sub(r"\x1b\[[0-9;]*m", "", log)
    matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", text, re.I)
    require(matches and int(matches[-1][0]) > 0 and matches[-1][0] == matches[-1][1],
            "native performance baseline did not confirm full GPU layer offload")
    return {"offloaded": int(matches[-1][0]), "total": int(matches[-1][1]), "backend": "vulkan"}


def zinc_kv_storage(log):
    lines = [line for line in log.splitlines() if "KV cache:" in line]
    require(any(re.search(r"\bf16\b", line) for line in lines), "ZINC did not confirm F16 KV storage")
    return "f16"


def comparison_eligibility(samples, scenarios):
    eligible = {}
    for scenario in scenarios:
        measured = [s for s in samples if s["scenario"] == scenario["id"] and not s["warmup"]]
        eligible[scenario["id"]] = {s["engine"] for s in measured} == {"zinc", "llama"} and all(s["output_budget_complete"] for s in measured)
    return eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    required = next(p["models"] for p in policy()["gpu_profiles"] if p["id"] == "vulkan-intel-xe2")
    for name in ("zinc-tree", "zinc-source", "llama-bin", "llama-source", "oracle", "oracle-source", "worker-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=required, default=required)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if sys.platform != "linux" or min(args.rounds, args.runs, args.warmup) < 1 or len(args.models) != len(set(args.models)):
        parser.error("run on the remote Linux GPU worker with positive counts and unique models")
    for name in ("zinc_tree", "zinc_source", "llama_bin", "llama_source", "oracle", "oracle_source", "worker_config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    worker = json.loads(args.worker_config.read_text())
    args.device = args.zinc_device = worker["device"]
    require(type(args.device) is int and args.device >= 0, "worker GPU index is invalid")
    require(gpu.reference_device_backend(worker["reference_args"], "vulkan") == "vulkan", "a native Vulkan oracle is required")
    args.oracle_device = "Vulkan" + str(args.device)
    require(args.oracle_device in worker["reference_args"], "reference and ZINC must select the same physical GPU")
    args.backend = "vulkan"
    os.umask(0o077)
    install_signal_handlers()
    # The shared helper installs TERM/INT; SSH disconnects also need unwinding.
    signal.signal(signal.SIGHUP, interrupted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "zinc-intel-performance-v1", "started_at": now(), "passed": False,
              "release_qualification": False, "required_models": required, "selected_models": args.models,
              "complete_model_matrix": False, "complete_performance_matrix": False, "models": [],
              "settings": {"backend": "vulkan", "target": "intel", "kv_cache": "f16", "mtp": "off", "slots": 1,
                           "prompt_cache": False, "rounds": args.rounds, "runs": args.runs, "warmup": args.warmup,
                           "temperature": 0, "top_p": 1, "seed": 0, "context": "pinned model qualification limit",
                           "ttft": "separate warmed 16-token streaming probe per workload/round", "startup": "observed model startup; OS caches are retained",
                           "startup_poll_interval_ms": 500}}
    def save():
        write_json(args.output, report)
    save()
    try:
        pin = json.loads((ROOT / "config/benchmark-reference.json").read_text())["commit"]
        head = gpu.output(["bun", str(ROOT / "tools/benchmark_reference.mjs"), "head"], timeout=60).strip()
        require(head == pin, "refresh and build the latest native performance reference before starting a campaign")
        report["latest_reference_check"] = {"commit": pin, "observed_head": head, "checked_at": now(), "scope": "frozen for this campaign"}
        env = gpu.packaged_environment()
        env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", ZINC_MTP="0", ZINC_KV_CACHE="f16", ZINC_KV_F16="1", ZINC_KV_Q8="0", ZINC_PROMPT_CACHE="0")
        report["reference"] = reference_identity(args.llama_bin, args.llama_source, env, pin)
        report["oracle"] = reference_identity(args.oracle, args.oracle_source, env, policy()["reference"]["commit"])
        zinc = args.zinc_tree / "bin/zinc"
        source = gpu.output(["git", "-C", str(args.zinc_source), "rev-parse", "HEAD"]).strip()
        require(not gpu.output(["git", "-C", str(args.zinc_source), "status", "--porcelain", "--untracked-files=no"]).strip(), "ZINC source is modified")
        version = gpu.output([str(zinc), "--version"], env).strip()
        require(source[:7] in version, "ZINC archive differs from the recorded source")
        report["zinc"] = {"commit": source, "sha256": sha256(zinc), "version": version}
        report["harness_sha256"] = {name: sha256(ROOT / name) for name in HARNESS}
        report["harness_commit"] = gpu.output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip()
        require(not gpu.output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]).strip(), "benchmark harness is modified")
        with worker_reservation(Path(worker["lock_directory"])):
            idle()
            # Keep private request/server logs for profiling and failed-run diagnosis.
            work = Path(tempfile.mkdtemp(prefix="intel-matrix-", dir=args.output.parent))
            report["hardware"] = gpu.hardware_evidence(gpu.output([str(zinc), "--check", "-d", str(args.device)], env, timeout=360), "intel-xe2", args.device)
            print("Correctness: 376 numerical shader cases", flush=True)
            gpu.kernel_reference(args.zinc_tree, args.oracle, args.oracle_source, args.device, work, env)
            report["kernel_reference"] = {"passed": True, "cases": 376}
            for identifier in args.models:
                model_pin = next(m for m in policy()["models"] if m["id"] == identifier)
                args.model, args.context = Path(worker["models"][identifier]), model_pin["context"]
                require(args.model.stat().st_size == model_pin["size_bytes"] and sha256(args.model) == model_pin["sha256"], "model bytes differ from policy: " + identifier)
                current = {"id": identifier, "sha256": model_pin["sha256"], "context": args.context, "passed": False,
                           "samples": [], "stream_samples": [], "processes": []}
                report["models"].append(current)
                model_work = work / identifier
                model_work.mkdir()
                print("Correctness: " + identifier, flush=True)
                gpu.tokenizer_parity(args.zinc_tree, args.oracle, args.model, model_work, env)
                current["correctness"] = correctness(args, model_work, env, model_pin)
                scenarios = run_json(["bun", str(ROOT / "tools/intel_workloads.mjs"), identifier])
                current["scenarios"] = [{k: v for k, v in row.items() if k != "body"} | {"request_sha256": text_hash(json.dumps(row["body"], sort_keys=True))} for row in scenarios]
                identities, counts, number = {}, {}, 0
                for round_index in range(args.rounds):
                    for engine in (["zinc", "llama"] if round_index % 2 == 0 else ["llama", "zinc"]):
                        idle()
                        port = gpu.free_port()
                        if engine == "zinc":
                            command = [str(zinc), "-m", str(args.model), "-d", str(args.device), "--context", str(args.context), "--parallel", "1", "--port", str(port)]
                            active_env = env
                        else:
                            command = [str(args.llama_bin / "llama-server"), "-m", str(args.model), "--device", args.oracle_device, "-ngl", "999", "--host", "127.0.0.1", "--port", str(port),
                                       "-c", str(args.context), "--parallel", "1", "-ctk", "f16", "-ctv", "f16", "-b", "4096", "-ub", "1024", "--flash-attn", "on",
                                       "--spec-type", "none", "--no-cache-prompt", "-lv", "4", "-t", "4", "-tb", "4"]
                            active_env = gpu.reference_environment(args.llama_bin, env)
                        log = model_work / f"{round_index}-{engine}.log"
                        print(f"Benchmark {identifier}, round {round_index + 1}/{args.rounds}: {engine}", flush=True)
                        started = time.monotonic()
                        with gpu.process(command, log, args.zinc_source, active_env) as child, MemoryTracker(child.pid) as memory:
                            gpu.ready(child, port)
                            process = {"engine": engine, "round": round_index + 1, "startup_ms": (time.monotonic() - started) * 1000}
                            if engine == "llama":
                                process["gpu_offload"] = native_gpu_layers(log.read_text(errors="replace"))
                            else:
                                process["kv_cache"] = zinc_kv_storage(log.read_text(errors="replace"))
                            current["processes"].append(process)
                            for scenario in scenarios:
                                for iteration in range(args.warmup + args.runs):
                                    number += 1
                                    # Natural EOS is retained as evidence, never a full-budget win.
                                    sample = measure(engine, "off", child, port, log, scenario, model_work, number, require_full_budget=False)
                                    key = (engine, scenario["id"])
                                    require(identities.setdefault(key, sample["content_sha256"]) == sample["content_sha256"], "repeated complete outputs differ")
                                    observed = counts.setdefault(scenario["id"], [])
                                    observed.append(sample["promptTokens"])
                                    require(max(observed) - min(observed) <= 1, "engines processed different prompt counts")
                                    current["samples"].append({"engine": engine, "round": round_index + 1, "scenario": scenario["id"], "warmup": iteration < args.warmup, **sample})
                                    save()
                                probe = stream_probe(port, scenario["body"])
                                current["stream_samples"].append({"engine": engine, "round": round_index + 1, "scenario": scenario["id"], **probe})
                            process["memory"] = memory.snapshot()
                        idle()
                        save()
                current["comparison_eligible"] = comparison_eligibility(current["samples"], scenarios)
                current["passed"] = True
                save()
            require(reference_identity(args.llama_bin, args.llama_source, env, pin) == report["reference"], "performance reference changed")
            require(reference_identity(args.oracle, args.oracle_source, env, policy()["reference"]["commit"]) == report["oracle"], "correctness oracle changed")
            require(sha256(zinc) == report["zinc"]["sha256"], "ZINC binary changed")
            require(all(sha256(ROOT / name) == digest for name, digest in report["harness_sha256"].items()), "benchmark harness changed")
            idle()
        complete_models = set(args.models) == set(required)
        report.update(passed=True, complete_model_matrix=complete_models,
                      complete_performance_matrix=complete_models and all(all(m["comparison_eligible"].values()) for m in report["models"]),
                      cleanup="no GPU process remains")
    except BaseException as error:
        report["error"] = str(error) if type(error) in (ValueError, AssertionError) else type(error).__name__
        raise
    finally:
        report["completed_at"] = now()
        save()


if __name__ == "__main__":
    main()
