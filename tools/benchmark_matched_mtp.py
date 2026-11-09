#!/usr/bin/env python3
"""Run on the GPU host: matched MTP on/on and off/off, owned server groups only."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import release_gpu as gpu
from release_lib import install_signal_handlers, sha256, worker_reservation, write_json

ROOT = Path(__file__).resolve().parent.parent


def idle():
    names = subprocess.check_output(["ps", "-eo", "comm="], text=True).splitlines()
    if any(Path(name.strip()).name in ("zinc", "llama-server", "llama-cli", "llama-bench") for name in names):
        raise ValueError("GPU worker is busy; no process was interrupted")


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def run_json(command):
    return json.loads(subprocess.check_output(command, text=True, timeout=60))


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def measure(engine, mode, child, port, log, scenario, directory, number, require_full_budget=True, completion_mode="chat"):
    if completion_mode not in ("chat", "raw"):
        raise ValueError("unknown completion measurement mode")
    offset = log.stat().st_size
    started = time.monotonic()
    endpoint = "/v1/completions" if completion_mode == "raw" else "/v1/chat/completions"
    with gpu.request(port, endpoint, scenario["body"], timeout=600) as response:
        body = json.load(response)
    latency = (time.monotonic() - started) * 1000
    if body.get("error"):
        raise ValueError("engine returned an API error")
    choice = body.get("choices", [{}])[0]
    content = (choice.get("text") if completion_mode == "raw" else choice.get("message", {}).get("content")) or ""
    if not content or "\ufffd" in content:
        raise ValueError("empty or corrupted generated content")
    body_file = directory / f"{number}.json"
    log_file = directory / f"{number}.log"
    body_file.write_text(json.dumps(body))
    deadline = time.monotonic() + 5
    while True:
        assert child.poll() is None, "server died during measurement"
        delta = log.read_bytes()[offset:].decode(errors="replace")
        if engine != "zinc" or ("Generated " in delta and (mode == "off" or "request accepted" in delta)):
            break
        if time.monotonic() > deadline:
            raise ValueError("request-specific generation/MTP telemetry was not produced")
        time.sleep(0.02)
    log_file.write_text(delta)
    parsed = run_json(["bun", str(ROOT / "tools/parse_mtp_sample.mjs"), engine, str(body_file), str(log_file)])
    active = parsed.get("speculative")
    if (mode == "on") != bool(active and active.get("drafted", 0) > 0):
        raise ValueError("requested MTP mode did not match observed drafting telemetry")
    if not parsed.get("prefillTokens") or parsed["prefillTokens"] < parsed["promptTokens"] - 1:
        raise ValueError("prompt cache reduced measured prefill work")
    complete = parsed["generatedTokens"] == scenario["max_tokens"] and choice.get("finish_reason") == "length"
    if not (0 < parsed["generatedTokens"] <= scenario["max_tokens"]) or choice.get("finish_reason") not in ("length", "stop"):
        raise ValueError("generation returned invalid token accounting or finish reason")
    if require_full_budget and not complete:
        raise ValueError("generation did not reach the common output budget")
    return {**parsed, "totalLatencyMs": latency, "totalTps": (parsed["promptTokens"] + parsed["generatedTokens"]) / (latency / 1000), "content_sha256": text_hash(content),
            "content_bytes": len(content.encode()), "finish_reason": choice.get("finish_reason"), "output_budget_complete": complete}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zinc", type=Path, required=True)
    parser.add_argument("--zinc-source", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("vulkan", "rocm"), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--llama-device", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--drafts", type=int, default=3)
    parser.add_argument("--engines", nargs="+", choices=("zinc", "llama"), default=["zinc", "llama"])
    parser.add_argument("--record-output-drift", action="store_true", help="retain mode differences as failed controls; repeats must still be deterministic")
    parser.add_argument("--scenarios", nargs="+", choices=("core", "context-medium", "context-long", "decode-extended"))
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("run this GPU harness on the remote Linux worker")
    if min(args.rounds, args.runs, args.warmup, args.drafts) < 1 or args.drafts > 16:
        parser.error("rounds, runs, warmup and drafts must be positive; drafts <= 16")
    for name in ("zinc", "zinc_source", "llama_server", "llama_source", "model", "output"):
        setattr(args, name, getattr(args, name).resolve())
    install_signal_handlers()
    signal.signal(signal.SIGHUP, interrupted)
    report = {"schema_version": "zinc-matched-mtp-v1", "backend": args.backend,
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "passed": False, "protocol_passed": False, "release_qualification": False, "samples": [], "output_drift": [],
              "settings": {"context": 4096, "slots": 1, "kv_cache": "f32", "drafts": args.drafts,
                           "runs": args.runs, "warmup": args.warmup, "rounds": args.rounds,
                           "temperature": 0, "top_p": 1, "seed": 0, "prompt_cache": False, "engines": args.engines}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        write_json(args.output, report)
    save()
    try:
        pin = json.loads((ROOT / "config/benchmark-reference.json").read_text())["commit"]
        source_pin = subprocess.check_output(["git", "-C", str(args.llama_source), "rev-parse", "HEAD"], text=True).strip()
        assert source_pin == pin, "reference source is not the pinned latest revision"
        assert not subprocess.check_output(["git", "-C", str(args.llama_source), "status", "--porcelain", "--untracked-files=no"], text=True).strip(), "reference source is modified"
        reference = gpu.reference_artifacts(args.llama_server.parent)
        env = gpu.packaged_environment()
        ref_env = gpu.reference_environment(args.llama_server.parent, env)
        version = subprocess.check_output([str(args.llama_server), "--version"], env=ref_env, stderr=subprocess.STDOUT, text=True)
        assert pin[:7] in version, "reference executable version differs from its checkout"
        report["reference"] = {"commit": pin, "version": version.strip(), **reference}
        assert subprocess.run(["git", "-C", str(args.zinc_source), "diff", "--quiet", "HEAD", "--", "src", "build.zig", "build.zig.zon"]).returncode == 0, "ZINC build source is modified"
        report["zinc"] = {"commit": subprocess.check_output(["git", "-C", str(args.zinc_source), "rev-parse", "HEAD"], text=True).strip(), "sha256": sha256(args.zinc)}
        model_pin = next(model for model in json.loads((ROOT / "config/release/targets.json").read_text())["models"] if model["id"] == "qwen38-27b-q4k-m")
        assert args.model.stat().st_size == model_pin["size_bytes"] and sha256(args.model) == model_pin["sha256"], "model pin differs"
        report["model"] = {key: model_pin[key] for key in ("id", "sha256", "size_bytes")}
        report["harness_sha256"] = {name: sha256(ROOT / name) for name in ("tools/benchmark_matched_mtp.py", "tools/parse_mtp_sample.mjs", "tools/mtp_workloads.mjs", "tools/performance_suite.mjs", "tools/benchmark_speculation.mjs", "tools/release_gpu.py", "tools/release_lib.py", "config/benchmark-reference.json")}
        scenarios = run_json(["bun", str(ROOT / "tools/mtp_workloads.mjs")])
        if args.scenarios:
            scenarios = [scenario for scenario in scenarios if scenario["id"] in args.scenarios]
        report["scenarios"] = [{key: value for key, value in scenario.items() if key != "body"} | {"request_sha256": text_hash(json.dumps(scenario["body"], sort_keys=True))} for scenario in scenarios]
        save()
        identities = {}
        mode_identities = {}
        prompt_counts = {}
        args.reservation.parent.mkdir(parents=True, exist_ok=True)
        with worker_reservation(args.reservation):
            idle()
            work = Path(tempfile.mkdtemp(prefix="matched-mtp-work-", dir=args.output.parent))
            number = 0
            for round_index in range(args.rounds):
                treatments = [(engine, mode) for engine, mode in [("zinc", "off"), ("llama", "off"), ("zinc", "on"), ("llama", "on")] if engine in args.engines]
                if round_index % 2:
                    treatments.reverse()
                for engine, mode in treatments:
                    idle()
                    port = gpu.free_port()
                    active_env = dict(env if engine == "zinc" else ref_env)
                    active_env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
                    if args.backend == "rocm":
                        active_env["ROCR_VISIBLE_DEVICES"] = str(args.device)
                    else:
                        active_env["RADV_PERFTEST"] = "coop_matrix"
                    if engine == "zinc":
                        active_env.update(ZINC_MTP="1" if mode == "on" else "0", ZINC_MTP_DRAFTS=str(args.drafts), ZINC_KV_CACHE="f32", ZINC_KV_F16="0", ZINC_KV_Q8="0", ZINC_PROMPT_CACHE="0")
                        command = [str(args.zinc), "-m", str(args.model), "--context", "4096", "--parallel", "1", "-d", "0" if args.backend == "rocm" else str(args.device), "--port", str(port)]
                    else:
                        command = [str(args.llama_server), "-m", str(args.model), "--host", "127.0.0.1", "--port", str(port), "--device", args.llama_device, "-ngl", "999", "--ctx-size", "4096", "--parallel", "1", "-ctk", "f32", "-ctv", "f32", "-ctkd", "f32", "-ctvd", "f32", "--spec-draft-sampling", "greedy", "-b", "4096", "-ub", "1024", "--flash-attn", "on", "-lv", "4", "--no-cache-prompt", "-t", "4", "-tb", "4", "--spec-type", "draft-mtp" if mode == "on" else "none", "--spec-draft-n-max", str(args.drafts), "--spec-draft-p-min", "0"]
                    log = work / f"{round_index}-{engine}-{mode}.log"
                    print(f"Round {round_index + 1}/{args.rounds}: {engine} MTP {mode}", flush=True)
                    with gpu.process(command, log, args.zinc_source, active_env) as child:
                        gpu.ready(child, port)
                        startup = log.read_text(errors="replace")
                        if engine == "zinc":
                            cache_lines = [line for line in startup.splitlines() if "KV cache:" in line]
                            if not any("f32" in line for line in cache_lines):
                                raise ValueError("ZINC did not confirm F32 KV storage")
                        for scenario in scenarios:
                            for iteration in range(args.warmup + args.runs):
                                number += 1
                                sample = measure(engine, mode, child, port, log, scenario, work, number)
                                identity = (engine, scenario["id"], mode)
                                previous = identities.setdefault(identity, sample["content_sha256"])
                                if previous != sample["content_sha256"]:
                                    raise ValueError(f"{engine} output differs between repeated requests: {scenario['id']} MTP {mode}")
                                mode_identity = (engine, scenario["id"])
                                prior_mode, prior_hash = mode_identities.setdefault(mode_identity, (mode, sample["content_sha256"]))
                                if prior_mode != mode and prior_hash != sample["content_sha256"]:
                                    if not args.record_output_drift:
                                        raise ValueError(f"{engine} output differs between MTP modes: {scenario['id']}")
                                    drift = {"engine": engine, "scenario": scenario["id"], "off_sha256": prior_hash if prior_mode == "off" else sample["content_sha256"], "on_sha256": sample["content_sha256"] if mode == "on" else prior_hash}
                                    if drift not in report["output_drift"]:
                                        report["output_drift"].append(drift)
                                counts = prompt_counts.setdefault(scenario["id"], [])
                                counts.append(sample["promptTokens"])
                                if max(counts) - min(counts) > 1:
                                    raise ValueError("engines did not process matching prompt token counts")
                                report["samples"].append({"engine": engine, "mtp": mode, "round": round_index + 1, "scenario": scenario["id"], "warmup": iteration < args.warmup, **sample})
                                save()
                    idle()
            assert gpu.reference_artifacts(args.llama_server.parent) == reference, "reference libraries changed during benchmark"
            assert sha256(args.zinc) == report["zinc"]["sha256"], "ZINC binary changed during benchmark"
            report.update(protocol_passed=True, passed=not report["output_drift"], cleanup="no GPU process remains")
    except BaseException as error:
        report["error"] = str(error) if isinstance(error, (ValueError, AssertionError)) else type(error).__name__
        raise
    finally:
        report["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
