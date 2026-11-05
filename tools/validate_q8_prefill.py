#!/usr/bin/env python3
"""Remote Intel GPU regression: compare batched and serial full-vocabulary logits."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import tempfile

import release_gpu as gpu
from benchmark_matched_mtp import idle, interrupted
from release_lib import install_signal_handlers, policy, sha256, worker_reservation, write_json

PREFILL_TOKEN_COUNTS = (1, 2, 15, 16, 31, 32, 33, 95, 96, 97, 257, 383, 384, 385)


class Q8RegressionError(ValueError):
    """A static diagnostic owned by this validator, safe for public reports."""


def require(condition, message):
    if not condition:
        raise Q8RegressionError(message)


def validated_logits(log, tokens):
    # validate mode falls back silently when the model is ineligible. Requiring
    # this record makes the old Q8_0 eligibility rejection a decisive failure.
    rows = re.findall(r"prefillBatched validate\[(ok|exceeded)\]: last-token logits "
                      r"max_abs_diff=(\S+).*? nonfinite=(\d+) tol=(\S+) n_tokens=(\d+)", log)
    require(len(rows) == 1, "the Q8 batched-versus-serial comparison did not run exactly once")
    status, maximum, nonfinite, tolerance, count = rows[0]
    maximum, tolerance = float(maximum), float(tolerance)
    require(status == "ok" and int(nonfinite) == 0 and int(count) == tokens and
            math.isfinite(maximum) and math.isfinite(tolerance) and
            0 <= maximum <= tolerance <= 1e-3,
            "Q8 batched prefill differs from the serial full-vocabulary reference")
    return {"prompt_tokens": int(count), "maximum_absolute_logit_difference": maximum,
            "tolerance": tolerance, "nonfinite": int(nonfinite)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("zinc-tree", "worker-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--experimental-opt-in", action="store_true", help="exercise a pre-default experimental build")
    args = parser.parse_args()
    require(sys.platform == "linux", "run this regression on the remote Linux GPU worker")
    os.umask(0o077)
    install_signal_handlers()
    signal.signal(signal.SIGHUP, interrupted)
    worker = json.loads(args.worker_config.read_text())
    model_pin = next(m for m in policy()["models"] if m["id"] == "qwen3-0.6b-q8-0")
    model = Path(worker["models"][model_pin["id"]])
    device = worker["device"]
    require(type(device) is int and device >= 0, "worker GPU index is invalid")
    require(model.stat().st_size == model_pin["size_bytes"] and sha256(model) == model_pin["sha256"],
            "Q8 model bytes differ from the qualification fixture")
    tree, output = args.zinc_tree.resolve(), args.output.resolve()
    require(not output.exists(), "preserve the previous Q8 regression report before rerunning")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = gpu.packaged_environment()
    # This isolates the F32 matrix/attention arithmetic. Rounding intermediate
    # K/V to F16 amplifies small summation-order differences across layers
    # (two-token control: 0.003924 with F16 versus 0.000033 with F32). Keep the
    # same numerical tolerance; qualify actual F16 inference separately against
    # complete native continuations, chunk boundaries and API recovery.
    env.update(ZINC_BATCHED_PREFILL="validate",
               ZINC_MTP="0", ZINC_KV_CACHE="f32", ZINC_KV_F16="0", ZINC_KV_Q8="0",
               ZINC_PROMPT_CACHE="0")
    if args.experimental_opt_in:
        env["ZINC_INTEL_BATCHED_PREFILL"] = "1"
    report = {"schema_version": "zinc-q8-prefill-regression-v1", "passed": False,
              "release_qualification": False, "model": model_pin["id"],
              "kv_cache": "f32", "scope": "numerical; F16 native/API qualification is a separate gate",
              "experimental_opt_in": args.experimental_opt_in,
              "model_sha256": model_pin["sha256"], "binary_sha256": sha256(tree / "bin/zinc"),
              "q8_gemm_shader_sha256": sha256(tree / "share/zinc/shaders/mul_mm_q8_0.spv"),
              "cases": []}
    active_count = None
    try:
        with worker_reservation(Path(worker["lock_directory"])):
            idle()
            report["hardware"] = gpu.hardware_evidence(
                gpu.output([str(tree / "bin/zinc"), "--check", "-d", str(device)], env, timeout=360), "intel-xe2", device)
            work = Path(tempfile.mkdtemp(prefix="q8-prefill-", dir=output.parent))
            # Exercise tile/chunk boundaries and a single-token dispatch.
            # Nonzero KV positions need the separate native/API gates.
            for count in PREFILL_TOKEN_COUNTS:
                active_count = count
                prompt = " ".join(["word"] * count)
                prompt_file = work / f"prompt-{count}.txt"
                prompt_file.write_text(prompt)
                tokens = gpu.token_ids(gpu.output([str(tree / "bin/zinc-tokenize"), str(model), str(prompt_file)], env))
                require(len(tokens) == count, "the Q8 fixture did not exercise its intended token boundary")
                log = work / f"validate-{count}.log"
                command = [str(tree / "bin/zinc"), "-m", str(model), "-d", str(device), "--raw",
                           "--prompt", prompt, "-n", "1", "-c", str(model_pin["context"])]
                with gpu.process(command, log, work, env) as child:
                    require(child.wait(timeout=360) == 0, "Q8 prefill regression process failed")
                text = log.read_text(errors="replace")
                require(any(re.search(r"\bKV cache:.*\bf32\b", line) for line in text.splitlines()),
                        "numerical Q8 validation did not confirm F32 KV storage")
                report["cases"].append(validated_logits(text, count))
                idle()
                write_json(output, report)
            active_count = None
            report.update(passed=True, cleanup="no GPU process remains")
    except BaseException as error:
        # Keep completed cases and the boundary that failed. Unexpected process
        # errors, including generic ValueError, can contain private worker
        # paths/commands. Only this validator's static diagnostics are public.
        report["passed"] = False
        report.pop("cleanup", None)
        report["failure"] = {
            "prompt_tokens": active_count,
            "error_type": type(error).__name__,
            "reason": str(error) if type(error) is Q8RegressionError else type(error).__name__,
        }
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            report["interrupted"] = True
        raise
    finally:
        write_json(output, report)


if __name__ == "__main__":
    main()
