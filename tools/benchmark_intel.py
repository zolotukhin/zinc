#!/usr/bin/env python3
"""Validate Intel Vulkan inference, then compare matched Vulkan or SYCL servers."""

import argparse
from contextlib import contextmanager
import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import tempfile
import time

import release_gpu as gpu
import intel_sycl_environment as sycl
from intel_native_probe import validate_probe
from benchmark_gemma_rocm import correctness, now, reference_identity
from benchmark_matched_mtp import idle, interrupted, measure, run_json, text_hash
from benchmark_metrics import MemoryTracker, stream_probe
from release_lib import ROOT, install_signal_handlers, policy, reference_agreement, require, sha256, worker_reservation, write_json

HARNESS = ("tools/benchmark_intel.py", "tools/benchmark_metrics.py", "tools/intel_workloads.mjs",
           "tools/benchmark_gemma_rocm.py", "tools/benchmark_matched_mtp.py", "tools/parse_mtp_sample.mjs",
           "tools/performance_suite.mjs", "tools/benchmark_speculation.mjs", "tools/benchmark_reference.mjs",
           "tools/release_gpu.py", "tools/release_lib.py", "config/benchmark-reference.json", "config/release/targets.json")
HARNESS += ("tools/validate_dmmv_shared_cache.c", "tools/validate_q8_decode.c", "tools/validate_gemma_router.c", "tools/validate_attention_window.c")
HARNESS += ("tools/validate_gemma_geglu.c",)
HARNESS += ("tools/intel_sycl_environment.py", "tools/intel_native_probe.py", "tools/intel_sycl_probe.cpp", "tools/intel_probe_json.h")


@contextmanager
def process_observations(record, collectors):
    """Save diagnostics inside the owned process context, before its cleanup."""
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        observation_error = None
        for key, collect in collectors:
            try:
                record[key] = collect()
            except Exception as error:
                # A dying process can lose /proc entries. Retain that fact
                # without replacing the measurement/cancellation that failed.
                record[key + "_error"] = type(error).__name__
                if not failed and observation_error is None:
                    observation_error = (key, error)
        if observation_error is not None:
            key, error = observation_error
            # The model-level ValueError handler saves this record and honors
            # --keep-going. A raw probe TimeoutExpired/RuntimeError would skip
            # that handler and abort the remaining model matrix.
            raise ValueError(f"required process {key} observation unavailable ({type(error).__name__})") from error


def reference_selection(selected, head, selected_at=None, current=None):
    """Freeze the observed upstream pin before a long build, never its successor."""
    pin = selected.get("commit")
    require(isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{40}", pin), "invalid performance reference pin")
    require(isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}", head), "invalid observed upstream HEAD")
    current = current or datetime.datetime.now(datetime.timezone.utc)
    if selected_at is None:
        require(head == pin, "refresh and build the latest native performance reference before starting a campaign")
        return {"commit": pin, "observed_head": head, "checked_at": current.isoformat(), "scope": "run-start"}
    require(selected_at == selected.get("checked_at_utc"), "campaign selection differs from the recorded pin")
    try:
        timestamp = datetime.datetime.fromisoformat(selected_at.replace("Z", "+00:00"))
        require(timestamp.tzinfo is not None, "reference selection needs an explicit timezone")
        age = current - timestamp
    except (TypeError, AttributeError, ValueError) as error:
        raise ValueError("invalid reference selection timestamp") from error
    require(datetime.timedelta(0) <= age <= datetime.timedelta(hours=2), "select a fresh reference before building the campaign")
    # observed_head is the selection-time pin recorded by benchmark_reference;
    # retain the live HEAD separately if upstream advanced during compilation.
    return {"commit": pin, "observed_head": pin, "checked_at": selected_at,
            "scope": "campaign-before-build", "observed_head_at_run": head}


def native_gpu_layers(log, backend="vulkan"):
    require(backend in ("vulkan", "sycl"), "unsupported Intel performance reference backend")
    text = re.sub(r"\x1b\[[0-9;]*m", "", log)
    matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", text, re.I)
    require(matches and int(matches[-1][0]) > 0 and matches[-1][0] == matches[-1][1],
            "native performance baseline did not confirm full GPU layer offload")
    if backend == "sycl":
        require(re.search(r"\bSYCL\d+\b[^\n]*model buffer", text, re.I),
                "native performance baseline did not confirm a SYCL model buffer")
    return {"offloaded": int(matches[-1][0]), "total": int(matches[-1][1]), "backend": backend}


def zinc_kv_storage(log):
    lines = [line for line in log.splitlines() if "KV cache:" in line]
    require(any(re.search(r"\bf16\b", line) for line in lines), "ZINC did not confirm F16 KV storage")
    return "f16"


def packaged_shaders(tree):
    files = {str(path.relative_to(tree)): sha256(path) for path in sorted(Path(tree).rglob("*.spv")) if path.is_file()}
    require(files, "ZINC artifact has no packaged shaders")
    return files


def performance_reference_command(args, port):
    return [str(args.llama_bin / "llama-server"), "-m", str(args.model), "--device", args.reference_device,
            "-ngl", "999", "--host", "127.0.0.1", "--port", str(port), "-c", str(args.context), "--parallel", "1",
            "-ctk", "f16", "-ctv", "f16", "-b", "4096", "-ub", "1024", "--flash-attn", "on",
            "--spec-type", "none", "--no-cache-prompt", "--cache-ram", "0", "-lv", "4", "-t", "4", "-tb", "4"]


def raw_scenarios(args, scenarios, work, env):
    """Render once with the frozen oracle, then send identical raw bytes."""
    port = gpu.free_port()
    generation = policy()["reference"]["generation"]
    command = [str(args.oracle / "llama-server"), "-m", str(args.model), "--device", args.oracle_device, "-ngl", "999",
               "-c", str(args.context), "--parallel", "1", "-b", str(generation["batch_size"]), "-ub", str(generation["micro_batch"]),
               "-ctk", generation["kv_cache"], "-ctv", generation["kv_cache"], "--flash-attn", "off",
               "--spec-type", "none", "--no-cache-prompt", "--host", "127.0.0.1", "--port", str(port), "-t", "4", "-tb", "4"]
    result = []
    with gpu.process(command, work / "render-workloads.log", work, gpu.reference_environment(args.oracle, env)) as child:
        gpu.ready(child, port, require_ok=True)
        for scenario in scenarios:
            rendered = gpu.json_request(port, "/apply-template", {"messages": scenario["body"]["messages"],
                                        "chat_template_kwargs": {"enable_thinking": False}}).get("prompt")
            require(isinstance(rendered, str) and rendered, "oracle did not render the shared workload")
            result.append({**scenario, "prompt_mode": "raw", "rendered_prompt": rendered,
                           "body": {"prompt": rendered, "max_tokens": scenario["max_tokens"], "temperature": 0,
                                    "top_p": 1, "seed": 0, "cache_prompt": False}})
    return result


def native_reference_validation(args, work, env, reference_env, record=None):
    """Check the timed SYCL configuration against the frozen Vulkan oracle."""
    # Attach this object to the parent before calling. A rejected configuration
    # must retain its obtained continuations and live runtime, without becoming
    # an accepted baseline just because the process returned usable text.
    # Each retry needs a fresh record; clearing a used one would erase evidence.
    record = {} if record is None else record
    require(isinstance(record, dict) and not record, "native preflight record must be fresh")
    record.update(passed=False, backend="sycl", oracle_backend="vulkan", oracle_commit=policy()["reference"]["commit"],
                  performance_configuration_checked=False, validation_stage="startup", cases=[])
    samples = json.loads((work / "reference-samples.json").read_text())
    require(samples, "native reference validation needs frozen oracle samples")
    port, cases, pending = gpu.free_port(), record["cases"], []
    log = work / "native-reference-validation.log"
    with gpu.process(performance_reference_command(args, port), log, work,
                     gpu.reference_environment(args.llama_bin, reference_env)) as child:
        with process_observations(record, [("runtime", lambda: sycl.runtime_evidence(child.pid, args.sycl_prefix))]):
            gpu.ready(child, port, require_ok=True)
            record["gpu_offload"] = native_gpu_layers(log.read_text(errors="replace"), "sycl")
            record["performance_configuration_checked"] = True
            record["validation_stage"] = "native_continuations"
            for index, sample in enumerate(samples):
                prompt, expected = sample["prompt"], sample["expected"]
                wanted = gpu.reference_generated_ids(expected)
                actual = gpu.json_request(port, "/completion", {"prompt": prompt, "n_predict": len(wanted),
                                          "temperature": 0, "n_probs": 2, "cache_prompt": False})
                write_json(work / f"native-reference-case-{index}.json", {"prompt": prompt, "expected": expected, "actual": actual})
                case = {"reference_ids": wanted, "native_ids": [], "content": actual.get("content"), "comparison": {"tokens": 0, "near_ties": []}}
                cases.append(case)
                ids = gpu.reference_generated_ids(actual)
                case["native_ids"] = ids
                comparison = case["comparison"]
                comparison["tokens"] = len(ids)
                require(len(ids) == len(wanted), "SYCL reference continuation count differs")
                if ids == wanted:
                    require(actual.get("content") == expected.get("content"), "SYCL reference continuation text differs")
                else:
                    tie = gpu.reference_near_tie(ids, expected)
                    comparison["near_ties"].append(tie)
                    pending.append((prompt, actual, ids, tie))
    record["validation_stage"] = "aggregate_agreement"
    record["agreement"] = reference_agreement([case["comparison"] for case in cases], expected_cases=len(samples))
    if pending:
        record["validation_stage"] = "oracle_branches"
        # The native process has exited before this oracle launch. A close
        # first choice still requires an independently checked complete tail.
        generation = policy()["reference"]["generation"]
        port = gpu.free_port()
        command = [str(args.oracle / "llama-server"), "-m", str(args.model), "--device", args.oracle_device, "-ngl", "999",
                   "-c", str(args.context), "--parallel", "1", "-b", str(generation["batch_size"]), "-ub", str(generation["micro_batch"]),
                   "-ctk", generation["kv_cache"], "-ctv", generation["kv_cache"], "--flash-attn", "off",
                   "--spec-type", "none", "--no-cache-prompt", "--host", "127.0.0.1", "--port", str(port), "-t", "4", "-tb", "4"]
        with gpu.process(command, work / "native-reference-near-ties.log", work,
                         gpu.reference_environment(args.oracle, env)) as child:
            gpu.ready(child, port, require_ok=True)
            for prompt, actual, ids, tie in pending:
                tokens = gpu.json_request(port, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True}).get("tokens")
                require(isinstance(tokens, list) and tokens and all(type(token) is int and token >= 0 for token in tokens),
                        "native reference branch prompt tokens are missing")
                start = tie["position"] + 1
                if start < len(ids):
                    tail = gpu.json_request(port, "/completion", {"prompt": tokens + ids[:start], "n_predict": len(ids) - start,
                                            "temperature": 0, "n_probs": 2, "cache_prompt": False})
                    require(gpu.reference_generated_ids(tail) == ids[start:], "SYCL continuation after the near tie differs")
                text = gpu.json_request(port, "/detokenize", {"tokens": ids}).get("content")
                require(text == actual.get("content"), "SYCL reference near-tie text differs")
    record.update(passed=True, validation_stage="complete")
    return record


def raw_stream_parity(transcript, baseline):
    """Require complete text and usage, not merely a nonempty first chunk."""
    require(len(transcript) <= 1024 * 1024, "raw stream exceeded its transcript limit")
    frames = [line[5:].lstrip(" ") for line in transcript.decode("utf-8", errors="strict").splitlines() if line.startswith("data:")]
    require(frames and frames[-1] == "[DONE]" and frames.count("[DONE]") == 1, "raw stream did not terminate")
    text, finish, usage, identity = [], None, None, None
    for frame in frames[:-1]:
        event = json.loads(frame)
        require(not event.get("error") and event.get("object") == "text_completion", "raw stream returned an error or a chat event")
        require(event.get("model") == baseline.get("model"), "raw stream changed its model")
        current = (event.get("id"), event.get("created"))
        require(isinstance(current[0], str) and current[0] and type(current[1]) is int, "raw stream identity is missing")
        identity = current if identity is None else identity
        require(current == identity, "raw stream identity changed")
        choices = event.get("choices")
        require(isinstance(choices, list) and len(choices) <= 1, "raw stream choices are invalid")
        if not choices:
            require(finish is not None and usage is None and event.get("usage") == baseline.get("usage"), "raw stream usage differs")
            usage = event["usage"]
            continue
        require(finish is None and usage is None, "raw stream continued after its finish")
        choice = choices[0]
        require(choice.get("index") == 0 and isinstance(choice.get("text"), str) and "delta" not in choice, "raw stream text is missing")
        text.append(choice["text"])
        finish = choice.get("finish_reason")
        require(finish in (None, "length", "stop") and (finish is None or choice["text"] == ""), "raw stream finish is invalid")
    require(finish == baseline["choices"][0]["finish_reason"] and usage == baseline.get("usage") and usage is not None,
            "raw stream finish or usage differs")
    content = "".join(text)
    require(content == baseline["choices"][0]["text"] and "\ufffd" not in content, "complete raw JSON/SSE text differs")
    return {"content_sha256": text_hash(content), "usage": usage, "finish_reason": finish}


def raw_stream_validation(args, work, env):
    samples = json.loads((work / "reference-samples.json").read_text())
    port, cases = gpu.free_port(), []
    command = [str(args.zinc_tree / "bin/zinc"), "-m", str(args.model), "-d", str(args.device),
               "--context", str(args.context), "--parallel", "1", "--port", str(port)]
    with gpu.process(command, work / "raw-stream-validation.log", work, env) as child:
        gpu.ready(child, port, require_ok=True)
        for sample in samples:
            body = {"prompt": sample["prompt"], "max_tokens": policy()["reference"]["agreement"]["tokens_per_case"],
                    "temperature": 0, "cache_prompt": False}
            baseline = gpu.json_request(port, "/v1/completions", {**body, "stream": False})
            expected = sample["expected"]
            accepted = [expected["content"], expected.get("validated_content")]
            require(baseline["choices"][0]["text"] in accepted, "raw stream JSON control differs from frozen oracle")
            with gpu.request(port, "/v1/completions", {**body, "stream": True, "stream_options": {"include_usage": True}}) as response:
                require("text/event-stream" in response.headers.get("Content-Type", ""), "raw streaming returned JSON")
                cases.append(raw_stream_parity(response.read(1024 * 1024 + 1), baseline))
        # Close only after real generated text, then check full native answers on
        # the same server; a healthy HTTP response cannot prove state recovery.
        with gpu.request(port, "/v1/completions", {"prompt": samples[0]["prompt"], "max_tokens": 256,
                         "temperature": 0, "stream": True}) as response:
            cancelled = False
            for _ in range(1024):
                line = response.readline(1024 * 1024 + 1).decode("utf-8", errors="strict").strip()
                require(line != "data: [DONE]", "raw generation completed before cancellation")
                if line.startswith("data:"):
                    event = json.loads(line[5:].lstrip(" "))
                    require(not event.get("error"), "raw stream failed before cancellation")
                    if any(choice.get("text") and not choice.get("finish_reason") for choice in event.get("choices", [])):
                        cancelled = True
                        break
            require(cancelled, "raw cancellation did not observe generated text")
        gpu.reference_http_parity(port, {"temperature": 0, "stream": False}, [(s["prompt"], s["expected"]) for s in samples])
    return {"passed": True, "cases": cases, "complete_native_replay_after_cancellation": True}


def comparison_eligibility(samples, scenarios):
    eligible = {}
    for scenario in scenarios:
        measured = [s for s in samples if s["scenario"] == scenario["id"] and not s["warmup"]]
        eligible[scenario["id"]] = {s["engine"] for s in measured} == {"zinc", "llama"} and all(s["output_budget_complete"] for s in measured)
    return eligible


def response_identity(collection, row):
    """Text equality alone cannot prove stable token accounting or completion."""
    require(collection in ("samples", "stream_samples") and isinstance(row, dict), "invalid repeatability sample")
    digest, size = row.get("content_sha256"), row.get("content_bytes")
    require(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest)
            and type(size) is int and size > 0, "invalid repeatability content identity")
    identity = {"content_sha256": digest, "content_bytes": size}
    if collection == "samples":
        for field in ("promptTokens", "generatedTokens"):
            value = row.get(field)
            require(type(value) is int and value > 0, "invalid repeatability token accounting")
            identity[field] = value
        require(row.get("finish_reason") in ("length", "stop"), "invalid repeatability finish reason")
        identity["finish_reason"] = row["finish_reason"]
    else:
        cap = row.get("output_cap")
        require(type(cap) is int and cap > 0, "invalid repeatability streaming cap")
        identity["output_cap"] = cap
    return identity


def repeated_response(collection, row, identities, index):
    identity = response_identity(collection, row)
    require(row.get("engine") in ("zinc", "llama") and isinstance(row.get("scenario"), str)
            and row["scenario"], "invalid repeatability engine or workload")
    # Keep the warmup and all subsequent server lifetimes in this identity.
    # Streams have a separate output budget. Different engines may produce
    # different text, but each must repeat its own complete response.
    key = (collection, row["engine"], row["scenario"])
    expected, first_index = identities.setdefault(key, (identity, index))
    if identity != expected:
        return {"collection": collection, "engine": row["engine"], "scenario": row["scenario"],
                "first_sample_index": first_index, "sample_index": index, "expected": expected, "actual": identity,
                "changed_fields": [field for field in expected if identity[field] != expected[field]]}
    return None


def repeatability_verification(model):
    """Rebuild stability from retained rows, never a cached passing summary."""
    result = {"passed": False, "full_responses_checked": 0, "streams_checked": 0, "reason": None}
    identities = {}
    for collection, counter in (("samples", "full_responses_checked"), ("stream_samples", "streams_checked")):
        rows = model.get(collection)
        if not isinstance(rows, list) or not rows:
            result["reason"] = "missing repeatability samples"
            return result
        for index, row in enumerate(rows):
            result[counter] += 1
            try:
                mismatch = repeated_response(collection, row, identities, index)
            except ValueError as error:
                result.update(collection=collection, sample_index=index, reason=str(error))
                return result
            if mismatch is not None:
                result.update(mismatch, reason="repeated complete outputs differ")
                return result
    result["passed"] = True
    return result


def retain_measurement(model, collection, row, identities, counts=None):
    # Retain the rejected request too. Otherwise the public report would stop
    # just before the divergent output and require private logs to explain it.
    model[collection].append(row)
    mismatch = repeated_response(collection, row, identities, len(model[collection]) - 1)
    if mismatch is not None:
        model.setdefault("repeatability_failures", []).append(mismatch)
        raise ValueError(row["engine"] + " " + row["scenario"] + " repeated complete outputs differ")
    if counts is not None:
        observed = counts.setdefault(row["scenario"], [])
        observed.append(row["promptTokens"])
        require(max(observed) - min(observed) <= 1, "engines processed different prompt counts")


def metric_comparison(rows, field, rounds, runs, higher_is_better=False, eligible=True, resolution=0):
    """Compare every matched round; missing counters never become zero-byte wins."""
    result = {"field": field, "higher_is_better": higher_is_better, "rounds": []}
    for number in range(1, rounds + 1):
        values = {engine: [] for engine in ("zinc", "llama")}
        for row in rows:
            if row.get("round") != number or row.get("engine") not in values:
                continue
            value = row
            for part in field.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if field.startswith("memory.observed_peak.gpu_") and not row.get("memory", {}).get("gpu_memory_available"):
                value = None
            values[row["engine"]].append(value)
        paired = {engine: None for engine in values}
        valid_counts = {}
        for engine, samples in values.items():
            # Zero is meaningful for memory, but invalid for timing/rates.
            valid = [v for v in samples if type(v) in (int, float) and math.isfinite(v)
                     and (v >= 0 if field.startswith("memory.") else v > 0)]
            valid_counts[engine] = len(valid)
            # Extra samples are retained but disqualify this round: choosing a
            # convenient subset would permit cherry-picking a transient rerun.
            if len(samples) == runs and len(valid) == runs:
                paired[engine] = statistics.median(valid)
        zinc, llama = paired["zinc"], paired["llama"]
        outcome, ratio = "unavailable", None
        if eligible and zinc is not None and llama is not None:
            if zinc == llama:
                outcome, ratio = "tied", 1.0
            else:
                outcome = "better" if (zinc > llama) == higher_is_better else "worse"
                denominator = llama if higher_is_better else zinc
                if denominator > 0:
                    ratio = (zinc if higher_is_better else llama) / denominator
                if abs(zinc - llama) <= resolution:
                    outcome = "unresolved"
        result["rounds"].append({"round": number, "zinc": zinc, "llama": llama,
                                 "zinc_advantage_ratio": ratio, "outcome": outcome,
                                 "sample_counts": {"expected_per_engine": runs, **{e: len(v) for e, v in values.items()}},
                                 "valid_sample_counts": valid_counts})
    outcomes = {row["outcome"] for row in result["rounds"]}
    result["outcome"] = next(iter(outcomes)) if len(outcomes) == 1 else "mixed" if outcomes else "unavailable"
    result["all_rounds_better"] = bool(outcomes) and outcomes == {"better"}
    return result


def gpu_process_verification(model, rounds, reference_backend="vulkan"):
    """Require recorded GPU offload for every measured server, not just labels."""
    expected = {(engine, number) for engine in ("zinc", "llama") for number in range(1, rounds + 1)}
    processes = model.get("processes", [])
    result = {"passed": False, "expected_servers": len(expected), "observed_servers": len(processes), "reason": None}
    def reject(reason):
        result["reason"] = reason
        return result
    if reference_backend not in ("vulkan", "sycl"):
        return reject("unsupported reference backend")
    # Warmup requests reuse the measured server. Extra/restarted servers are
    # separate attempts, not evidence that can be silently discarded.
    if len(processes) != len(expected):
        return reject("missing or extra measured-server records")
    seen = set()
    for process in processes:
        if type(process.get("round")) is not int:
            return reject("invalid server round")
        key = (process.get("engine"), process.get("round"))
        if key not in expected or key in seen:
            return reject("unexpected or duplicate engine/round record")
        seen.add(key)
        if key[0] == "zinc":
            if process.get("kv_cache") != "f16":
                return reject("ZINC server did not confirm F16 KV")
        else:
            offload = process.get("gpu_offload", {})
            if not (isinstance(offload, dict) and offload.get("backend") == reference_backend and type(offload.get("offloaded")) is int
                    and type(offload.get("total")) is int and offload["offloaded"] > 0
                    and offload["offloaded"] == offload["total"]):
                return reject("baseline server did not confirm complete " + ("SYCL" if reference_backend == "sycl" else "Vulkan") + " GPU offload")
            if reference_backend == "sycl":
                runtime = process.get("native_runtime", {})
                libraries = runtime.get("library_sha256", {}) if isinstance(runtime, dict) else {}
                if not (isinstance(runtime, dict) and runtime.get("runtime") == "level_zero" and isinstance(libraries, dict) and libraries
                        and all(isinstance(name, str) and isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest)
                                for name, digest in libraries.items())
                        and any(name.startswith("libsycl.so") for name in libraries)
                        and any(name.startswith("libur_adapter_level_zero") for name in libraries)):
                    return reject("baseline server did not prove its live SYCL/Level Zero libraries")
    result["passed"] = seen == expected
    return result


def performance_comparison(model, settings):
    rounds, runs = settings["rounds"], settings["runs"]
    # Every server carries its own resolution. Missing historical metadata is
    # conservatively 500 ms, even when combined with a newer 10 ms campaign.
    polls = [settings.get("startup_poll_interval_ms", 500)]
    polls.extend(p.get("startup_poll_interval_ms", 500) for p in model["processes"])
    startup_resolution = max(polls) if all(type(p) in (int, float) and 0 < p <= 360_000 for p in polls) else None
    dimensions = []
    measured = [s for s in model["samples"] if s.get("warmup") is False]
    eligible = comparison_eligibility(model["samples"], model["scenarios"])
    for scenario in model["scenarios"]:
        name = scenario["id"]
        samples = [s for s in measured if s["scenario"] == name]
        for field in ("prefillTps", "decodeTps", "totalLatencyMs"):
            dimensions.append({"scenario": name, **metric_comparison(samples, field, rounds, runs,
                higher_is_better=field.endswith("Tps"), eligible=eligible[name])})
        streams = [s for s in model["stream_samples"] if s["scenario"] == name]
        dimensions.append({"scenario": name, **metric_comparison(streams, "first_usable_content_ms", rounds, 1,
            eligible=eligible[name])})
    for field in ("startup_ms", "memory.observed_peak.host_peak_rss_bytes",
                  "memory.observed_peak.gpu_resident_vram_bytes", "memory.observed_peak.gpu_total_vram_bytes"):
        # Use the recorded resolution, including historical 500 ms campaigns.
        # Finer polling in new runs cannot turn old coarse samples into wins.
        dimensions.append({"scenario": None, **metric_comparison(model["processes"], field, rounds, 1,
            eligible=field != "startup_ms" or startup_resolution is not None,
            resolution=startup_resolution if field == "startup_ms" and startup_resolution is not None else 0)})
    complete_workloads = (len(model["scenarios"]) == 4
                          and set(eligible) == {"core", "context-medium", "context-long", "decode-extended"})
    reference_backend = settings.get("reference_backend", "vulkan")
    matched_settings = (settings.get("backend") == "vulkan" and settings.get("target") == "intel"
                        and settings.get("kv_cache") == "f16" and settings.get("mtp") == "off"
                        and settings.get("slots") == 1 and settings.get("prompt_cache") is False
                        and reference_backend in ("vulkan", "sycl")
                        and (reference_backend != "sycl" or settings.get("native_stack_verified") is True and settings.get("protocol") == "raw"))
    measured_dimensions = sum(bool(d["rounds"]) and all(r["zinc"] is not None and r["llama"] is not None
                                                      for r in d["rounds"]) for d in dimensions)
    correctness = model.get("correctness")
    correct = isinstance(correctness, dict) and correctness.get("passed") is True
    native_validation = model.get("native_reference_validation")
    reference_correct = reference_backend != "sycl" or (isinstance(native_validation, dict) and native_validation.get("passed") is True)
    gpu_verification = gpu_process_verification(model, rounds, reference_backend)
    gpu_verified = gpu_verification["passed"]
    repeatability = repeatability_verification(model)
    headline = (model.get("passed") is True and correct and reference_correct and gpu_verified and repeatability["passed"] and matched_settings and startup_resolution is not None and complete_workloads
                and rounds >= 3 and runs >= 3 and settings["warmup"] >= 1 and all(eligible.values())
                and bool(dimensions) and measured_dimensions == len(dimensions))
    return {"scope": "observed matched single-slot MTP-off rounds; not concurrency or statistical confidence",
            "zinc_backend": "vulkan", "reference_backend": reference_backend,
            "inference_correctness_verified": correct, "gpu_offload_verified": gpu_verified,
            "native_reference_correctness_verified": reference_correct,
            "gpu_offload_verification": gpu_verification,
            "response_repeatability_verified": repeatability["passed"], "repeatability_verification": repeatability,
            "matched_settings_verified": matched_settings,
            "startup_resolution_ms": startup_resolution,
            "headline_eligible": headline, "comparison_eligible": eligible, "dimensions": dimensions,
            "measured_dimensions": measured_dimensions, "dimension_count": len(dimensions),
            "all_dimensions_better": headline and bool(dimensions) and all(d["all_rounds_better"] for d in dimensions)}


def campaign_outcome(report):
    """Refresh model summaries and derive coverage from completed results."""
    # Neither CLI selection nor a narrowed report declaration can redefine
    # "all models"; the immutable campaign policy is the authority.
    required = next(profile["models"] for profile in policy()["gpu_profiles"] if profile["id"] == "vulkan-intel-xe2")
    declared, selected, models = report["required_models"], report["selected_models"], report["models"]
    observed = [model.get("id") for model in models]
    complete = (report.get("passed") is True and report.get("error") is None
                and bool(required) and len(set(required)) == len(required)
                and len(declared) == len(required) and set(declared) == set(required)
                and len(selected) == len(required) and set(selected) == set(required)
                and len(observed) == len(required) and set(observed) == set(required)
                and all(model.get("passed") is True and isinstance(model.get("correctness"), dict)
                        and model["correctness"].get("passed") is True for model in models))
    # Recompute from the samples so stale per-model summary flags cannot turn
    # a missing, duplicated or losing model into an all-model win.
    comparisons = [performance_comparison(model, report["settings"]) for model in models] if complete else []
    for model, comparison in zip(models, comparisons):
        model["performance_comparison"] = comparison
        model["comparison_eligible"] = comparison["comparison_eligible"]
    workloads = {"core", "context-medium", "context-long", "decode-extended"}
    hardware = report.get("hardware")
    stable_hardware = (report.get("hardware_unchanged") is True and isinstance(hardware, dict)
                       and bool(hardware) and hardware == report.get("hardware_after"))
    performance_complete = (complete and stable_hardware and all(
        comparison["gpu_offload_verified"] and comparison["matched_settings_verified"] and comparison["native_reference_correctness_verified"]
        and comparison["response_repeatability_verified"]
        and set(comparison["comparison_eligible"]) == workloads
        and all(comparison["comparison_eligible"].values()) for comparison in comparisons))
    reference = report.get("reference") if isinstance(report.get("reference"), dict) else {}
    latest = report.get("latest_reference_check") if isinstance(report.get("latest_reference_check"), dict) else {}
    commit = reference.get("commit")
    latest_verified = (isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None
                       and latest.get("commit") == latest.get("observed_head") == commit
                       and isinstance(latest.get("checked_at"), str) and bool(latest["checked_at"]))
    return {"complete_model_matrix": complete, "complete_performance_matrix": performance_complete,
            "all_models_all_dimensions_better": (performance_complete and latest_verified
                                                and all(c["all_dimensions_better"] for c in comparisons))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    required = next(p["models"] for p in policy()["gpu_profiles"] if p["id"] == "vulkan-intel-xe2")
    for name in ("zinc-tree", "zinc-source", "llama-bin", "llama-source", "oracle", "oracle-source", "worker-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=required, default=required)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reference-selected-at", help="Reuse the recorded upstream selection from before this campaign's builds (at most two hours old)")
    parser.add_argument("--reference-backend", choices=("vulkan", "sycl"), default="vulkan")
    parser.add_argument("--reference-device", help="explicit llama.cpp device for the performance reference")
    parser.add_argument("--sycl-prefix", type=Path, help="isolated official Intel package environment")
    parser.add_argument("--sycl-probe", type=Path, help="native readiness probe compiled in the selected environment")
    parser.add_argument("--sycl-graphs", action="store_true", help="measure the opt-in native SYCL graph configuration separately")
    parser.add_argument("--sycl-native-attention", action="store_true", help="measure native SYCL attention instead of oneDNN SDPA as a separate treatment")
    parser.add_argument("--sycl-disable-dnn", action="store_true", help="measure the explicit oneDNN-disabled control separately")
    parser.add_argument("--keep-going", action="store_true", help="retain a failed model and attempt the remaining matrix; the campaign still fails")
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
    if args.reference_backend == "sycl":
        require(args.sycl_prefix and args.sycl_probe and args.reference_device
                and re.fullmatch(r"SYCL\d+", args.reference_device), "SYCL reference needs a prefix, probe and explicit SYCL device")
        args.sycl_prefix, args.sycl_probe = args.sycl_prefix.resolve(), args.sycl_probe.resolve()
    else:
        require(not (args.sycl_graphs or args.sycl_native_attention or args.sycl_disable_dnn), "native tuning controls require a SYCL reference")
        require(args.reference_device in (None, args.oracle_device), "Vulkan performance reference must use the oracle GPU")
        args.reference_device = args.oracle_device
    os.umask(0o077)
    install_signal_handlers()
    # The shared helper installs TERM/INT; SSH disconnects also need unwinding.
    signal.signal(signal.SIGHUP, interrupted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "zinc-intel-performance-v1", "started_at": now(), "passed": False,
              "release_qualification": False, "required_models": required, "selected_models": args.models,
              "complete_model_matrix": False, "complete_performance_matrix": False, "models": [],
              "all_models_all_dimensions_better": False, "hardware_unchanged": False,
              "settings": {"backend": "vulkan", "target": "intel", "protocol": "raw", "kv_cache": "f16", "mtp": "off", "slots": 1,
                           "reference_backend": args.reference_backend, "native_stack_verified": False,
                           "prompt_cache": False, "rounds": args.rounds, "runs": args.runs, "warmup": args.warmup,
                           "temperature": 0, "top_p": 1, "seed": 0, "context": "pinned model qualification limit",
                           "ttft": "separate warmed 16-token streaming probe per workload/round", "startup": "observed model startup; OS caches are retained",
                           "startup_poll_interval_ms": 10}}
    def save():
        write_json(args.output, report)
    save()
    try:
        selected = json.loads((ROOT / "config/benchmark-reference.json").read_text())
        pin = selected["commit"]
        head = gpu.output(["bun", str(ROOT / "tools/benchmark_reference.mjs"), "head"], timeout=60).strip()
        report["latest_reference_check"] = reference_selection(selected, head, args.reference_selected_at)
        env = gpu.packaged_environment()
        env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", ZINC_MTP="0", ZINC_KV_CACHE="f16", ZINC_KV_F16="1", ZINC_KV_Q8="0", ZINC_PROMPT_CACHE="0")
        reference_env = sycl.environment(args.sycl_prefix, env, graphs=args.sycl_graphs,
                                        dnn=not args.sycl_disable_dnn, onednn_attention=not args.sycl_native_attention) if args.reference_backend == "sycl" else env
        zinc = args.zinc_tree / "bin/zinc"
        source = gpu.output(["git", "-C", str(args.zinc_source), "rev-parse", "HEAD"]).strip()
        require(not gpu.output(["git", "-C", str(args.zinc_source), "status", "--porcelain", "--untracked-files=no"]).strip(), "ZINC source is modified")
        version = gpu.output([str(zinc), "--version"], env).strip()
        require(source[:7] in version, "ZINC archive differs from the recorded source")
        report["zinc"] = {"commit": source, "sha256": sha256(zinc), "version": version, "shader_sha256": packaged_shaders(args.zinc_tree)}
        report["harness_sha256"] = {name: sha256(ROOT / name) for name in HARNESS}
        report["harness_commit"] = gpu.output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip()
        require(not gpu.output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]).strip(), "benchmark harness is modified")
        with worker_reservation(Path(worker["lock_directory"])):
            idle()
            # Even version/device discovery stays inside the worker reservation.
            report["reference"] = reference_identity(args.llama_bin, args.llama_source, reference_env, pin)
            report["oracle"] = reference_identity(args.oracle, args.oracle_source, env, policy()["reference"]["commit"])
            # Keep private request/server logs for profiling and failed-run diagnosis.
            work = Path(tempfile.mkdtemp(prefix="intel-matrix-", dir=args.output.parent))
            report["hardware"] = gpu.hardware_evidence(gpu.output([str(zinc), "--check", "-d", str(args.device)], env, timeout=360), "intel-xe2", args.device)
            if args.reference_backend == "sycl":
                devices = [p for p in Path("/sys/class/drm").glob("renderD*/device/vendor") if p.read_text().strip() == "0x8086"]
                require(len(devices) == 1, "cross-backend comparison requires exactly one physical Intel render device")
                report["native_stack"] = sycl.stack_manifest(args.sycl_prefix)
                report["native_build"] = sycl.build_manifest(args.llama_bin, args.llama_source)
                report["settings"]["native_runtime"] = {key: reference_env[key] for key in (
                    "GGML_SYCL_ENABLE_GRAPH", "GGML_SYCL_ENABLE_DNN", "GGML_SYCL_FA_ONEDNN")}
                report["native_probe"] = validate_probe(json.loads(gpu.output([str(args.sycl_probe)], reference_env, timeout=180)))
                report["native_opencl"] = json.loads(gpu.output([sys.executable, str(ROOT / "tools/intel_sycl_environment.py"),
                                      "--probe-opencl", str(args.sycl_prefix / "lib/libOpenCL.so.1")], reference_env, timeout=60))
                require(report["native_opencl"]["device_name"] == report["native_probe"]["device_name"],
                        "OpenCL and Level Zero GPU identities differ")
                report["settings"]["native_stack_verified"] = True
            print("Correctness: 376 numerical shader cases", flush=True)
            report["kernel_reference"] = gpu.kernel_reference(args.zinc_tree, args.oracle, args.oracle_source, args.device, work, env)
            report["q8_decode_reference"] = gpu.q8_decode_reference(args.zinc_tree, args.oracle, args.oracle_source, args.device, work, env)
            gemma_geglu = gpu.gemma_geglu_reference(args.zinc_tree, args.oracle, args.oracle_source, args.device, work, env)
            if gemma_geglu is not None:
                report["gemma_geglu_reference"] = gemma_geglu
            for identifier in args.models:
                model_pin = next(m for m in policy()["models"] if m["id"] == identifier)
                args.model, args.context = Path(worker["models"][identifier]), model_pin["context"]
                require(args.model.stat().st_size == model_pin["size_bytes"] and sha256(args.model) == model_pin["sha256"], "model bytes differ from policy: " + identifier)
                current = {"id": identifier, "sha256": model_pin["sha256"], "context": args.context, "passed": False,
                           "samples": [], "stream_samples": [], "processes": []}
                report["models"].append(current)
                model_work = work / identifier
                model_work.mkdir()
                try:
                    print("Correctness: " + identifier, flush=True)
                    gpu.tokenizer_parity(args.zinc_tree, args.oracle, args.model, model_work, env)
                    current["correctness"] = correctness(args, model_work, env, model_pin)
                    current["raw_streaming"] = raw_stream_validation(args, model_work, env)
                    if args.reference_backend == "sycl":
                        print("Correctness: timed SYCL configuration versus frozen Vulkan oracle", flush=True)
                        current["native_reference_validation"] = {}
                        native_reference_validation(args, model_work, env, reference_env, current["native_reference_validation"])
                    scenarios = run_json(["bun", str(ROOT / "tools/intel_workloads.mjs"), identifier])
                    scenarios = raw_scenarios(args, scenarios, model_work, env)
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
                                command = performance_reference_command(args, port)
                                active_env = gpu.reference_environment(args.llama_bin, reference_env)
                            log = model_work / f"{round_index}-{engine}.log"
                            print(f"Benchmark {identifier}, round {round_index + 1}/{args.rounds}: {engine}", flush=True)
                            started = time.monotonic()
                            with gpu.process(command, log, args.zinc_source, active_env) as child, MemoryTracker(child.pid) as memory:
                                gpu.ready(child, port, poll_interval=report["settings"]["startup_poll_interval_ms"] / 1000, require_ok=True)
                                process = {"engine": engine, "round": round_index + 1, "startup_ms": (time.monotonic() - started) * 1000,
                                           "startup_poll_interval_ms": report["settings"]["startup_poll_interval_ms"]}
                                if engine == "llama":
                                    process["gpu_offload"] = native_gpu_layers(log.read_text(errors="replace"), args.reference_backend)
                                else:
                                    process["kv_cache"] = zinc_kv_storage(log.read_text(errors="replace"))
                                current["processes"].append(process)
                                collectors = [("memory", memory.snapshot)]
                                if engine == "llama" and args.reference_backend == "sycl":
                                    collectors.append(("native_runtime", lambda pid=child.pid, prefix=args.sycl_prefix: sycl.runtime_evidence(pid, prefix)))
                                with process_observations(process, collectors):
                                    for scenario in scenarios:
                                        for iteration in range(args.warmup + args.runs):
                                            number += 1
                                            # Natural EOS is retained as evidence, never a full-budget win.
                                            sample = measure(engine, "off", child, port, log, scenario, model_work, number, require_full_budget=False, completion_mode="raw")
                                            retain_measurement(current, "samples", {"engine": engine, "round": round_index + 1,
                                                               "scenario": scenario["id"], "warmup": iteration < args.warmup, **sample}, identities, counts)
                                            save()
                                        probe = stream_probe(port, scenario["body"], completion_mode="raw")
                                        retain_measurement(current, "stream_samples", {"engine": engine, "round": round_index + 1,
                                                           "scenario": scenario["id"], **probe}, identities)
                            idle()
                            save()
                    current["comparison_eligible"] = comparison_eligibility(current["samples"], scenarios)
                    current["passed"] = True
                    current["performance_comparison"] = performance_comparison(current, report["settings"])
                    save()
                except (ValueError, AssertionError, OSError) as error:
                    current["passed"] = False
                    current["error"] = str(error) if type(error) in (ValueError, AssertionError) else type(error).__name__
                    save()
                    if not args.keep_going:
                        raise
                    print("Model failed: " + identifier + "; retained attempt and continuing", flush=True)
                    idle()
            require(reference_identity(args.llama_bin, args.llama_source, reference_env, pin) == report["reference"], "performance reference changed")
            require(reference_identity(args.oracle, args.oracle_source, env, policy()["reference"]["commit"]) == report["oracle"], "correctness oracle changed")
            require(sha256(zinc) == report["zinc"]["sha256"], "ZINC binary changed")
            require(packaged_shaders(args.zinc_tree) == report["zinc"]["shader_sha256"], "ZINC packaged shaders changed")
            require(all(sha256(ROOT / name) == digest for name, digest in report["harness_sha256"].items()), "benchmark harness changed")
            idle()
            # Every new server loads the active driver again. A stack upgrade
            # during the campaign invalidates a comparison just like changed
            # executable bytes; keep upgrades between controlled campaigns.
            report["hardware_after"] = gpu.hardware_evidence(gpu.output([str(zinc), "--check", "-d", str(args.device)], env, timeout=360), "intel-xe2", args.device)
            require(report["hardware_after"] == report["hardware"], "GPU, kernel or Vulkan driver changed during the campaign")
            report["hardware_unchanged"] = True
            if args.reference_backend == "sycl":
                report["native_stack_after"] = sycl.stack_manifest(args.sycl_prefix)
                report["native_build_after"] = sycl.build_manifest(args.llama_bin, args.llama_source)
                report["native_probe_after"] = validate_probe(json.loads(gpu.output([str(args.sycl_probe)], reference_env, timeout=180)))
                report["native_opencl_after"] = json.loads(gpu.output([sys.executable, str(ROOT / "tools/intel_sycl_environment.py"),
                                            "--probe-opencl", str(args.sycl_prefix / "lib/libOpenCL.so.1")], reference_env, timeout=60))
                require(report["native_stack_after"] == report["native_stack"] and report["native_probe_after"] == report["native_probe"]
                        and report["native_build_after"] == report["native_build"] and report["native_opencl_after"] == report["native_opencl"],
                        "native runtime, packages or GPU changed during the campaign")
        report.update(passed=all(model["passed"] for model in report["models"]), cleanup="no GPU process remains")
        report.update(campaign_outcome(report))
        require(report["passed"], "one or more model checks failed; all attempts are retained")
    except BaseException as error:
        report["error"] = str(error) if type(error) in (ValueError, AssertionError) else type(error).__name__
        raise
    finally:
        report["completed_at"] = now()
        save()


if __name__ == "__main__":
    main()
