"""CPU fixtures for Intel measurement, never a GPU or a model load."""
import copy
import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import benchmark_intel
import benchmark_metrics
import benchmark_matched_mtp
from benchmark_gemma_rocm import correctness_prompts


class IntelMetricsTest(unittest.TestCase):
    def test_observations_are_retained_after_rejected_measurements_and_cancellation(self):
        for failure in (ValueError("complete output changed"), KeyboardInterrupt()):
            record = {"passed": False}
            memory, runtime = Mock(return_value={"peak": 123}), Mock(return_value={"runtime": "level_zero"})
            with self.assertRaises(type(failure)) as caught:
                with benchmark_intel.process_observations(record, [("memory", memory), ("native_runtime", runtime)]):
                    memory.assert_not_called(); runtime.assert_not_called()
                    raise failure
            self.assertIs(caught.exception, failure)
            self.assertEqual(record, {"passed": False, "memory": {"peak": 123}, "native_runtime": {"runtime": "level_zero"}})
            memory.assert_called_once(); runtime.assert_called_once()

    def test_missing_observations_do_not_mask_the_primary_failure_or_skip_other_collectors(self):
        failure = ValueError("complete output changed")
        record, runtime = {}, Mock(return_value={"runtime": "level_zero"})
        with self.assertRaises(ValueError) as caught:
            with benchmark_intel.process_observations(record, [("memory", Mock(side_effect=OSError("private diagnostic"))), ("native_runtime", runtime)]):
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(record, {"memory_error": "OSError", "native_runtime": {"runtime": "level_zero"}})
        runtime.assert_called_once()

    def test_successful_measurements_still_fail_if_required_observation_is_missing(self):
        record = {}
        failure = ValueError("native runtime proof unavailable")
        with self.assertRaises(ValueError) as caught:
            with benchmark_intel.process_observations(record, [("native_runtime", Mock(side_effect=failure)), ("memory", lambda: {"peak": 123})]):
                pass
        self.assertIs(caught.exception, failure)
        self.assertEqual(record, {"native_runtime_error": "ValueError", "memory": {"peak": 123}})
        record = {}
        with benchmark_intel.process_observations(record, [("memory", lambda: {"peak": 123})]):
            self.assertEqual(record, {})
        self.assertEqual(record, {"memory": {"peak": 123}})

    def test_reference_is_frozen_at_observed_selection_before_building(self):
        current = datetime.datetime(2026, 11, 9, 20, tzinfo=datetime.timezone.utc)
        timestamp = "2026-11-09T19:00:00Z"
        selected = {"commit": "a" * 40, "checked_at_utc": timestamp}
        check = benchmark_intel.reference_selection(selected, "b" * 40, timestamp, current)
        self.assertEqual(check, {"commit": "a" * 40, "observed_head": "a" * 40,
                         "checked_at": timestamp, "scope": "campaign-before-build", "observed_head_at_run": "b" * 40})
        # The default remains a live run-start check; a moving upstream cannot
        # silently relabel the selected source as the newer revision.
        with self.assertRaisesRegex(ValueError, "latest native"):
            benchmark_intel.reference_selection(selected, "b" * 40, current=current)
        self.assertEqual(benchmark_intel.reference_selection(selected, "a" * 40, current=current)["scope"], "run-start")
        for invalid in ("2026-11-09T17:59:59Z", "2026-11-09T20:00:01Z", "2026-11-09T19:00:00", "invalid", 1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                benchmark_intel.reference_selection({**selected, "checked_at_utc": invalid}, "b" * 40, invalid, current)
        with self.assertRaisesRegex(ValueError, "differs"):
            benchmark_intel.reference_selection(selected, "b" * 40, "2026-11-09T19:01:00Z", current)
        for pin, head in (("bad", "a" * 40), ("a" * 40, "bad")):
            with self.assertRaises(ValueError):
                benchmark_intel.reference_selection({**selected, "commit": pin}, head, timestamp, current)

    def test_rejected_repetitions_and_prompt_mismatches_remain_in_the_evidence(self):
        model, identities, counts = {"samples": [], "stream_samples": []}, {}, {}
        first = {"engine": "zinc", "scenario": "core", "round": 1, "content_sha256": "a", "promptTokens": 42}
        benchmark_intel.retain_measurement(model, "samples", first, identities, counts)
        # Different engines need not generate identical text. A new server of
        # the same engine still has to reproduce the complete prior output.
        benchmark_intel.retain_measurement(model, "samples", {**first, "engine": "llama", "content_sha256": "b"}, identities, counts)
        rejected = {**first, "round": 2, "content_sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "zinc core repeated"):
            benchmark_intel.retain_measurement(model, "samples", rejected, identities, counts)
        self.assertEqual(model["samples"][-1], rejected)
        rejected = {**first, "promptTokens": 45}
        with self.assertRaisesRegex(ValueError, "different prompt counts"):
            benchmark_intel.retain_measurement(model, "samples", rejected, identities, counts)
        self.assertEqual(model["samples"][-1], rejected)
        self.assertEqual(len(model["samples"]), 4)
        # The 16-token stream is a separate output budget and identity.
        benchmark_intel.retain_measurement(model, "stream_samples", {**first, "content_sha256": "stream"}, identities)
        self.assertEqual(len(model["stream_samples"]), 1)

    def test_raw_completions_use_text_and_full_prefill_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); log = root / "server.log"; log.write_text("")
            body = {"choices": [{"text": "raw answer", "finish_reason": "length"}]}
            parsed = {"prefillTokens": 20, "promptTokens": 20, "generatedTokens": 16}
            with patch.object(benchmark_matched_mtp.gpu, "request", side_effect=lambda *a, **k: io.BytesIO(json.dumps(body).encode())) as request, \
                    patch.object(benchmark_matched_mtp, "run_json", return_value=parsed):
                sample = benchmark_matched_mtp.measure("llama", "off", Mock(poll=Mock(return_value=None)), 1, log,
                            {"body": {"prompt": "same bytes", "cache_prompt": False}, "max_tokens": 16}, root, 1, completion_mode="raw")
                self.assertEqual(request.call_args.args[1], "/v1/completions")
                self.assertTrue(sample["output_budget_complete"])
                parsed["prefillTokens"] = 1
                with self.assertRaisesRegex(ValueError, "cache reduced"):
                    benchmark_matched_mtp.measure("llama", "off", Mock(poll=Mock(return_value=None)), 1, log,
                                {"body": {"prompt": "same bytes"}, "max_tokens": 16}, root, 2, completion_mode="raw")

    def test_raw_streaming_reads_text_chunks_and_retains_the_common_cap(self):
        events = [{"choices": [{"text": " "}]}, {"choices": [{"text": "Hello"}]}, {"choices": [], "usage": {}}]
        frames = b"".join(b"data: " + json.dumps(x).encode() + b"\n\n" for x in events) + b"data: [DONE]\n\n"
        ticks = iter(range(1000))
        with patch.object(benchmark_metrics.gpu, "request", return_value=io.BytesIO(frames)) as request:
            result = benchmark_metrics.stream_probe(1, {"prompt": "same bytes", "max_tokens": 256},
                        clock=lambda: next(ticks) / 1000, completion_mode="raw")
            self.assertEqual(request.call_args.args[1], "/v1/completions")
            self.assertEqual(request.call_args.args[2]["max_tokens"], 16)
            self.assertEqual(result["content_bytes"], len(" Hello"))

    def test_shader_fingerprints_detect_overlay_changes_and_never_include_host_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); shader = root / "kernel.spv"; shader.write_bytes(b"original")
            before = benchmark_intel.packaged_shaders(root)
            self.assertEqual(set(before), {"kernel.spv"}); self.assertNotIn(str(root), json.dumps(before))
            shader.write_bytes(b"modified")
            self.assertNotEqual(before, benchmark_intel.packaged_shaders(root))
            shader.unlink()
            with self.assertRaisesRegex(ValueError, "no packaged shaders"):
                benchmark_intel.packaged_shaders(root)

    def test_sycl_comparison_needs_live_runtime_proof_and_native_model_correctness(self):
        model, settings = self.campaign()
        settings.update(reference_backend="sycl", native_stack_verified=True, protocol="raw")
        model["native_reference_validation"] = {"passed": True}
        for process in model["processes"]:
            if process["engine"] == "llama":
                process["gpu_offload"]["backend"] = "sycl"
                process["native_runtime"] = {"runtime": "level_zero", "library_sha256": {
                    "libsycl.so.9": "a" * 64, "libur_adapter_level_zero.so.0": "b" * 64}}
        result = benchmark_intel.performance_comparison(model, settings)
        self.assertTrue(result["headline_eligible"])
        self.assertEqual(result["reference_backend"], "sycl")
        self.assertEqual(result["zinc_backend"], "vulkan")
        self.assertFalse(benchmark_intel.performance_comparison(model, {**settings, "native_stack_verified": False})["headline_eligible"])
        del model["native_reference_validation"]
        self.assertFalse(benchmark_intel.performance_comparison(model, settings)["headline_eligible"])
        model["native_reference_validation"] = {"passed": True}
        native = next(p for p in model["processes"] if p["engine"] == "llama")
        native["native_runtime"] = {"runtime": "opencl", "library_sha256": {}}
        self.assertFalse(benchmark_intel.performance_comparison(model, settings)["gpu_offload_verified"])

    def test_sycl_label_requires_an_actual_sycl_model_buffer(self):
        log = 'offloaded 29/29 layers to GPU\nllama_model_load: SYCL0 model buffer size = 500 MiB\n'
        self.assertEqual(benchmark_intel.native_gpu_layers(log, "sycl")["backend"], "sycl")
        with self.assertRaisesRegex(ValueError, "SYCL model buffer"):
            benchmark_intel.native_gpu_layers(log.replace('SYCL0', 'Vulkan0'), "sycl")
        with self.assertRaisesRegex(ValueError, "full GPU"):
            benchmark_intel.native_gpu_layers(log.replace('29/29', '28/29'), "sycl")

    def test_health_poll_precision_does_not_shorten_the_startup_deadline(self):
        for interval in (0.5, 0.01):
            with self.subTest(interval=interval):
                clock = {"now": 0.0}
                def health(*args):
                    # Beyond 600 fast polls: the deadline must remain time-based.
                    if clock["now"] < 8.2:
                        raise OSError("not listening")
                    return {"status": "ok"}
                def sleep(seconds): clock["now"] += seconds
                with patch.object(benchmark_intel.gpu.time, "monotonic", side_effect=lambda: clock["now"]), \
                     patch.object(benchmark_intel.gpu.time, "sleep", side_effect=sleep), \
                     patch.object(benchmark_intel.gpu, "json_request", side_effect=health):
                    benchmark_intel.gpu.ready(Mock(poll=Mock(return_value=None)), 1, poll_interval=interval)
                self.assertGreaterEqual(clock["now"], 8.2)
                self.assertLessEqual(clock["now"], 8.2 + interval + 1e-6)

    def test_health_requires_ok_and_retains_the_release_poll_default(self):
        clock = {"now": 0.0}
        def sleep(seconds): clock["now"] += seconds
        with patch.object(benchmark_intel.gpu.time, "monotonic", side_effect=lambda: clock["now"]), \
             patch.object(benchmark_intel.gpu.time, "sleep", side_effect=sleep), \
             patch.object(benchmark_intel.gpu, "json_request", side_effect=[None, {}, {"status": "loading"}, {"status": "ok"}]) as health:
            benchmark_intel.gpu.ready(Mock(poll=Mock(return_value=None)), 1, require_ok=True)
        self.assertEqual(health.call_count, 4)
        self.assertEqual(clock["now"], 1.5)
        with patch.object(benchmark_intel.gpu, "json_request", return_value={}), patch.object(benchmark_intel.gpu.time, "sleep") as sleep:
            benchmark_intel.gpu.ready(Mock(poll=Mock(return_value=None)), 1)
            sleep.assert_not_called()

    def test_health_timeout_and_dead_process_cannot_supply_readiness(self):
        with patch.object(benchmark_intel.gpu, "json_request") as health:
            with self.assertRaisesRegex(ValueError, "exited before"):
                benchmark_intel.gpu.ready(Mock(poll=Mock(return_value=1)), 1)
            health.assert_not_called()
        clock = {"now": 0.0}
        def unavailable(*args):
            clock["now"] += 120
            raise OSError("unavailable")
        def sleep(seconds): clock["now"] += seconds
        with patch.object(benchmark_intel.gpu.time, "monotonic", side_effect=lambda: clock["now"]), \
             patch.object(benchmark_intel.gpu.time, "sleep", side_effect=sleep), \
             patch.object(benchmark_intel.gpu, "json_request", side_effect=unavailable) as health:
            with self.assertRaisesRegex(ValueError, "health timeout"):
                benchmark_intel.gpu.ready(Mock(poll=Mock(return_value=None)), 1, poll_interval=0.01)
        self.assertGreaterEqual(clock["now"], 360)
        self.assertLess(clock["now"], 480)

    def test_invalid_health_polling_cannot_busy_loop_or_sleep_forever(self):
        for interval in (0, -1, True, float("nan"), float("inf"), 361):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "poll interval"):
                benchmark_intel.gpu.ready(Mock(), 1, poll_interval=interval)

    def test_startup_comparison_uses_each_campaigns_recorded_resolution(self):
        model, settings = self.campaign()
        for row in model["processes"]:
            row["startup_ms"] = 120 if row["engine"] == "zinc" else 135
        def startup(active):
            return next(d for d in benchmark_intel.performance_comparison(model, active)["dimensions"] if d["field"] == "startup_ms")
        self.assertEqual(startup(settings)["outcome"], "unresolved")
        for row in model["processes"]: row["startup_poll_interval_ms"] = 10
        self.assertEqual(startup({**settings, "startup_poll_interval_ms": 10})["outcome"], "better")
        model["processes"][0]["startup_poll_interval_ms"] = 500
        self.assertEqual(startup({**settings, "startup_poll_interval_ms": 10})["outcome"], "unresolved")
        del model["processes"][0]["startup_poll_interval_ms"]
        self.assertEqual(startup({**settings, "startup_poll_interval_ms": 10})["outcome"], "unresolved")
        model["processes"][0]["startup_poll_interval_ms"] = 10
        historical = dict(settings); historical.pop("startup_poll_interval_ms")
        self.assertEqual(startup(historical)["outcome"], "unresolved")
        for row in model["processes"]:
            if row["engine"] == "llama": row["startup_ms"] = 125
        self.assertEqual(startup({**settings, "startup_poll_interval_ms": 10})["outcome"], "unresolved")

    def test_invalid_recorded_startup_resolution_cannot_establish_a_win(self):
        for invalid in (None, True, 0, -1, "10", float("nan"), float("inf"), 360_001):
            with self.subTest(resolution=invalid):
                model, settings = self.campaign()
                model["processes"][0]["startup_poll_interval_ms"] = invalid
                result = benchmark_intel.performance_comparison(model, settings)
                self.assertFalse(result["headline_eligible"])
                self.assertIsNone(result["startup_resolution_ms"])
                startup = next(d for d in result["dimensions"] if d["field"] == "startup_ms")
                self.assertEqual(startup["outcome"], "unavailable")

    def campaign(self):
        settings = {"rounds": 3, "runs": 3, "warmup": 1, "backend": "vulkan", "target": "intel",
                    "kv_cache": "f16", "mtp": "off", "slots": 1, "prompt_cache": False,
                    "startup_poll_interval_ms": 500}
        model = {"passed": True, "correctness": {"passed": True}, "samples": [], "stream_samples": [], "processes": [],
                 "scenarios": [{"id": s} for s in ("core", "context-medium", "context-long", "decode-extended")]}
        for number in range(1, 4):
            for engine in ("zinc", "llama"):
                better = engine == "zinc"
                for scenario in model["scenarios"]:
                    for warmup in (True, False, False, False):
                        model["samples"].append({"engine": engine, "round": number, "scenario": scenario["id"],
                            "warmup": warmup, "output_budget_complete": True, "prefillTps": 200 if better else 100,
                            "decodeTps": 20 if better else 10, "totalLatencyMs": 100 if better else 200})
                    model["stream_samples"].append({"engine": engine, "round": number, "scenario": scenario["id"],
                        "first_usable_content_ms": 10 if better else 20})
                model["processes"].append({"engine": engine, "round": number, "startup_ms": 1000 if better else 2000,
                    **({"kv_cache": "f16"} if better else {"gpu_offload": {"backend": "vulkan", "offloaded": 40, "total": 40}}),
                    "memory": {"gpu_memory_available": True, "observed_peak": {
                        "host_peak_rss_bytes": 100 if better else 200,
                        "gpu_resident_vram_bytes": 100 if better else 200,
                        "gpu_total_vram_bytes": 100 if better else 200}}})
        return model, settings

    def test_all_dimension_claim_requires_every_matched_round_and_ignores_warmups(self):
        model, settings = self.campaign()
        for row in model["samples"]:
            if row["warmup"]:
                row.update(prefillTps=0, decodeTps=0, totalLatencyMs=float("nan"), output_budget_complete=False)
        result = benchmark_intel.performance_comparison(model, settings)
        self.assertTrue(result["headline_eligible"])
        self.assertEqual(len(result["dimensions"]), 20)
        self.assertTrue(result["all_dimensions_better"])
        # A pooled median would hide a losing round behind two faster rounds.
        for row in model["samples"]:
            if row["engine"] == "zinc" and row["round"] == 3:
                row["decodeTps"] = 5
        result = benchmark_intel.performance_comparison(model, settings)
        self.assertFalse(result["all_dimensions_better"])
        decode = next(d for d in result["dimensions"] if d["scenario"] == "core" and d["field"] == "decodeTps")
        self.assertEqual(decode["outcome"], "mixed")
        self.assertEqual(decode["rounds"][2]["outcome"], "worse")

    def test_unknown_memory_diagnostic_counts_or_wrong_settings_cannot_claim_all_dimensions(self):
        original, settings = self.campaign()
        for change in ("missing_memory", "unavailable_memory", "missing_round", "duplicate_sample", "early_eos",
                       "one_round", "one_run", "wrong_kv", "mtp_on", "cached", "multiple_slots", "missing_workload", "duplicate_workload", "failed_model"):
            with self.subTest(change=change):
                model, active = copy.deepcopy(original), dict(settings)
                if change == "missing_memory":
                    del model["processes"][0]["memory"]["observed_peak"]["gpu_resident_vram_bytes"]
                elif change == "unavailable_memory":
                    model["processes"][0]["memory"]["gpu_memory_available"] = False
                elif change == "missing_round":
                    model["stream_samples"] = [s for s in model["stream_samples"] if not (s["round"] == 3 and s["engine"] == "llama")]
                elif change == "duplicate_sample":
                    model["samples"].append(copy.deepcopy(model["samples"][1]))
                elif change == "early_eos":
                    model["samples"][1]["output_budget_complete"] = False
                elif change == "one_round": active["rounds"] = 1
                elif change == "one_run": active["runs"] = 1
                elif change == "wrong_kv": active["kv_cache"] = "f32"
                elif change == "mtp_on": active["mtp"] = "on"
                elif change == "cached": active["prompt_cache"] = True
                elif change == "multiple_slots": active["slots"] = 2
                elif change == "missing_workload": model["scenarios"].pop()
                elif change == "duplicate_workload": model["scenarios"].append(copy.deepcopy(model["scenarios"][0]))
                elif change == "failed_model": model["passed"] = False
                self.assertFalse(benchmark_intel.performance_comparison(model, active)["all_dimensions_better"])

    def test_zero_unknown_and_nonfinite_values_do_not_invent_speedups(self):
        for bad in (None, -1, 0, float("nan"), float("inf"), True):
            with self.subTest(value=bad):
                rows = [{"round": 1, "engine": "zinc", "value": bad}, {"round": 1, "engine": "llama", "value": 1}]
                result = benchmark_intel.metric_comparison(rows, "value", 1, 1)
                self.assertEqual(result["outcome"], "unavailable")
                self.assertIsNone(result["rounds"][0]["zinc_advantage_ratio"])
        rows = [{"round": 1, "engine": e, "memory": {"zero": 0}} for e in ("zinc", "llama")]
        result = benchmark_intel.metric_comparison(rows, "memory.zero", 1, 1)
        self.assertEqual(result["outcome"], "tied")
        self.assertFalse(result["all_rounds_better"])

    def test_cpu_partial_or_missing_gpu_evidence_and_unqualified_models_cannot_claim_a_win(self):
        original, settings = self.campaign()
        for change in ("missing_offload", "null_offload", "cpu", "partial", "zero_layers", "boolean_layers",
                       "wrong_backend", "missing_kv", "missing_correctness", "failed_correctness", "boolean_round"):
            with self.subTest(change=change):
                model = copy.deepcopy(original)
                zinc = next(p for p in model["processes"] if p["engine"] == "zinc")
                llama = next(p for p in model["processes"] if p["engine"] == "llama")
                if change == "missing_offload": del llama["gpu_offload"]
                elif change == "null_offload": llama["gpu_offload"] = None
                elif change == "cpu": llama["gpu_offload"].update(offloaded=0, backend="cpu")
                elif change == "partial": llama["gpu_offload"]["offloaded"] = 39
                elif change == "zero_layers": llama["gpu_offload"].update(offloaded=0, total=0)
                elif change == "boolean_layers": llama["gpu_offload"].update(offloaded=True, total=True)
                elif change == "wrong_backend": llama["gpu_offload"]["backend"] = "rocm"
                elif change == "missing_kv": del zinc["kv_cache"]
                elif change == "missing_correctness": del model["correctness"]
                elif change == "failed_correctness": model["correctness"]["passed"] = False
                elif change == "boolean_round": llama["round"] = True
                self.assertFalse(benchmark_intel.performance_comparison(model, settings)["all_dimensions_better"])

    def complete_campaign(self):
        model, settings = self.campaign()
        required = [m["id"] for m in benchmark_intel.policy()["models"]]
        hardware = {"gpu": "Intel GPU fixture", "os": "Linux fixture", "driver": "fixture"}
        return {"passed": True, "required_models": required, "selected_models": list(required),
                "models": [{**copy.deepcopy(model), "id": identifier} for identifier in required],
                "settings": settings, "hardware_unchanged": True, "hardware": hardware,
                "hardware_after": dict(hardware), "reference": {"commit": "a" * 40},
                "latest_reference_check": {"commit": "a" * 40, "observed_head": "a" * 40,
                                           "checked_at": "2026-11-07T00:00:00+00:00"}}

    def test_extra_server_records_stay_visible_and_explain_why_the_campaign_is_ineligible(self):
        model, settings = self.campaign()
        model["processes"].append(copy.deepcopy(model["processes"][0]))
        result = benchmark_intel.performance_comparison(model, settings)
        self.assertEqual(result["gpu_offload_verification"], {
            "passed": False, "expected_servers": 6, "observed_servers": 7,
            "reason": "missing or extra measured-server records"})
        self.assertFalse(result["headline_eligible"])
        self.assertEqual(len(model["processes"]), 7)

    def test_all_models_win_is_derived_from_completed_models_not_the_requested_list(self):
        report = self.complete_campaign()
        self.assertEqual(benchmark_intel.campaign_outcome(report), {
            "complete_model_matrix": True, "complete_performance_matrix": True,
            "all_models_all_dimensions_better": True})
        for change in ("missing_model", "duplicate_model", "extra_model", "failed_model", "failed_correctness",
                       "missing_correctness", "subset_requested", "duplicate_request", "empty_required", "narrowed_scope", "interrupted", "reported_error"):
            with self.subTest(change=change):
                active = copy.deepcopy(report)
                if change == "missing_model": active["models"].pop()
                elif change == "duplicate_model": active["models"][-1] = copy.deepcopy(active["models"][0])
                elif change == "extra_model": active["models"].append({**copy.deepcopy(active["models"][0]), "id": "extra"})
                elif change == "failed_model": active["models"][-1]["passed"] = False
                elif change == "failed_correctness": active["models"][-1]["correctness"]["passed"] = False
                elif change == "missing_correctness": del active["models"][-1]["correctness"]
                elif change == "subset_requested": active["selected_models"].pop()
                elif change == "duplicate_request": active["selected_models"][-1] = active["selected_models"][0]
                elif change == "empty_required": active.update(required_models=[], selected_models=[], models=[])
                elif change == "narrowed_scope":
                    active["required_models"].pop()
                    active["selected_models"].pop()
                    active["models"].pop()
                elif change == "interrupted": active["passed"] = False
                elif change == "reported_error": active["error"] = "interrupted after samples"
                result = benchmark_intel.campaign_outcome(active)
                self.assertFalse(result["complete_model_matrix"])
                self.assertFalse(result["complete_performance_matrix"])
                self.assertFalse(result["all_models_all_dimensions_better"])

    def test_stale_summary_flags_missing_workloads_driver_changes_and_unverified_latest_are_not_wins(self):
        original = self.complete_campaign()
        for change in ("losing_model", "missing_workload", "missing_samples", "missing_gpu", "driver_changed",
                       "missing_hardware", "missing_reference_check", "different_upstream_head", "missing_checked_at",
                       "null_reference", "null_reference_check", "wrong_settings"):
            with self.subTest(change=change):
                report = copy.deepcopy(original)
                # Stale cached flags may survive a rerun; inspect the actual samples.
                for model in report["models"]:
                    model.update(performance_comparison={"all_dimensions_better": True}, comparison_eligible={})
                model = report["models"][-1]
                if change == "losing_model":
                    for sample in model["samples"]:
                        if sample["engine"] == "zinc": sample["decodeTps"] = 1
                elif change == "missing_workload": model["scenarios"].pop()
                elif change == "missing_samples": model["samples"] = []
                elif change == "missing_gpu": model["processes"][-1].pop("gpu_offload")
                elif change == "driver_changed": report["hardware_after"]["driver"] = "different"
                elif change == "missing_hardware": del report["hardware"]
                elif change == "missing_reference_check": del report["latest_reference_check"]
                elif change == "different_upstream_head": report["latest_reference_check"]["observed_head"] = "b" * 40
                elif change == "missing_checked_at": del report["latest_reference_check"]["checked_at"]
                elif change == "null_reference": report["reference"] = None
                elif change == "null_reference_check": report["latest_reference_check"] = None
                elif change == "wrong_settings": report["settings"]["kv_cache"] = "f32"
                self.assertFalse(benchmark_intel.campaign_outcome(report)["all_models_all_dimensions_better"])
                if change == "losing_model":
                    self.assertFalse(report["models"][-1]["performance_comparison"]["all_dimensions_better"])

    def test_missing_counters_cannot_be_headline_eligible(self):
        model, settings = self.campaign()
        del model["processes"][0]["memory"]["observed_peak"]["gpu_resident_vram_bytes"]
        result = benchmark_intel.performance_comparison(model, settings)
        self.assertFalse(result["headline_eligible"])
        self.assertEqual(result["measured_dimensions"], 19)
        self.assertEqual(result["dimension_count"], 20)

    def test_count_diagnostics_distinguish_missing_extra_and_invalid_samples(self):
        zinc = {"round": 1, "engine": "zinc", "value": 1}
        llama = {"round": 1, "engine": "llama", "value": 2}
        cases = [([zinc], {"zinc": 1, "llama": 0}, {"zinc": 1, "llama": 0}),
                 ([zinc, zinc, llama], {"zinc": 2, "llama": 1}, {"zinc": 2, "llama": 1}),
                 ([{**zinc, "value": None}, llama], {"zinc": 1, "llama": 1}, {"zinc": 0, "llama": 1})]
        for rows, observed, valid in cases:
            with self.subTest(observed=observed, valid=valid):
                result = benchmark_intel.metric_comparison(rows, "value", 1, 1)
                self.assertEqual(result["outcome"], "unavailable")
                self.assertEqual(result["rounds"][0]["sample_counts"], {"expected_per_engine": 1, **observed})
                self.assertEqual(result["rounds"][0]["valid_sample_counts"], valid)

    def test_startup_differences_within_the_polling_resolution_are_unresolved(self):
        model, settings = self.campaign()
        for row in model["processes"]:
            row["startup_ms"] = 1000 if row["engine"] == "zinc" else 1200
        result = benchmark_intel.performance_comparison(model, settings)
        startup = next(d for d in result["dimensions"] if d["field"] == "startup_ms")
        self.assertEqual(startup["outcome"], "unresolved")
        self.assertGreater(startup["rounds"][0]["zinc_advantage_ratio"], 1)
        self.assertFalse(result["all_dimensions_better"])

    def test_retained_real_diagnostic_is_not_relabelled_as_a_decode_win(self):
        path = Path(__file__).resolve().parents[1] / "site/public/zinc/validation/intel-vulkan-performance-diagnostic-2026-11-03.json"
        report = json.loads(path.read_text())["report"]
        self.assertTrue(report["passed"])
        for model in report["models"]:
            with self.subTest(model=model["id"]):
                result = benchmark_intel.performance_comparison(model, report["settings"])
                self.assertFalse(result["headline_eligible"])
                self.assertFalse(result["all_dimensions_better"])
                core_decode = next(d for d in result["dimensions"] if d["scenario"] == "core" and d["field"] == "decodeTps")
                self.assertEqual(core_decode["outcome"], "worse")

    def test_comparisons_require_both_measured_engines_and_ignore_warmups(self):
        def sample(engine, complete=True, warmup=False):
            return {"engine": engine, "scenario": "core", "warmup": warmup, "output_budget_complete": complete}
        scenarios = [{"id": "core"}]
        self.assertFalse(benchmark_intel.comparison_eligibility([], scenarios)["core"])
        self.assertFalse(benchmark_intel.comparison_eligibility([sample("zinc")], scenarios)["core"])
        both = [sample("zinc"), sample("llama")]
        self.assertTrue(benchmark_intel.comparison_eligibility(both + [sample("zinc", False, True)], scenarios)["core"])
        self.assertFalse(benchmark_intel.comparison_eligibility(both + [sample("llama", False)], scenarios)["core"])

    def test_generic_models_keep_short_checks_and_gemma_keeps_extended_windows(self):
        short = correctness_prompts({"id": "qwen35-9b-q4k-m"})
        windowed = correctness_prompts({"id": "gemma4-31b-q4k-m", "sliding_window": 1024})
        self.assertEqual(len(short), 3)
        self.assertEqual(windowed[:3], short)
        self.assertEqual(len(windowed), 6)
        self.assertTrue(all(prompt.count("word ") > 1024 for prompt in windowed[3:]))

    def test_drm_memory_counts_unique_clients_and_excludes_private_identifiers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = root / "123"
            (process / "fdinfo").mkdir(parents=True)
            (process / "status").write_text("VmRSS:\t12 kB\nVmHWM:\t20 kB\nRssAnon:\t8 kB\nRssFile:\t4 kB\n")
            first = "drm-driver: xe\ndrm-client-id: 17\ndrm-pdev: <private-id>\ndrm-total-vram0: 2 MiB\ndrm-resident-vram0: 128 KiB\ndrm-resident-system: 64\n"
            for descriptor in ("3", "4"):
                (process / "fdinfo" / descriptor).write_text(first)
            (process / "fdinfo/5").write_text(first.replace("client-id: 17", "client-id: 18"))
            (process / "fdinfo/6").write_text("pos: 0\n")
            result = benchmark_metrics.memory_snapshot(123, root)
            self.assertEqual(result["gpu_total_vram_bytes"], 4 * 1024 * 1024)
            self.assertEqual(result["gpu_resident_vram_bytes"], 256 * 1024)
            self.assertEqual(result["gpu_resident_system_bytes"], 128)
            self.assertEqual(result["host_peak_rss_bytes"], 20 * 1024)
            self.assertEqual(result["host_file_bytes"], 4 * 1024)
            self.assertNotIn("private-id", json.dumps(result))
            self.assertNotIn("client-id", json.dumps(result))

    def test_missing_gpu_stats_are_unknown_and_units_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = root / "123"
            (process / "fdinfo").mkdir(parents=True)
            (process / "status").write_text("VmRSS: 12 kB\n")
            (process / "fdinfo/3").write_text("drm-client-id: 1\ndrm-total-vram0: 3 unknown\ndrm-resident-vram0: -1 KiB\n")
            result = benchmark_metrics.memory_snapshot(123, root)
            self.assertEqual(result, {"host_rss_bytes": 12 * 1024})
            self.assertEqual(benchmark_metrics.memory_snapshot(999, root), {})

    def test_tracker_retains_peaks_and_does_not_invent_vram_measurements(self):
        with patch.object(benchmark_metrics, "memory_snapshot", side_effect=[{"host_rss_bytes": 10}, {"host_rss_bytes": 5}]):
            tracker = benchmark_metrics.MemoryTracker(123)
            tracker.sample()
            result = tracker.snapshot()
            self.assertEqual(result["observed_peak"]["host_rss_bytes"], 10)
            self.assertFalse(result["gpu_memory_available"])

    def probe(self, frames):
        response = io.BytesIO(frames)
        ticks = iter(range(1000))
        with patch.object(benchmark_metrics.gpu, "request", return_value=response) as request:
            result = benchmark_metrics.stream_probe(1, {"max_tokens": 256, "messages": []}, clock=lambda: next(ticks) / 1000)
            payload = request.call_args.args[2]
            self.assertEqual(payload["max_tokens"], 16)
            self.assertTrue(payload["stream"])
            return result

    def test_ttft_ignores_roles_nulls_heartbeats_and_whitespace_until_usable_content(self):
        events = [{"choices": [{"delta": {"role": "assistant", "content": None}}]},
                  {"choices": [{"delta": {"content": " \n"}}]},
                  {"choices": [{"delta": {"content": "Hello"}}]},
                  {"choices": [{"delta": {"content": " world"}, "finish_reason": "length"}]}]
        frames = b": heartbeat\r\n\r\n" + b"".join(b"data: " + json.dumps(event).encode() + b"\r\n\r\n" for event in events) + b"data: [DONE]\n\n"
        result = self.probe(frames)
        self.assertGreater(result["first_usable_content_ms"], 0)
        self.assertGreaterEqual(result["total_latency_ms"], result["first_usable_content_ms"])
        self.assertEqual(result["content_bytes"], len(" \nHello world"))

    def test_ttft_fails_closed_on_empty_unterminated_error_or_unbounded_streams(self):
        for frames in (b"data: [DONE]\n\n", b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
                       b'data: {"error":{"message":"failed"}}\n\n', b"x" * (1024 * 1024 + 1)):
            with self.subTest(size=len(frames)), self.assertRaises(ValueError):
                self.probe(frames)

    def test_a_terminator_received_after_the_deadline_cannot_pass(self):
        frames = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
        elapsed = [0]
        class LateTerminator(io.BytesIO):
            def readline(self, *args):
                line = super().readline(*args)
                if self.tell() == len(frames):
                    elapsed[0] = 11
                return line
        with patch.object(benchmark_metrics.gpu, "request", return_value=LateTerminator(frames)):
            with self.assertRaisesRegex(ValueError, "deadline"):
                benchmark_metrics.stream_probe(1, {}, timeout=10, clock=lambda: elapsed[0])

    def test_native_baseline_requires_full_gpu_offload(self):
        self.assertEqual(benchmark_intel.native_gpu_layers("load: offloaded 65/65 layers to GPU"),
                         {"offloaded": 65, "total": 65, "backend": "vulkan"})
        for text in ("CPU-only model", "offloaded 0/65 layers to GPU", "offloaded 12/65 layers to GPU"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                benchmark_intel.native_gpu_layers(text)

    def test_zinc_must_confirm_the_matched_kv_type(self):
        self.assertEqual(benchmark_intel.zinc_kv_storage("info: KV cache: 128 MB (f16)"), "f16")
        for text in ("", "info: KV cache: 256 MB (f32)", "f16 model weights"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                benchmark_intel.zinc_kv_storage(text)

    def test_natural_eos_is_retained_only_when_explicitly_requested_and_never_full_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log = directory / "server.log"
            log.write_text("")
            body = {"choices": [{"message": {"content": "A complete short answer."}, "finish_reason": "stop"}]}
            parsed = {"prefillTokens": 20, "promptTokens": 20, "generatedTokens": 12}
            def request(*args, **kwargs):
                return io.BytesIO(json.dumps(body).encode())
            with patch.object(benchmark_matched_mtp.gpu, "request", side_effect=request), patch.object(benchmark_matched_mtp, "run_json", return_value=parsed):
                child = Mock()
                child.poll.return_value = None
                args = ("llama", "off", child, 1, log, {"body": {}, "max_tokens": 16}, directory, 1)
                with self.assertRaisesRegex(ValueError, "common output budget"):
                    benchmark_matched_mtp.measure(*args)
                sample = benchmark_matched_mtp.measure(*args, require_full_budget=False)
                self.assertFalse(sample["output_budget_complete"])
                self.assertEqual(sample["finish_reason"], "stop")
                body["choices"][0]["finish_reason"] = "length"
                parsed["generatedTokens"] = 16
                self.assertTrue(benchmark_matched_mtp.measure(*args)["output_budget_complete"])
                parsed["generatedTokens"] = 17
                with self.assertRaisesRegex(ValueError, "invalid token accounting"):
                    benchmark_matched_mtp.measure(*args, require_full_budget=False)


if __name__ == "__main__":
    unittest.main()
