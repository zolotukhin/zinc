"""CPU fixtures for Intel measurement, never a GPU or a model load."""
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
