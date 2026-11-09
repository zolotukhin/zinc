"""Model-oracle protocol fixtures. All servers and GPU operations are mocked."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import benchmark_intel as benchmark


class NativeReferenceTest(unittest.TestCase):
    def test_workloads_are_rendered_once_with_the_oracle_and_become_identical_raw_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(oracle=root / "oracle", model=root / "model.gguf", oracle_device="Vulkan0", context=4096)
            scenarios = [{"id": "core", "max_tokens": 64, "body": {"messages": [{"role": "user", "content": "question"}]}}]
            original = copy.deepcopy(scenarios)
            with patch.object(benchmark.gpu, "process", return_value=contextlib.nullcontext(Mock())), \
                    patch.object(benchmark.gpu, "ready"), patch.object(benchmark.gpu, "free_port", return_value=1), \
                    patch.object(benchmark.gpu, "json_request", return_value={"prompt": "rendered exact bytes"}) as request:
                result = benchmark.raw_scenarios(args, scenarios, root, {})
            self.assertEqual(scenarios, original)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(result[0]["body"], {"prompt": "rendered exact bytes", "max_tokens": 64,
                             "temperature": 0, "top_p": 1, "seed": 0, "cache_prompt": False})
            self.assertEqual(result[0]["prompt_mode"], "raw")

    def test_raw_stream_requires_complete_text_identity_usage_and_terminator(self):
        baseline = {"model": "fixture", "choices": [{"text": "café 東京", "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 42, "completion_tokens": 16, "total_tokens": 58}}
        common = {"id": "cmpl-test", "created": 123, "model": "fixture", "object": "text_completion"}
        events = [{**common, "choices": [{"index": 0, "text": "café ", "finish_reason": None}]},
                  {**common, "choices": [{"index": 0, "text": "東京", "finish_reason": None}]},
                  {**common, "choices": [{"index": 0, "text": "", "finish_reason": "length"}]},
                  {**common, "choices": [], "usage": baseline["usage"]}]
        def transcript(rows, done=True):
            return ("".join("data: " + json.dumps(x, ensure_ascii=False) + "\n\n" for x in rows)
                    + ("data: [DONE]\n\n" if done else "")).encode()
        self.assertEqual(benchmark.raw_stream_parity(transcript(events), baseline)["usage"], baseline["usage"])
        for mutation in ("text", "identity", "usage", "finish", "chat", "late-content", "missing-usage"):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(events)
                if mutation == "text": rows[1]["choices"][0]["text"] = "wrong"
                if mutation == "identity": rows[1]["id"] = "other"
                if mutation == "usage": rows[3]["usage"]["completion_tokens"] = 15
                if mutation == "finish": rows[2]["choices"][0]["finish_reason"] = "stop"
                if mutation == "chat": rows[0]["object"] = "chat.completion.chunk"
                if mutation == "late-content": rows.append(rows[0])
                if mutation == "missing-usage": rows.pop()
                with self.assertRaises(ValueError): benchmark.raw_stream_parity(transcript(rows), baseline)
        with self.assertRaisesRegex(ValueError, "terminate"):
            benchmark.raw_stream_parity(transcript(events, done=False), baseline)
        with self.assertRaisesRegex(ValueError, "terminate"):
            benchmark.raw_stream_parity(json.dumps(baseline).encode(), baseline)

    def expected(self, ids=None, content="primary"):
        ids = [1] * 16 if ids is None else ids
        return {"content": content, "completion_probabilities": [
            {"id": token, "top_logprobs": [{"id": 1, "logprob": -0.10}, {"id": 2, "logprob": -0.11}]} for token in ids]}

    def execute(self, mode="near-tie"):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            samples = [{"prompt": f"prompt {i}", "expected": self.expected()} for i in range(3)]
            (work / "reference-samples.json").write_text(json.dumps(samples))
            args = SimpleNamespace(llama_bin=work / "native", oracle=work / "oracle", model=work / "model.gguf",
                                   context=4096, reference_device="SYCL0", oracle_device="Vulkan0", sycl_prefix=work / "prefix")
            active, commands, requests = [], [], []
            @contextlib.contextmanager
            def process(command, log, cwd, env):
                self.assertEqual(active, [], "oracle and native processes must not overlap")
                active.append(command); commands.append(command)
                log.write_text("offloaded 29/29 layers to GPU\nSYCL0 model buffer size = 500 MiB\n")
                try: yield SimpleNamespace(pid=123)
                finally: active.clear()
            generated = {"count": 0}
            def request(port, path, body):
                requests.append((path, copy.deepcopy(body)))
                if path == "/completion" and isinstance(body["prompt"], str):
                    index = generated["count"]; generated["count"] += 1
                    if index == 0:
                        if mode == "short": return self.expected([1], "primary")
                        if mode == "wrong-text": return self.expected(content="different text")
                        return self.expected([2] + [1] * 15, "branch")
                    return self.expected()
                if path == "/tokenize": return {"tokens": [99]}
                if path == "/completion":
                    self.assertEqual(body["prompt"], [99, 2])
                    self.assertEqual(body["n_predict"], 15)
                    return self.expected([3 if mode == "bad-tail" else 1] * 15)
                if path == "/detokenize": return {"content": "branch"}
                raise AssertionError("unexpected protocol request")
            runtime = {"runtime": "level_zero", "library_sha256": {"libsycl.so.9": "a" * 64}}
            with patch.object(benchmark.gpu, "process", side_effect=process), patch.object(benchmark.gpu, "ready"), \
                    patch.object(benchmark.gpu, "free_port", return_value=1), patch.object(benchmark.gpu, "json_request", side_effect=request), \
                    patch.object(benchmark.sycl, "runtime_evidence", return_value=runtime):
                try:
                    result = benchmark.native_reference_validation(args, work, {}, {})
                finally:
                    self.assertEqual(active, [])
                    # Even a count/text/suffix failure keeps the actual native
                    # response for diagnosis; it must not vanish on exception.
                    saved = json.loads((work / "native-reference-case-0.json").read_text())
                    self.assertIn("actual", saved)
                    self.assertIn("expected", saved)
            return result, commands, requests

    def test_close_choice_requires_independent_oracle_suffix_and_complete_text(self):
        result, commands, requests = self.execute()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["cases"]), 3)
        self.assertEqual(len(commands), 2)
        self.assertIn("SYCL0", commands[0]); self.assertIn("Vulkan0", commands[1])
        self.assertEqual(commands[0][commands[0].index("-ctk") + 1], "f16")
        self.assertEqual(commands[1][commands[1].index("-ctk") + 1], "f32")
        self.assertEqual(commands[1][commands[1].index("-ub") + 1], "1")
        self.assertTrue(any(path == "/detokenize" for path, _ in requests))

    def test_close_first_choice_cannot_excuse_a_later_wrong_token(self):
        with self.assertRaisesRegex(ValueError, "after the near tie"):
            self.execute("bad-tail")

    def test_incomplete_output_or_changed_text_cannot_validate_the_baseline(self):
        for mode, message in (("short", "count differs"), ("wrong-text", "text differs")):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, message):
                self.execute(mode)


if __name__ == "__main__":
    unittest.main()
