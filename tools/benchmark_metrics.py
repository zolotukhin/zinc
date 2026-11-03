"""Process memory and first usable SSE content for GPU benchmark workers."""

import hashlib
import json
from pathlib import Path
import re
import threading
import time

import release_gpu as gpu
from release_lib import require


def memory_snapshot(pid, proc_root=Path("/proc")):
    result = {}
    root = proc_root / str(pid)
    try:
        status = (root / "status").read_text()
        for field, name in (("VmRSS", "host_rss_bytes"), ("VmHWM", "host_peak_rss_bytes"),
                            ("RssAnon", "host_anonymous_bytes"), ("RssFile", "host_file_bytes")):
            match = re.search(r"^" + field + r":\s*(\d+) kB$", status, re.M)
            if match:
                result[name] = int(match[1]) * 1024
        clients = set()
        for descriptor in (root / "fdinfo").iterdir():
            try:
                text = descriptor.read_text()
            except (OSError, UnicodeError):
                continue
            fields = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
            if "drm-client-id" not in fields:
                continue
            # These identifiers are used only to avoid counting duplicated
            # descriptors. Neither client IDs nor PCI addresses enter reports.
            identity = tuple(fields.get(key, "").strip() for key in ("drm-driver", "drm-pdev", "drm-client-id"))
            if identity in clients:
                continue
            clients.add(identity)
            for key, value in fields.items():
                match = re.fullmatch(r"drm-(total|resident)-(vram\d*|system|gtt)", key)
                number = re.fullmatch(r"\s*(\d+)\s*(KiB|MiB|B)?\s*", value)
                if not match or not number:
                    continue
                region = "vram" if match[2].startswith("vram") else match[2]
                name = "gpu_" + match[1] + "_" + region + "_bytes"
                amount = int(number[1]) * {None: 1, "B": 1, "KiB": 1024, "MiB": 1024 * 1024}[number[2]]
                result[name] = result.get(name, 0) + amount
    except (OSError, UnicodeError):
        pass
    return result


class MemoryTracker:
    """Observed DRM peaks sampled at 50 ms; kernel VmHWM retains the RSS peak."""
    def __init__(self, pid, interval=0.05):
        self.pid, self.interval = pid, interval
        self.peak = {}
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        values = memory_snapshot(self.pid)
        with self.lock:
            for key, value in values.items():
                self.peak[key] = max(self.peak.get(key, 0), value)

    def run(self):
        while not self.stopped.wait(self.interval):
            self.sample()

    def snapshot(self):
        self.sample()
        with self.lock:
            return {"observed_peak": dict(self.peak), "sample_interval_ms": round(self.interval * 1000),
                    "gpu_memory_available": "gpu_resident_vram_bytes" in self.peak,
                    "source": "Linux process status and DRM fdinfo"}

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join(timeout=2)
        self.sample()


def stream_probe(port, body, timeout=600, clock=time.monotonic):
    # A separate short probe measures first usable content without replacing
    # the ordinary full-output phase/latency samples with short generations.
    payload = {**body, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 16}
    started, first, content, done = clock(), None, [], False
    data = []
    with gpu.request(port, "/v1/chat/completions", payload, timeout=timeout) as response:
        received = 0
        while True:
            require(clock() - started < timeout, "streaming benchmark exceeded its deadline")
            raw = response.readline(1024 * 1024 + 1)
            if not raw:
                break
            received += len(raw)
            require(received <= 1024 * 1024, "streaming benchmark exceeded its transcript limit")
            line = raw.decode("utf-8", errors="strict").rstrip("\r\n")
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
                continue
            if line or not data:
                continue
            event, data = "\n".join(data), []
            if event == "[DONE]":
                done = True
                break
            item = json.loads(event)
            require("error" not in item, "streaming benchmark returned an error")
            choices = item.get("choices", [])
            if choices:
                text = choices[0].get("delta", {}).get("content")
                if text is None:
                    continue
                require(isinstance(text, str), "streaming content is not text")
                if text:
                    content.append(text)
                    if first is None and text.strip():
                        first = (clock() - started) * 1000
    require(done and first is not None, "streaming benchmark lacked usable content or its terminator")
    text = "".join(content)
    return {"first_usable_content_ms": first, "total_latency_ms": (clock() - started) * 1000,
            "output_cap": 16, "content_sha256": hashlib.sha256(text.encode()).hexdigest(), "content_bytes": len(text.encode())}
