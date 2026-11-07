# Zig 0.17 migration and RDNA validation

ZINC now pins Zig 0.17.1. `./scripts/zig.sh` selects the pinned compiler, downloads
the official archive when necessary, and verifies its SHA256. It preserves an
existing system compiler. `ZINC_TOOLCHAIN_DIR` overrides the toolchain cache.

The migration moves host code to process-provided `std.Io`, native file and
network I/O, `std.Io` synchronization and bounded task groups, translated C
binding modules, and the new reflection/build APIs. Entrypoints use the process
allocator, including the tokenizer benchmark. Tokenization stores initial symbols in one allocation per
chunk, while preserving the merge algorithm and token IDs. JSON responses and
SSE frames use vectored writes; SSE disables Nagle buffering and advertises the
server's actual one-request connection lifetime.

The measurements below describe the initial migration with Zig 0.17.0;
they are not measurements of the 0.17.1 patch upgrade. A subsequent
[BPE optimization](https://github.com/zolotukhin/zinc/blob/main/docs/ZIG_0_17_BPE.md), measured against that already optimized
0.17 build, further reduces Gemma's long-prompt first-output latency by 34%
and fixes ordinary angle-bracket token handling.

Safety testing also caught two migration issues: clearing an ArrayList poisoned
a still-borrowed streaming delta, and the new file streaming API reports EOF as
an error. The streaming detector preserves its borrowed bytes until the next
feed, and the file adapter preserves zero-byte EOF for checksum/read loops.

## Comparison

The baseline is unmodified main at `29bc350ac4cbf9f110ec628b9e177ea04ac816fb`,
built with Zig 0.15.2 and ReleaseFast. The candidate starts from that same source
and uses Zig 0.17.0. GPU kernels, quantization, expert counts and sampling
settings are unchanged.

Measurements ran on a Ryzen 7 9800X3D and Radeon AI PRO R9700 (GFX1201), with
Mesa 25.0.7 and ROCm 7.2.4. Each comparison uses the same GGUF, context budget,
prompts, greedy sampling and warmup count. A reusable server handles each
model's scenarios; server order reverses in the second round. Server-side
prefill/decode timings and client-side latency are recorded separately. The
runner hashes binaries, shader trees, models and response content, checks
output parity, and stops every server it starts.

Final Vulkan medians use three independent server rounds and three measured
requests per scenario in each round:

| Long-context first visible text | Zig 0.15.2 | Zig 0.17.0 |         Change |
| ------------------------------- | ---------: | ---------: | -------------: |
| Qwen 3.5 9B                     |  1386.6 ms |  1357.8 ms | 2.1% less time |
| Qwen 3.6 35B-A3B                |  1312.2 ms |  1261.5 ms | 3.9% less time |
| Gemma 4 26B-A4B                 |  1845.7 ms |  1809.1 ms | 2.0% less time |

GPU decode throughput remains effectively level: the worst Vulkan server round
was 0.18% below its matching baseline, within the 2% guard. These gains include the
allocator and tokenizer changes; the initial compiler-only port was neutral
on GPU decode.
All compared outputs match within each backend and scenario.

ROCm's worst server round was 1.50% below its decode baseline, also within the
2% guard. Its 800-token prefill throughput improved in both measured rounds:

| Model            | Round 1 | Round 2 |
| ---------------- | ------: | ------: |
| Qwen 3.5 9B      |   +4.6% |   +4.3% |
| Qwen 3.6 35B-A3B |  +38.3% |   +5.0% |
| Gemma 4 26B-A4B  |   +1.2% |   +0.9% |

Qwen 35B baseline prefill varied substantially between server rounds. The
per-round gains show that variation; the pooled 19% increase is not a stable
performance estimate.

Locked GPU power profiles produced intermittent slow runs in both versions,
including the unchanged 0.15.2 baseline. Switching to automatic power management
restored full speed within the same slow process. Core and memory clock readings
alone did not identify the slow state. Final comparisons therefore use the
node's normal automatic power policy. The runner also checks each independent
server round against its matching baseline, rather than allowing an aggregate
median to conceal a slow restart. Diagnostic samples are retained alongside
the final measurements.

The CPU-only tokenizer benchmark parses the vocabulary once, warms encoding,
then times 50 encodings per corpus and build, in two alternating rounds. It uses
the allocator used by each production build and verifies every timed token
sequence against its reference:

| Model            |   Code | Mixed text | Prose | Multilingual text |
| ---------------- | -----: | ---------: | ----: | ----------------: |
| Qwen 3.5 9B      | 102.6× |      48.1× | 72.3× |             22.7× |
| Qwen 3.6 35B-A3B | 103.3× |      48.8× | 71.8× |             22.7× |
| Gemma 4 26B-A4B  |  12.6× |       4.1× |  1.7× |              5.3× |

These are prompt-encoding speedups, separate from GPU token generation.

## Validation

- 703 Zig tests in Debug, ReleaseSafe, ReleaseFast and ReleaseSmall.
- 652 Bun tests, including required native API struct-layout extraction.
- Linux Vulkan, hot-decode benchmark and ZINC_RT builds in all four modes.
- ROCm ReleaseFast inference on Qwen 9B, Qwen 35B-A3B and Gemma 26B-A4B.
- Q5_1 route-column GPU results match the scalar CPU reference on five shapes,
  including ragged routes, accumulation and 128 experts.
- CUDA host code checks for the main executable, smoke test, loader test,
  decode driver and debug driver; no NVIDIA runtime performance claim.
- macOS Metal executables and tests compile in all four modes. GPU workloads
  run on the remote node; the workstation GPU is not used.
- 60 tokenizer comparisons against llama.cpp: 20 corpora each on Qwen 9B,
  Qwen 35B-A3B and Gemma 26B-A4B, all matching.
- OpenAI API and SDK suites on Qwen and Gemma: streaming/non-streaming,
  sequential/overlapping streams, active/queued health, cancellation, raw
  completions, invalid requests and missing endpoints.
- Eight GGUFs compared against the 0.15.2 executable, including Qwen 27B,
  Qwen 3.8 27B, Gemma 31B, Muse Glimmer and GPT-OSS beyond the primary matrix.
- 240 measured requests per build from six concurrent API clients after warmup;
  every response matches and resident memory remains stable.
- Real HTTP model downloads into an isolated cache: content-length, chunked,
  cache reuse, checksum rejection and truncated-body rejection. File contents,
  manifests, activation and partial-file cleanup are checked. A separate real
  HTTPS download checks certificate validation, redirects, SHA256, manifest and
  active-model selection.
- Network tests cover UTF-8 SSE framing and a 512 KiB response with a constrained
  socket send buffer. Executor tests cover zero, one and three worker threads.

CI pins 0.17.1 and checks the four optimization modes, hot-decode and CUDA entrypoints,
ZINC_RT and Metal compilation. The site build also installs the compiler and
headers required by its native API documentation probe.

Unsuccessful Q4_K loop/specialization changes, Q8 specialization, fused SSM
projection and fence-spinning experiments were excluded after repeated runs
showed no sustained decode gain. The shipped shader sources are unchanged.

## Reproduce

Run model and benchmark commands on the designated GPU host. Build the frozen
baseline in a separate checkout with Zig 0.15.2.

To reproduce these measurements, check out the recorded 0.17.0 candidate source
and run the commands below. Its wrapper selects that compiler; current main
selects the 0.17.1 patch release.

```sh
./scripts/zig.sh build -Doptimize=ReleaseFast
./scripts/zig.sh build test check-cuda -Doptimize=ReleaseSafe
./scripts/zig.sh build tokenize-tool tokenize-bench -Doptimize=ReleaseFast

python3 tools/compare_builds.py \
  --baseline "$ZINC_BASELINE" --candidate "$PWD" \
  --model-root "$ZINC_MODELS" --device "$ZINC_DEVICE" \
  --models Qwen3.5-9B-Q4_K_M.gguf \
  --stream --runs 3 --warmup 2 --rounds 2 \
  --max-decode-regression-pct 2 --output comparison.json

./zig-out/bin/zinc-tokenize-bench "$MODEL_GGUF" "$CORPUS_FILE" 50
```

The example preserves the existing power policy. The optional `--lock-clocks`
flag requires sysfs write access and restores the prior setting; it is useful
for diagnostics, but locked profiles caused instability on this test node.

Measurement dumps use lossless gzip compression with uncompressed SHA256 hashes.
Raw samples, exact tokenizer corpora and the matching 0.15.2 tokenizer benchmark
source are in `docs/validation/zig-0.17/`. The source fixture is stored as text
and compiled against the frozen baseline's tokenizer. Comparisons retain their
binary hashes, so results remain attributable to the measured executable.
