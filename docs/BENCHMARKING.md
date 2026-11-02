# Benchmarking ZINC

How to reproduce ZINC's published numbers, run an ad-hoc llama.cpp baseline, and use the per-kernel hot-bench. Apple Silicon notes at the bottom.

## Published benchmarks (the canonical numbers)

The [benchmark dashboard](https://zolotukhin.ai/zinc/benchmarks/) renders the measurements in `site/src/data/zinc-performance.json`, collected by `tools/performance_suite.mjs`. Runs pair ZINC against llama.cpp on the same hardware and model file across four workloads: quick chat (`core`), coding review (`context-medium`), incident context (`context-long`), and long coding draft (`decode-extended`). Production is published with a manual Cloudflare Pages upload after validation; a push to `main` does not deploy it.

### Inspect a published result

1. Select a platform and read **both backend labels**. AMD Vulkan and ROCm are separate targets. Open **Exact configuration & methodology** for the recorded build, llama.cpp device, linked source commits, binary checksum when available, and run policy.
2. Select a workload. Expand a model's **Prompt, timings & source** for its user prompt, both engines' prompt/output counts, requested cap, sample counts, phase times, request latency, and model-specific source revisions. Model dates and revisions can differ within a partially refreshed target.
3. Download the [published JSON](https://zolotukhin.ai/zinc/benchmarks.json) for individual samples, output previews, provenance, and MTP variants. The JSON is the same artifact used by the page. **Copy link to this view** preserves the backend, workload, and MTP setting.

The overall ratio is `llama.cpp phase time / ZINC phase time`, where each engine's phase time is `prefill_tokens / prefill_tps + generated_tokens / decode_tps`. It excludes HTTP, process startup, and model loading. `1.00×` is parity; larger ratios favor ZINC. The summary divides the sum of comparable llama.cpp phase times by the sum of ZINC phase times, rather than averaging ratios. Rate summaries are means of model medians. Missing measurements are not zeroes.

The page excludes mismatched or unrecorded MTP settings, mismatched prompt counts, unequal or incomplete output budgets, flagged previews, and incomplete phase telemetry from overall comparisons. The recorded prompt-parity check allows a one-token BOS accounting difference. A preview is a sanity check, not an evaluation of answer quality. Older Metal, Intel, and CUDA records used ZINC CLI runs against a baseline that preferred a persistent server; the page labels those protocols as historical diagnostics. Their request latencies must not be compared as if both engines were reusable servers.

Follow the runner link at the **recorded revision** to reproduce historical flags, templates, and system messages. Current defaults may have changed. Some older artifacts lack driver versions, model checksums, or exact build metadata; the page says **Not recorded** instead of filling those gaps from today's machine. Reproducing those records exactly may require a new run with the missing metadata captured.

The historical Intel baseline's recorded short commit `43c0ac9` could not be resolved in the upstream llama.cpp repository. It is preserved as text and labeled unavailable, rather than linked to a broken source page. The R9700 hardware metadata has also been corrected from 640 to **576 GB/s**, matching [AMD's datasheet](https://www.amd.com/content/dam/amd/en/documents/partner-hub/radeon-pro/radeon-ai-pro-r9700-datasheet.pdf). Recorded timings and sample values were not changed.

Remote machine details are intentionally supplied by `.env` or CLI flags. Do not commit public hostnames, public IP addresses, private SSH ports, or one-off benchmark node aliases into this document or into published site data.

### Fair comparison contract

Use the suite's server-vs-server path for any headline "ZINC vs baseline" claim on RDNA. That means one reusable ZINC server per model, one reusable baseline server per model, the same GGUF, the same prompt matrix, the same warmup/run count, and server-side timing for prefill/decode. Do not compare a one-shot ZINC CLI run against a warmed baseline server and call that a result; the CLI path is useful for local engine diagnostics only.

Both engines must read the same prompt. Chat requests therefore carry an explicit system turn and `enable_thinking: false` (top level for ZINC, `chat_template_kwargs` for llama.cpp). Without them the two servers render different prompts from the same request: ZINC injects its own short system turn and closes the think block, while the GGUF Jinja that llama.cpp runs can add a long reasoning directive and open one (Qwen 3.8 read 47–66 prompt tokens in ZINC against 87 in llama.cpp). Every comparison records both prompt token counts and a `prompt_parity` flag, and the dashboard shows how many rows matched.

`prompt_parity` holds only when the two token counts differ by at most one (BOS accounting). A looser allowance once hid a Gemma 4 tokenizer bug that turned a 353-token prompt into 357.

### Check the output against llama.cpp before publishing

Fast is only interesting if it is right, and ZINC agreeing with itself proves nothing: a Gemma 4 tokenizer that merged by score instead of rank, and a Vulkan gate/up kernel that skipped half of every Q4_K block, both stayed "token-identical" from run to run for months while the model read misspelled words and its hidden states drifted from the reference. Before publishing a model's numbers, or after any tokenizer or kernel change, run the parity check on the GPU machine with no server holding the GPU:

```bash
zig build -Doptimize=ReleaseFast && zig build tokenize-tool -Doptimize=ReleaseFast
tools/llama_parity.sh --zinc-dir zig-out --llama-dir <llama.cpp>/build/bin \
  --llama-args "--device <llama-device> -ngl 99" \
  --prompt-file <long prompt> <model.gguf>...
```

It compares `zinc-tokenize` with `llama-tokenize --ids` over a built-in multilingual/code corpus, then lets llama-server render each chat prompt and has both engines greedily continue the identical text, comparing the first token and the continuation. Any difference exits non-zero. To localize a numeric difference, dump per-layer residuals with `ZINC_LAYER_DIAG_POS=<position>` (decode and the grouped Gemma prefill both support it) and compare them with `llama-eval-callback` rows (`l_out-N`, and stage tensors such as `attn_out-N`, `ffn_moe_out-N`).

Compare like with like: a ROCm ZINC row is measured against llama.cpp's HIP backend, a Vulkan row against llama.cpp's Vulkan backend, both from the same llama.cpp commit. Speculative decoding (NextN/MTP) is published as a separate variant; the headline bars compare both engines without it.

The RDNA ZINC requests intentionally omit the OpenAI `model` field. The model is selected by the server process at launch with `-m <gguf>`; sending `model: "q"` to ZINC can trigger managed-model routing instead of measuring the loaded GGUF.

```bash
# Metal target (Apple Silicon, runs locally)
bun tools/performance_suite.mjs --target metal

# AMD Vulkan, remote node from .env. Select the baseline device from
# vulkaninfo and set ZINC_RDNA_LLAMA_DEVICE to that device's llama.cpp name.
bun tools/performance_suite.mjs --target rdna --phase all --mtp off \
  --rdna-sync --rdna-build --rdna-start-llama --rdna-backend vulkan \
  --rdna-llama-device "${ZINC_RDNA_LLAMA_DEVICE:?Set the llama.cpp Vulkan device}" \
  --runs 3 --warmup 1 --no-site-write --output /tmp/zinc-vulkan.json

# Pick a specific RDNA node when .env defines ZINC_RDNA1_* and ZINC_RDNA2_*.
bun tools/performance_suite.mjs --target rdna --rdna-node rdna2 --rdna-sync --rdna-build --rdna-backend vulkan

# Intel Arc target (separate remote node from .env)
bun tools/performance_suite.mjs --target intel --intel-sync --intel-build --intel-start-llama

# Temporary password-only Intel nodes can use a process environment secret.
# Do not commit the value; prefer a throwaway shell export or password file.
export ZINC_INTEL_SSH_PASSWORD='<temporary-password>'
bun tools/performance_suite.mjs --target intel --intel-sync --intel-build --intel-start-llama

# Everything in one run
bun tools/performance_suite.mjs --target all --rdna-sync --rdna-build --rdna-start-llama
```

Defaults are 1 warmup + 3 measured runs per scenario; pass `--runs N --warmup M` to override. Pass `--no-site-write` to leave `site/src/data/zinc-performance.json` alone (the run still emits a `/tmp` artifact via `--output`). The suite stamps `provenance.zinc.version` from `git describe --dirty`; when `--rdna-sync` is used, provenance comes from the local tree that was synced, not from the remote workdir's stale `.git` metadata. Commit (or stash) before publishing — a `-dirty` tag means reviewers cannot reproduce the exact tree.

The run count, warmup count, timeout and RDNA device-index arguments require complete unsigned decimal safe integers. Leading zeroes are accepted. Fractions, exponents, signs, surrounding whitespace, suffixes and unsafe integers fail before a benchmark starts. Run count must be positive; the other three retain their existing zero behavior. The environment-based RDNA device selector follows the same validation.

RDNA suite runs sync into `/root/zinc-bench` by default. Keep that checkout isolated from optimization loops and ad-hoc experiments; sharing `/root/zinc` can overwrite `zig-out/bin/zinc` mid-suite and invalidate later model rows. The suite verifies `zinc --version` reports the requested backend before RDNA measurements, and before every ZINC scenario, so a stale or overwritten binary fails loudly instead of publishing mixed Vulkan/ZINC_RT data.

`bun tools/performance_suite.mjs --help` lists every flag (target subset, model filtering, baseline binary overrides, remote workdir / libc / model-root overrides, etc.).

For password-auth Intel nodes, the suite reads `ZINC_INTEL_SSH_PASSWORD`, `ZINC_INTEL_SSH_PASSWORD_ENV`, or `ZINC_INTEL_SSH_PASSWORD_FILE` and drives `ssh`/`rsync` through `SSH_ASKPASS`. The generic `loops/optimize_gpu.ts` loop accepts the same variables, plus the `ZINC_GPU_*` equivalents. The generated benchmark commands reference only the env-var name or file path, not the password itself. Remove the temporary secret after the node is converted to key-based SSH.

The default RDNA suite covers Gemma 4 26B-A4B Q4_K_M, Gemma 4 31B Q4_K_M, Qwen 3.5 9B Q4_K_M, Qwen 3.6 35B-A3B UD Q4_K_XL, and Qwen 3.8 27B Q4_K_M. The small-Qwen row is `Qwen3.5-9B-Q4_K_M.gguf`, not the older Qwen 3 8B GGUF.

## Latest llama.cpp and matched MTP pairs

New comparisons use the upstream revision in
[`config/benchmark-reference.json`](../config/benchmark-reference.json).
Before a campaign, refresh the pin, then build both `llama-server` and
`llama-tokenize` from that exact checkout for each native GPU backend:

```bash
bun tools/benchmark_reference.mjs update
# Build the pinned checkout on the remote worker; see the commands below.
bun tools/benchmark_reference.mjs check
```

The main suite checks upstream HEAD before collecting a new baseline and
requires its executable to identify the pin and an exact SHA256. If upstream
advances, refresh and rebuild before the next campaign. Freeze the revision
through an active campaign. Release correctness qualification has its own
separate pin in `config/release/targets.json`; refreshing the performance
reference does not revise old release evidence.

Current llama.cpp supports the embedded NextN block directly. MTP on means
`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`; off means
`--spec-type none`. A separate exported draft model is unnecessary for this
Qwen 3.8 GGUF. See the upstream [speculative decoding guide](https://github.com/ggml-org/llama.cpp/blob/785322d17f2d51e5185cf996a1f4e46d87640841/docs/speculative.md).

The suite accepts `--mtp on|off` and `--mtp-drafts N` for `--target rdna`.
It configures both engines and requires request-specific drafting telemetry:
ZINC's acceptance counters and llama.cpp's `timings.draft_n` /
`timings.draft_n_accepted`. A flag alone is insufficient evidence that MTP ran.
Every sample in an on run must show drafting; an off run must not show it.

The dashboard's off view selects **both** engines' off measurements; the on
view selects **both** engines' on measurements. Unknown or different settings
retain their raw speeds and dates but cannot produce a ratio. The README
chart applies the same guard. Historical MTP-on/off comparisons are not
relabeled as matched measurements.

Gemma's [Vulkan revalidation protocol](VULKAN_GEMMA_SLIDING_WINDOW.md#repeating-the-benchmark-after-a-correctness-fix)
runs the numerical, tokenizer, complete continuation and recovery gates before
the repeated server comparisons. Its [fresh AMD and Intel results](VULKAN_GEMMA_SLIDING_WINDOW.md#fresh-vulkan-measurements)
record the performance baseline and correctness oracle separately. Publication
replaces only a passing model's rows and preserves its complete old record and
caution in measurement history.

### Controlled 27B campaign

[`tools/benchmark_matched_mtp.py`](../tools/benchmark_matched_mtp.py) uses the
canonical suite's four prompts, payloads, parsers and summary statistics. Run
it **on the remote Linux GPU worker**. The worker must be idle; an existing
GPU process or reservation fails without terminating anything. Use the
scheduler's shared reservation location, not ZINC's native process-lock file.
Each server and its children are owned and reaped inside the invocation,
including failures and cancellation.

Build the reference on that worker from the performance pin. Select one
backend per build directory:

```bash
# LLAMA_SOURCE and LLAMA_BUILD are operator-supplied paths on the worker.
# Check out the commit from config/benchmark-reference.json first.
cmake -S "$LLAMA_SOURCE" -B "$LLAMA_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$LLAMA_BUILD" -j4 --target llama-server llama-tokenize

# For ROCm use a separate build with -DGGML_HIP=ON instead of Vulkan,
# -DAMDGPU_TARGETS=<gpu-target>, and the ROCm clang/clang++ compilers.
```

Build ZINC ReleaseFast for the same backend, then run:

```bash
python3 tools/benchmark_matched_mtp.py \
  --backend vulkan --device "$ZINC_GPU_DEVICE" \
  --llama-device "$LLAMA_GPU_DEVICE" \
  --zinc "$ZINC_BINARY" --zinc-source "$ZINC_SOURCE" \
  --llama-server "$LLAMA_BUILD/bin/llama-server" --llama-source "$LLAMA_SOURCE" \
  --model "$MODEL_GGUF" --reservation "$GPU_RESERVATION" \
  --rounds 3 --runs 3 --warmup 1 --drafts 3 --output "$REPORT_JSON"
```

Use `--backend rocm` with the ROCm builds and device selectors for HIP.
Both engines use the identical pinned Qwen 3.8 27B Q4_K_M file, F32 K/V,
4,096 context tokens, one slot, temperature zero, seed zero, full prompt
prefill, and common output caps. F32 is explicitly selected through
`ZINC_KV_F16=0 ZINC_KV_Q8=0` for Vulkan and `ZINC_KV_CACHE=f32` for ROCm;
the runner requires ZINC's startup log to confirm it. The ordinary suite's
`--rdna-llama-kv` option now sets both engines' cache storage as well.

Three rounds alternate treatment order. Each treatment keeps a server alive
across all workloads, excludes one warmup per workload and measures three
requests. That is nine measured samples per workload/engine/mode, or
144 measured requests plus 48 warmups per backend. HTTP latency is captured
separately from engine phase timing. All responses must reach the common
output budget, have matching prompt counts within one BOS token, and preserve
each engine's complete greedy output across repeats and MTP modes. This is a
bounded correctness control, not a replacement for release qualification.

The default runner stops on a greedy-output difference. For an investigation,
`--record-output-drift` retains mode differences while still rejecting
nondeterministic repeats. Such a report has `protocol_passed: true` only if the
full timing matrix completed, and `passed: false` when any output control
failed. Its `output_drift` entries remain failures; completing measurements
does not clear them or establish that either wording is incorrect.
`--engines` and `--scenarios` select diagnostic subsets, which cannot qualify
a complete four-workload campaign.

The report records model/source/binary/library/harness hashes and acceptance
counts. Reference libraries and the ZINC binary must remain byte-identical
through the campaign. ZINC's draft vocabulary may be restricted to 98,304
rows; llama.cpp uses its default draft vocabulary. Both verify with the full
target model. Report these implementation choices alongside matched mode
and draft limits rather than treating MTP-off as the competing MTP result.

The [November 2026 matched campaign](MATCHED_MTP_LLAMA_CPP.md) retains the
complete Vulkan and ROCm evidence, including failed exact-output controls.
For MTP-on phase comparisons, record whether the prefill timer includes draft
priming. Different or unknown coverage excludes the prefill and summed-phase
ratios; full HTTP latency includes priming for both engines.

Publish completed reports with `tools/merge_matched_mtp.mjs --report <file>`
(repeat for both backends) and `--harness-commit <full-frozen-commit>`. The
publisher verifies hashes against that immutable source. Failed output
controls require explicit `--include-diagnostics` and keep their on-mode
ratios excluded; they cannot clear valid off-mode results or historical
model-wide cautions. This produces the site's paired engine variants and
public raw campaign files.

The legacy `merge_mtp_variant.mjs` helper accepts a freshly measured baseline
for the off variant. Reusing an old baseline with missing or different MTP
settings cannot create a comparison.

## Ad-hoc llama.cpp baseline on the RDNA4 node

Use this when you want a one-off llama.cpp number outside the perf suite — for example, to validate a Mesa or driver change before doing a full publish run.

**Model on the RDNA4 test node**: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22.4 GiB, MoE 35B/3B active).

The reference setup on the RDNA4 test node:

### Test node setup (critical for reproducing the baseline)

```bash
# 1. Mesa must be 25.0.7 (25.2.8 causes ~14% RADV regression)
dpkg -l mesa-vulkan-drivers  # should show 25.0.7-0ubuntu0.24.04.2
# Pinned in /etc/apt/preferences.d/mesa-pin to prevent auto-upgrade

# 2. GECC disabled (amdgpu.ras_enable=0 in /etc/default/grub)
cat /sys/module/amdgpu/parameters/ras_enable  # should show 0

# 3. RADV_PERFTEST=coop_matrix set in llama-server.service
#    Without this, cooperative matrix is disabled → scalar fallback

# 4. llama.cpp must be recorded from the benchmark binary:
#    /root/llama.cpp/build/bin/llama-server --version
#    The published artifact records the exact version and commit under
#    provenance.llama_cpp. Rebuild and republish when changing it.

# 5. Server flags (in /etc/systemd/system/llama-server.service):
#    -ngl 99 --device Vulkan<N> --parallel 4 -c 32768
#    -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 --mlock --flash-attn on
#    Pick the discrete GPU index from `vulkaninfo --summary`; on mixed APU+dGPU
#    nodes this may be Vulkan1 rather than Vulkan0.
```

### Measure llama.cpp

```bash
source .env

# Start server (if not running)
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "systemctl start llama-server && sleep 15"

# Warmup + 3 benchmark runs via OpenAI API
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST '
  curl -s http://localhost:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" > /dev/null
  for i in 1 2 3; do
    out=$(curl -s http://localhost:8088/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":256,\"stream\":false}" \
    )
    gen=$(printf "%s" "$out" | jq -r ".timings.predicted_per_second // 0")
    prompt=$(printf "%s" "$out" | jq -r ".timings.prompt_per_second // 0")
    printf "Run %d: gen %s tok/s | prompt %s tok/s\n" "$i" "$gen" "$prompt"
  done
'
# Treat this as a one-off sanity check. The canonical values are the
# medians in site/src/data/zinc-performance.json and on /zinc/benchmarks.
```

## ROCm server comparison

Use the same full server-vs-server suite for ROCm claims:

```bash
ZINC_MTP=0 bun tools/performance_suite.mjs \
  --target rdna \
  --phase all \
  --rdna-backend rocm \
  --rdna-sync \
  --rdna-build \
  --rdna-start-llama \
  --rdna-llama-device "${ZINC_RDNA_LLAMA_DEVICE:?Set the llama.cpp ROCm device}" \
  --runs 3 --warmup 1 --no-site-write --output /tmp/zinc-rocm.json
```

The published ROCm target records both backend identities in its methodology.
ZINC runs its native ROCm/HIP backend; the comparison server uses the selected
llama.cpp device on the same Radeon GPU and the same GGUF. Published
ROCm rows use `--rdna-llama-device ROCm0` with a llama.cpp HIP build
(`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201`) at the same commit as the Vulkan
baseline, passed through `--rdna-llama-server`. A Vulkan device such as
`Vulkan0` gives a cross-backend baseline, which is not the headline
comparison. Never relabel one as the other. The benchmark artifact records the exact choice, binary checksum, and
revision.

The current results and raw samples live on the
[benchmark dashboard](https://zolotukhin.ai/zinc/benchmarks/#rdna-rocm). The
older `llama-bench` pp/tg microbench archive has been retired so it cannot be
mistaken for the reusable-server comparison.

## Measure ZINC (CLI diagnostics only)

Use this when you need quick engine logs, token traces, or a narrow sanity check. Do not use this path for published prefill/decode comparisons against a persistent baseline server.

```bash
source .env

# Sync source to test node
rsync -az --delete --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' \
  --exclude '.DS_Store' --exclude 'site' \
  -e "ssh -p $ZINC_PORT" . $ZINC_USER@$ZINC_HOST:/root/zinc/

# Build and run
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "cd /root/zinc && zig build -Doptimize=ReleaseFast && \
  RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is'"

# Key output lines:
#   info(forward): Prefill complete: N tokens in X ms (Y tok/s)
#   info(forward): Generated N tokens in X ms — Y tok/s (Z ms/tok)
```

## Measure ZINC (HTTP)

Use the HTTP benchmarks for end-to-end API latency, queueing behavior, or to compare the chat endpoint against the raw completions path. For headline RDNA comparisons, prefer `tools/performance_suite.mjs`; it starts/stops both engines and extracts comparable server-side timings.

Caveats:

1. Bench a clean node. Other `zinc`, `llama-server`, and `llama-cli` processes on the RDNA4 host contaminate latency and throughput.
2. `POST /v1/chat/completions` is an end-user latency benchmark, not a pure decode-throughput benchmark. The chat route applies templates and stop handling, so many prompts stop after only a handful of tokens.
3. Use `POST /v1/completions` for sustained HTTP decode throughput.
4. ZINC server generation is still serialized. With `concurrency > 1`, aggregate throughput stays roughly flat while per-request latency grows because requests queue behind one active decode.

Clean-server setup:

```bash
source .env

# 1. Stop stale GPU users on the test node.
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  pkill -f 'zig-out/bin/zinc' || true; \
  pkill -f 'llama-server' || true; \
  pkill -f 'llama-cli' || true"

# 2. Sync, build, and restart one clean ZINC server on :9090.
rsync -az --delete --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' \
  --exclude '.DS_Store' --exclude 'site' \
  -e "ssh -p $ZINC_PORT" . $ZINC_USER@$ZINC_HOST:/root/zinc/

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && zig build -Doptimize=ReleaseFast && \
  nohup env RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
    -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --port 9090 >/tmp/zinc_9090.log 2>&1 < /dev/null &"

# 3. Wait for health.
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  until curl -fsS http://127.0.0.1:9090/health >/dev/null; do sleep 1; done; \
  curl -sS http://127.0.0.1:9090/health"
```

Chat-endpoint latency matrix:

```bash
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  /root/.bun/bin/bun tools/benchmark_api.mjs \
    --base http://127.0.0.1:9090/v1 \
    --mode chat \
    --output /tmp/zinc_api_chat_benchmark.json"
```

Raw sustained throughput:

```bash
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  /root/.bun/bin/bun tools/benchmark_api.mjs \
    --base http://127.0.0.1:9090/v1 \
    --mode raw \
    --output /tmp/zinc_api_raw_benchmark.json"
```

## Latest single-stream reference

Current numbers for every (target, model, scenario) combination live on the dashboard: [zolotukhin.ai/zinc/benchmarks](https://zolotukhin.ai/zinc/benchmarks). The dashboard is regenerated by `tools/performance_suite.mjs` on every publish run and reports prefill, decode, and a combined prompt-plus-decode "overall" ratio against the same-machine llama.cpp baseline.

## Hot-bench: per-kernel microbenchmarks

Use the dedicated microbenchmark when whole-model decode says "MoE", "shared expert", or `ssm_delta_net` is hot and you need exact per-kernel numbers plus `RADV_DEBUG=shaderstats` feedback.

Caveat: hot-bench rotates across multiple buffer sets to reduce cache-hot bias, but treat its GB/s as a kernel-comparison signal, not a final whole-model DRAM bandwidth number.

```bash
source .env

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --iterations 200 --warmup 25"
```

Single case + shader stats:

```bash
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd /root/zinc && \
  RADV_DEBUG=shaderstats zig build hot-bench -Doptimize=ReleaseFast -- \
    --model /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --case ssm_delta"
```

Available cases: `q8_router`, `q8_shared_gate_up`, `q8_shared_down`, `q8_ssm_out`, `ssm_delta`.

## Troubleshooting

If llama.cpp baseline drops below ~100 tok/s, check in order:

1. **Mesa version** — `dpkg -l mesa-vulkan-drivers` must show 25.0.7 (not 25.2.8).
2. **GECC** — `cat /sys/module/amdgpu/parameters/ras_enable` must show 0.
3. **coop_matrix** — server log must show `matrix cores: KHR_coopmat`.
4. **Reboot** — Mesa/driver changes need a reboot to take full effect.
5. **DPM stuck low** — long-running GPU processes can hold the R9700 in low-DPM. A reboot restores peak clocks. (Effort 10 baselines were corrupted by this for ~22 days.)
6. **Dirty benchmark node** — stop stray `zinc` / `llama-*` processes before comparing runs.
7. **Wrong endpoint for the question** — `/v1/chat/completions` for chat latency and queueing, `/v1/completions` for sustained HTTP decode throughput.
8. **Early chat stops** — if chat completions are ending after a handful of tokens, change the prompt or switch to `/v1/completions`.
