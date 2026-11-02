# Agent benchmark reference

Read [AGENTS.md](https://github.com/zolotukhin/zinc/blob/main/AGENTS.md) first. Its workload placement, process ownership,
and permission rules apply to every recipe here. The March 2026 baseline and
driver pins below are historical reproduction notes; verify the current stack
for a new campaign.

## Running Benchmarks

### Baseline: 107 tok/s (2026-03-26)

The reference baseline is llama.cpp server on the RDNA4 test node with this exact configuration. These March 2026 numbers describe that historical campaign. New campaigns must refresh the performance reference and match MTP settings as described in [BENCHMARKING.md](BENCHMARKING.md#latest-llamacpp-and-matched-mtp-pairs).

Headline RDNA comparisons must use the fair server-vs-server harness in
`tools/performance_suite.mjs`: one reusable ZINC server per model, one reusable
llama.cpp server per model, same GGUF, same scenario matrix, same warmup/run
count, and server-side prefill/decode timings. Do not compare a one-shot ZINC
CLI run against a warmed llama.cpp server when deciding whether a metric is
beaten. CLI runs are diagnostics only.

**Model**: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (20.7 GiB, MoE 35B/3B active)
**Baseline result**: 107 tok/s decode (with reasoning), 223 tok/s prefill

### Test node setup (critical for reproducing baseline)

```bash
# 1. Mesa must be 25.0.7 (25.2.8 causes ~14% RADV regression)
dpkg -l mesa-vulkan-drivers  # should show 25.0.7-0ubuntu0.24.04.2
# Pinned in /etc/apt/preferences.d/mesa-pin to prevent auto-upgrade

# 2. GECC disabled (amdgpu.ras_enable=0 in /etc/default/grub)
cat /sys/module/amdgpu/parameters/ras_enable  # should show 0

# 3. RADV_PERFTEST=coop_matrix set in llama-server.service
#    Without this, cooperative matrix is disabled → scalar fallback

# 4. llama.cpp build 3306dba, built with:
#    cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
#      -DCMAKE_CXX_FLAGS='-O3 -march=znver4' -DCMAKE_C_FLAGS='-O3 -march=znver4'

# 5. Server flags (in /etc/systemd/system/llama-server.service):
#    -ngl 99 --device Vulkan<N> --parallel 4 -c 32768
#    -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 --mlock --flash-attn on
#    Pick the discrete GPU index from `vulkaninfo --summary`; on mixed APU+dGPU
#    nodes this may be Vulkan1 rather than Vulkan0.
```

### Run comparable measurements

Use the repository harness for current server comparisons. It owns the
servers it starts and records the model, source, warmups, run counts, and
per-scenario timings:

```bash
source .env
bun tools/performance_suite.mjs \
  --target rdna --phase all --rdna-sync --rdna-build \
  --runs 3 --warmup 1 --no-site-write --output /tmp/zinc-rdna-suite.json
```

For HTTP latency diagnostics, run the following as one shell command **on the
remote node**, with `ZINC_WORKDIR` and `ZINC_MODEL` set to your checkout and GGUF.
Inspect existing GPU users first; stop only processes owned by your task.
Do not restart another workload or a long-running shared service.

```bash
set -eu
for name in zinc llama-server llama-cli llama-bench; do
  if pgrep -x "$name" >/dev/null; then
    printf 'A GPU process is already running; inspect it before continuing.\n'
    exit 1
  fi
done
cd "${ZINC_WORKDIR:?Set your remote checkout}"
./scripts/zig.sh build -Doptimize=ReleaseFast
server_pid=
cleanup() {
  if [ -n "$server_pid" ]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m "${ZINC_MODEL:?Set your remote GGUF path}" --port 9090 \
  > /tmp/zinc-api-benchmark.log 2>&1 &
server_pid=$!
for attempt in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:9090/health >/dev/null; then break; fi
  kill -0 "$server_pid"
  sleep 1
done
curl -fsS http://127.0.0.1:9090/health
bun tools/benchmark_api.mjs --base http://127.0.0.1:9090/v1 \
  --mode chat --output /tmp/zinc-api-chat.json
bun tools/benchmark_api.mjs --base http://127.0.0.1:9090/v1 \
  --mode raw --output /tmp/zinc-api-raw.json
```

Chat requests include templates and stop handling and often end after a few
tokens. Use raw completions for sustained decode diagnostics. Queueing and
batching depend on the backend; report latency and aggregate throughput
separately. A one-shot CLI run is a diagnostic, not a server comparison.

### Measure hot decode kernels directly

Use the dedicated microbenchmark when whole-model decode says “MoE”, “shared
expert”, or `ssm_delta_net` is hot and you need exact per-kernel numbers plus
`RADV_DEBUG=shaderstats` feedback.

Important caveat:

- the current hot-bench path rotates across multiple buffer sets to reduce the
  worst cache-hot bias
- still treat its reported GB/s as a kernel-comparison signal, not as the final
  whole-model DRAM bandwidth number

```bash
source .env

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd \"${ZINC_WORKDIR:?Set ZINC_WORKDIR}\" && \
  ./scripts/zig.sh build hot-bench -Doptimize=ReleaseFast -- \
    --model model.gguf \
    --iterations 200 --warmup 25"
```

Focused single-case runs:

```bash
source .env

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd \"${ZINC_WORKDIR:?Set ZINC_WORKDIR}\" && \
  ./scripts/zig.sh build hot-bench -Doptimize=ReleaseFast -- \
    --model model.gguf \
    --case q8_router"

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd \"${ZINC_WORKDIR:?Set ZINC_WORKDIR}\" && \
  ./scripts/zig.sh build hot-bench -Doptimize=ReleaseFast -- \
    --model model.gguf \
    --case q8_shared_gate_up"

ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST "\
  cd \"${ZINC_WORKDIR:?Set ZINC_WORKDIR}\" && \
  RADV_DEBUG=shaderstats ./scripts/zig.sh build hot-bench -Doptimize=ReleaseFast -- \
    --model model.gguf \
    --case ssm_delta"
```

Available cases right now:

- `q8_router`
- `q8_shared_gate_up`
- `q8_shared_down`
- `q8_ssm_out`
- `ssm_delta`

### Troubleshooting performance

If llama.cpp baseline drops below ~100 tok/s, check in order:

1. **Mesa version** — `dpkg -l mesa-vulkan-drivers` must show 25.0.7 (not 25.2.8)
2. **GECC** — `cat /sys/module/amdgpu/parameters/ras_enable` must show 0
3. **coop_matrix** — server log must show `matrix cores: KHR_coopmat`
4. **Reboot** — Mesa/driver changes need a reboot to take full effect
5. **Dirty benchmark node** — stop stray `zinc` / `llama-*` processes before comparing runs
6. **Wrong endpoint for the question** — use `/v1/chat/completions` for chat latency and queueing, `/v1/completions` for sustained HTTP decode throughput
7. **Early chat stops** — if chat completions are ending after a handful of tokens, change the prompt or switch to `/v1/completions`; otherwise the reported completion TPS is mostly prompt+HTTP overhead
