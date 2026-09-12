# Effort 34 — RDNA4 context window: 22K → 262K on one 32 GB card

Qwen 3.8 27B Q4_K_M, AMD R9700 (32,624 MiB, RADV GFX1201), Vulkan backend.
Companion page: https://claude.ai/code/artifact/839cab94-f3eb-4f8d-8fe7-fd063ff7ca96

## Result

| | before | after |
|---|---:|---:|
| default context | 22,016 | 168,960 |
| ceiling | 30,851 | 262,144 (`ZINC_MTP=0`) / 226,934 (MTP on) |
| KV cache per token | 520 KB | 68 KB |

Commits: `28f417f0` (layers + f16 + capture cap), `30c0f9b4` (full window),
`16940d24` (query-tiled attention, opt-in), `d64be7a9` (harness DPM fix).

## The arithmetic

The card holds 32,624 MiB and the weights are ~16 GiB, so context is whatever is
left. The cache was costing 4x what the model needs and storing it at double
width:

- **Only 17 of 65 layers attend.** Full attention runs every 4th layer (16 of
  64) plus the appended NextN block. The Vulkan engine allocated a cache for all
  65; the 48 DeltaNet layers keep recurrent state and never read it. The Metal
  backend already allocated correctly (`isFullAttentionLayer`) but the *shared*
  planner counted every layer, so Metal was under-sizing its context too.
  `memory_plan.kvLayerCount()` now teaches both. → 3.82x
- **f32 → f16.** `compute/kv_dtype.zig` decides once at init; `shaderName()`
  swaps in `_f16kv` shader variants so no dispatch site changed. → 2x
- **The NextN block's 276 MiB of weights** are not uploaded when MTP is off
  (`loader.isAppendedNextnTensor`). That is the last 266 MiB between 257,884 and
  the full 262,144. → +4,260 tokens

The f16 conversion is mechanical because element indices are in units of 4
elements, never bytes: an `f16vec4` holds the same four elements a `vec4` did, so
every base/offset computation carries over verbatim. Readers wrap loads in
`vec4(...)`, writers wrap stores in `float16_t(...)`.

## Two bugs that cost the most time

**Every cache writer must be converted or it corrupts silently.** There are five
(`kv_cache_write`, `kv_cache_write_batched`, `qk_norm_rope_kv_write{,_batched}`,
`k_norm_rope_kv_write_batched`) and two readers (`flash_attn{,_batched}`). The
single-token `kv_cache_write` was missed. It serves the MoE decode path *and
every NextN draft step*, so Qwen3.6 35B A3B produced wrong tokens outright while
the 27B only lost 9 points of draft acceptance (54% → 45%) — which reads exactly
like "f16 is imprecise" and is not. With all five converted, output is
bit-identical to f32 on both models.

**Both VRAM accountants walked the GGUF tensor list, not the loaded tensors.**
`forward.tensorBytes()` and `model_manager.tensorBytes()` summed
`model.gguf_file.tensors.items`, so the 276 MiB the loader had just skipped
stayed subtracted from the KV budget and the context did not move at all. They
walk `model.tensors` now.

## Speculative decoding at depth

The prime capture buffer was capped at 2,048 rows, so any longer prompt silently
fell back to ordinary decode. The cliff is sharp: **78.2 tok/s at a 1,568-token
prompt, 34.3 at 2,467.** Cap is 32,768 now (`ZINC_MTP_CAPTURE_ROWS`), which costs
671 MB of reserve — about 5K tokens of context — and is what makes speculation
reach the depths the bigger window opens up.

At 262,144 speculation cannot fit: the NextN block is a 17th attending layer and
its cache is a full GiB at that length. That is the whole reason `ZINC_MTP=0` is
required for the full window.

## Against llama.cpp (same card, same GGUF, same driver)

| | ZINC | llama.cpp |
|---|---:|---:|
| max context | 262,144 | 262,144 |
| cache per token | 68 KB | ~64 KB |
| decode, short prompt | **55.0** | 31.0 |
| decode, 20K context | **44.2** | 28.9 |
| decode, 262K context | **33.9** | 30.5 |
| prefill, 20K context | 362.7 | 459.3 |

Both prefill numbers are clean (idle card, single request, 2026-09-10). The
sections at the end of this log carry the decomposition.

## Open: prefill

A 20K prefill costs 59.6 s — dense FFN 23.2 s, DeltaNet 19.0 s, attention 16.9 s.
Converted to arithmetic throughput:

- dense FFN gate/up (Q4_K): **32.5 TFLOP/s** — healthy
- dense FFN down (Q6_K): 25.1 TFLOP/s
- DeltaNet QKV (Q6_K): **12.2 TFLOP/s** — the outlier, and note the dense down is
  also Q6_K, so this is that specific path, not the quant format
- DeltaNet out-proj: 13.2 TFLOP/s
- attention: **4.7 TFLOP/s**

Attention is structurally wrong for prefill: `flash_attn_batched` dispatches one
workgroup per (head, query) and each walks its single query's keys alone, so a
loaded K vector produces exactly one dot product and the 6 query heads sharing a
KV head each re-read the same keys. `flash_attn_batched_qt` (committed, opt-in
via `ZINC_FA_QUERY_TILE=1`) gives one workgroup 4 consecutive queries so each
loaded K/V row feeds 4 dot products.

**It has no clean A/B yet** — the node was serving a model during every attempt,
so a 48-token prompt read 158 vs 161 tok/s, which measures the contention. Also
note its output is *not* bit-identical to the untiled kernel and is not supposed
to be: per-query arithmetic is the same but summed serially rather than through
the untiled kernel's 16-way tree. Judge it on needle retrieval and speed.

Rough budget to overtake 466 tok/s: attention at 15 TFLOP/s saves ~11 s and the
Q6_K projections at Q4_K efficiency save ~7 s, which is 59.6 → ~42 s = 480 tok/s.

## Measurement gotchas (both cost a wrong conclusion this session)

- **The perf suite forced `power_dpm_force_performance_level=high`,** which locks
  a nominal DPM state and gives up boost: 4.1–4.4% slower for ZINC and 3.3–3.5%
  for llama.cpp, on every historical RDNA number. Fixed in `d64be7a9`; the pin
  came from the 9070 XT, a different card whose SCLK idled at 0 MHz.
- **The RDNA target benchmarks the server over the OpenAI API, not the CLI,** so
  its prompts carry ~19 chat-template tokens, which lowers draft acceptance. A
  published row reads ~1 tok/s under the same text through `--chat --prompt`.

## Running it

| launch | context | decode | speculation |
|---|---:|---:|---|
| no `-c` | 168,960 | ~55 | on |
| `-c 226934` | 225,729 | ~55 | on |
| `ZINC_MTP=0 -c 262144` | 261,022 | ~34 | off |
| `ZINC_PREFILL_SCRATCH_MB=0 ZINC_MTP=0 -c 262144` | 262,144 | ~34 | off |

The two trimmed rows are the honest numbers as of 2026-09-11: the context plan
now reserves prefill scratch, and at the ceiling it shrinks that reserve to its
80 MB floor (256-token chunks) before trimming. The nominal 262,144 only ever fit
because scratch was unreserved and spilled to system memory during long
prefills; `ZINC_PREFILL_SCRATCH_MB=0` restores exactly that behaviour.

The long-lived 9090 server on the node is pinned at 32k by a launcher invoked
over SSH from outside the node — nothing on the node carries that flag and the
process tree ends at sshd, so every respawn returns at 32k until that command
changes.

Switches: `ZINC_KV_F16=0` (f32 cache), `ZINC_MTP=0`, `ZINC_MTP_CAPTURE_ROWS`,
`ZINC_FA_QUERY_TILE=1`.

Tests at the end of this effort: Zig 617/618 (1 skip), bun 639 pass / 3 skip.

## Prefill, measured 2026-09-10 (idle card, 20,043-token prompt)

The card was finally free, so the prefill work has real numbers. Both of my
predictions were wrong in instructive ways.

| | 20K prefill |
|---|---:|
| unchunked (previous behaviour) | 336.7 tok/s |
| chunked, 1 GB scratch budget | 352.6 |
| chunked, 2 GB | 356.1 |
| chunked + query-tiled attention | 255.6 |
| **llama.cpp, clean single request** | **459.3** |

**llama.cpp's prefill is 459 tok/s** — 19,934 tokens in 43.4 s on an idle card.
The earlier 165 tok/s reading was contention from my own profiling job; the 466
reading was genuine. Quote 459.

**Chunking is worth ~5%, not the fix I claimed.** The scratch spill was real
(6.1 GiB wanted, 5.1 GiB free) but it was never the dominant cost. Re-profiling
with chunking on shows why: attention is now ~43% of prefill (was 28%), dense FFN
~35%, DeltaNet ~21%. Chunking cut the linear work; attention is quadratic and
untouched by it, so its share grew.

**The query-tiled attention kernel is a 28% regression** (356 -> 256) and stays
off. The reuse argument was right and the implementation was wrong: it spends
~21 KB of LDS per workgroup against the original's ~4 KB, which collapses
occupancy, and it has 4 accumulator chains where the original unrolls 16. Memory
reuse bought at the price of latency hiding is a losing trade on this part.

**What a correct attention kernel needs:** keep the 16-way ILP and small LDS
while getting reuse. The promising shape is grouping the `q_per_kv` = 6 query
heads that share a KV head (one workgroup = 1 query position x 6 heads): the same
K load feeds 6 dot products, Q costs 6 KB of LDS instead of 16, and the output
accumulators stay in registers (6 vec4 = 24 VGPRs per lane) rather than LDS.
Budget: attention is ~24 s of the 57 s; llama.cpp is 13.6 s ahead, so attention
needs roughly a 2.2x speedup to close the gap on its own.

**Test hygiene note:** the needle assertion in the first A/B reported MISS on
every run because it only generated 8 tokens — not enough for the model to
answer. It proved nothing. Generate >= 40 before trusting it.

## Where the prefill gap actually is (2026-09-10, measured)

Three attention rewrites, all measured on an idle card against a 20,043-token
prompt:

| attempt | hypothesis | result |
|---|---|---:|
| query tiling (4 queries/workgroup) | bandwidth, via query reuse | −28% |
| head grouping, 64 threads | bandwidth, via 6:1 GQA reuse | **+2.8%** |
| head grouping, 384 threads | wave occupancy | −7% |

Cutting K/V traffic sixfold moved the total 2.8%; raising waves-in-flight
sixfold made it worse. Only the middle one is committed (`1d3975e9`).

**Do not trust the prefill phase profiler for attribution.** With grouping on it
reports attention *slower* (1.87 ms/token vs 1.66) while the pass is *faster*
(55.4 s vs 56.4), and its per-phase figures sum to 3.83 ms/token against an
actual 2.81. The "attention is 43% of prefill" figure it produced is what sent me
into three kernels; the real share is 22%.

**The honest decomposition** comes from the length curve, since attention is the
only quadratic term. Chunking and grouping on:

| prompt | ms/token |
|---:|---:|
| 4,099 | 2.282 |
| 8,086 | 2.377 |
| 16,040 | 2.613 |
| 20,043 | 2.777 |

Least squares on T/n = a + b·n gives a = 2.141 ms/token and b = 3.08e-5
ms/token², predicting 55.3 s against 55.7 measured. At 20K that is:

- **linear (projections, FFN, DeltaNet): 42.9 s, 78%**
- **quadratic (attention): 12.4 s, 22%**

llama.cpp does the whole 20K pass in 43.4 s = 2.177 ms/token. **Our linear term
alone is 2.141 ms/token** — essentially their entire pass. So the per-token work
is at parity and the whole 12.3 s gap is attention: 79 TFLOP in 12.4 s = 6.4
TFLOP/s, against the ~32 TFLOP/s the dense FFN reaches on the same silicon.

**What that means for the next attempt.** Attention needs ~5x, not 20%, so
variants of the current shape cannot get there — the three above moved it by at
most 13% of the phase. The shape itself is the problem: one workgroup per query
position walking its whole causal range. A FlashAttention-2 style kernel tiles a
block of queries against a block of keys, keeps the accumulators in registers
across the key loop, and stages K/V tiles in LDS once per query block. That is a
focused piece of work with a clear target (6.4 -> ~30 TFLOP/s), and it is the
only remaining item between us and llama.cpp on prefill.

## Both sides decomposed (2026-09-11)

The same length-curve fit on llama.cpp, same prompts, idle card:

| at 20K | ZINC | llama.cpp |
|---|---:|---:|
| linear (per-token) | 2.141 ms/tok → 42.8 s | 1.956 ms/tok → 39.1 s |
| quadratic (attention) | 3.08e-5 → 12.3 s | 0.99e-5 → 4.0 s |

So their attention is 3.1x faster, not free, and the 12 s gap is **8.3 s of
attention plus 3.7 s of linear work**. Matching their attention alone lands at
46.8 s — still 3.7 s behind. Beating them needs attention *and* ~9% off the
linear term.

The linear side: the DeltaNet QKV projection already runs the DP4a Q8_1 path in
prefill (`qwenDenseSsmProjDp4aEnabled`, n_tokens >= 32), so its 12.2 TFLOP/s
against the dense down's 25.1 on the same Q6_K kernel is efficiency at K=5120
versus K=17408 — a per-tile overhead that a 3.4x shorter K loop cannot amortize —
not a missing switch.

**The reference scalar attention kernel's shape for head size 256 on AMD**
(derived from its tuning selection): workgroup 256 = 4 independent wave64s,
each wave owning Br/row_split = 4 query rows against Bc = 32 key columns per
step, head dim split 8 ways across lanes (each lane holds a 32-dim slice), no
LDS staging of K/V, the 4x4 score tile and the 4x8-vec4 output tile in
registers, and — the part that matters — **each 8-lane column group keeps its
own online softmax** so the inner loop has no cross-lane reduction; the 8
partials merge once at the end through shuffles. Per K vec4 loaded: 4 FMAs
(rows); per Q vec4 read: 4 FMAs (columns). `flash_attn_batched_tile.comp` is
that shape on our paged f16 cache with causal masking done arithmetically and
GQA heads packed as rows (r = q·gqa + h) so heads sharing a KV head share a wave.

**Result of that shape** (`flash_attn_batched_tile`, 20,043-token prompt, idle
card, both runs retrieve the needle):

| | 20K prefill |
|---|---:|
| head-grouped kernel (previous best) | 360.2 tok/s |
| **register-tiled kernel** | **397.5** |
| llama.cpp | 459.3 |

55.6 s → 50.4 s: attention from ~12.3 s to ~7.1 s (1.7x). Gap to llama.cpp is
1.16x, from 1.29x. First attention change of four that moved the needle, and it
is the one that changed the algorithm's shape rather than its memory traffic or
occupancy.

**Four waves sharing Q + branch-free loop** (`271f71de`): 397.5 → **414.6 tok/s**
at 20K (48.3 s). Gap to llama.cpp 1.11x. Both taken from the reference's
configuration: a 256-thread workgroup stages one 16-row Q tile behind a single
barrier so its four waves stream the same K/V through L0/L1, and invalid columns
load from a clamped address and are masked to -inf instead of branched around.

**Linear term, tried and rejected:** `dmmv.zig` selects the BM64/BN64 Q6_K Q8_1
kernel for K=5120 only when `N == 64` (the K=17408 gate accepts any 64-aligned
N), so prefill chunks take the BM32 ragged kernel. Widening the gate and padding
long batches to 64 columns measured **410.3 vs 414.6** at 20K and 441 vs 450 at
1.5K — the BM64 tile is not faster at this M/K, so the QKV projection's 12.2
TFLOP/s is not a tile-selection accident. Reverted. The remaining ~3.7 s of
linear gap is diffuse GEMM efficiency (effort 15 territory), not a switch.

## Standing at the end of this effort (2026-09-11)

| 20K prefill | tok/s | pass |
|---|---:|---:|
| start of the prefill work (unchunked, original attention) | 335 | 59.6 s |
| + scratch chunking | 352.6 | 56.8 s |
| + head-grouped attention | 362.7 | 55.3 s |
| + register-tiled attention | 397.5 | 50.4 s |
| **+ 4 waves sharing Q, branch-free** | **414.6** | **48.3 s** |
| llama.cpp | 459.3 | 43.4 s |

Decode still leads at every length (55.0 / 44.2 / 33.9 vs 31.0 / 28.9 / 30.5),
context at parity (262,144). Prefill gap 1.29x → 1.11x; the rest is ~3 s of
attention (theirs is 4.0 s, ours ~5 s now) and ~3.7 s of linear work.

Next levers, in order: (1) attention — 8 rows per wave or a 64-column step to
raise register-tile reuse further, and f16 packed math for the score products,
which the reference uses when accumulation stays f32; (2) linear — the DeltaNet
QKV (Q6_K, K=5120) and out-proj (Q5_K, K=6144) GEMMs at ~12-13 TFLOP/s against
the dense FFN's 25-32, a kernel-efficiency problem at short K rather than a gate.

## 262K usability pass (2026-09-11)

What "usable at 262K" needs, checked one by one on the card:

1. **Fits.** With scratch reserved the honest ceiling is 261,022 (MTP off) or
   225,729 (MTP on); see the table above. Scratch auto-reduces to its floor
   before the context is trimmed (`Prefill scratch budget reduced from 1024 MB
   to 80 MB so the requested 262144-token context fits`).
2. **Turn-to-turn reuse — fixed and verified.** Two problems found with the
   session cache, measured over the API, both now closed (`routes.zig`):
   - **Thinking off (the tool-calling / bot path): the cache never hit.** The
     engine's transcript keeps the empty `<think>` scaffold the model actually
     saw in its prompt, while the next request re-renders that assistant turn
     without it, so `startsWith` diverged at the scaffold and every turn
     re-prefilled the whole conversation. With thinking on it is worse: the
     re-render drops or *re-tokenizes* the reasoning block, so no re-render can
     ever match. For a recurrent model a partial prefix is useless (the DeltaNet
     state exists only at the end), so the exact engine transcript is the only
     thing continuable. The fix stops comparing re-renders: the cache stores
     the **engine sequence** (prompt + generated + the template's end-of-turn
     tail, recovered from the canonical render by `transcriptTailAfter`), and
     the next request is matched by `alignEngineTranscript`, which lets the
     engine hold tokens the re-render dropped (up to 8,192 at a spot, each skip
     confirmed by 4 matching tokens) and then **splices** the request's new
     tail onto the stored transcript. The model continues exactly the
     conversation it generated against.
   - **Speculation was lost across a stored turn.** The end-of-turn tail is
     prefilled after the reply without the draft block, so the next turn found
     the NextN cache one step short. `warmChatReuseCache` now extends it
     (`mtpResetCapture` + `mtpPrimeSuffix`) and primes after a rebuild.

   Verified 2026-09-11, two turns of one session over the API:

   | | turn 2 log | decode |
   |---|---|---:|
   | thinking off | `spliced: engine_only_tokens=4` → `hit: reused=66 appended=27` → `primed 27 appended rows over a 66-token reused prefix in 2.9 ms` | 77 tok/s |
   | thinking on | `hit: reused=102 appended=25` → `primed 25 appended rows … in 2.5 ms` | 69 tok/s |

   Both answered the needle. `ZINC_MTP_REUSE` now defaults on. The chat handler
   reports canonical prompt token counts to the client; only the engine sees
   the spliced sequence.
3. **Speculation at 262K** cannot fit at f16: 17 attending layers x 262,144 x
   64 KiB = 17,408 MiB of cache against 16,303 MiB of weights on a 32,624 MiB
   card. **q8_0 KV** (32-element blocks, f16 scale) is 9,248 MiB, which leaves
   ~7 GiB for the draft block's cache, its 671 MB capture buffer and a full
   1 GB scratch budget — and halves attention's KV traffic at long context.
   Sites: 5 writers (`kv_cache_write{,_batched,_single}`, `qk_norm_rope_kv_write
   {,_batched}`, `k_norm_rope_kv_write_batched`) and the readers (`flash_attn`,
   `flash_attn_batched*`). The tiled prefill kernel is a natural fit: each lane
   already owns a 32-dim slice = one q8 block. Writers quantize per 32-block
   with a subgroup-clustered max over the 32 lanes that hold the block (the
   rope writer's lane→element mapping keeps a block inside one wave). This is
   the next lever; it is what makes 262K + speculation possible.
4. **A 226K-token conversation over the API, measured** (`ZINC_MTP=0 -c
   262144`, thinking off, `session_id` set):

   | | wall | |
   |---|---:|---|
   | turn 1: full prefill of 225,584 tokens | 1,387 s (23.1 min) | 162.7 tok/s, 256-token chunks |
   | **turn 2: reused prefix + 29 new tokens** | **8 s** | `spliced engine_only_tokens=4` → `hit reused=225594 appended=29`; needle at 60% depth answered |

   The second turn of a conversation that fills most of the window costs
   seconds instead of another 23 minutes. Prefill at this depth is slower than
   the 20K figure predicts (~18.7 min): 881 chunks of 256 re-stream the weights
   (~25 s) and the DP4a GEMMs lose efficiency at 256 columns.

   **Decode at a genuinely full window is 8.3 tok/s (120 ms/token), not the
   33.9 quoted above for "262K context".** That earlier number, and llama.cpp's
   30.5 beside it, were taken with the window allocated but nearly empty, so
   attention had nothing to read. With 225K tokens resident each decoded token
   reads the whole 14.8 GB cache across the 17 attending layers; at 576 GB/s
   that plus the weights bounds decode near ~19 tok/s, and the split-K decode
   kernel reaches ~164 GB/s effective. Two levers, both real: the decode
   attention kernel at depth, and q8_0 KV, which halves that traffic and is
   also what lets speculation fit at 262K. llama.cpp at the same resident depth
   is unmeasured (it would cost another ~20-minute prefill on the card).

   Gotcha: `/root/mk_needle.py` counts ~18 tokens per line but the real rate is
   3.44 bytes/token, so its "250K" file was ~332K tokens and the server's
   capacity check rejected it with a 400 (correctly). Size prompts by bytes.

## q8_0 KV cache (2026-09-12, implementation)

`ZINC_KV_Q8=1` stores the cache as q8_0 blocks: 32 int8 + one f16 scale, padded
to 36 bytes (9 uints) per 32 elements. Rows are block-aligned because head_dim
and kv_dim are multiples of 32, so the vec4 index i every shader already uses
lives in block i>>3 at uint i&7, and the block's scale is uint 8. No new device
features: readers unpack int8 quads from plain uints
(`unpackSnorm4x8(u) * 127 * scale`), writers quantize a block with
`subgroupClusteredMax` over the 32 lanes that hold it and pack four lanes per
uint with `subgroupShuffleDown`. Every store loop runs in uniform control flow
with a lane predicate, since the quantization is a subgroup operation; `active`
is a GLSL reserved word, hence `lane_on`.

- `compute/kv_dtype.zig`: `Layout {f32, f16, q8}`, `bytesPer32Elems()` (128 /
  64 / 36), `rowBytes()`, `shaderName()` → `_q8kv`. The memory plan sizes the
  cache per 32-element block (`profileWithKvBlockBytes`).
- readers: `flash_attn_q8kv` (decode), `flash_attn_batched_tile_q8kv` (prefill;
  under q8 every batched attention routes to the tiled kernel, so head_dim
  must be 256). Generated by `scratchpad/mk_q8kv.py`.
- writers (hand-written): `qk_norm_rope_kv_write{,_batched}_q8kv`,
  `kv_cache_write{,_batched,_single}_q8kv`. `k_norm_rope_kv_write_batched`
  (Gemma) has no q8 variant.
- per token: 17 layers × 2 × 1024 elements × 36/32 B = 39,168 B (f16: 65,536).
  At 262,144 tokens: 9.6 GiB against 16 GiB.

## llama.cpp at full depth (2026-09-12, measured)

Same 224,022-token prompt, `llama-server -c 262144 -np 1`, idle card:

| 224K tokens resident | ZINC (f16, MTP off) | llama.cpp |
|---|---:|---:|
| prefill | 1,387 s · 162.7 tok/s | **1,095 s · 204.5 tok/s** |
| decode at depth | 8.3 tok/s | **17.75 tok/s** |
| second turn (35 new tokens) | 8 s | **3 s** (its prompt cache reused the prefix) |

They lead on all three at full depth; decode by 2.1x. Their turn-1 reply was an
empty think block (thinking on by default in their template), so correctness at
depth is measured separately by the multi-needle eval with thinking disabled on
both engines (`chat_template_kwargs.enable_thinking=false` for llama.cpp).

**q8 validation (2026-09-12).** First run fell to a generic prefill path
(233 tok/s, capture count 2× the prompt, speculation skipped) because the
accelerated dense-hybrid prefill gates tested for the untiled batched kernel,
which q8 does not load; `batchedAttentionAvailable()` now accepts the tiled
kernel. With that:

| needle prompt | f16 | q8 | answer |
|---|---:|---:|---|
| 1,568 tokens: prefill / decode | 444.6 / 78.0 tok/s | 443.7 / 76.0 | same |
| 20,043 tokens: prefill / decode | 412.6 / 44.1 | 369.2 / 39.8 | same, primed |

~10% instruction cost at 20K on both sides (two loads per vec4 in the
generated readers); the byte saving only pays at depth, which the 226K stage
measures. The decode reader's 16-wide K loop is hand-tuned afterwards to load
each block's scale once (`flash_attn_q8kv.comp`, "HAND-TUNED SIBLING").

**The GPU reset near the ceiling, explained.** The first q8 run at the full
window with speculation on (which q8 makes fit: `requested 262144, reserved
262144`, cache 9,792 MB, scratch at its full 1 GB so chunks of 3,276) died
mid-prefill with `radv/amdgpu: the context is lost … guilty of a hard recovery`
and dmesg `ring comp_1.1.0 timeout` — the compute-ring watchdog, not a memory
fault (the prime-capture bounds are checked). Each prefill chunk's attention
layer is one submission, and its work grows with the depth it attends over; a
3,276-token chunk at 200K depth exceeded the driver's limit, while the f16 run
at the same depth with 256-token chunks (scratch floor) ran for 23 minutes
without incident. Same mechanism as the very first reset, a 26.7K prompt
submitted whole before chunking existed. Chunks are now bounded by
query × key pairs as well as by scratch (`ZINC_PREFILL_ATTN_PAIRS`, default
3e8: 1,500 tokens at 200K depth, scratch-limited when shallow).

## Decode at depth: the grouped kernel (2026-09-12)

flash_attn dispatches one workgroup per query head, so with 6:1 GQA every K/V
row is streamed six times per decoded token (~88 GB at 226K resident across the
16 attending layers). That is why halving the cache's bytes (q8) did nothing for
decode at depth: the traffic is the redundancy, not the format.
`flash_attn_gqa` (+ f16kv/q8kv variants, split-K compatible, same partial/LSE
layout so the merge pass is unchanged) gives one workgroup a KV head and the six
query heads that read it. Host gate: head_dim 256, n_heads == 6·n_kv_heads,
`ZINC_FA_GQA_DECODE` (default on when applicable).

The first version used `acc[HEADS]`-style arrays indexed in loops bounded by a
specialization constant and one accumulation chain per head — and measured
**18.9 tok/s at 20K with speculation off against 29.0 for the per-head kernel**
(q8: 18.0). At 20K depth attention is a fifth of the per-token cost, so the
test can only show overhead, and the overhead was a third of the token. The
rewrite is hand-unrolled for six heads, keeps scores as vec4 groups of four
keys, takes four keys per V step into two chains per head, and stays at ~13 KB
of LDS. Measured next, together with per-chunk speculation priming
(`mtp_prime_during_prefill`: the draft block is primed chunk by chunk during a
chunked prefill, carrying the last normalized row across chunks exactly as a
reused prefix does, so speculation no longer stops at the 32,768-row capture
cap) and the text-level cache fallback (`spliceByText`: when a reply's generated
tokens do not re-tokenize to the canonical render, compare the transcript as
text minus closed think blocks and tokenize only the new tail).

**The grouped decode kernel, rewritten:** 26.0 tok/s at 20K with speculation
off against 28.4 per-head (v1: 18.9). Still behind, and the reason is now
visible: at 20K attention is a fifth of the token and neither kernel is
traffic-bound, so grouping cannot win there — and at 226K the split-K dispatch
is 24 heads × 4 chunks = 96 workgroups each walking 56K keys serially, a
fraction of the card, which grouping shrinks to 16. The occupancy, not the
bytes, is what bounds decode at depth. `n_chunks` is now a push constant
(flash_attn, flash_attn_gqa, flash_attn_split_merge; the fused o-proj merge
keeps the base count and is skipped otherwise) and `splitKChunksForSeq` scales
it with the resident context — about 2,048 keys per chunk, up to 64 chunks
(`ZINC_FA_SPLIT_K_KEYS`, `ZINC_FA_SPLIT_K_MAX`; the partial buffer is sized for
64). Grouped decode stays opt-in (`ZINC_FA_GQA_DECODE=1`) until the depth A/B.

## 226K with speculation and the text fallback (2026-09-12, measured)

q8, `-c 262144`, per-chunk priming, per-head decode with the base 4 chunks:

| | wall | log |
|---|---:|---|
| turn 1, 225,584 tokens | 2,113 s | every 3,276-row chunk primed (~280 ms each), no reset |
| turn 2, reused | 9 s | `spliced engine_only_tokens=4` → hit → `primed 29 appended rows … in 2.8 ms` → drafts 5/6 accepted |
| turn 3, reused | **23 s** (was 2,091 s) | `spliced by text: tail=24` → hit → 6/8 accepted |

Session reuse now holds at full depth on both turns, and speculation primes at
any depth. But decode with speculation at depth is **5.0 tok/s against 7.7
without**: `target=1508.8 ms over 3 cycles` — each verify step (a 3-token
batch through the tiled prefill kernel, which has no split-K) is ~500 ms,
because 3 queries × 6 heads = 18 rows fill 8 workgroups that walk all 226K keys
serially. The same occupancy disease as decode, in the other kernel. Next:
route small batches at depth through the split-K decode kernel per query
(q/o offsets in its push constants), so a verify costs ~3 decode attentions.

## Prefill at depth: the int8-dot kernel for the q8 cache (2026-09-12)

At 226K the tiled prefill kernel sustains ~11 TFLOP/s of f32 FMA (10 PFLOP of
attention in ~900 s at f16; the generated q8 reader ~1,600 s) against llama.cpp's
~17. A q8 cache holds int8 keys already, so `flash_attn_batched_tile_q8mmq`
computes Q·K^T on `dotPacked4x8` — four multiply-adds per instruction: a lane
owns one 32-element block of the head dimension instead of one vec4 of every
block, Q is quantized once per (row, block) into int8 in LDS with the attention
scale folded in, and a (row, column) partial is 8 int8-dot instructions times
the two block scales, against 32 f32 FMAs. The 8 dim-lanes' partials sum by the
same butterfly. PV stays f32 for now (V dequantized per block). Opt-in
(`ZINC_FA_Q8_MMQ=1`) until the needle and speed checks land; the same trick is
the plan for the q8 decode kernel once the split-K chunk scaling has been
measured.
The decode side gets the same treatment in `flash_attn_q8mmq` (split-K, per
head): Q quantized once per block into 64 uints of LDS, then 64 int8-dot
instructions per key instead of 256 f32 FMAs. Selected with the same
`ZINC_FA_Q8_MMQ=1` for the split-K decode dispatch and for small batches at
depth. Both are measured behind the depth A/Bs.

## Full-context eval, first run (2026-09-12): 1/5 against 5/5, and why

Five facts planted at 5/25/50/75/95% depth in a ~197K-token document, asked one
per turn with thinking off on both engines (`/root/fullctx_eval.py`):

| | ZINC (q8, spec on, build c8b39c4a) | llama.cpp |
|---|---:|---:|
| correct | **1 / 5** | **5 / 5** |
| prefill 197K | 1,666 s | 893 s (221 tok/s) |
| per question | 12 s, or a 1,670 s re-prefill | 1 s (28 new tokens) |
| decode at depth | 2.3–7.2 tok/s | 18.5 tok/s |

Every question answered from a **cache hit** came back as `!!!!…` — token 0,
the signature of NaN logits — and every question that fell back to a full
re-prefill was answered correctly. The same hit-then-answer sequence was
correct in the 226K three-turn run on the previous build, so the delta was the
depth-scaled split-K: `flash_attn_split_merge` sized its LDS arrays for 16
chunks ("a generous ceiling") while the split dispatch now issues up to 64 at
depth, so lanes 16–63 wrote past them. Fixed (`MAX_CHUNKS = 64`, matching
`flash_attn_max_split_chunks`); the ZINC half of the eval reruns after the
depth A/Bs. llama.cpp's numbers stand: correct at every depth, 1-second turns
through its prompt cache, 18.5 tok/s with 197K resident.

## Decode at depth: occupancy confirmed (2026-09-12)

100K resident tokens, f16, speculation off, 8-token generations after a reused
prefix, merge fixed:

| decode at 100K depth | tok/s |
|---|---:|
| per-head kernel, 4 split-K chunks (old default) | 14.6 |
| per-head kernel, depth-scaled chunks (48) | **24.1** |
| grouped kernel, depth-scaled chunks | 23.9 |

Same kernel, same bytes: 1.65x from workgroups in flight alone. The grouped
kernel adds nothing once occupancy is fixed, so it stays opt-in. (These runs'
needle check missed on all three variants alike — the "vault access code"
phrasing draws a refusal with thinking off; the depth prompts now use the
neutral "project codename" fact, and correctness at depth is judged by the
five-fact eval rerun.)

**Speculation at depth, with the verify batch on the decode kernel** (100K
resident, q8, depth-scaled chunks, 48-token generations after a reused prefix):

| | per verify cycle | tok/s |
|---|---:|---:|
| verify through the tiled prefill kernel | 188 ms | slower than plain decode |
| verify through the split-K decode kernel, per query | **96 ms** | **26.3** |
| same, grouped decode kernel | 94 ms | 26.6 |

Plain decode at the same depth: 24.1 (scaled chunks) / 14.6 (the old 4 chunks).

**Staged block scales: null.** q8 prefill at 20K 370.5 tok/s against 369.2 with
the generated reader — the scale reloads were not the q8 penalty; the int8-dot
kernel (stage 6) is the test that matters for q8 prefill. The same 20K run
shows the decode changes paying at moderate depth too: **57.2 tok/s with
speculation on the q8 cache** (39.8 before scaled chunks and the verify-batch
routing; f16 was 44.1), needle found.

**100K prefill, q8 vs f16** (server, speculation off): q8 231.5 tok/s (430 s),
f16 287.0 tok/s (347 s) — the staged-scale reader changed nothing at depth
either, and its q8 run missed the needle where f16 found it (q8 on the
generated reader had answered correctly at 226K). No gain and a correctness
question: reverted to the generated reader. The int8-dot kernel is the q8
prefill lever.

**int8-dot kernels, measured at 20K (q8 cache, needle found at 1.5K and 20K):**

| q8, 20K | prefill | decode (spec on) |
|---|---:|---:|
| generated f32-dot reader | 356.6 tok/s | 56.9 |
| **int8-dot (`ZINC_FA_Q8_MMQ`)** | **397.9** | **63.9** |
| f16 cache, for reference | 412.6 | — |

Attention is about a fifth of a 20K pass, so an 11.6% whole-pass gain means
the attention itself roughly doubled. Now the default for a q8 cache; the 100K
prefill and the depth decode A/B follow.

**int8-dot prefill at 100K** (server, speculation off): 288.3 tok/s (345.5 s)
against 231.5 for the f32-dot q8 reader and 287.0 for the f16 cache — q8
prefill at depth is at f16 parity, so the cache that lets speculation fit at
262K no longer costs prefill. PV stays f32: with int8 V the f16 conversion
costs what packed f16 FMAs save, so the next prefill lever at depth is the
chunk cap (`ZINC_PREFILL_ATTN_PAIRS`, 1,500-token chunks at 200K bite the
GEMMs), to be raised cautiously now that the attention job is faster.

**int8-dot decode at 100K** (q8, speculation on, verify on the decode kernel,
64-token generations): f32-dot 29.6 tok/s (verify cycle 96 ms) → **int8-dot
39.1 tok/s** (74 ms). Decode at 100K resident went 14.6 → 39.1 over the day.
Open question: both stage-7 runs missed the neutral needle while the
speculation-off runs at 100K found it and the 226K speculation-on run (verify
through the tiled kernel) was correct — the verify routing at large chunk
counts is the suspect; the eval rerun and an isolation run (routing on/off,
replies printed) settle it.

## Full-context eval, rerun on the fixed build (2026-09-12)

q8 cache, speculation on, int8-dot kernels, depth-scaled split-K, verify on the
decode kernel (`5268d26b` + merge fix), same 197K document and questions:

| | ZINC | llama.cpp |
|---|---:|---:|
| correct | **3 / 5** | 5 / 5 |
| prefill 197,241 tokens | 1,016 s (194 tok/s) — was 1,666 | 893 s (221 tok/s) |
| per question | 6 s, six of six from cache hits | 1 s |
| decode at 197K resident | 22.4–29.5 tok/s | 18.5 |

The two misses are truncations, not wrong retrievals: `88-` for 88-Q-3105 and
`23.` for 23.5 degrees, each "Generated 3 tokens" — the model emitted
end-of-sequence right after a hyphen or a decimal point. The same shape appeared
in the first eval (`88-` after a full re-prefill on the old build) and in the
very first f16, speculation-off 226K run (`PLUM-4417-` for turn 1), so it
predates every kernel from today; at 20K the same prompts answer in full. The
100K runs split by speculation: off → full answers (f16 and int8-dot q8), on →
misses (stage 7, both decode kernels). Stage 9 prints the replies with
speculation off, on with the verify routing, and on with the tiled verify, at
100K, to separate speculation from numerics at depth. Candidates if it is
numerics: RoPE angles for the high-frequency dims computed as
`cos(position * freq)` in f32 (position 200K puts the argument near 2e5 rad,
where a GLSL range reduction keeps only a couple of digits of the fraction).

**Isolation (stage 9, 100K, q8 int8-dot):** speculation off, on with the verify
routing, on with the tiled verify — all three answer `PLUM-4417-` and stop.
Speculation is not the cause; the truncation is in plain greedy decode at
depth, and the eval shows it depends on the needle's depth (5–50% facts
answered in full, 75–95% truncated). Remaining candidates: q8 quantization
noise on far keys (int8 Q × int8 K, ~1% per score, enough to blunt a lone
needle's softmax peak among 100K keys) or the session-reuse path. Stage 10
runs the five-fact document at 100K on q8 and on f16 through a session, plus
the deep fact as a single request.

**Depth truncation: q8 and f16 are identical** (stage 10, 100K, speculation
off): both answer 5/25/50% in full and truncate 75% (`88-` for 88-Q-3105) and
95% (`23.`), and for both the 75% fact answered **correctly as a single request**
(`88-Q-3105`). So it is not q8, not the kernels (f16 does the same), not
speculation, not retrieval (the fact is there). It tracks the multi-turn
session, but the single-request control changed two things at once — it dropped
the reuse/splice path *and* the prior Q&A history. Stage 12 sends the identical
full transcript through the reuse path (session_id) and through a fresh
monolithic prefill (no session_id), and the monolithic one to llama.cpp: if
ZINC-monolithic answers and ZINC-spliced truncates it is the splice; if both
truncate it is how ZINC decodes that exact context, and the llama.cpp run on the
same tokens says whether that is an engine gap.

**20K control (stage 11): 5/5 on both q8+speculation and f16**, same five-fact
session flow, same q4/q5 facts that truncate at 100K (`88-Q-3105`, `23.5
degrees` both answered in full). So the machinery is sound and the facts are
answerable in-session; the truncation is purely a function of depth. Combined
with q8 == f16 at 100K, the suspect narrows to something identical across cache
formats that grows with depth — the split-K online-softmax merge (f32 partials
either way, ~48 chunks at 100K vs ~10 at 20K) is the leading candidate, and it
is code this session changed. Stage 12 (splice vs monolithic vs llama.cpp) and
then a chunk-count sweep decide it.

**Stage 12: the identical transcript decodes correctly everywhere.** doc + q1–q3
+ clean answers + q4, fed at 100K: ZINC monolithic, ZINC "spliced", and
llama.cpp monolithic all answer `88-Q-3105` (9 tokens). So ZINC decodes that
exact 100K context as well as llama.cpp — not an engine gap, not genuine model
behaviour on that context. Caveat: the "spliced" run used a fresh session_id, so
it had no resident prefix to splice against and was really a second monolithic
prefill. The truncation appears only in a *live* multi-turn session, where the
engine's stored sequence (the model's own generated answers + end-of-turn tails,
carried across turns) is what later turns continue — so the splice/reuse path is
building a context that differs from the canonical one. Stage 10's log hints at
it: `engine_only_tokens=20` and `=24` on the later turns, where the first turn
was `=4` (just the empty think scaffold). Stage 13 runs the live five-fact flow
at 100K with reuse **off** (full re-prefill each turn): 5/5 pins it on the splice
path; a miss means the model's own multi-turn history truncates it and llama.cpp
on the same live flow says whether that is shared.

## Root cause: the reuse path, not q8/kernels/speculation (2026-09-12)

Stage 13 — the live five-fact flow at 100K with reuse **off** (full re-prefill
each turn, clean history) — scores **5/5**, against 3/5 for the same flow with
reuse on. Everything else held equal. So the deep-needle truncation is the
session reuse path, and the scaffold-accumulation idea is too small to be it
(20 tokens in 100K). What a fresh prefill rebuilds and reuse carries, on this
hybrid model, is the **DeltaNet recurrent state** (48 of 65 layers): it is f32
regardless of KV format, which is exactly why q8 and f16 truncate identically,
and it is absent from single requests (stage 12) and reuse-off (stage 13). Next:
read how chat reuse carries or recomputes DeltaNet state across turns, since that
determines whether the fix is a recompute trigger or a drift correction.

### Standing scoreboard at depth (2026-09-12)

| 100K–226K resident | ZINC | llama.cpp |
|---|---:|---:|
| decode, speculation on | 26–39 tok/s | 18.5 |
| prefill (q8, int8-dot) | 288 tok/s | ~290 |
| turn-to-turn reuse | 6–9 s | 1 s |
| deep-needle eval | 3/5 (reuse) · 5/5 (fresh) | 5/5 |

Decode and prefill at depth are now at or ahead of llama.cpp; the one open gap is
reuse-path correctness on deep needles.

## The reuse-path truncation is a design tension, not a bug (2026-09-12)

Confirmed in code. With thinking off, `transportAssistantContent` prepends the
empty `<think>\n\n</think>\n\n` to every stored assistant turn on purpose, so
the reused token sequence matches the resident KV *and* DeltaNet recurrent state
— which the model actually decoded with those scaffolds present. That match is
required for reuse to be correct. A cache hit then sets `state.position` to the
reused length and prefills only the new tail (`routes.zig` ~2786), carrying the
recurrent state forward; a fresh request rebuilds it from zero.

The catch is that the canonical multi-turn render — what a fresh prefill and
llama.cpp feed — drops *past* turns' scaffolds and keeps only the current one.
So the reuse path replays a sequence that gains one empty think block per turn,
and the 48 DeltaNet (linear-attention) layers have no softmax to suppress them:
each spurious segment leaves a persistent bias in the recurrent state. At 20K
it is harmless (5/5); past ~50% of a 100K context it tips deep-needle answers
into an early end-of-sequence (3/5). It is identical on q8 and f16 because the
recurrent state is f32 either way.

No free fix, and the direction is a real tradeoff:
  - **Canonical reuse:** drop historical scaffolds so the reused sequence
    matches the standard render. Correct, but the resident recurrent state was
    built over the scaffolded sequence, so it needs a rebuild — which is the
    full prefill reuse exists to avoid, exactly at the depth where reuse is most
    valuable.
  - **Bounded divergence:** force a clean re-prefill once the accumulated
    scaffolds at deep context cross a threshold. Keeps reuse for the common case
    (few turns / short context), pays one full prefill occasionally.
  - **No per-turn scaffold:** stop injecting the empty scaffold at generation so
    nothing accumulates. Largest change; touches the tool-calling suppression
    path the scaffold was added for, needs that regression test.

Left for a deliberate follow-up rather than a rushed patch. The reuse path stays
on: it is correct at the depths a bot actually runs, and it is what makes 262K
usable turn to turn.

## Fix: recurrent-state checkpoints for chat reuse (2026-09-12)

The design tension resolves the way llama.cpp resolves it on hybrid models:
checkpoint the recurrent state at a turn boundary and roll back to it. Each
turn the handler prefills the prompt up to the end of the last
`<|im_start|>assistant\n` (canonical history), primes the draft block over it,
takes a checkpoint — every SSM layer's conv and recurrent state copied to a
device-local buffer (~150 MB), the conv ring offsets, the position, and the
draft block's carried row — stores that canonical prefix as the session entry,
then prefills the generation prompt (scaffold or thinking header) and primes it.
The next turn matches by plain prefix, restores the checkpoint, and prefills
the canonical suffix: previous answer, end of turn, new question, new header.
The reused context is therefore exactly a fresh render; nothing accumulates in
the DeltaNet state, thinking-on sessions drop past reasoning canonically, and
the token/text aligners are no longer needed on this path (kept for legacy
entries). Engine: `ssmCheckpointTake/Restore/Position/Invalidate`
(`forward.zig`, SSM buffers now `TRANSFER_SRC` too); server:
`storeCheckpointed`, `checkpointPosFor`, `lastAssistantHeaderEnd`, and the split
prefill in the chat handler. Verified next at 20K, 100K and 197K.

**Checkpointed reuse, measured** (q8, speculation on, five facts, reuse on):

| | 20K | 100K | 197K |
|---|---:|---:|---:|
| before (transcript replay) | 5/5 | 3/5 | 3/5 |
| **after (checkpoint + canonical suffix)** | **5/5** | **5/5** | **4/5** |
| per question | 1 s | 2 s | 3–4 s |

The deep facts that truncated (75%, 95%) now answer at every depth. The one
remaining miss moved to the shallow 5% fact at 197K (`PLUM-4417-`, the same
hyphen-then-stop shape), and every miss so far has been with speculation on
while every speculation-off run at depth has been 5/5 — so the last suspect is
the verify batch's numerics (batched DMMV columns and the 3-token SSM step
accumulate in a different order than single-token decode) flipping a near-tie
between continuing a code and ending the turn. Stage 15 runs the 197K flow with
speculation off and again with it on.

**Stage 15 → 16: a client retry broke checkpointed reuse (fixed, 7db40a1b).**
Stage 15's SSH session died mid-run, and re-driving the live server from a
fresh client re-sent turn 0. That retry exposed three bugs in the checkpoint
path, each invisible in the clean stage-14 flow:

1. The restore refused any prompt matched by *text* rather than exact tokens,
   even with the entry length, checkpoint position and engine position all
   equal. After one text splice every later question fell to a full 197K
   re-prefill: 17 minutes per question.
2. A turn whose canonical history ends exactly at the restored checkpoint (the
   retried turn) took no new checkpoint, so the transcript path replaced the
   entry and the session lost its checkpoint.
3. The stored prefix sliced the canonical prompt with the engine's indices;
   after a splice those differ, so the entry held the wrong tokens (and could
   run past the end of the shorter prompt).

The fix restores whenever the reused length equals the checkpoint position,
re-checkpoints a turn that ends at the checkpoint, and stores the tokens the
engine processed. Stage 16 reruns the 197K five-fact flow with the retry in
both speculation modes.

**Stage 16 result.** With the retry fixed (7 restores, 0 fallbacks, 1 full
prefill, 3–4 s per question), the 197K flow scores **4/5 in both speculation
modes with the identical miss**: q1 (5% depth) → `PLUM-4417-`. Speculation is
exonerated. The miss belongs to the reuse flow: the same document answered 5/5
as a single prompt. The 5% fact is also the *farthest* from the question, so it
is the one with the thinnest attention margin; the reuse flow's only numeric
difference from a single prompt is that the short appended suffix attends
through the split-K decode kernel (int8-dot for q8) instead of the prefill tile
kernel. Stage 17 asks the fact last (order 2,3,4,5,1,1) to separate "first
restore after the chunked prefill" from "the fact itself"; stage 18 A/Bs
f32-dot decode attention, suffix attention through the tile kernel, and the
f16 cache.

**Stage 17: the 5% fact asked last answers in full, twice — 6/6.** Order
2,3,4,5,1,1 at 197K with checkpointed reuse: every answer correct, q1 →
`PLUM-4417-ZEBRA` both times, 7 restores, 0 fallbacks. So the fact is
retrievable through the reuse flow; the miss is specific to it being the
*first* question after the turn-0 prefill (a single prompt answers it first).
Stage 18 asks 1,2,5 with one numeric path changed per run.

**Stage 18a: f32-dot attention does not change it.** `ZINC_FA_Q8_MMQ=0`, order
1,2,5: q1 still `PLUM-4417-`, q2/q5 correct. The int8 Q quantization is
exonerated; and the f32-dot q8 prefill took 1665 s against 1016 s with
int8-dot, so the int8 path is worth 64% on the 197K prefill. Stage 19 asks
order 1,1,2,1 to separate "first restore after the chunked prefill" from "no
prior answer in the context".

**Stage 18b: the suffix through the prefill tile kernel does not change it
either** (`ZINC_FA_SMALL_BATCH_DECODE=0`, order 1,2,5: q1 `PLUM-4417-`, q2/q5
correct). Attention kernels are exonerated. Stage 20 (after 19) disables the
checkpoint (`ZINC_CHAT_CHECKPOINT=0`, transcript reuse): if q1 then answers,
the checkpoint take/restore itself is numerically off.

**Stage 18c: the f16 cache does not change it either** (order 1,2,5: q1
`PLUM-4417-`, q2/q5 correct). Four numeric configurations (int8-dot, f32-dot,
suffix through the tile kernel, f16 cache) give the identical cut, so the miss
is not numeric noise; something discrete about "q1 first after the turn-0
prefill through reuse" decides it. Stage 19 (order 1,1,2,1), stage 20
(transcript reuse instead of the checkpoint) and stage 21 (top-2 logit
margins at each generated token) are queued.

**Stage 19: order 1,1,2,1 → MISS, MISS, OK, MISS.** Once the cut answer is in
the history the model repeats it, and one correct answer in between does not
undo it; stage 17's full answers came after four complete answers. So "asked
later" was in-context priming, not a first-restore effect. Every checkpoint-flow
configuration cuts q1 identically, while the old transcript-reuse flow (the
previous turn carrying its empty think scaffold) answered it in full — the
model's decision on this farthest fact sits on an edge that the rendering of
the previous turn moves. Stage 20b: a barrier-ordered checkpoint copy first
(the take had no barrier against the prefill's dispatches), then the top-2
margins at each generated token, then transcript reuse on the same build.

**Stage 20b: the barrier changes nothing, and the margins settle it.** With
`ZINC_LOG_TOPK=1` (CPU argmax, speculation off) the greedy tokens of the 5%
answer at 197K are `PL` `UM` `-` `4` `4` `1` `7` `-` with margins of 3–9
logits, and then:

    top=EOS logit=19.067   second='Z' logit=18.921   margin=0.146

The cut is a **0.146-logit near-tie** between ending the turn and continuing
the code; every other token in the run has a margin of 1–13. That is why four
numeric configurations agree (their differences are below 0.15 logits), why
four prior complete answers or the old scaffold-carrying rendering flip it,
and why llama.cpp's f16 pipeline lands on the other side. It is a model
decision on the farthest fact, not an engine defect. The checkpoint-copy
barrier (74ff61f7) stays as hygiene. The 197K first-question score is 4/5 on
this margin; with any prior complete answer in the context it is 5/5.

**Stage 20b, transcript reuse on the same build: q1 `PLUM-4417-ZEBRA`, 3/3.**
With `ZINC_CHAT_CHECKPOINT=0` the previous turn is replayed with its empty
think scaffold, and the 0.146-logit tie lands on the other side. The canonical
rendering (what the checkpoint path and llama.cpp both feed the model) is the
right one to keep; which side of a 0.15-logit tie an engine's numerics land on
is not a correctness property. Correctness thread closed: at 197K the
first-question score is 4/5 on that one tie and 5/5 otherwise; every deep fact
answers; every turn after the first is 3–4 s.

**Stage 23: the chunk budget is not the lever.** `ZINC_PREFILL_ATTN_PAIRS=5e8`
(chunks stay 3,276 tokens to full depth): 197K prefill 1009 s vs 1015 s, no
watchdog. Length-curve split of the two engines (20K and 197K points):

| | linear ms/token | attention ms/token per 1K depth |
|---|---:|---:|
| ZINC | 2.10 | 0.0155 |
| llama.cpp | 1.91 | 0.0133 |

At 197K attention is 59% of ZINC's time and its per-pair cost is 14% above
llama.cpp's; that is the remaining prefill gap. The per-question latency gap
(3–4 s vs 1 s at 197K) is the other open metric; the suffix prefill is
~0.75 s and generation ~0.4 s, so the rest is tokenizing and matching the
1.1 MB prompt each turn.
Stage 23b: 2 GB scratch (6,553-token chunks) + 5e8 pairs: 1007 s. Chunk size
is worth under 1%. Next lever is inside the tile kernel: per 32-key step each
lane spends 128 multiplies rescaling its output accumulators by the running-max
correction, which is exactly 1 once the max has settled — now skipped on a
wave-uniform `subgroupAny(max_changed)` (bf8f78d0, exact). The larger lever is
the PV side: dequantizing V (int8 → f32 with a scale) costs about as many
instructions as the 512 FMAs it feeds; packed-f16 PV with per-tile f32 flush
would cut that ~2.5× and is the candidate after the A/B.

**Stage 24: where a question turn's 4 s go at 197K.**

| step | time |
|---|---:|
| tokenize the 771 KB rendered prompt | 2.5–2.6 s |
| reuse matching (exact prefix + checkpoint restore) | < 0.1 s |
| suffix prefill (29–36 tokens at depth) | 0.6–0.7 s |
| generation (4–8 tokens) | 0.2–0.3 s |

Tokenization is two thirds of the turn. Fix: cache the previous request's
rendered text and tokens per session; a new prompt shares a byte prefix up to
the previous assistant header, and `<|im_start|>` is a hard BPE boundary, so
the cached tokens up to the k-th `<|im_start|>` plus an encode of the tail
equal a full encode exactly. Expected turn: ~1.3 s (llama.cpp: 1 s).

**Stage 25: the rescale skip changes nothing** — 87.6K-token prefill, q8:
289.2 → 289.0 tok/s. Removing an eighth of the per-step ALU work moves nothing,
so the tile kernel is not ALU-bound; a rough budget confirms it (the 197K
attention would take ~25 s of pure ALU and ~55 GB/s of VRAM, against ~600 s
observed). It is latency-bound: q8 blocks are 36 bytes, so each K or V block is
nine scalar dword loads, and 4 waves × ~90 VGPRs leave few waves per CU to
cover them. That is also why the PV instruction-count rewrite would not pay.
The lever there is memory-level parallelism (LDS-staged K/V with cooperative
vectorized loads and a software-pipelined next tile), which is the shape of
llama.cpp's scalar FA — a kernel-architecture change, deferred. The rescale
skip stays (exact, harmless).

**Stage 26: the tokenization cache verifies.** 197K, four turns after the
first, `ZINC_TOKENIZE_CACHE_VERIFY=1`: every cached tokenization equals the
full encode (197266, 197298, 197323, 197356 tokens). Stage 27 measures the turn
without the verification encode.

**Stage 27: a question turn at 197K is now 1 s** (was 4 s; llama.cpp 1 s).
Tokenization 2484 ms → 1 ms from the cache; what remains is the suffix prefill
(0.6–0.7 s) and generation (0.2–0.3 s). Per-question latency is at parity.

Scorecard after stage 27 (Qwen 3.8 27B Q4_K_M, R9700, ZINC vs llama.cpp):
decode ahead everywhere (55.8 vs 31.0 short; 22–29 vs 18.5 at 197K); prefill
ahead at short and medium prompts, parity at 100K, behind 7–12% at 197K
(attention kernel latency-bound; MTP priming 6% of it); per-question latency
1 s vs 1 s; five facts 5/5 vs 5/5 at 100K, 4/5 vs 5/5 at 197K on a 0.146-logit
tie. The 20K prefill (415 vs 459) splits by the length curve into linear
2.10 vs 1.91 ms/token and attention 0.31 vs 0.27 — the linear layers carry
most of that gap; next: a per-kernel profile of a 20K prefill.

**Stage 28/29: the 20K prefill, per sub-phase** (CLI, `ZINC_PREFILL_PROFILE=1`,
17,120 tokens, 404 tok/s = 2.47 ms/token; the profiler sums to ~109%):

| phase | ms/token | share | useful rate |
|---|---:|---:|---:|
| dense FFN gate/up (Q4_K + Q6_K layers) | 0.698 | 28% | ~33 TFLOPS |
| dense FFN down (Q4_K + Q6_K) | 0.506 | 20% | ~23 TFLOPS |
| **DeltaNet qkv projection (Q6_K, K=5120)** | **0.483** | **20%** | **~8 TFLOPS** |
| attention layers (projections + flash attention) | 0.640 | 26% | flash ≈ 0.26 |
| DeltaNet out projection (Q5_K) | 0.147 | 6% | ~14 TFLOPS |
| DeltaNet z projection | 0.134 | 5% | ~15 TFLOPS |
| DeltaNet scan + conv + norms | 0.075 | 3% | |

The recurrent scan is cheap (0.049 ms); the qkv projection is the outlier at a
third of the FFN GEMMs' efficiency. At FFN-down efficiency it would take
~0.17 ms: −0.31 ms/token = −12% of the 20K prefill (405 → ~460 tok/s, llama.cpp
459) and ~6% at 197K.

**Stage 30: 64-row tiles for the K=5120 projections: +3.7% at 17K** (404 →
419 tok/s). The BM64 DP4a variants for the DeltaNet qkv (Q6_K, Q8_1 input) and
z (Q4_K) projections were routed only when N was exactly 64; now any N that is
a multiple of 64 takes them, and prefill chunks are sized to a multiple of 64
so every full chunk qualifies (3,276 → 3,264 tokens). z: 0.134 → 0.092
ms/token; qkv: 0.483 → 0.431. Correct: 20K five-fact 5/5 (a truncated 226K
document ties its first fact at 0.090 logits on the previous binary too — not
the retile). qkv still runs at ~12 TFLOPS against the Q4_K z projection's 33
with the same K and tile: the Q6_K DP4a kernels read weights with byte loads
(`uint8_t a_data[]`, 36 loads per 16 elements); Q6_K GEMMs are 45% of the
prefill (qkv 0.43 + FFN-down 0.30 + gate/up 0.35 ms/token). Next: 16-bit weight
loads in the four Q6_K DP4a kernels.

**Stage 33: 16-bit weight loads in the Q6_K DP4a kernels: +4% at 17K** (419 →
436 tok/s; cumulative 404 → 436, +7.8%). FFN-down Q6_K 0.301 → 0.236 ms/token
(−22%); qkv only 0.431 → 0.416; gate/up Q6_K unchanged (0.36). Bit-identical
math (same integers), 20K five-fact 5/5 with identical answers. The qkv kernel
is therefore not load-instruction-bound either; its Q8_1 activation tile may be
read column-major (uncoalesced across the 64 columns) where the FFN kernels
read a block-interleaved layout.
100K prefill with the retile + 16-bit loads: 289 → 303.7 tok/s (+5%). The
GGUF says the 48 DeltaNet qkv projections are 24 × Q6_K and 24 × Q4_K (the
DP4a qkv path accepts only Q6_K/Q5_K), ffn_down is 33 × Q6_K + 32 × Q4_K,
gate/up all Q4_K, ssm_out Q5_K, attention q/k/o Q4_K, v 9 × Q6_K.

**Stage 35: the Q4_K half of the qkv projections through the DP4a GEMM: +10%.**
Splitting the qkv phase by weight type showed the 24 Q6_K layers at 0.117
ms/token (21 TFLOPS) and the 24 Q4_K layers at 0.302 (8 TFLOPS): the Q4_K
ones fell through to the batched DMMV chunks because the qkv DP4a path only
took Q6_K/Q5_K. Routing them through the Q4_K z-projection DP4a GEMM (generic
in M): 0.302 → 0.083 ms/token; 17K prefill 432 → **475 tok/s**. Cumulative
today: 404 → 475 (+17.5%). 20K five-fact 5/5.
The exact 20,031-token comparison prompt: **470.4 tok/s vs llama.cpp 459**
(was 414.6). ZINC now leads the 20K prefill. 100K/197K re-measure queued.

**Stage 36: depth re-measure with the three GEMM fixes** (q8 cache,
speculation priming on):

| prefill | before | now | llama.cpp |
|---|---:|---:|---:|
| 20,031 tokens | 414.6 tok/s | **470.4** | 459 |
| 87,592 tokens | 289 | **323.7** | ~290 |
| 197,239 tokens | 1015 s (194 tok/s) | **918 s (214.9)** | 893 s (221) |

ZINC now leads the prefill at 20K and 100K; at 197K the gap is 2.7% (was 12%),
and 6% of that prefill is the NextN priming that llama.cpp does not do.

**Stage 37: the attention layer, per sub-phase, at 17K** (ms/token): flash
attention 0.477, q/k/v/gate projections 0.081, o-projection 0.045, head norms
+ rope 0.002, KV write 0.000. The projections are already on DP4a; the flash
kernel is 20% of the whole prefill at 17K and the majority at depth. Its Of
accumulators are 4 rows × 8 vec4 = 128 VGPRs per lane before anything else,
which is the occupancy problem behind "latency-bound".

**Stage 38: RADV shader stats for the tile attention kernel**
(`RADV_DEBUG=shaderstats`): 252 VGPRs, 0 spills, 6 subgroups per SIMD, 3,798
instructions (3,234 VALU, 72 VMEM), LDS 4,608 B. Six waves per SIMD is the
latency wall, and each of a workgroup's 4 waves fetches the same 32-key K/V
tile from memory itself (4× redundant, 36-byte per-lane gathers). Next: stage
the tile through LDS once per workgroup with coalesced cooperative loads
(behind ZINC_FA_TILE_LDS for A/B; the per-lane math and accumulation order stay
identical, so outputs should be bit-identical).

