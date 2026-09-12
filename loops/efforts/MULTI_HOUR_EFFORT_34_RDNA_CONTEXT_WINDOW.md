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

