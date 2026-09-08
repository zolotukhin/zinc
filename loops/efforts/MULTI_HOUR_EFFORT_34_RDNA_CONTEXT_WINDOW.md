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
| prefill, 20K context | 335 | 466 *(unverified)* |

**The prefill number for llama.cpp is not trustworthy yet.** One clean run gave
466 tok/s; a second run across four prompt lengths gave a consistent 165 tok/s,
but that one overlapped a profiling job of mine. Re-measure on an idle card with
a fresh single-request server before quoting it. Our own 335 is solid.

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
| `-c 226934` | 226,934 | ~55 | on |
| `ZINC_MTP=0 -c 262144` | 262,144 | ~34 | off |

The long-lived 9090 server on the node is pinned at 32k by a launcher invoked
over SSH from outside the node — nothing on the node carries that flag and the
process tree ends at sshd, so every respawn returns at 32k until that command
changes.

Switches: `ZINC_KV_F16=0` (f32 cache), `ZINC_MTP=0`, `ZINC_MTP_CAPTURE_ROWS`,
`ZINC_FA_QUERY_TILE=1`.

Tests at the end of this effort: Zig 617/618 (1 skip), bun 639 pass / 3 skip.
