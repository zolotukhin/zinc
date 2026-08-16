---
title: "How ZINC got a CUDA backend and beat llama.cpp decode on NVIDIA"
seoTitle: "ZINC CUDA Backend: beating llama.cpp decode on RTX 4090/5090"
date: "2026-06-07"
tags:
  - zinc
  - nvidia
  - cuda
  - rtx-5090
  - rtx-4090
  - blackwell
  - wsl2
  - local-llm
  - llm-inference
  - gpu-kernels
  - dmmv
  - nvrtc
  - cuda-streams
  - qwen3
  - gemma
keywords:
  - ZINC CUDA backend
  - ZINC NVIDIA support
  - local LLM RTX 5090 decode
  - CUDA decode beats llama.cpp
  - NVRTC runtime kernel compile
  - dmmv_q4k_fast CUDA
  - CUDA stream event decode ring
  - WSL2 CUDA only no Vulkan
  - Blackwell sm_120 inference
  - dp4a int8 matvec
  - GPU boost clock sync bound decode
  - 5 of 5 catalog coherent CUDA
faqs:
  - question: "Why did ZINC need a new CUDA backend instead of reusing Vulkan?"
    answer: "On the WSL2 box the only Vulkan device is llvmpipe, a CPU software rasterizer. CUDA is the only path that actually reaches the RTX 4090 and 5090. ZINC was Vulkan/Metal-only, so a native src/cuda backend was the only way its kernels could touch the hardware at all."
  - question: "How fast is ZINC decode on NVIDIA versus llama.cpp?"
    answer: "On Qwen3.5-9B Q4_K_M on an RTX 4090, ZINC decode reached about 104 tok/s after the async CUstream/CUevent ring landed, ahead of llama.cpp's 97 tok/s on the same model file. The correctness-first build was about 18 to 27 tok/s before the sync overhead was removed."
  - question: "What was the single most expensive bug?"
    answer: "Qwen3.5 packs its attention gate into the attn_q tensor interleaved per head, as [Q, gate] blocks, not as one contiguous [Q-all, gate-all] split. A contiguous split fed the wrong rows in as the gate, sigmoid saturated to about 1, attention was never gated down, and the residual stream blew up about 3.3x at the first attention layer. Per-layer residual diffing against a llama.cpp eval-callback isolated it."
  - question: "Why did fast kernels not immediately make decode fast?"
    answer: "The matvecs hit 72 to 90 percent of the 5090's 1792 GB/s bandwidth, but decode stayed bimodal at roughly 66 or 22 tok/s. The per-token sync gaps starved the GPU clock: the SM ran at a 525 MHz median during sync-bound decode versus 2520 MHz during sustained prefill. The async ring fixed the throughput and the clock starvation at once."
  - question: "Which models run coherently on the NVIDIA backend?"
    answer: "All five rows of that historical catalog: Qwen3.5-9B, a 27B dense Qwen checkpoint, Qwen3.6-35B-A3B (MoE), Gemma-4-31B (dense), and Gemma-4-26B-A4B (MoE). The qwen family came first; the gemma family needed the baked (1+weight) RMSNorm convention, per-layer mixed attention geometry, K-as-V global layers, and a shared-plus-routed MoE FFN."
excerpt: "ZINC is a Vulkan and Metal inference engine. On the WSL2 NVIDIA box the only Vulkan device is llvmpipe on the CPU, so ZINC could not touch the RTX 4090 or 5090 at all. This is how we wrote a CUDA backend by mirroring the Metal shim, found a per-head attention-gate bug by diffing residuals against llama.cpp, discovered that per-token sync gaps were starving the GPU boost clock, and shipped an async stream/event ring that pushed Qwen3.5-9B decode past llama.cpp."
seoDescription: "How ZINC added a native CUDA backend on WSL2 RTX 4090/5090: mirroring the Metal shim, NVRTC kernels, a per-head attention-gate bug, a sync-bound boost-clock discovery, and an async stream ring that beat llama.cpp decode."
---

The short version: ZINC is a Vulkan and Metal engine, and on the NVIDIA box it could not run on the GPU at all. Not slowly. Not at half speed. The kernels had no device to dispatch to.

The reason was mundane and total. The box is an RTX 5090 plus an RTX 4090 under WSL2, and WSL2 exposes exactly one Vulkan device: `llvmpipe`, a software rasterizer that runs on the CPU. There is no NVIDIA Vulkan ICD in the passthrough and none installable without sudo. So ZINC's entire GPU story — 110 compute shaders, a tuned DMMV path, a whole Metal mirror — pointed at hardware it could not see.

CUDA was the only door. This is the story of writing a `src/cuda` backend that walked through it, and what it cost to get from "no device" to **104 tok/s on Qwen3.5-9B, ahead of llama.cpp's 97** on the same RTX 4090.

## The measured result

Same box throughout: RTX 4090 for the decode and coherence numbers, RTX 5090 (sm_120, 32 GB, 1792 GB/s) for the kernel and prefill research, `ReleaseFast`, NVRTC-compiled kernels, the catalog GGUF files.

| Qwen3.5-9B Q4_K_M decode, RTX 4090 | tok/s |
| --- | ---: |
| correctness-first build (sync-bound) | ~18–27 |
| `*_fast` matvecs wired, boost-starved | ~66 / ~22 (bimodal) |
| async stream/event ring | **~98–104** |
| llama.cpp CUDA, same model file | 97 |

<img class="diagram-visual" src="/blog/2026-06-07-cuda-decode-journey.svg" alt="Horizontal bar chart of Qwen3.5-9B decode throughput on an RTX 4090 across four ZINC builds and llama.cpp. The correctness-first build sits near 22 tok/s, the fast-matvec build is bimodal around 66 and 22, the async ring reaches about 104 tok/s, and llama.cpp is at 97 tok/s, so the async ring crosses the llama.cpp line." loading="lazy" />

The interesting part is not the final number. It is that each plateau had a different cause, and the wrong fix for each one looked obvious.

## Mirroring Metal instead of porting Vulkan

ZINC's backends are OS-keyed: Linux picks Vulkan, macOS picks Metal. The first real change was a build-option selector so `cuda` and `vulkan` could coexist on Linux.

The second decision was which backend to copy. Vulkan was the obvious source — same OS, same SPIR-V-shaped shaders. Metal was the better source. Metal's seam is cleaner: a thin `shim.h`/`shim.m` ABI, an async `commitAsync` / `wait` / `releaseCompleted` command model, explicit buffer and pipeline objects. That maps onto CUDA almost one-to-one:

- Metal command queue and `commitAsync` → CUDA **streams plus events**.
- Vulkan `dotPacked4x8AccSatEXT` (int8 dot product) → CUDA **`__dp4a`**.
- Vulkan subgroup ops → warp intrinsics (`__shfl_sync`, `__ballot_sync`, `__reduce_*_sync`).
- wave64 reductions → warp32, reusing the existing wave32 fallback path as the template.
- A library of `.comp` shaders → one `kernels.cu`, **compiled at runtime by NVRTC** for the exact `sm_120` / `sm_89` target.

Crucially, ZINC's GEMMs never used cooperative matrix. They use the packed int8 dot. That meant the first correct token did not need tensor cores at all — `__dp4a` and a faithful Metal mirror got a token-for-token match against llama.cpp on the first dense forward.

The whole seam — `cuda_shim.h`/`cuda_shim.c` over the CUDA Driver API and NVRTC, plus Zig wrappers for device, buffer, pipeline, and command — was proven with a standalone smoke test before a single model weight was loaded: device selection by highest compute capability (so it always picks the 5090 for research), staged host/device copies, an NVRTC compile for `sm_120`, a push-constant dispatch, sync and async commits, and `__dp4a` returning 70.

## The bug that cost the most: a gate hiding inside attn_q

The first multi-layer forward produced finite numbers and confident garbage. The residual norm was sane through layers 0, 1, 2, then tripled at layer 3 — the first attention layer.

Qwen3.5 packs its attention gate **into** the `attn_q` tensor, which therefore outputs `2 * q_dim`. The trap is the layout: it is interleaved **per head**, as `[Q(head_dim) | gate(head_dim)]` blocks with stride `2 * head_dim`, not as one contiguous `[Q_all | gate_all]` split. A contiguous split feeds the wrong rows in as the gate. `sigmoid(≈large) ≈ 1`, so the gate never attenuates anything, and the post-attention residual blows up ~3.3x — exactly at L3.

The fix was a `deinterleave_qgate` kernel that splits the packed projection into contiguous `q_buf` and `gate_buf`. The fix that *found* it was the technique used for everything afterward: a debug build that dumps per-layer residual norms, diffed against a **llama.cpp eval-callback** with `n_gpu_layers=0` so tensor data is host-readable. L0–L2 matched; L3 diverged; the bug had nowhere to hide. SSM, FFN, embedding, and the tail were correct from the start.

## The plateau that was not a kernel problem

With the forward correct, decode landed at a correctness-first **18–27 tok/s**. The naive matvecs were the obvious suspect: they re-read each Q4_K block header 256 times per block and hit only **12–15%** of the 5090's 1792 GB/s.

So the matvecs got rewritten. The `*_fast` family loads each superblock header once, uses coalesced `float4`/`uint32` loads, and ports the tuned Vulkan layout. An overnight block-size autotune found that the optimal CUDA block is **M-dependent** — `block=64`, not the prototype's 256, wins for the large-M matvecs that dominate decode.

<img class="diagram-visual" src="/blog/2026-06-07-cuda-matvec-peak.svg" alt="Bar chart of CUDA matvec bandwidth as a percentage of the RTX 5090's 1792 GB/s peak. The naive dmmv sits at about 13 percent, dmmv_q4k_fast at 72 percent, dmmv_q6k_fast at 85 percent, and the autotuned block-64 Q6_K LM head at 90 percent." loading="lazy" />

| 5090 matvec | % of 1792 GB/s peak |
| --- | ---: |
| naive `dmmv_q4k` | ~13% |
| `dmmv_q4k_fast` | 72% |
| `dmmv_q6k_fast` (the real LM head) | 85% |
| autotuned `block=64`, Q6_K LM head | **90%** (1613 GB/s) |

The matvecs were now bandwidth-bound. Decode got faster — and got *weird*. It was bimodal, snapping between roughly **66 and 22 tok/s** on identical runs. That is not a kernel signature. That is a scheduling signature.

## The real bottleneck was the clock

The profiler told the truth: the 32-layer stack was 91% of per-token time, with **~65 `commitAndWait` calls per token at ~0.85 ms each**. Decode was not compute-bound. It was **sync-bound** — the CPU blocked on the GPU 65 times per token, and in the gaps the GPU did the sensible thing and dropped its clock.

The clock evidence is the whole story in one chart. During sync-bound decode the SM clock median was **525 MHz**. During sustained prefill — the same GPU, same model — it was **2520 MHz**. The sync gaps were starving the boost. The fast matvecs were running at quarter clock.

<img class="diagram-visual" src="/blog/2026-06-07-cuda-boost-clock.svg" alt="A two-panel diagram. The left panel contrasts SM clock under sync-bound decode at 525 MHz against sustained prefill at 2520 MHz, nearly five times higher. The right panel shows the per-token command pattern collapsing from about 65 blocking commitAndWait calls to a single async submit when the stream/event ring lands." loading="lazy" />

So the async stream/event ring was never just a throughput optimization. It helps **twice**: it removes the ~64 per-token syncs (≈45 ms of pure blocking), and it keeps the GPU loaded so it *holds boost*. Dense layer ops now `commitAsync` onto the context's single auto-ordered `CUstream`; the CPU never blocks per op; the tail's lone `commitAndWait` drains the stream and frees the events. One submit per token instead of sixty-five.

The bimodal 22–66 became a steady **98–104 tok/s**, past llama.cpp's 97, with correctness unchanged — a single ordered stream serializes the GPU identically, it just stops the CPU from blocking. The bottleneck moved *off* the GPU entirely: what is left per token is CPU glue — embed dequant and the argmax readback.

The async path is gated on `n_experts == 0` so the MoE models keep the proven sync path, and a one-command regression harness (`validate_cuda_decode.sh`, which embeds and builds the llama.cpp references, runs argmax + multi-prompt generation + logit fidelity, and exits non-zero on any drift) guards every change.

## Prefill is a different machine

Decode is bandwidth and latency. Prefill is arithmetic. A naive "dequant once, batch the token columns" kernel is **A-traffic-bound** — each activation is re-read once per output row, ~33 GB of activations at T=512 — so column batching alone fails. Prefill wants a real 2D-tiled GEMM: a block computes a `BM×BT` output tile, stages `A[BT,BK]` and dequantized `W[BM,BK]` into shared memory, and reuses each.

The register-blocked v2 GEMM — a 64×64 output tile, 256 threads each computing a 4×4 register micro-tile — gets there:

| RTX 5090 prefill GEMM, T=512 | speedup vs matvec×T |
| --- | ---: |
| `gemm_q4k_tiled` (v1, 2D shared tiles) | 2.1–2.4x |
| `gemm_q4k_tiled_v2` (register-blocked) | **5.9x** (9254 GFLOP/s) |
| `gemm_q6k_tiled_v2` | 6.1x |
| `gemm_q5k_tiled_v2` | 5.6x |

That is ~9000 GFLOP/s in fp32 — and still only ~9% of the card's fp32 peak, because at this size the GEMM is occupancy- and latency-bound, not compute-bound.

There is one more lever, and it is honest about its ceiling. Dequantizing the weights to an fp16 scratch buffer **once** and then running a pure fp16 `wmma` tensor-core GEMM measures **2.04–2.24x over the fp32 v2** (17978 vs 8812 GFLOP/s). It is not the 3–6x a tensor-core headline promises, because even the fp16 path stays at ~4% of the fp16 peak — reaching the true peak needs a cuBLAS-class async-pipelined kernel. And it is gated on one build-flag decision: NVRTC needs `-I/usr/local/cuda/include` for `mma.h` and `cuda_fp16.h`, so the fp16/tensor-core kernels cannot live in the shared `kernels.cu` until that flag is added. That is the next real prefill win, and it is a one-line change away from being unblocked.

## Five of five, and the model family that keeps ZINC honest

The foundation made the qwen family token-correct: Qwen3.5-9B and the 27B dense Qwen checkpoint both answer "Paris," and Qwen3.6-35B-A3B — a hybrid MoE-plus-SSM model — runs coherently through its delta-net SSM layers, sigmoid-gated shared expert, and top-k routing.

Gemma was the family that, as ever, [keeps ZINC honest](/blog/2026-06-02-gemma-is-the-model-family-that-keeps-zinc-honest/). The gemma-4 forward needed a pile of things qwen never punished: the RMSNorm `(1 + weight)` offset is **baked into the GGUF weights**, so the norm reuses the standard `×weight` kernel rather than re-adding the one — and the small post-attention norm weights make that the difference between coherent and garbage. It needed per-layer mixed attention geometry (local sliding-window layers at head_dim 256, global layers at 512 every sixth layer, sharing K as V), proportional RoPE on the global layers, scaled embeddings, and a shared-dense-plus-128-routed-expert MoE FFN for the 26B-A4B.

That closed the set: **all five catalog models — three qwen, two gemma — generate coherent text on the NVIDIA backend**, validated by a catalog-wide harness.

## What this backend is now

ZINC went from *cannot see the NVIDIA GPU* to a CUDA backend that is decode-competitive with llama.cpp, has a tuned bandwidth-bound matvec set, a register-blocked prefill GEMM family, and a quantified tensor-core ceiling waiting on one build flag. The recurring lesson is the boring one: every plateau looked like a kernel problem and was usually a scheduling problem. The fast matvecs mattered, but the move that actually beat llama.cpp was removing sixty-four syncs so the card would run at full clock.

The remaining levers are known and sized: wire the NVRTC `-I` flag for ~2.2x tensor-core prefill, move the per-token CPU glue (embed dequant, argmax readback) onto the GPU, and bring the async ring to the MoE path. None of them need new hardware. They need the same thing the rest of this did — a profiler, a llama.cpp reference, and the discipline to diff against it.
