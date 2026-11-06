# ZINC_RT M1 Stall Analysis - 2026-05-15

Status: M1 to M2 migration, RDNA4 R9700 node.

## Baseline Observed

- Vulkan backend: ~115 tok/s decode on `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`.
- ZINC_RT scalar path: ~47-48 tok/s decode, coherent output.
- ZINC_RT evidence in the benchmark is currently limited to PM4/AMDGPU-CS token/model-value copy gates:
  - `execution_tier=t1_pm4`
  - `direct_token_boundary=amdgpu_cs_copy_data`
  - `direct_model_ops=1`
  - `consumed_gpu_model_value=1`

## Remote Evidence

Remote node:

```text
Linux <worker> 6.17.0-23-generic
/sys/module/amdgpu/parameters/user_queue = -1
T2 UMQ forced run: compute_userq_unavailable / compute_userq_slots_missing
```

Delayed `perf record` over the decode window showed the scalar path is dominated by `forward_zinc_rt.matvecRawDirectSerial`, with samples landing primarily in the re-quantized `Q4_0` dot loop. The top sampled source lines were in `src/zinc_rt/isa/cpu_zig/dequant.zig::dotQ4_0RowUnchecked`.

## Attempts That Did Not Land

These were tested on the remote and reverted because they either regressed speed or broke coherence:

- Wider `Q4_0` dot unroll: no measurable gain; median stayed ~46-47 tok/s.
- Decode MoE top-k lowered from 2 to 1 after prefill: output became `Paris France, (11-)`, which fails the benchmark coherence heuristic.
- Skipping the shared MoE expert: output collapsed to whitespace/`2`; incoherent.
- Direct LM-head argmax without materializing logits: preserved tokens but regressed the median.
- Fusing each expert's gate/up scalar matvecs inside the worker: preserved tokens but did not improve decode.
- Disabling `Q8_0 -> Q4_0` LM-head re-quantization: preserved tokens but regressed to ~40 tok/s.
- Forcing `-Dcpu=znver4`: no material improvement over the default remote build.

## Conclusion

More scalar CPU micro-tuning is unlikely to close the 2.4x gap to Vulkan. The next useful M1/M2 step should lower a real executed model slice behind the direct runtime path, ideally the smallest DMMV row or a tiny row range with a benchmark-visible comparison against the scalar value. If T2 is revisited, it must first prove real `USERQ_CREATE` availability on this node; kernel version alone is insufficient here.
