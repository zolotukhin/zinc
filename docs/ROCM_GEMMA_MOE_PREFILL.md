# ROCm Gemma MoE prefill correctness

The extended reference gate exposed a Gemma 4 26B-A4B mismatch in the ROCm
batched prefill path. All 48 short reference choices matched. After a 1,136-token
prefix, the copper continuation began with `Here` instead of `Three`. The
reference log-probability gap was about 2.274, beyond the unchanged 0.05
near-tie allowance. This result cannot qualify a release archive.

Controls retained the same packaged executable, weights, prompt and native
reference. Disabling only Q8 projection prefill or F16 attention did not fix
the continuation. Full serial prefill matched all 16 choices, including when
prompt-internal tokens skipped the unused output head.

ROCm Gemma MoE therefore uses serial prefill by default through the shared
CLI/single-slot HTTP entrypoint. It selects that path before allocating batched
scratch or executing a layer. Prompt positions remain absolute; only the final
prompt token computes the next-token logits. Dense Gemma, Muse, Qwen and CUDA
retain their existing routing.

`ZINC_ROCM_GEMMA_MOE_BATCHED_PREFILL=1` explicitly enables the unqualified batched
path and logs a warning. It is outside the release correctness claim. More
batched-stage isolation is needed before enabling it by default again; the
serial control does not identify one particular batched kernel as the cause.

CPU policy tests guard the model boundary. Qualification still requires the
three complete short continuations and three continuations beyond Gemma's
1,024-token window, through CLI and HTTP, followed by request recovery. No
reference tolerances, prompts or agreement limits were weakened.
