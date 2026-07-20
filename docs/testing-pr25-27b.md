# Verifying PR #25 on Qwen 3.6 27B

PR #25 chunks long prompts through the layer-major batched prefill on
Apple Silicon/Metal. The 9B half is verified (see the PR description); the
27B half is only architecturally reasoned about — it needs to actually run
on a machine with enough disk *and unified memory* for the 27B model before
merging.

**Hardware requirement, found by trying this on a 16 GiB Mac:** disk space
alone is not enough. The Q4_K_M weights are ~16 GB by themselves, and Metal
caps usable GPU memory at 85% of `recommendedMaxWorkingSetSize` (see
`memoryPlanningBudget` in `forward_metal.zig`). On a 16 GiB machine that
budget comes out to ~10 GB — smaller than the weights alone — so the engine
fails fast with `ContextLengthDoesNotFit` before any prompt is even
processed, regardless of `-c`. This isn't a PR #25 bug; the same failure
happens on an unpatched `main` build. You need a Mac with meaningfully more
unified memory than the model's own weight size — 32 GB is the practical
floor, 64 GB+ gives comfortable context headroom.

This is the same verification loop that caught a real bug on the 9B side
(a KV-cache byte/block offset mismatch that produced fast, wrong output),
adapted for the 27B's numbers. `PR25` and `main` below mean: build the
`metal/qwen9b-chunked-prefill` branch, and build `main` (or the commit
right before PR #25) as the baseline to diff against.

## 1. Build two binaries

```bash
git fetch origin
git worktree add /tmp/zinc-baseline main
git worktree add /tmp/zinc-pr25 metal/qwen9b-chunked-prefill

cd /tmp/zinc-baseline && zig build -Doptimize=ReleaseFast
cd /tmp/zinc-pr25     && zig build -Doptimize=ReleaseFast
```

Keep both `zig-out/bin/zinc` binaries around; you'll run both.

## 2. Get the model

```bash
/tmp/zinc-pr25/zig-out/bin/zinc model pull qwen36-27b-q4k-m
```

~16.8 GB download, cached under `~/.cache/zig/zinc` /
`~/Library/Caches/zinc` (both binaries share the same cache, so pull once).

Before running anything, sanity-check the machine can actually load the
model at all:

```bash
/tmp/zinc-pr25/zig-out/bin/zinc --model-id qwen36-27b-q4k-m --prompt "Hello" -n 5
```

If this fails with `ContextLengthDoesNotFit` / `No decode context fits
within N GiB Metal planning budget`, the machine doesn't have enough
unified memory for this model — see the hardware requirement note above.
No further steps here will work until that's resolved; this is unrelated
to PR #25.

## 3. Prompt lengths to test

The 27B's own single-shot batched-prefill ceiling is **40 tokens** (not
256, like the 9B) — a separately-validated number this PR didn't touch,
just chunks around. Test lengths that specifically exercise the new
chunking logic:

| length | why |
|---:|---|
| 45 | one chunk (40) + a 5-token remainder — below the 27B's own 32-token minimum, must fall back to per-token decode for that remainder |
| 80 | exactly two full 40-token chunks, no remainder |
| 90 | two full chunks + a 10-token remainder (same fallback case as 45, at a later position) |
| 200 | five chunks, stresses repeated position-carry across many boundaries |

Generate a prompt of a given token count with something like:

```bash
python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 20)"   # ~180 words ≈ tune to hit the target token count
```

Token count isn't exact from word count — check the actual count in the
`Prefill: N tokens in ...` log line and adjust the repeat count until you
land close to the target (within a few tokens is fine; the point is
crossing 40/80 boundaries, not hitting them exactly).

## 4. Compare outputs (must be byte-identical)

For each length:

```bash
PROMPT="..."   # from step 3

/tmp/zinc-baseline/zig-out/bin/zinc --model-id qwen36-27b-q4k-m \
  --prompt "$PROMPT" -n 32 > /tmp/base_out.txt 2>&1

/tmp/zinc-pr25/zig-out/bin/zinc --model-id qwen36-27b-q4k-m \
  --prompt "$PROMPT" -n 32 > /tmp/pr25_out.txt 2>&1

diff <(grep Output /tmp/base_out.txt) <(grep Output /tmp/pr25_out.txt) \
  && echo "IDENTICAL" || echo "DIFFER — stop here, this is a real bug"
```

If any length differs, do **not** merge — that's exactly the shape of bug
PR #25 already found once on the 9B.

## 5. Cross-chunk recall (proves chunk N attends chunk 1's KV, not just that tokens decode)

```bash
PROMPT=$(python3 -c "
filler = 'The museum catalog lists many unremarkable exhibits from the early colonial period, including pottery shards, farming tools, and faded textiles. ' * 20
print('Important: the secret code is 7391. Remember it. ' + filler + ' Question: what is the secret code mentioned at the very beginning? Answer with just the number:')
")

/tmp/zinc-baseline/zig-out/bin/zinc --model-id qwen36-27b-q4k-m --prompt "$PROMPT" -n 16
/tmp/zinc-pr25/zig-out/bin/zinc     --model-id qwen36-27b-q4k-m --prompt "$PROMPT" -n 16
```

Both should answer `7391`, and the two outputs should match. Adjust the
`* 20` multiplier so the prompt lands comfortably over 80–120 tokens (at
least two chunk boundaries between the planted fact and the question).

## 6. Confirm the fast path actually engaged (not a silent fallback)

```bash
/tmp/zinc-pr25/zig-out/bin/zinc --model-id qwen36-27b-q4k-m \
  --prompt "$PROMPT" -n 8 --profile 2>&1 | grep 'layer-major prefill:'
```

Expect `reason complete` and `layers <n_layers>/<n_layers>` (all layers
materialized, no `replay_layer_tokens`). One naming quirk: this log line
always says `qwen35-9b layer-major prefill:` even when the loaded model is
the 27B — that's a pre-existing label from when this code path was 9B-only,
not a sign you're on the wrong model. Ignore the "9b" in the text; check
`layers`/`reason` instead.

## 7. Throughput (optional but useful)

```bash
/tmp/zinc-baseline/zig-out/bin/zinc --model-id qwen36-27b-q4k-m --prompt "$PROMPT" -n 4 2>&1 | grep Prefill
/tmp/zinc-pr25/zig-out/bin/zinc     --model-id qwen36-27b-q4k-m --prompt "$PROMPT" -n 4 2>&1 | grep Prefill
```

Expect a real speedup at 80+ tokens (baseline falls to per-token replay
past 40 tokens today; PR #25 keeps it batched). Not required for
correctness, but a large regression here would also be worth reporting.

## Reporting back

Paste the diff results from step 4, the two recall answers from step 5,
and the `--profile` line from step 6 into the PR thread. If everything
matches, that's what unblocks merging the 27B half.
