# Frozen repository acceptance checks

These are the unchanged task definitions and external functional checks from
the [2026-10-26 OpenCode repository trial](../../OPENCODE_E2E_REPORT.md).
The coding client could edit its candidate worktree but could not edit these
external checks, the verifier, configuration or context hook.

Run these from a disposable Zinc checkout with Bun 1.4.4 and Zig 0.17.0.
Set `ACCEPTANCE_DIR` to a separate copy of this directory and `CANDIDATE_DIR`
to the candidate checkout. Keep the checks outside the agent's editable tree.

```bash
bun "$ACCEPTANCE_DIR/strict-numbers.mjs" "$CANDIDATE_DIR"

./scripts/zig.sh test --dep discovery \
  -Mroot="$ACCEPTANCE_DIR/discovery-utf8.zig" \
  -Mdiscovery="$CANDIDATE_DIR/src/server/model_discovery.zig"

bun "$ACCEPTANCE_DIR/auto-context.mjs" "$CANDIDATE_DIR"
```

The first and third commands import or execute project tooling. Review a
candidate before running it. These checks do not start inference or load a
model. The auto-context check creates a temporary loopback discovery fixture;
optional `ZINC_E2E_LIVE_BASE` and `ZINC_E2E_LIVE_CONTEXT` add a request to an
already-running server. Do not put runtime endpoint values in tracked files.

The trial also ran the relevant existing Bun tests, the full CPU project
suite, and CUDA host compilation for the Zig change. Before inference, all
three external checks failed on the original source while existing CPU tests
passed. Each condition used a fresh worktree at the commit in `manifest.json`.

These checks were incomplete: the first two did not require the requested new
regression tests and documentation; model-selection edge cases were missing
from the third. A green check therefore did not establish complete delivery.
The report retains those original outcomes and records the later review
separately. Permanent reviewer-authored regressions live beside the production
tooling and in `src/server/model_discovery.zig`; they do not retroactively
change the trial's acceptance criteria.

The portable result JSON publishes hashes of these exact files. Host-specific
launchers, raw traces, session state, endpoints and machine paths remain private.
