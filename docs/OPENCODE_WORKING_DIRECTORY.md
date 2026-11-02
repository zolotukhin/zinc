# OpenCode working-directory correction

The Intel Gemma trace exposed a launch defect independent of inference: an
absolute file edit reached the fixture, but relative shell commands could not
find its `src` or `test` directories. Increasing the deadline let the client
search the wrong workspace for longer.

Launching a child with `cwd` does not rewrite its inherited `PWD`. OpenCode
uses that environment variable to select its workspace; the
[2.0.21 run implementation](https://github.com/anomalyco/opencode/blob/v2.0.21/packages/cli/src/run/run.ts)
chooses `process.env.PWD` before `process.cwd()`. Absolute tool paths had hidden
the mismatch in the earlier protocol fixtures.

The setup helper, independent verification and evaluator now align both
values with the selected project at launch. The inherited environment and
saved portable configuration remain unchanged. This corrects where tools run;
it supplies no task implementation or repaired model output.

The actual-client CPU regression deliberately supplies a stale `PWD` pointing
to another owned directory. It reads and writes relative files, checks the
shell's project marker, runs independent acceptance tests, and verifies that
the other directory's sentinel is unchanged. Both OpenCode 1.19.4 and 2.0.21
fail the pre-fix relative-tools control and pass all three protocol fixtures
after the fix. The pinned reference checkout and frozen harness files were
also audited after the original Intel trials and remain unchanged.

These CPU results establish launch and tool behavior without model inference
or a performance claim. Earlier Intel coding trials retain their exact outcomes
and flawed launch context. The corrected real-GPU matrix must establish model
coding readiness separately.
