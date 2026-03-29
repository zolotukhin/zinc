# ZINC Roadmap

This is a lightweight public roadmap for contributors. It is not a guarantee of order, but it shows where outside help is most useful.

## Now

Current priorities:

- make the core Linux AMD path more stable on real hardware
- improve Windows and cross-platform build reliability
- harden the OpenAI-compatible server path
- validate and prioritize support for the next smaller Qwen3.5 variant after the current 35B-A3B target
- improve test coverage for scheduler, API, tokenizer, graph, and model loading code
- make bug reports and regressions easier to reproduce
- keep RDNA4 performance work measurable and benchmark-driven

Good contributions here:

- build fixes
- reproducible bug reports
- small correctness fixes with tests
- docs improvements that reduce setup friction
- benchmark tooling and diagnostics

## Next

Near-term work:

- better profiling and graph/runtime inspection
- more contributor-friendly benchmark workflows
- broader model validation across supported GGUF architectures
- make smaller Qwen3.5-family models first-class validation targets in docs, CI, and benchmark workflows
- stronger CI coverage across more environments
- more complete API compatibility and client examples

Good contributions here:

- profiler UX
- validation scripts
- API tests
- docs and examples

## Later

Longer-horizon directions:

- more mature continuous batching and serving behavior
- deeper TurboQuant validation and tuning
- broader hardware coverage beyond the primary RDNA3/RDNA4 target
- richer visual tooling around execution graphs and performance traces
- packaging and distribution improvements for non-core developer users

## What helps most from the community

The most useful outside input is usually one of:

- exact reproductions on hardware the maintainer does not have
- measured before/after performance data
- focused tests for edge cases
- small, well-scoped PRs instead of broad refactors

If you want to help, start with [CONTRIBUTING.md](../CONTRIBUTING.md) and open an issue with concrete hardware, commands, logs, and expected behavior.
