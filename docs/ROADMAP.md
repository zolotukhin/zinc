# ZINC Roadmap

This is a lightweight public roadmap for contributors. It is not a guarantee of order, but it shows where outside help is most useful.

## Now

Current priorities:

- make the core Linux AMD path more stable on real hardware
- improve Windows and cross-platform build reliability
- harden the OpenAI-compatible server path
- keep Qwen 3.8 27B correctness and performance coverage stable across RDNA4 and Metal
- harden Intel Arc as an official Linux Vulkan target now that the full catalog matrix is published
- improve test coverage for scheduler, API, tokenizer, graph, and model loading code
- make bug reports and regressions easier to reproduce
- keep RDNA4, Intel Arc, and Metal performance work measurable and benchmark-driven

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
- expand the perf suite catalog beyond the current Qwen 3.6 and Gemma 4 targets
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
- broader hardware coverage beyond the primary RDNA3/RDNA4 and Intel Arc targets
- richer visual tooling around execution graphs and performance traces
- packaging and distribution improvements for non-core developer users

## What helps most from the community

The most useful outside input is usually one of:

- exact reproductions on hardware the maintainer does not have
- measured before/after performance data
- focused tests for edge cases
- small, well-scoped PRs instead of broad refactors

If you want to help, start with [CONTRIBUTING.md](https://github.com/zolotukhin/zinc/blob/main/CONTRIBUTING.md) and open an issue with concrete hardware, commands, logs, and expected behavior.
