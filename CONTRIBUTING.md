# Contributing to ZINC

ZINC is still moving quickly. The most useful contributions are the ones that make the project easier to run, easier to debug, and easier to trust on real hardware.

## Before you start

- Read [README.md](./README.md) for the current project scope.
- Read [docs/GETTING_STARTED.md](./docs/GETTING_STARTED.md) if you are setting up the project for the first time.
- Read [docs/ROADMAP.md](./docs/ROADMAP.md) if you want to understand where help is most useful.
- If you are touching runtime behavior, read the relevant docs in [docs/](./docs/).

## Good first contributions

These are usually the easiest ways to help:

- documentation fixes and missing examples
- test coverage for parser, scheduler, API, and graph code
- bug reproduction cases with exact hardware and commands
- Windows and non-RDNA Linux build fixes
- benchmark tooling and better diagnostics
- small API or UX improvements that do not change core architecture

If you want a good first issue, look for `good first issue` or `help wanted` labels once they are available.

## Development setup

Core tools:

- Zig 0.15.2+
- Bun 1.x for the TypeScript tooling and tests
- Vulkan headers/loader
- `glslc` on Linux if you want shader compilation

Typical local workflow:

```bash
git clone https://github.com/zolotukhin/zinc.git
cd zinc
zig build
zig build test
```

On Linux, `zig build` also compiles shaders. On macOS, shader compilation is skipped and GPU inference is not the primary target environment.

## Making changes

Keep pull requests focused. A small, well-explained fix lands faster than a broad refactor with unclear impact.

For code changes:

1. reproduce the bug or baseline behavior first
2. make the smallest change that actually fixes the problem
3. run the relevant tests
4. include the exact commands you used in the PR

For performance changes:

1. include the baseline measurement
2. include the new measurement
3. include hardware, driver, model, quantization, and command
4. call out tradeoffs such as memory usage, complexity, or reduced portability

## Testing expectations

Before opening a PR, run at minimum:

```bash
zig build test
```

`zig build test` now runs both the Zig test suite and the Bun test suite.

If you want to run the Bun suite by itself:

```bash
bun test
```

If you changed build logic, Vulkan setup, shaders, or platform support, include the exact build command and platform you validated.

## Reporting bugs well

The fastest way to get help is to make the issue reproducible.

Please include:

- operating system and version
- GPU model
- Vulkan driver/runtime details
- Zig version
- exact command you ran
- exact model file and quantization
- full error output or log snippet
- whether the issue reproduces on the latest `main`

For performance bugs, include both expected and actual tok/s.

## Pull request guidelines

- explain what changed and why
- link the issue if one exists
- keep unrelated cleanup out of the PR
- note any architectural tradeoffs explicitly
- include screenshots only when they help explain behavior

## Communication

The README links to the project Discord. Use GitHub issues for actionable bugs and feature requests, and use Discord for faster discussion, early feedback, and testing coordination.

## Code of conduct

By participating, you agree to the expectations in [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

## Project boundaries

Please ask before making changes that alter core architecture or project direction, especially:

- compute graph IR shape
- model architecture support
- GGUF parsing behavior
- Vulkan device initialization and selection

These areas tend to have wider compatibility and maintenance impact than they first appear to.
