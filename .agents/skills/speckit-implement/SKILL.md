---
name: speckit-implement
description: Use when working in this Spec Kit repository and the user wants to execute the active tasks.md plan, respecting checklist gates, task dependencies, and phase-by-phase implementation.
---

# Speckit Implement

Use this skill for the Spec Kit `implement` phase.

1. Read `../../../.claude/commands/speckit.implement.md` and treat it as the authoritative workflow.
2. Treat the current user request as the `$ARGUMENTS` referenced there.
3. Translate slash-command references to Codex skill names such as `speckit-specify`, `speckit-clarify`, `speckit-plan`, `speckit-tasks`, `speckit-analyze`, `speckit-implement`, `speckit-checklist`, `speckit-constitution`, and `speckit-taskstoissues`.
4. Use the repo-local `.specify/scripts/bash/*.sh` and `.specify/templates/*.md` files the workflow calls for.
5. Preserve the source workflow's execution rules, including checklist gating, phased task execution, and marking completed tasks as `[X]` in `tasks.md`.
6. Do not create Codex custom prompt files. Codex support in this repo is skill-based.
