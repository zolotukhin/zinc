---
name: speckit-tasks
description: Use when working in this Spec Kit repository and the user wants to generate an actionable, dependency-ordered tasks.md from the active spec, plan, and related design artifacts.
---

# Speckit Tasks

Use this skill for the Spec Kit `tasks` phase.

1. Read `../../../.claude/commands/speckit.tasks.md` and treat it as the authoritative workflow.
2. Treat the current user request as the `$ARGUMENTS` referenced there.
3. Translate slash-command references to Codex skill names such as `speckit-specify`, `speckit-clarify`, `speckit-plan`, `speckit-tasks`, `speckit-analyze`, `speckit-implement`, `speckit-checklist`, `speckit-constitution`, and `speckit-taskstoissues`.
4. Use the repo-local `.specify/scripts/bash/*.sh` and `.specify/templates/*.md` files the workflow calls for.
5. Preserve the source workflow's task-format and organization rules, especially story grouping, dependency ordering, and exact file paths.
6. Do not create Codex custom prompt files. Codex support in this repo is skill-based.
