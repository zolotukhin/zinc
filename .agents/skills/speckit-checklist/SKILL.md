---
name: speckit-checklist
description: Use when working in this Spec Kit repository and the user wants a requirements-quality checklist for the active feature, generated from the current spec, plan, and tasks artifacts.
---

# Speckit Checklist

Use this skill for the Spec Kit `checklist` phase.

1. Read `../../../.claude/commands/speckit.checklist.md` and treat it as the authoritative workflow.
2. Treat the current user request as the `$ARGUMENTS` referenced there.
3. Translate slash-command references to Codex skill names such as `speckit-specify`, `speckit-clarify`, `speckit-plan`, `speckit-tasks`, `speckit-analyze`, `speckit-implement`, `speckit-checklist`, `speckit-constitution`, and `speckit-taskstoissues`.
4. Use the repo-local `.specify/scripts/bash/*.sh` and `.specify/templates/*.md` files the workflow calls for.
5. Preserve the checklist intent from the source workflow: validate the quality of written requirements, not implementation behavior.
6. Do not create Codex custom prompt files. Codex support in this repo is skill-based.
