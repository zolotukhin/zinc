---
name: speckit-constitution
description: Use when working in this Spec Kit repository and the user wants to create or amend the project constitution in .specify/memory/constitution.md and keep dependent templates aligned.
---

# Speckit Constitution

Use this skill for the Spec Kit `constitution` phase.

1. Read `../../../.claude/commands/speckit.constitution.md` and treat it as the authoritative workflow.
2. Treat the current user request as the `$ARGUMENTS` referenced there.
3. Translate slash-command references to Codex skill names such as `speckit-specify`, `speckit-clarify`, `speckit-plan`, `speckit-tasks`, `speckit-analyze`, `speckit-implement`, `speckit-checklist`, `speckit-constitution`, and `speckit-taskstoissues`.
4. Use the repo-local `.specify/scripts/bash/*.sh` and `.specify/templates/*.md` files the workflow calls for.
5. Apply amendments directly to `.specify/memory/constitution.md` and keep the sync report and template follow-through required by the source workflow.
6. Do not create Codex custom prompt files. Codex support in this repo is skill-based.
