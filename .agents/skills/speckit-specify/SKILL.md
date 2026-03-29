---
name: speckit-specify
description: Use when working in this Spec Kit repository and the user wants to create or update a feature specification from a natural-language request, including the feature spec and its requirements checklist artifacts.
---

# Speckit Specify

Use this skill for the Spec Kit `specify` phase.

1. Read `../../../.claude/commands/speckit.specify.md` and treat it as the authoritative workflow.
2. Treat the current user request as the `$ARGUMENTS` referenced there. When the source workflow talks about text entered after `/speckit.specify`, it means the current user prompt.
3. Translate slash-command references to Codex skill names such as `speckit-specify`, `speckit-clarify`, `speckit-plan`, `speckit-tasks`, `speckit-analyze`, `speckit-implement`, `speckit-checklist`, `speckit-constitution`, and `speckit-taskstoissues`.
4. Use the repo-local `.specify/scripts/bash/*.sh` and `.specify/templates/*.md` files the workflow calls for.
5. Keep the spec workflow source-of-truth in the existing Spec Kit templates and command doc. Do not create parallel Codex prompt files.
6. Do not create Codex custom prompt files. Codex support in this repo is skill-based.
