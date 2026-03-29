---
name: speckit-analyze
description: Use when working in this Spec Kit repository and the user wants a read-only consistency and coverage analysis across the active feature's spec.md, plan.md, and tasks.md before implementation.
---

# Speckit Analyze

Use this skill for the Spec Kit `analyze` phase.

1. Read `../../../.claude/commands/speckit.analyze.md` and treat it as the authoritative workflow.
2. Treat the current user request as the `$ARGUMENTS` referenced there.
3. Translate slash-command references to Codex skill names such as `speckit-specify`, `speckit-clarify`, `speckit-plan`, `speckit-tasks`, `speckit-analyze`, `speckit-implement`, `speckit-checklist`, `speckit-constitution`, and `speckit-taskstoissues`.
4. Use the repo-local `.specify/scripts/bash/*.sh` and `.specify/templates/*.md` files the workflow calls for.
5. Keep this skill strictly read-only. Do not modify files unless the user explicitly asks for follow-up remediation outside this skill.
6. Do not create Codex custom prompt files. Codex support in this repo is skill-based.
