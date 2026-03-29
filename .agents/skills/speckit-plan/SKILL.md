---
name: speckit-plan
description: Use when working in this Spec Kit repository and the user wants to turn the active feature spec into an implementation plan and design artifacts such as research.md, data-model.md, contracts, and quickstart.md.
---

# Speckit Plan

Use this skill for the Spec Kit `plan` phase.

1. Read `../../../.claude/commands/speckit.plan.md` and treat it as the authoritative workflow.
2. Treat the current user request as the `$ARGUMENTS` referenced there.
3. Translate slash-command references to Codex skill names such as `speckit-specify`, `speckit-clarify`, `speckit-plan`, `speckit-tasks`, `speckit-analyze`, `speckit-implement`, `speckit-checklist`, `speckit-constitution`, and `speckit-taskstoissues`.
4. Use the repo-local `.specify/scripts/bash/*.sh` and `.specify/templates/*.md` files the workflow calls for.
5. Override the Claude-specific agent-context step from the source workflow: when it asks for `.specify/scripts/bash/update-agent-context.sh claude`, run `.specify/scripts/bash/update-agent-context.sh codex` instead.
6. Do not create Codex custom prompt files. Codex support in this repo is skill-based.
