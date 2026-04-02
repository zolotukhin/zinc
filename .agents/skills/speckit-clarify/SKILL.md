---
name: speckit-clarify
description: Use when working in this Spec Kit repository and the user wants to resolve ambiguity in the active feature spec by asking targeted clarification questions and writing the accepted answers back into spec.md.
---

# Speckit Clarify

Use this skill for the Spec Kit `clarify` phase.

1. Read `../../../.claude/commands/speckit.clarify.md` and treat it as the authoritative workflow.
2. Treat the current user request as the `$ARGUMENTS` referenced there.
3. Translate slash-command references to Codex skill names such as `speckit-specify`, `speckit-clarify`, `speckit-plan`, `speckit-tasks`, `speckit-analyze`, `speckit-implement`, `speckit-checklist`, `speckit-constitution`, and `speckit-taskstoissues`.
4. Use the repo-local `.specify/scripts/bash/*.sh` and `.specify/templates/*.md` files the workflow calls for.
5. Follow the source workflow's interaction model: ask one clarification question at a time, keep the total within the documented limit, and update the spec incrementally after each accepted answer.
6. Do not create Codex custom prompt files. Codex support in this repo is skill-based.
