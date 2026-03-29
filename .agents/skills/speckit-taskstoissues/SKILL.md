---
name: speckit-taskstoissues
description: Use when working in this Spec Kit repository and the user wants to convert the active tasks.md into dependency-ordered GitHub issues for the repository's GitHub remote.
---

# Speckit Tasks To Issues

Use this skill for the Spec Kit `taskstoissues` phase.

1. Read `../../../.claude/commands/speckit.taskstoissues.md` and treat it as the authoritative workflow.
2. Treat the current user request as the `$ARGUMENTS` referenced there.
3. Translate slash-command references to Codex skill names such as `speckit-specify`, `speckit-clarify`, `speckit-plan`, `speckit-tasks`, `speckit-analyze`, `speckit-implement`, `speckit-checklist`, `speckit-constitution`, and `speckit-taskstoissues`.
4. Use the repo-local `.specify/scripts/bash/*.sh` and `.specify/templates/*.md` files the workflow calls for.
5. Only proceed when the repository remote is GitHub and the required GitHub issue-creation tooling is available in the environment. Otherwise stop and report the missing prerequisite.
6. Do not create Codex custom prompt files. Codex support in this repo is skill-based.
