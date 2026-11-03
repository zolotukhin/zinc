"""Bound review input without allowing published evidence to hide source changes."""

import re
import shlex


def _path(block):
    header = block.splitlines()[0]
    try:
        fields = shlex.split(header)
        if len(fields) == 4 and fields[3].startswith("b/"):
            return fields[3][2:]
    except ValueError:
        pass
    # Keep an unfamiliar header visible instead of silently dropping its patch.
    return header


def _priority(path):
    if (path.startswith("site/public/zinc/validation/")
            or path == "site/src/data/zinc-performance.json"
            or path.endswith((".lock", ".spv"))):
        return 2
    if path.endswith((".md", ".mdx", ".txt")) and path != "AGENTS.md":
        return 1
    return 0


def _bounded(text, limit, notice):
    if len(text) <= limit:
        return text
    if limit < len(notice):
        return notice[:limit]
    return text[:limit - len(notice)] + notice


def _bounded_inventory(text, limit):
    if len(text) <= limit:
        return text
    notice = "\n[Inventory truncated; additional files exist in the full PR.]\n"
    lines = []
    used = len(notice)
    for line in text.splitlines(keepends=True):
        if used + len(line) > limit:
            break
        lines.append(line)
        used += len(line)
    return "".join(lines) + notice


def _allocations(sizes, budget):
    """Retain small patches in full, then share the rest between larger ones."""
    assigned = [0] * len(sizes)
    pending = list(range(len(sizes)))
    while pending:
        share, remainder = divmod(budget, len(pending))
        complete = [index for index in pending if sizes[index] <= share]
        if not complete:
            for position, index in enumerate(pending):
                assigned[index] = share + (position < remainder)
            break
        for index in complete:
            assigned[index] = sizes[index]
            budget -= sizes[index]
            pending.remove(index)
    return assigned


def build_review_context(diff, limit=120000, per_file_limit=120000):
    """List changed files and allocate patch space to code before generated data.

    The total includes inventory and omission notices. Sharing each priority
    group's remaining budget prevents one large source file from hiding tests
    later in the diff. Missing context never means a file was absent from the PR.
    """
    if limit < 512 or per_file_limit < 128:
        raise ValueError("Review limits are too small to describe omitted context")
    starts = [match.start() for match in re.finditer(r"^diff --git ", diff, re.M)]
    if not starts:
        return _bounded("Unstructured diff; no changed-file inventory was available.\n\n" + diff,
                        limit, "\n[Review input truncated.]\n")
    blocks = [diff[start:end] for start, end in zip(starts, starts[1:] + [len(diff)])]
    entries = [(_priority(_path(block)), _path(block), block) for block in blocks]
    labels = ("code/config", "documentation", "generated/evidence")
    inventory = "Changed-file inventory: " + str(len(entries)) + " files in the full PR diff.\n"
    inventory += "".join(f"- {path} [{labels[priority]}]\n" for priority, path, _ in entries)
    inventory = _bounded_inventory(inventory, min(16000, limit // 3))
    prefix = inventory + "\nPatches are ordered by review priority.\n\n"
    suffix = ("\nReview scope: patches may be truncated or omitted. Use the inventory to identify "
              "changed files; do not infer missing implementation or tests from omitted context.\n")
    budget = limit - len(prefix) - len(suffix)
    chunks = []
    for priority in range(3):
        group = [entry for entry in entries if entry[0] == priority]
        sizes = [min(len(block), per_file_limit) for _, _, block in group]
        allowances = _allocations(sizes, budget)
        for (_, _, block), allowance in zip(group, allowances):
            if allowance < min(128, len(block)):
                continue
            chunk = _bounded(block, allowance, "\n[Patch truncated for review; see full diff.]\n")
            chunks.append(chunk)
            budget -= len(chunk)
    return prefix + "".join(chunks) + suffix
