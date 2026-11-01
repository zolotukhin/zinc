"""Maintainer-side release configuration audit; never reads secret values."""

import datetime
import json
import os
import re
import subprocess

from release_lib import ROOT, environment_main_only, environment_requires_reviewers, policy, require, source_sha, version
from release_sign import SIGNING_SECRETS


class InspectionError(ValueError):
    pass


class GitHub:
    def run(self, args, body=None):
        try:
            result = subprocess.run(["gh", *args], cwd=ROOT, input=None if body is None else json.dumps(body),
                                    capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            raise InspectionError("GitHub CLI unavailable or timed out; check gh authentication and retry") from None
        # CLI stderr, URLs and exception argv can expose private runner details
        # or credentials. Only fixed messages cross the reporting boundary.
        if result.returncode:
            raise InspectionError("GitHub request failed; check gh authentication and repository read/admin permissions")
        return result.stdout

    def request(self, path, paginate=False, body=None):
        args = ["api", "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2022-11-28"]
        if body is not None:
            args += ["--method", "POST", "--input", "-"]
        if paginate:
            args += ["--paginate", "--slurp"]
            path += "?per_page=30"
        try:
            return json.loads(self.run([*args, path], body=body))
        except (TypeError, ValueError) as error:
            if isinstance(error, InspectionError):
                raise
            raise InspectionError("GitHub returned an invalid response; retry the configuration audit") from None

    def collection(self, path, key, identity="name"):
        pages = self.request(path, paginate=True)
        valid = isinstance(pages, list) and bool(pages)
        rows, totals = [], []
        for page in pages if valid else []:
            if not isinstance(page, dict) or not isinstance(page.get(key), list) or type(page.get("total_count")) is not int:
                valid = False
                break
            rows.extend(page[key])
            totals.append(page["total_count"])
        # Reject detectable missing pages, inconsistent counts and duplicates.
        # GitHub pagination is not atomic: same-count replacements can escape
        # these checks. Hardware jobs must still validate the selected worker.
        if (not valid or any(total != len(rows) for total in totals) or
                not all(isinstance(row, dict) and type(row.get(identity)) is (int if identity == "id" else str) for row in rows) or
                len({row[identity] for row in rows}) != len(rows)):
            raise InspectionError("GitHub inventory was incomplete or changed during pagination; retry the audit")
        return rows


def repository(value, github):
    if not value:
        value = os.environ.get("GITHUB_REPOSITORY") or github.run(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]).strip()
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", value)
            and value.split("/")[1] not in (".", ".."), "repository must be owner/repo")
    return value


def check(name, passed, message, **fields):
    return {"name": name, "passed": bool(passed), "message": message, **fields}


def runner_checks(profiles, variables, runners):
    # The API error already blocks the audit. Avoid eight duplicate failures
    # or claiming assignments are missing when their metadata is unreadable.
    if variables is None or runners is None:
        return []
    roles = {}
    for profile in profiles:
        roles.setdefault(profile["runner_variable"], []).append(profile["id"])
    checks = []
    for name, assigned in roles.items():
        fields = {"profiles": assigned}
        raw = variables.get(name)
        if not raw:
            checks.append(check(name, False, "Configure this repository variable as a JSON label array.", **fields))
            continue
        try:
            labels = json.loads(raw)
            require(isinstance(labels, list) and labels and all(isinstance(label, str) and label.strip() for label in labels), "invalid labels")
        except (TypeError, ValueError):
            checks.append(check(name, False, "Replace the invalid assignment with a nonempty JSON array of labels.", **fields))
            continue
        # GitHub label matching is case-insensitive and requires EVERY label.
        # Busy online workers are eligible; GitHub queues until they are idle.
        labels = {label.casefold() for label in labels}
        matches = [runner for runner in runners if labels <= {label["name"].casefold() for label in runner["labels"]}]
        online = [runner for runner in matches if runner["status"] == "online"]
        idle = sum(not runner["busy"] for runner in online)
        message = ("Online matching worker available." if online else
                   "Matching workers are offline; bring the assigned worker online." if matches else
                   "No registered worker matches all assigned labels; provision or correct the assignment.")
        checks.append(check(name, bool(online), message, matching=len(matches), online=len(online), idle=idle, **fields))
    return checks


def audit(github, repo, data):
    checks = []
    base = f"repos/{repo}/"

    def inspect(name, operation):
        try:
            return operation()
        except InspectionError as error:
            checks.append(check(name, False, str(error)))
            return None

    def variables():
        rows = github.collection(base + "actions/variables", "variables")
        if not all(isinstance(row.get("value"), str) for row in rows):
            raise InspectionError("Runner variable values could not be inspected; retry the audit")
        return {row["name"]: row["value"] for row in rows}

    def runners():
        rows = github.collection(base + "actions/runners", "runners", "id")
        for row in rows:
            labels = row.get("labels")
            if (row.get("status") not in ("online", "offline") or type(row.get("busy")) is not bool or
                    not isinstance(labels, list) or not all(isinstance(label, dict) and isinstance(label.get("name"), str) for label in labels)):
                raise InspectionError("Runner inventory could not be inspected; retry the audit")
        return rows

    def read_environment(name):
        value = github.request(base + "environments/" + name)
        if (not isinstance(value, dict) or "deployment_branch_policy" not in value or
                not isinstance(value.get("protection_rules"), list)):
            raise InspectionError("Environment metadata was invalid; retry the configuration audit")
        return value

    assignments = inspect("runner variables API", variables)
    inventory = inspect("runner inventory API", runners)
    checks.extend(runner_checks(data["gpu_profiles"], assignments, inventory))
    for name in ("release-workers", "release-signing", "release"):
        environment = inspect(name + " environment API", lambda: read_environment(name))
        if environment is None:
            continue
        branches = inspect(name + " branch policies API", lambda: github.collection(base + "environments/" + name + "/deployment-branch-policies", "branch_policies", "id"))
        if branches is not None:
            # collection() already proved every page's total_count equals the
            # complete row count, before sharing this predicate with publishing.
            main_only = environment_main_only(environment, branches)
            checks.append(check(name + " branches", main_only, "Restricted to the main branch." if main_only else
                                "Restrict this environment to one custom branch policy: main (branch, not tag)."))
        if name == "release":
            reviewers = environment_requires_reviewers(environment)
            checks.append(check("release approval", reviewers, "Publication requires reviewer approval." if reviewers else
                                "Configure at least one required reviewer on the release environment."))

    secrets = inspect("Apple signing secrets API", lambda: github.collection(base + "environments/release-signing/secrets", "secrets"))
    missing = sorted(set(SIGNING_SECRETS) - {row["name"] for row in secrets}) if secrets is not None else None
    if missing is not None:
        checks.append(check("Apple signing secrets", missing == [],
                            "Configure the missing secrets in the release-signing environment." if missing else
                            "All required secret names are present; validity is checked by the signing job.", missing=missing))
    inventory_counts = None if inventory is None else {"registered": len(inventory),
        "online": sum(row["status"] == "online" for row in inventory),
        "idle": sum(row["status"] == "online" and not row["busy"] for row in inventory)}
    return {"schema_version": 1, "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "configuration_ready": all(row["passed"] for row in checks), "runner_inventory": inventory_counts, "checks": checks,
            "qualification": "Configuration snapshot only; signing, private worker setup and all nine packaged hardware profiles still require candidate qualification."}


def resolve_candidate_source(github, repo, requested):
    """Freeze main once, then prove the selected commit belongs to its history."""
    base = f"repos/{repo}/"
    main = github.request(base + "git/ref/heads/main")
    obj = main.get("object") if isinstance(main, dict) else None
    if (not isinstance(main, dict) or main.get("ref") != "refs/heads/main" or
            not isinstance(obj, dict) or obj.get("type") != "commit" or
            not isinstance(obj.get("sha"), str) or not re.fullmatch(r"[0-9a-f]{40}", obj["sha"])):
        raise InspectionError("Could not resolve the main branch to a source commit")
    frozen_main = obj["sha"]
    if requested == "main" or requested == frozen_main:
        return frozen_main
    source_sha(requested)
    commit = github.request(base + "commits/" + requested)
    if not isinstance(commit, dict) or commit.get("sha") != requested:
        raise InspectionError("Candidate source did not resolve to the requested commit")
    comparison = github.request(base + "compare/" + requested + "..." + frozen_main + "?per_page=1")
    if (not isinstance(comparison, dict) or comparison.get("status") not in ("ahead", "identical") or
            type(comparison.get("behind_by")) is not int or comparison["behind_by"] != 0 or
            not isinstance(comparison.get("base_commit"), dict) or comparison["base_commit"].get("sha") != requested or
            not isinstance(comparison.get("merge_base_commit"), dict) or comparison["merge_base_commit"].get("sha") != requested):
        raise InspectionError("Candidate source must be a verified ancestor of main; no workflow was dispatched")
    return requested


def dispatch_candidate(github, repo, release_version, commit):
    # Request the ID from THIS dispatch. Searching recent runs can accidentally
    # associate another maintainer's concurrent run with our request.
    result = github.request(f"repos/{repo}/actions/workflows/release-candidate.yml/dispatches", body={
        "ref": "main", "inputs": {"version": release_version, "source": commit}, "return_run_details": True,
    })
    if not isinstance(result, dict) or type(result.get("workflow_run_id")) is not int or result["workflow_run_id"] <= 0:
        raise InspectionError("Dispatch response did not identify a workflow run; inspect Actions before retrying")
    # Ignore all response URLs and other metadata at the redaction boundary.
    return result["workflow_run_id"]


def render(report):
    lines = ["Release configuration: " + ("READY" if report["configuration_ready"] else "BLOCKED")]
    inventory = report["runner_inventory"]
    if inventory is not None:
        lines.append(f"Runner inventory: {inventory['registered']} registered, {inventory['online']} online, {inventory['idle']} idle.")
    for row in report["checks"]:
        line = ("PASS " if row["passed"] else "FAIL ") + row["name"] + ": " + row["message"]
        if "online" in row:
            line += f" ({row['online']} online, {row['idle']} idle)"
        if row.get("missing"):
            line += " Missing: " + ", ".join(row["missing"])
        lines.append(line)
    lines.append(report["qualification"])
    if "dispatch_confirmed" in report:
        lines.append(f"Candidate run {report['candidate_run_id']} created; execution and validation are still pending." if report["dispatch_confirmed"] else
                     "Dispatch was not confirmed; inspect Actions runs before retrying." if report["dispatch_attempted"] else
                     "Candidate workflow was not dispatched because a preflight check failed.")
    return "\n".join(lines)


def run(args):
    github = GitHub()
    repo = repository(args.repo, github)
    if args.dispatch:
        version(args.dispatch)
        if args.source != "main":
            source_sha(args.source)
    else:
        require(args.source == "main", "--source requires --dispatch")
    report = audit(github, repo, policy())
    if args.dispatch:
        report["dispatch_attempted"] = False
        report["dispatch_confirmed"] = False
        if report["configuration_ready"]:
            try:
                report["source_commit"] = resolve_candidate_source(github, repo, args.source)
            except InspectionError as error:
                report["checks"].append(check("candidate source", False, str(error)))
            else:
                report["checks"].append(check("candidate source", True, "Frozen source commit verified on main."))
                report["dispatch_attempted"] = True
                try:
                    report["candidate_run_id"] = dispatch_candidate(github, repo, args.dispatch, report["source_commit"])
                    report["dispatch_confirmed"] = True
                except InspectionError as error:
                    report["checks"].append(check("candidate dispatch", False, str(error)))
    print(json.dumps(report, indent=2) if args.json else render(report))
    if not report["configuration_ready"] or (args.dispatch and not report["dispatch_confirmed"]):
        raise SystemExit(1)
