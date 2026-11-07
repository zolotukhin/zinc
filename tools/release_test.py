"""Release contract tests. Fixtures execute CPU-only C programs, never a GPU/model."""

import contextlib
import copy
import base64
import io
import importlib.util
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import release
import release_gpu
import release_gpu_group
import release_lib
import release_publish
import release_preflight
import release_sign
import release_ssh
import release_worker
import release_worker_guard
from release_lib import (
    ROOT, archive_name, check_metadata, extract_archive, inventory, policy,
    read_json, sha256, target, version, write_json,
)

COMMIT = "1" * 40
PREFLIGHT_MAIN = "2" * 40
NATIVE = "macos-aarch64-metal" if sys.platform == "darwin" else "linux-x86_64-vulkan"
_launcher_spec = importlib.util.spec_from_file_location("release_worker_launcher", ROOT / "scripts/release_worker/launch.py")
worker_launcher = importlib.util.module_from_spec(_launcher_spec)
_launcher_spec.loader.exec_module(worker_launcher)


class PreflightGitHub(release_preflight.GitHub):
    """Exercise real parsing/matching with synthetic, deliberately private data."""
    def __init__(self):
        self.commands = []
        self.requests = []
        self.responses = {}
        variables, runners = [], []
        roles = list(dict.fromkeys(row["runner_variable"] for row in policy()["gpu_profiles"]))
        for index, role in enumerate(roles):
            labels = ["self-hosted", f"<private-label-{index}>"]
            variables.append({"name": role, "value": json.dumps(labels)})
            runners.append({"id": index + 1, "name": "<private-runner>", "status": "online", "busy": False,
                            "labels": [{"name": label.upper()} for label in labels]})
        self.rows("actions/variables", "variables", variables)
        self.rows("actions/runners", "runners", runners)
        for name in ("release", "release-signing", "release-workers"):
            self.responses[f"environments/{name}"] = {
                "deployment_branch_policy": {"custom_branch_policies": True, "protected_branches": False},
                "protection_rules": [{"type": "required_reviewers", "reviewers": [{"type": "User", "reviewer": {"id": 1, "login": "<private-reviewer>"}}]}],
            }
            self.rows(f"environments/{name}/deployment-branch-policies", "branch_policies", [{"id": 1, "name": "main", "type": "branch"}])
        self.rows("environments/release-signing/secrets", "secrets", [{"name": name} for name in release_sign.SIGNING_SECRETS])
        self.responses["git/ref/heads/main"] = {"ref": "refs/heads/main", "object": {"type": "commit", "sha": PREFLIGHT_MAIN}}
        self.responses["commits/" + COMMIT] = {"sha": COMMIT}
        self.responses[f"compare/{COMMIT}...{PREFLIGHT_MAIN}"] = {
            "status": "ahead", "behind_by": 0, "base_commit": {"sha": COMMIT}, "merge_base_commit": {"sha": COMMIT},
        }
        self.responses["actions/workflows/release-candidate.yml/dispatches"] = {
            "workflow_run_id": 42, "html_url": "<private-response-url>", "run_url": "<private-api-url>",
        }

    def rows(self, path, key, rows):
        self.responses[path] = [{"total_count": len(rows), key: rows}]

    def run(self, args, body=None):
        self.commands.append(args)
        path = args[-1].removeprefix("repos/owner/repo/").split("?")[0]
        self.requests.append((path, body))
        value = self.responses[path]
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


class ReleasePreflightTest(unittest.TestCase):
    def full_policy(self):
        # Retain coverage of signing and future expanded matrices separately
        # from today's Linux-only launch policy.
        data = policy()
        data.pop("release", None)
        return data

    def audit(self, github):
        return release_preflight.audit(github, "owner/repo", self.full_policy())

    def role(self, report, name="ZINC_RUNNER_METAL_APPLE7"):
        return next(row for row in report["checks"] if row["name"] == name)

    def args(self, **overrides):
        return SimpleNamespace(**{"repo": "owner/repo", "json": True, "dispatch": None, "source": "main", **overrides})

    def run_preflight(self, github, expected_status=0, **overrides):
        output = io.StringIO()
        with patch.object(release_preflight, "GitHub", return_value=github), \
                patch.object(release_preflight, "policy", return_value=self.full_policy()), contextlib.redirect_stdout(output):
            if expected_status:
                with self.assertRaises(SystemExit) as error:
                    release_preflight.run(self.args(**overrides))
                self.assertEqual(error.exception.code, expected_status)
            else:
                release_preflight.run(self.args(**overrides))
        return json.loads(output.getvalue())

    def test_ready_is_read_only_and_keeps_all_nine_profiles_in_eight_roles(self):
        github = PreflightGitHub()
        report = self.run_preflight(github)
        self.assertTrue(report["configuration_ready"])
        self.assertEqual(report["runner_inventory"], {"registered": 8, "online": 8, "idle": 8})
        roles = [row for row in report["checks"] if "profiles" in row]
        self.assertEqual(len(roles), 8)
        self.assertEqual(sum(len(row["profiles"]) for row in roles), 9)
        self.assertEqual(self.role(report, "ZINC_RUNNER_RDNA4")["profiles"], ["vulkan-rdna4", "rocm-rdna4"])
        self.assertTrue(all(command[0] == "api" and "--method" not in command for command in github.commands))
        self.assertIn("candidate qualification", report["qualification"])

    def test_linux_launch_requires_only_two_roles_and_never_reads_apple_configuration(self):
        github = PreflightGitHub()
        variables = github.responses["actions/variables"][0]["variables"]
        allowed = [row for row in variables if row["name"] in ("ZINC_RUNNER_RDNA4", "ZINC_RUNNER_INTEL_XE2")]
        labels = [set(json.loads(row["value"])) for row in allowed]
        runners = github.responses["actions/runners"][0]["runners"]
        github.rows("actions/variables", "variables", allowed)
        github.rows("actions/runners", "runners", [r for r in runners if any(label <= {l["name"].lower() for l in r["labels"]} for label in labels)])
        report = release_preflight.audit(github, "owner/repo", policy())
        self.assertTrue(report["configuration_ready"])
        self.assertEqual(report["runner_inventory"]["registered"], 2)
        self.assertEqual(sum(len(row["profiles"]) for row in report["checks"] if "profiles" in row), 3)
        self.assertTrue(all("release-signing" not in path for path, _ in github.requests))
        github.rows("actions/variables", "variables", [row for row in allowed if row["name"] != "ZINC_RUNNER_INTEL_XE2"])
        self.assertFalse(release_preflight.audit(github, "owner/repo", policy())["configuration_ready"])

    def test_label_matching_is_case_insensitive_and_requires_every_label(self):
        github = PreflightGitHub()
        self.assertTrue(self.audit(github)["configuration_ready"])
        github.responses["actions/runners"][0]["runners"][0]["labels"].pop()
        report = self.audit(github)
        self.assertFalse(report["configuration_ready"])
        self.assertEqual(self.role(report)["matching"], 0)

    def test_offline_workers_block_but_busy_online_workers_are_eligible(self):
        github = PreflightGitHub()
        runner = github.responses["actions/runners"][0]["runners"][0]
        runner["status"] = "offline"
        report = self.audit(github)
        self.assertFalse(report["configuration_ready"])
        self.assertEqual(self.role(report)["matching"], 1)
        self.assertEqual(self.role(report)["online"], 0)
        runner.update(status="online", busy=True)
        report = self.audit(github)
        self.assertTrue(report["configuration_ready"])
        self.assertEqual(self.role(report)["idle"], 0)

    def test_missing_or_invalid_runner_assignments_never_qualify(self):
        github = PreflightGitHub()
        for raw in (None, "", "[]", "{}", "[null]", '[" "]', "<private-invalid-json>"):
            rows = [] if raw is None else [{"name": "ZINC_RUNNER_METAL_APPLE7", "value": raw}]
            github.rows("actions/variables", "variables", rows)
            with self.subTest(raw=raw):
                report = self.audit(github)
                self.assertFalse(report["configuration_ready"])
                self.assertFalse(self.role(report)["passed"])
                self.assertNotIn("<private-invalid-json>", json.dumps(report))

    def test_environment_requires_only_main_branch_and_publication_reviewer(self):
        for name in ("release", "release-signing", "release-workers"):
            for branch_rows in ([], [{"id": 1, "name": "*", "type": "branch"}],
                                [{"id": 1, "name": "main", "type": "tag"}],
                                [{"id": 1, "name": "main", "type": "branch"}, {"id": 2, "name": "v*", "type": "tag"}]):
                github = PreflightGitHub()
                github.rows(f"environments/{name}/deployment-branch-policies", "branch_policies", branch_rows)
                with self.subTest(name=name, branches=branch_rows):
                    self.assertFalse(self.audit(github)["configuration_ready"])
            for restriction in (None, {}, {"protected_branches": True, "custom_branch_policies": False}):
                github = PreflightGitHub()
                github.responses[f"environments/{name}"]["deployment_branch_policy"] = restriction
                with self.subTest(name=name, restriction=restriction):
                    self.assertFalse(self.audit(github)["configuration_ready"])
        for reviewers in ([], [None], [{}], [{"type": "User", "reviewer": {"id": None}}]):
            github = PreflightGitHub()
            github.responses["environments/release"]["protection_rules"] = [{"type": "required_reviewers", "reviewers": reviewers}]
            with self.subTest(reviewers=reviewers):
                self.assertFalse(self.audit(github)["configuration_ready"])

    def test_signing_requires_all_six_names_in_the_environment(self):
        github = PreflightGitHub()
        github.rows("environments/release-signing/secrets", "secrets", [{"name": release_sign.SIGNING_SECRETS[0]}])
        report = self.audit(github)
        signing = self.role(report, "Apple signing secrets")
        self.assertFalse(report["configuration_ready"])
        self.assertEqual(set(signing["missing"]), set(release_sign.SIGNING_SECRETS[1:]))
        # Repository-level credentials do not satisfy the protected environment contract.
        self.assertFalse(any(command[-1].startswith("repos/owner/repo/actions/secrets") for command in github.commands))

    def test_preflight_and_publication_agree_on_one_complete_branch_policy(self):
        main = {"id": 1, "name": "main", "type": "branch"}
        tag = {"id": 2, "name": "v*", "type": "tag"}
        for total, rows, accepted in ((1, [main], True), (2, [main], False), (1, [main, tag], False), (2, [main, tag], False)):
            github = PreflightGitHub()
            response = {"total_count": total, "branch_policies": rows}
            github.responses["environments/release/deployment-branch-policies"] = [response]
            with self.subTest(total=total, rows=rows):
                self.assertEqual(self.audit(github)["configuration_ready"], accepted)
                environment = github.responses["environments/release"]
                if accepted:
                    release_publish.validate_environment(environment, response)
                else:
                    with self.assertRaises(ValueError):
                        release_publish.validate_environment(environment, response)

    def test_human_and_json_reports_never_disclose_private_api_fields(self):
        github = PreflightGitHub()
        github.responses["environments/release-signing/secrets"][0]["secrets"].append({"name": "<unrelated-secret>", "value": "<secret-value>"})
        github.responses["environments/release-signing/secrets"][0]["total_count"] += 1
        report = self.audit(github)
        output = (json.dumps(report) + release_preflight.render(report)).casefold()
        for marker in ("<private-", "<unrelated-secret>", "<secret-value>"):
            self.assertNotIn(marker, output)

    def test_pagination_collects_all_pages_and_rejects_incomplete_or_duplicate_data(self):
        github = PreflightGitHub()
        github.responses["fixture"] = [{"total_count": 2, "runners": [{"id": 1}]}, {"total_count": 2, "runners": [{"id": 2}]}]
        self.assertEqual(github.collection("repos/owner/repo/fixture", "runners", "id"), [{"id": 1}, {"id": 2}])
        self.assertIn("--paginate", github.commands[-1])
        self.assertIn("--slurp", github.commands[-1])
        self.assertTrue(github.commands[-1][-1].endswith("?per_page=30"))
        for payload in ([], {}, [{"total_count": 2, "runners": [{"id": 1}]}],
                        [{"total_count": 2, "runners": [{"id": 1}, {"id": 1}]}],
                        [{"total_count": 1, "runners": [{"id": 1}]}, {"total_count": 2, "runners": [{"id": 2}]}]):
            github.responses["fixture"] = payload
            with self.subTest(payload=payload), self.assertRaises(release_preflight.InspectionError):
                github.collection("repos/owner/repo/fixture", "runners", "id")

    def test_unreadable_and_malformed_api_data_fail_closed_and_redact_errors(self):
        for outcome in (SimpleNamespace(returncode=1, stdout="<private-response>", stderr="<private-stderr>"),
                        SimpleNamespace(returncode=0, stdout="<invalid-private-json>", stderr=""),
                        OSError("<private-path>"), subprocess.TimeoutExpired(["gh", "<private-argv>"], 60)):
            kwargs = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
            with self.subTest(outcome=type(outcome)), patch.object(release_preflight.subprocess, "run", **kwargs):
                report = self.audit(release_preflight.GitHub())
                self.assertFalse(report["configuration_ready"])
                output = (json.dumps(report) + release_preflight.render(report)).casefold()
                for marker in ("<private", "<invalid-private-json>"):
                    self.assertNotIn(marker, output)
        for key, value in (("labels", None), ("labels", [{"name": None}]), ("status", "unknown"), ("busy", None)):
            github = PreflightGitHub()
            github.responses["actions/runners"][0]["runners"][0][key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(self.audit(github)["configuration_ready"])

    def test_blocked_configuration_never_dispatches(self):
        github = PreflightGitHub()
        github.rows("actions/runners", "runners", [])
        report = self.run_preflight(github, expected_status=1, dispatch="0.1.0-rc.1")
        self.assertFalse(report["dispatch_attempted"])
        self.assertFalse(report["dispatch_confirmed"])
        self.assertFalse(any(body is not None for _, body in github.requests))

    def test_dispatch_uses_trusted_main_workflow_and_explicit_frozen_source(self):
        github = PreflightGitHub()
        report = self.run_preflight(github, dispatch="0.1.0-rc.1", source=COMMIT)
        self.assertTrue(report["dispatch_confirmed"])
        self.assertEqual(report["source_commit"], COMMIT)
        self.assertEqual(report["candidate_run_id"], 42)
        self.assertEqual(github.requests[-1], ("actions/workflows/release-candidate.yml/dispatches", {
            "ref": "main", "inputs": {"version": "0.1.0-rc.1", "source": COMMIT}, "return_run_details": True,
        }))
        self.assertNotIn("<private-", json.dumps(report) + release_preflight.render(report))
        self.assertIn("execution and validation are still pending", release_preflight.render(report))

    def test_main_is_frozen_before_dispatch_instead_of_passing_a_moving_branch(self):
        github = PreflightGitHub()
        report = self.run_preflight(github, dispatch="0.1.0-rc.1")
        self.assertEqual(report["source_commit"], PREFLIGHT_MAIN)
        self.assertEqual(github.requests[-1][1]["inputs"]["source"], PREFLIGHT_MAIN)
        self.assertFalse(any(path.startswith(("commits/", "compare/")) for path, _ in github.requests))

    def test_nonexistent_or_unrelated_sources_never_dispatch(self):
        for endpoint, value in (
                ("git/ref/heads/main", {}),
                ("commits/" + COMMIT, release_preflight.InspectionError("Commit unavailable")),
                ("commits/" + COMMIT, {"sha": PREFLIGHT_MAIN}),
                (f"compare/{COMMIT}...{PREFLIGHT_MAIN}", {"status": "behind", "behind_by": 1}),
                (f"compare/{COMMIT}...{PREFLIGHT_MAIN}", {"status": "diverged", "behind_by": 1}),
                (f"compare/{COMMIT}...{PREFLIGHT_MAIN}", {"status": "ahead", "behind_by": 0, "base_commit": {"sha": COMMIT}, "merge_base_commit": {"sha": PREFLIGHT_MAIN}})):
            github = PreflightGitHub()
            github.responses[endpoint] = value
            with self.subTest(endpoint=endpoint, value=value):
                report = self.run_preflight(github, expected_status=1, dispatch="0.1.0-rc.1", source=COMMIT)
                self.assertTrue(report["configuration_ready"])
                self.assertFalse(report["dispatch_attempted"])
                self.assertFalse(report["dispatch_confirmed"])
                self.assertFalse(any(body is not None for _, body in github.requests))

    def test_missing_or_invalid_run_id_is_not_a_confirmed_dispatch_and_is_not_retried(self):
        for response in (None, {}, {"workflow_run_id": True}, {"workflow_run_id": "42"}, {"workflow_run_id": 0}):
            github = PreflightGitHub()
            github.responses["actions/workflows/release-candidate.yml/dispatches"] = response
            with self.subTest(response=response):
                report = self.run_preflight(github, expected_status=1, dispatch="0.1.0-rc.1")
                self.assertTrue(report["dispatch_attempted"])
                self.assertFalse(report["dispatch_confirmed"])
                self.assertNotIn("candidate_run_id", report)
                self.assertEqual(sum(body is not None for _, body in github.requests), 1)

    def test_failed_dispatch_is_reported_without_retrying(self):
        github = PreflightGitHub()
        github.responses["actions/workflows/release-candidate.yml/dispatches"] = release_preflight.InspectionError("GitHub request failed")
        report = self.run_preflight(github, expected_status=1, dispatch="0.1.0-rc.1")
        self.assertTrue(report["configuration_ready"])
        self.assertTrue(report["dispatch_attempted"])
        self.assertFalse(report["dispatch_confirmed"])
        self.assertEqual(sum(body is not None for _, body in github.requests), 1)

    def test_unreadable_environment_does_not_create_duplicate_derived_failures(self):
        for value in ({}, [], release_preflight.InspectionError("Environment unavailable")):
            github = PreflightGitHub()
            github.responses["environments/release"] = value
            with self.subTest(value=value):
                report = self.audit(github)
                failures = [row["name"] for row in report["checks"] if not row["passed"]]
                self.assertEqual(failures, ["release environment API"])
                self.assertFalse(any(path == "environments/release/deployment-branch-policies" for path, _ in github.requests))
        github = PreflightGitHub()
        github.responses["environments/release/deployment-branch-policies"] = release_preflight.InspectionError("Branch policies unavailable")
        failures = [row["name"] for row in self.audit(github)["checks"] if not row["passed"]]
        self.assertEqual(failures, ["release branch policies API"])

    def test_dispatch_body_uses_json_stdin_and_requests_correlated_run_details(self):
        github = release_preflight.GitHub()
        response = SimpleNamespace(returncode=0, stdout='{"workflow_run_id":42}', stderr="")
        with patch.object(release_preflight.subprocess, "run", return_value=response) as command:
            self.assertEqual(release_preflight.dispatch_candidate(github, "owner/repo", "0.1.0-rc.1", COMMIT), 42)
        args, kwargs = command.call_args
        self.assertIn("--input", args[0])
        self.assertIn("X-GitHub-Api-Version: 2022-11-28", args[0])
        self.assertEqual(json.loads(kwargs["input"]), {
            "ref": "main", "inputs": {"version": "0.1.0-rc.1", "source": COMMIT}, "return_run_details": True,
        })

    def test_invalid_dispatch_inputs_are_rejected_before_api_or_workflow_calls(self):
        for args in ({"dispatch": "../0.1.0"}, {"dispatch": "0.1.0", "source": "feature-branch"},
                     {"source": COMMIT}, {"repo": "owner/../repo"}):
            github = PreflightGitHub()
            with self.subTest(args=args), patch.object(release_preflight, "GitHub", return_value=github), self.assertRaises(ValueError):
                release_preflight.run(self.args(**args))
            self.assertFalse(github.commands)

    def test_repository_uses_explicit_selection_then_actions_context_then_current_gh_repo(self):
        github = release_preflight.GitHub()
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "context/repo"}), patch.object(github, "run") as cli:
            self.assertEqual(release_preflight.repository("explicit/repo", github), "explicit/repo")
            self.assertEqual(release_preflight.repository(None, github), "context/repo")
            cli.assert_not_called()
        with patch.dict(os.environ, {}, clear=True), patch.object(github, "run", return_value="owner/repo\n") as cli:
            self.assertEqual(release_preflight.repository(None, github), "owner/repo")
            cli.assert_called_once_with(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"])


class ReleaseTest(unittest.TestCase):
    def test_ssh_controller_has_no_local_gpu_pid_namespace_or_reference_mounts(self):
        config = {"name": "zinc-release-intel-ssh", "mode": "ssh-controller", "uid": 10001, "gid": 10001,
                  "image": "sha256:" + "a" * 64, "gpu_devices": [], "gpu_groups": [],
                  "repository": "zolotukhin/zinc", "labels": "zinc-intel-ssh",
                  "mounts": [{"source": str(self.root / "key"), "target": "/run/secrets/qualification-key"}]}
        command = worker_launcher.container_args(config)
        self.assertNotIn("--pid=host", command)
        self.assertFalse(any(arg.startswith("--device=") or "VK_DRIVER" in arg or "LD_LIBRARY_PATH" in arg for arg in command))
        self.assertTrue(command[command.index("--mount") + 1].endswith(",readonly"))
        for changed in ({"gpu_devices": ["/dev/dri"]}, {"gpu_groups": [100]},
                        {"mounts": [{"source": str(self.root), "target": "/reference/source"}]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                worker_launcher.container_args({**config, **changed})

    def test_packaged_binaries_do_not_inherit_controller_or_actions_tokens(self):
        env = {"GH_TOKEN": "<secret>", "GITHUB_TOKEN": "<secret>", "ACTIONS_RUNTIME_TOKEN": "<secret>",
               "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "<secret>", "ZINC_RELEASE_WORKER_CONFIG": "<private-config>",
               "GITHUB_RUN_ID": "123", "PATH": os.defpath}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(release_gpu.packaged_environment(), {"GITHUB_RUN_ID": "123", "PATH": os.defpath})

    def test_worker_launcher_limits_host_access_and_keeps_inputs_read_only(self):
        config = {"name": "zinc-release-fixture", "uid": 10001, "gid": 10001, "image": "sha256:" + "a" * 64,
                  "gpu_devices": ["/dev/dri"], "gpu_groups": [100], "repository": "example/zinc", "labels": "zinc-fixture",
                  "mounts": [{"source": str(self.root / "model"), "target": "/models/qwen3-0.6b-q8-0/model.gguf"},
                             {"source": str(self.root / "work"), "target": "/runner"}]}
        args = worker_launcher.container_args(config)
        self.assertIn("--cap-drop=ALL", args)
        self.assertIn("--security-opt=no-new-privileges", args)
        self.assertIn("--read-only", args)
        # Package probes and the numerical reference execute from private /tmp.
        self.assertIn("exec", args[args.index("--tmpfs") + 1].split(":", 1)[1].split(","))
        mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "--mount"]
        self.assertTrue(mounts[0].endswith(",readonly"))
        self.assertFalse(mounts[1].endswith(",readonly"))
        for changed in ({"uid": 0}, {"gid": 0}, {"image": "ubuntu:latest"},
                        {"gpu_devices": ["/dev"]},
                        {"mounts": [{"source": "/", "target": "/runner"}]},
                        {"mounts": [{"source": "/var/run/docker.sock", "target": "/runner"}]},
                        {"mounts": [{"source": str(self.root), "target": "/etc"}]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                worker_launcher.container_args({**config, **changed})

    def test_reference_loader_uses_selected_build_without_changing_parent_environment(self):
        env = {"LD_LIBRARY_PATH": "/other-reference:/toolkit", "FIXTURE": "kept"}
        with patch.object(release_gpu.sys, "platform", "linux"):
            selected = release_gpu.reference_environment(Path("/selected-reference"), env)
        self.assertEqual(selected["LD_LIBRARY_PATH"], "/selected-reference:/other-reference:/toolkit")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/other-reference:/toolkit")
        self.assertEqual(selected["FIXTURE"], "kept")
        with patch.object(release_gpu.sys, "platform", "darwin"):
            self.assertEqual(release_gpu.reference_environment(Path("/selected-reference"), env), env)

    def test_packaged_environment_removes_inherited_model_and_library_overrides(self):
        inherited = {"PATH": "<system-tools>", "ZINC_PREFILL_CHUNK": "16", "ZINC_FORCE_CPU_ARGMAX": "1",
                     **{key: "<other-build>" for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_FALLBACK_LIBRARY_PATH")}}
        with patch.dict(os.environ, inherited, clear=True):
            self.assertEqual(release_gpu.packaged_environment(), {"PATH": "<system-tools>"})
            self.assertEqual(dict(os.environ), inherited)

    def test_packaged_kernel_gate_requires_cache_router_and_window_reference_suites(self):
        markers = {"validate-shared-cache": "PASS: 84 cache-boundary/reference/offset/stride/guard cases",
                   "validate-gemma-router": "PASS: 36 Gemma router workgroup/reference/guard/replay cases",
                   "validate-attention-window": "PASS: 256 sliding-window/page/reference/guard/replay cases"}
        for missing in (None, *markers):
            launched, reaped = [], []
            @contextlib.contextmanager
            def child(command, log, cwd, env):
                name = Path(command[0]).name
                launched.append(name)
                Path(log).write_text(markers[name] if name != missing else "incomplete suite")
                try:
                    yield SimpleNamespace(wait=lambda timeout: 0)
                finally:
                    reaped.append(name)
            with self.subTest(missing=missing), patch.object(release_gpu, "output", return_value=""), patch.object(release_gpu, "process", side_effect=child):
                if missing is None:
                    release_gpu.kernel_reference(self.root, self.root, self.root, 0, self.root, {})
                else:
                    with self.assertRaisesRegex(ValueError, "reference suite was incomplete"):
                        release_gpu.kernel_reference(self.root, self.root, self.root, 0, self.root, {})
            self.assertEqual(launched, list(markers) if missing is None else list(markers)[:list(markers).index(missing) + 1])
            self.assertEqual(reaped, launched)

    def test_intel_q8_gate_requires_all_cases_and_records_the_tested_package_shaders(self):
        shader_dir = self.root / "share/zinc/shaders"
        shader_dir.mkdir(parents=True)
        for name in ("dmmv_q8_0", "dmmv_q8_0_intel_rows"):
            (shader_dir / (name + ".spv")).write_bytes(name.encode())
        marker = "PASS: 180 Q8 native-reference/offset/accumulation/guard cases\nPASS: 90 Q8 identical-reduction pairs"
        for status, text in ((1, marker), (0, ""), (0, marker.replace("180", "179")),
                             (0, marker + "\n" + marker), (0, marker.replace("90", "89")),
                             (0, marker.replace("180", "60").replace("90", "30")),
                             (0, marker.splitlines()[0]), (0, marker)):
            reaped = []
            @contextlib.contextmanager
            def child(command, log, cwd, env):
                self.assertEqual(Path(command[0]).name, "validate-q8-decode")
                self.assertEqual(env["ZINC_GPU"], "3")
                self.assertEqual(env["ZINC_SHADER_DIR"], str(shader_dir))
                Path(log).write_text(text + "\n")
                try:
                    yield SimpleNamespace(wait=lambda timeout: status)
                finally:
                    reaped.append(True)
            with self.subTest(status=status, text=text), patch.object(release_gpu, "output", return_value="") as compile_tool, \
                    patch.object(release_gpu, "process", side_effect=child):
                if status or text != marker:
                    with self.assertRaisesRegex(ValueError, "reference validation failed|reference suite was incomplete"):
                        release_gpu.q8_decode_reference(self.root, self.root, self.root, 3, self.root, {})
                else:
                    result = release_gpu.q8_decode_reference(self.root, self.root, self.root, 3, self.root, {})
                    self.assertEqual(result["cases"], 180)
                    self.assertEqual(result["bitwise_pairs"], 90)
                    self.assertTrue(result["passed"])
                    self.assertEqual(result["shader_sha256"], {name: sha256(shader_dir / (name + ".spv"))
                                     for name in ("dmmv_q8_0", "dmmv_q8_0_intel_rows")})
                    self.assertEqual(result["validator_source_sha256"], sha256(ROOT / "tools/validate_q8_decode.c"))
                    self.assertEqual(result["shared_fixture_source_sha256"], sha256(ROOT / "tools/validate_dmmv_shared_cache.c"))
            self.assertEqual(compile_tool.call_count, 1)
            self.assertEqual(reaped, [True])

    def test_rocm_kernel_gate_rejects_wrong_embedded_source_failure_and_incomplete_success(self):
        (self.root / "bin").mkdir()
        binary = self.root / "bin/zinc"
        binary.write_bytes(b"another archive's embedded kernels")
        with patch.object(release_gpu, "output") as compile_tool, self.assertRaisesRegex(ValueError, "kernel source differs"):
            release_gpu.rocm_kernel_reference(self.root, self.root, self.root, 3, self.root, {})
        compile_tool.assert_not_called()
        binary.write_bytes(b"binary prefix" + (ROOT / "src/shaders/cuda/kernels.cu").read_bytes() + b"\0suffix")
        marker = "PASS 305 ROCm partial-tile/native-reference/route/quantizer/guard cases"
        for status, text in ((1, marker), (0, "PASS 304 ROCm partial-tile/native-reference/route/quantizer/guard cases"), (0, marker)):
            reaped = []
            @contextlib.contextmanager
            def child(command, log, cwd, env):
                self.assertEqual(cwd, ROOT)
                self.assertEqual(env["ZINC_GPU"], "3")
                Path(log).write_text(text + "\n")
                try:
                    yield SimpleNamespace(wait=lambda timeout: status)
                finally:
                    reaped.append(True)
            with self.subTest(status=status, text=text), patch.object(release_gpu, "output", return_value="") as compile_tool, \
                    patch.object(release_gpu, "process", side_effect=child):
                if status or text != marker:
                    with self.assertRaisesRegex(ValueError, "reference validation failed|reference suite was incomplete"):
                        release_gpu.rocm_kernel_reference(self.root, self.root, self.root, 3, self.root, {})
                else:
                    result = release_gpu.rocm_kernel_reference(self.root, self.root, self.root, 3, self.root, {})
                    self.assertEqual(result, {"passed": True, "cases": 305,
                        "kernel_source_sha256": sha256(ROOT / "src/shaders/cuda/kernels.cu"),
                        "validator_source_sha256": sha256(ROOT / "tools/validate_rocm_partial_tiles.c")})
            self.assertEqual(compile_tool.call_count, 3)
            self.assertEqual(reaped, [True])

    def test_sliding_window_reference_measures_prompts_beyond_the_mask_boundary(self):
        samples = [("rendered prompt " + str(i), {"comparison": {"tokens": 16, "near_ties": []}}) for i in range(3)]
        for count in (1025, 1024):
            with self.subTest(count=count), patch.object(release_gpu, "inference_parity", return_value=samples) as native, \
                    patch.object(release_gpu, "output", return_value=json.dumps([1] * count)), \
                    patch.object(release_gpu, "process", return_value=contextlib.nullcontext(SimpleNamespace())), \
                    patch.object(release_gpu, "ready"), patch.object(release_gpu, "free_port", return_value=0), \
                    patch.object(release_gpu, "json_request", return_value={"data": [{"id": "fixture", "active": True}]}), \
                    patch.object(release_gpu, "reference_http_parity") as http:
                if count == 1025:
                    result = release_gpu.sliding_window_reference(self.root, self.root, self.root / "fixture.gguf",
                        4096, 0, [], self.root, {}, 1024)
                    self.assertTrue(result["passed"])
                    self.assertEqual(result["minimum_prompt_tokens"], count)
                    self.assertEqual(result["reference_agreement"]["exact_tokens"], 48)
                    self.assertTrue(result["http_passed"])
                    http.assert_called_once()
                else:
                    with self.assertRaisesRegex(ValueError, "boundary was not exercised"):
                        release_gpu.sliding_window_reference(self.root, self.root, self.root / "fixture.gguf",
                            4096, 0, [], self.root, {}, 1024)
                prompts = native.call_args.kwargs["prompts"]
                self.assertEqual(len(prompts), 3)
                self.assertTrue(all(prompt.count("word ") == 1100 for prompt in prompts))

    def test_invalid_sliding_window_context_does_not_start_model_work(self):
        with patch.object(release_gpu, "inference_parity") as native:
            for window in (None, True, 0, -1, 4000):
                with self.subTest(window=window), self.assertRaisesRegex(ValueError, "invalid sliding-window"):
                    release_gpu.sliding_window_reference(self.root, self.root, self.root / "fixture.gguf",
                        4096, 0, [], self.root, {}, window)
            native.assert_not_called()

    def test_worker_guard_accepts_only_the_admin_selected_main_workflow_job(self):
        workflow = "example/zinc/.github/workflows/release-worker-check.yml@refs/heads/main"
        config = {"repository": "example/zinc", "actors": ["maintainer"],
                  "workflows": {workflow: {"events": ["workflow_dispatch"], "jobs": ["check"]}}}
        env = {"GITHUB_REPOSITORY": "example/zinc", "GITHUB_REF": "refs/heads/main", "GITHUB_ACTOR": "maintainer",
               "GITHUB_SHA": COMMIT, "GITHUB_RUN_ID": "123", "GITHUB_WORKFLOW_REF": workflow,
               "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_JOB": "check"}
        release_worker_guard.validate_context(config, env)
        for key in env:
            for value in ("", "<untrusted>"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    release_worker_guard.validate_context(config, {**env, key: value})
        with self.assertRaisesRegex(ValueError, "event"):
            release_worker_guard.validate_context(config, {**env, "GITHUB_EVENT_NAME": "pull_request_target"})

    def test_worker_preview_resolution_rejects_untrusted_failed_or_unproven_bytes(self):
        profile = release_worker.profile_for("vulkan-rdna4")
        run = {"id": 123, "repository": {"full_name": "zolotukhin/zinc"}, "path": ".github/workflows/release-preview.yml",
               "event": "push", "head_branch": "main", "status": "completed", "conclusion": "success", "head_sha": COMMIT}
        variables = {profile["runner_variable"]: '["self-hosted","linux","x64","zinc-rdna4"]'}
        result = release_worker.resolve_preview(profile, run, variables, COMMIT)
        self.assertEqual(result["version"], "0.1.0-dev.123")
        self.assertEqual(result["validation_commit"], COMMIT)
        self.assertEqual(result["runner_variable"], "ZINC_RUNNER_RDNA4")
        for changed in ({"event": "pull_request"}, {"head_branch": "feature"}, {"conclusion": "failure"},
                        {"head_sha": "2" * 40}, {"path": ".github/workflows/other.yml"}, {"id": True},
                        {"repository": None}, {"repository": {"full_name": "example/zinc"}}, {"path": None}, {"head_sha": None}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                release_worker.resolve_preview(profile, {**run, **changed}, variables, COMMIT)
        for labels in (None, "null", "[]", '["<private-selector>"]'):
            with self.subTest(labels=labels), self.assertRaises((ValueError, TypeError)):
                release_worker.resolve_preview(profile, run, {profile["runner_variable"]: labels}, COMMIT)

    def test_worker_older_preview_needs_immutable_ancestry_and_identical_policy(self):
        profile = release_worker.profile_for("vulkan-rdna4")
        run = {"id": 123, "repository": {"full_name": "zolotukhin/zinc"}, "path": ".github/workflows/release-preview.yml",
               "event": "workflow_dispatch", "head_branch": "main", "status": "completed", "conclusion": "success", "head_sha": COMMIT}
        variables = {profile["runner_variable"]: '["self-hosted","linux","x64","zinc-rdna4"]'}
        comparison = {"status": "ahead", "behind_by": 0, "base_commit": {"sha": COMMIT}, "merge_base_commit": {"sha": COMMIT}}
        digest = sha256(ROOT / "config/release/targets.json")
        result = release_worker.resolve_preview(profile, run, variables, PREFLIGHT_MAIN, comparison, digest)
        self.assertEqual((result["commit"], result["validation_commit"]), (COMMIT, PREFLIGHT_MAIN))
        for changed in (None, [], {}, {"status": "behind"}, {"status": "diverged"}, {"behind_by": 1},
                        {"behind_by": False}, {"base_commit": None}, {"base_commit": {"sha": PREFLIGHT_MAIN}},
                        {"merge_base_commit": {"sha": PREFLIGHT_MAIN}}):
            evidence = {**comparison, **changed} if isinstance(changed, dict) and changed else changed
            with self.subTest(comparison=changed), self.assertRaisesRegex(ValueError, "verified ancestor"):
                release_worker.resolve_preview(profile, run, variables, PREFLIGHT_MAIN, evidence, digest)
        for changed in (None, "", "0" * 64):
            with self.subTest(digest=changed), self.assertRaisesRegex(ValueError, "policy differs"):
                release_worker.resolve_preview(profile, run, variables, PREFLIGHT_MAIN, comparison, changed)

    def test_worker_resolver_fetches_frozen_policy_only_after_trusted_ancestry(self):
        run = {"id": 123, "repository": {"full_name": "zolotukhin/zinc"}, "path": ".github/workflows/release-preview.yml",
               "event": "push", "head_branch": "main", "status": "completed", "conclusion": "success", "head_sha": COMMIT}
        comparison = {"status": "ahead", "behind_by": 0, "base_commit": {"sha": COMMIT}, "merge_base_commit": {"sha": COMMIT}}
        frozen = {"type": "file", "path": "config/release/targets.json", "encoding": "base64",
                  "content": base64.encodebytes((ROOT / "config/release/targets.json").read_bytes()).decode()}
        args = SimpleNamespace(profile="vulkan-rdna4", preview_run="123")
        for failure in (None, "same_commit", "untrusted", "failed", "identity", "diverged", "policy", "malformed"):
            destination = self.root / (str(failure) + ".outputs")
            metadata = {**run, **({"event": "pull_request"} if failure == "untrusted" else
                                    {"conclusion": "failure"} if failure == "failed" else {"id": 456} if failure == "identity" else {})}
            ancestry = {**comparison, **({"status": "diverged"} if failure == "diverged" else {})}
            policy_file = {**frozen, **({"content": base64.b64encode(b"{}").decode()} if failure == "policy" else
                                      {"content": "%%invalid%%"} if failure == "malformed" else {})}
            with self.subTest(failure=failure), patch.dict(os.environ, {
                "GITHUB_SHA": COMMIT if failure == "same_commit" else PREFLIGHT_MAIN, "GITHUB_OUTPUT": str(destination),
                "RELEASE_VARIABLES": json.dumps({"ZINC_RUNNER_RDNA4": '["self-hosted","linux","x64","zinc-rdna4"]'}),
            }), patch.object(release_gpu, "output", side_effect=[COMMIT if failure == "same_commit" else PREFLIGHT_MAIN, ""]) as git, \
                    patch.object(release_worker, "github_json", side_effect=[metadata, ancestry, policy_file]) as api, contextlib.redirect_stdout(io.StringIO()):
                if failure in (None, "same_commit"):
                    release_worker.resolve(args)
                    output = dict(line.split("=", 1) for line in destination.read_text().splitlines())
                    self.assertEqual(output["commit"], COMMIT)
                    self.assertEqual(output["validation_commit"], COMMIT if failure else PREFLIGHT_MAIN)
                else:
                    with self.assertRaises(ValueError): release_worker.resolve(args)
                    self.assertFalse(destination.exists())
                expected = ["actions/runs/123"]
                if failure not in ("same_commit", "untrusted", "failed", "identity"):
                    expected += [f"compare/{COMMIT}...{PREFLIGHT_MAIN}?per_page=1"]
                    if failure != "diverged": expected += [f"contents/config/release/targets.json?ref={COMMIT}"]
                self.assertEqual([call.args[0] for call in api.call_args_list], expected)
                self.assertEqual(git.call_args_list[1].args[0], ["git", "-C", str(release_worker.ROOT), "status", "--porcelain", "--untracked-files=all"])

    def test_worker_resolver_rejects_wrong_or_dirty_checkout_before_using_its_policy(self):
        for failure in ("commit", "tracked", "untracked"):
            output = [COMMIT] if failure == "commit" else [PREFLIGHT_MAIN, " M tools/release_worker.py" if failure == "tracked" else "?? tools/unexpected.py"]
            with self.subTest(failure=failure), patch.dict(os.environ, {"GITHUB_SHA": PREFLIGHT_MAIN}), \
                    patch.object(release_gpu, "output", side_effect=output), patch.object(release_worker, "github_json") as api, \
                    self.assertRaisesRegex(ValueError, "validation checkout"):
                release_worker.resolve(SimpleNamespace(profile="vulkan-rdna4", preview_run="123"))
            api.assert_not_called()

    def test_worker_cli_requires_an_explicit_validation_commit(self):
        with patch.object(release_gpu, "output", return_value=COMMIT), self.assertRaises(ValueError):
            release_worker.validation_identity(None)
        result = subprocess.run([sys.executable, str(ROOT / "tools/release_worker.py"), "check", "--profile", "vulkan-rdna4",
                                 "--directory", str(self.root), "--version", "9.8.7", "--commit", COMMIT,
                                 "--output", str(self.root / "absent.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--validation-commit", result.stderr)
        self.assertFalse((self.root / "absent.json").exists())

    def test_worker_frozen_policy_rejects_non_file_and_unreadable_api_responses(self):
        response = {"type": "file", "path": "config/release/targets.json", "encoding": "base64", "content": "e30="}
        for changed in (None, [], {}, {"type": "dir"}, {"path": "other.json"}, {"encoding": "none"},
                        {"content": None}, {"content": "<invalid>"}):
            value = {**response, **changed} if isinstance(changed, dict) and changed else changed
            with self.subTest(response=changed), self.assertRaises(ValueError):
                release_worker.policy_digest_from_response(value)

    def test_worker_reservation_never_removes_an_occupied_reservation(self):
        path = self.root / "reservation"
        path.mkdir()
        with self.assertRaisesRegex(ValueError, "reserved"):
            with release_worker.reservation(path):
                self.fail("occupied reservation was entered")
        self.assertTrue(path.is_dir())
        path.rmdir()
        with patch.object(release_worker, "idle"), self.assertRaisesRegex(ValueError, "fixture"):
            with release_worker.reservation(path):
                raise ValueError("fixture")
        self.assertFalse(path.exists())

    def test_readiness_idle_checks_include_benchmarks_and_never_interrupt_processes(self):
        def status(command, **kwargs):
            self.assertEqual(command[:2], ["pgrep", "-x"])
            return SimpleNamespace(returncode=0 if command[2] == "llama-bench" else 1)
        with patch.object(release_worker.subprocess, "run", side_effect=status) as process, self.assertRaisesRegex(ValueError, "busy"):
            release_worker.idle()
        self.assertEqual({call.args[0][2] for call in process.call_args_list}, {"zinc", "llama-server", "llama-cli", "llama-bench"})

    def test_worker_reservation_refuses_a_live_owner_and_unknown_marked_contents(self):
        path = self.root / "reservation"
        with release_lib.worker_reservation(path):
            with self.assertRaisesRegex(ValueError, "reserved"):
                with release_lib.worker_reservation(path):
                    self.fail("live owner's reservation was entered")
            self.assertTrue(path.is_dir())
        self.assertFalse(path.exists())
        self.assertTrue((self.root / ".reservation.flock").is_file())
        path.mkdir()
        marker = path / ".zinc-flock-reservation-v1"
        marker.write_text("zinc-flock-reservation-v1\n")
        (path / "unrecognized").write_text("another owner's data")
        with self.assertRaisesRegex(ValueError, "legacy reservation"):
            with release_lib.worker_reservation(path):
                self.fail("unknown reservation contents were accepted")
        self.assertEqual((path / "unrecognized").read_text(), "another owner's data")

    def test_worker_reservation_recovers_a_hard_killed_marked_owner_without_interrupting_a_process(self):
        path = self.root / "hard-kill-reservation"
        code = ("import sys,time\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\n"
                "from release_lib import worker_reservation\n"
                "with worker_reservation(Path(sys.argv[2])):\n print('ready',flush=True)\n time.sleep(60)\n")
        child = subprocess.Popen([sys.executable, "-c", code, str(ROOT / "tools"), str(path)],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            self.assertTrue(select.select([child.stdout], [], [], 10)[0], "CPU reservation fixture did not start")
            self.assertEqual(child.stdout.readline(), b"ready\n")
            child.kill()
            child.wait(timeout=5)
            self.assertTrue(path.is_dir())
            with release_lib.worker_reservation(path):
                self.assertTrue(path.is_dir())
            self.assertFalse(path.exists())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdout.close()
            child.stderr.close()

    def test_reservation_atomic_publish_never_replaces_an_empty_legacy_owner(self):
        source, occupied = self.root / "prepared", self.root / "occupied"
        source.mkdir()
        occupied.mkdir()
        (source / "marker").write_text("prepared marker")
        with self.assertRaises(FileExistsError):
            release_lib.rename_worker_directory(source, occupied)
        self.assertEqual(list(occupied.iterdir()), [])
        self.assertTrue((source / "marker").is_file())

    def test_reservation_survives_hard_kills_during_preparation_and_retirement(self):
        for boundary in ("before_publish", "after_retire"):
            path = self.root / boundary
            code = ("import os,signal,sys\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\n"
                    "import release_lib as lib\noriginal=lib.rename_worker_directory\ncalls=0\n"
                    "def crash(source,target):\n global calls\n calls+=1\n"
                    " if calls==1 and sys.argv[3]=='before_publish':os.kill(os.getpid(),signal.SIGKILL)\n"
                    " original(source,target)\n"
                    " if calls==2 and sys.argv[3]=='after_retire':os.kill(os.getpid(),signal.SIGKILL)\n"
                    "lib.rename_worker_directory=crash\n"
                    "with lib.worker_reservation(Path(sys.argv[2])):pass\n")
            result = subprocess.run([sys.executable, "-c", code, str(ROOT / "tools"), str(path), boundary],
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, -9, boundary)
            self.assertFalse(path.exists(), boundary)
            with release_lib.worker_reservation(path):
                self.assertTrue(path.is_dir())
            self.assertFalse(path.exists())

    def test_worker_readiness_schema_cannot_be_aggregated_as_release_evidence(self):
        reports = self.root / "reports"
        reports.mkdir()
        write_json(reports / "worker.json", {"schema_version": release_worker.SCHEMA, "passed": True,
                                            "release_qualification": False})
        with patch.object(release, "verify_candidate", return_value={"artifacts": [], "required_profiles": []}), \
             self.assertRaisesRegex(ValueError, "unsupported GPU report schema"):
            release.aggregate(SimpleNamespace(directory=self.root, results=reports, version="9.8.7", commit=COMMIT))
        self.assertFalse((self.root / "validation.json").exists())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="zinc release test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def tree(self, release_version="9.8.7", exit_code=0):
        spec = target(NATIVE)
        tree = self.root / "release" / ("v" + release_version) / archive_name(release_version, NATIVE).removesuffix(".tar.gz")
        (tree / "bin").mkdir(parents=True)
        report = (f"zinc {release_version}\ncommit: {COMMIT}\n"
                  f"target: {'aarch64-macos-none' if sys.platform == 'darwin' else 'x86_64-linux-gnu'}\n"
                  f"optimize: fast\nbackends: {spec['backend']}\n")
        source = self.root / "fixture.c"
        source.write_text("#include <stdio.h>\nint main(void) { fputs(" + json.dumps(report) + f", stdout); return {exit_code}; }}\n")
        flags = ["-mmacosx-version-min=15.0", "-framework", "Metal", "-framework", "Foundation"] if sys.platform == "darwin" else ["-Wl,--no-as-needed", "-lvulkan"]
        subprocess.run(["cc", str(source), "-o", str(tree / "bin/zinc"), *flags], check=True, capture_output=True)
        shutil.copy2(tree / "bin/zinc", tree / "bin/zinc-tokenize")
        asset = "share/zinc/shaders/metal/fixture.metal" if sys.platform == "darwin" else "share/zinc/shaders/fixture.spv"
        (tree / asset).parent.mkdir(parents=True)
        (tree / asset).write_text("fixture shader\n")
        write_json(tree / "share/zinc/asset-manifest.json", {"schema_version": 1, "backend": spec["backend"], "files": [asset]})
        for name in ["README.md", "LICENSE"]:
            (tree / name).write_text(name + "\n")
        meta = {"schema_version": 1, "name": "zinc", "version": release_version, "tag": "v" + release_version,
                "commit": COMMIT, "target": NATIVE, "backend": spec["backend"], "cpu": "baseline",
                "optimize": "ReleaseFast", "minimum_os": spec["minimum_os"], "minimum_glibc": spec["minimum_glibc"]}
        write_json(tree / "VERSION.json", meta)
        receipt = {"format": "1", "version": release_version, "commit": COMMIT, "target": NATIVE,
                   "backend": spec["backend"], "minimum_os": spec["minimum_os"], "minimum_glibc": spec["minimum_glibc"],
                   "binary_sha256": sha256(tree / "bin/zinc")}
        (tree / "INSTALL-METADATA").write_text("".join(f"{k}={v}\n" for k, v in receipt.items()))
        self.refresh(tree)
        return tree

    def refresh(self, tree):
        if (tree / "FILES.sha256").exists():
            (tree / "FILES.sha256").unlink()
        (tree / "FILES.sha256").write_text("".join(f"{h}  {f}\n" for f, h in inventory(tree).items()))

    def bundle(self, tree, extras=(), duplicate_checksum=False):
        archive = tree.parent / (tree.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(tree, arcname=tree.name)
            for member in extras:
                stream.addfile(member, io.BytesIO(b"x" * member.size) if member.isfile() else None)
        line = sha256(archive) + "  " + archive.name + "\n"
        (tree.parent / "SHA256SUMS.txt").write_text(line * (2 if duplicate_checksum else 1))
        return archive

    def install(self, release_version="9.8.7", *args, **overrides):
        env = {**os.environ, "ZINC_VERSION": "v" + release_version,
               "ZINC_BASE_URL": (self.root / "release").as_uri(),
               "ZINC_INSTALL_DIR": str(self.root / "installation"), "ZINC_BIN_DIR": str(self.root / "commands")}
        env.pop("ZINC_BACKEND", None)
        env.update(overrides)
        return subprocess.run(["bash", str(ROOT / "scripts/install.sh"), *args], cwd=self.root,
                              env=env, capture_output=True, text=True, timeout=60)

    def assert_installed(self, expected):
        self.assertEqual((self.root / "installation/current").resolve().name,
                         archive_name(expected, NATIVE).removesuffix(".tar.gz"))

    def test_semver_rejects_paths_shell_text_and_numeric_prerelease_zeroes(self):
        for value in ["../0.1.0", "v0.1.0", "01.0.0", "1.2", "0.1.0-01", "0.1.0\n", "$(touch marker)"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                version(value)
        for value in ["0.1.0", "0.1.0-rc.1", "0.1.0+build-01", "1.2.3-0"]:
            self.assertEqual(version(value), value)

    def test_policy_keeps_preview_backends_but_requires_only_the_linux_launch_matrix(self):
        data = policy()
        self.assertEqual({x["backend"] for x in data["targets"]}, {"metal", "vulkan", "rocm"})
        self.assertEqual(len(data["gpu_profiles"]), 9)
        scoped = release_lib.release_policy(data)
        self.assertEqual([t["id"] for t in scoped["targets"]], ["linux-x86_64-vulkan", "linux-x86_64-rocm"])
        self.assertEqual([p["id"] for p in scoped["gpu_profiles"]], ["vulkan-rdna4", "vulkan-intel-xe2", "rocm-rdna4"])
        self.assertFalse(any(t["signing_required"] for t in scoped["targets"]))

    def test_vulkan_release_roles_include_the_nextn_model_that_exposes_verifier_drift(self):
        # The earlier six-model matrix omitted the catalog's NextN-bearing
        # Qwen 3.8. Its default decode must pass the same native reference gate.
        rows = release_lib.release_policy()["gpu_profiles"]
        for identifier in ("vulkan-rdna4", "vulkan-intel-xe2"):
            with self.subTest(profile=identifier):
                profile = next(row for row in rows if row["id"] == identifier)
                self.assertIn("qwen38-27b-q4k-m", profile["models"])

    def test_release_scope_rejects_unknown_duplicate_empty_or_uncovered_entries(self):
        data = policy()
        for scope in (None, [], {}, {"targets": [], "gpu_profiles": []},
                      {"targets": ["unknown"], "gpu_profiles": ["vulkan-rdna4"]},
                      {"targets": ["linux-x86_64-vulkan"], "gpu_profiles": ["vulkan-rdna4", "vulkan-rdna4"]},
                      {"targets": ["linux-x86_64-vulkan"], "gpu_profiles": [None]},
                      {"targets": ["linux-x86_64-vulkan"], "gpu_profiles": ["rocm-rdna4"]},
                      {"targets": ["linux-x86_64-vulkan", "linux-x86_64-rocm"], "gpu_profiles": ["vulkan-rdna4"]}):
            write_json(self.root / "policy.json", {**data, "release": scope})
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                release_lib.policy(self.root / "policy.json")

    def test_resolution_freezes_the_selected_matrix_and_only_requires_signing_for_selected_targets(self):
        for include_metal in (False, True):
            data = policy()
            if include_metal:
                data.pop("release")
            variables = {p["runner_variable"]: '["self-hosted","zinc-fixture"]' for p in release_lib.release_policy(data)["gpu_profiles"]}
            output = self.root / (str(include_metal) + ".outputs")
            with self.subTest(include_metal=include_metal), patch.object(release, "policy", return_value=data), \
                    patch.object(release, "command_output", side_effect=["", "1234567"]), \
                    patch.object(release, "frozen_checkout", return_value=COMMIT), patch.object(release, "git_revision", return_value=COMMIT), \
                    patch.object(release, "require_source_ancestor"), patch.object(release, "verify_release_sources"), \
                    patch.dict(os.environ, {"RELEASE_VARIABLES": json.dumps(variables), "GITHUB_OUTPUT": str(output)}), contextlib.redirect_stdout(io.StringIO()):
                release.resolve(SimpleNamespace(version="0.1.0-rc.1", ref="main"))
            values = dict(line.split("=", 1) for line in output.read_text().splitlines())
            self.assertEqual(values["signing_required"], str(include_metal).lower())
            self.assertEqual(len(json.loads(values["build_matrix"])["include"]), 3 if include_metal else 2)
            groups = json.loads(values["gpu_matrix"])["include"]
            self.assertEqual(len(groups), 8 if include_metal else 2)
            self.assertEqual(sum(len(group["profiles"]) for group in groups), 9 if include_metal else 3)
            self.assertEqual(next(group for group in groups if group["id"] == "rdna4")["profiles"], ["vulkan-rdna4", "rocm-rdna4"])
            self.assertEqual(next(group for group in groups if group["id"] == "rdna4")["timeout_minutes"], 315)
            self.assertEqual(next(group for group in groups if group["id"] == "intel-xe2")["timeout_minutes"], 165)
            self.assertTrue(all(group["timeout_minutes"] > 150 * len(group["profiles"]) for group in groups))
        with contextlib.redirect_stdout(io.StringIO()) as preview:
            release.matrix(SimpleNamespace(kind="build"))
        self.assertEqual(len(json.loads(preview.getvalue())["include"]), 3)

    def test_shared_worker_profiles_run_sequentially_and_failure_stops_the_group(self):
        for fail_at in (None, 0, 1, "interrupt", "timeout"):
            launched, active, workdirs = [], [], []
            @contextlib.contextmanager
            def child(command, log, work, env, grace_seconds):
                self.assertEqual(grace_seconds, release_gpu_group.CHILD_CLEANUP_GRACE_SECONDS)
                self.assertFalse(active)
                profile = command[command.index("--profile") + 1]
                self.assertTrue(Path(command[command.index("--directory") + 1]).is_absolute())
                self.assertEqual(Path(command[command.index("--output") + 1]).name, profile + ".json")
                self.assertTrue(Path(work).is_dir())
                workdirs.append(Path(work))
                index = len(launched)
                launched.append(profile)
                active.append(profile)
                def wait(timeout):
                    self.assertEqual(timeout, 150 * 60)
                    if fail_at == "interrupt":
                        raise KeyboardInterrupt()
                    if fail_at == "timeout":
                        raise subprocess.TimeoutExpired(command, timeout)
                    return int(index == fail_at)
                try:
                    yield SimpleNamespace(wait=wait)
                finally:
                    active.remove(profile)
            with self.subTest(fail_at=fail_at), patch.object(release_gpu_group.gpu, "process", side_effect=child), contextlib.redirect_stdout(io.StringIO()):
                if fail_at is None:
                    release_gpu_group.check_group("ZINC_RUNNER_RDNA4", self.root, self.root / "passed")
                elif fail_at == "interrupt":
                    with self.assertRaises(KeyboardInterrupt):
                        release_gpu_group.check_group("ZINC_RUNNER_RDNA4", self.root, self.root / "interrupted")
                elif fail_at == "timeout":
                    with self.assertRaisesRegex(ValueError, "timed out after 150 minutes: vulkan-rdna4"):
                        release_gpu_group.check_group("ZINC_RUNNER_RDNA4", self.root, self.root / "timed-out")
                else:
                    with self.assertRaisesRegex(ValueError, "packaged GPU profile failed"):
                        release_gpu_group.check_group("ZINC_RUNNER_RDNA4", self.root, self.root / ("failed-" + str(fail_at)))
            self.assertFalse(active)
            self.assertTrue(all(not path.exists() for path in workdirs))
            self.assertEqual(launched, ["vulkan-rdna4"] if fail_at in (0, "interrupt", "timeout") else ["vulkan-rdna4", "rocm-rdna4"])
        with patch.object(release_gpu_group.gpu, "process") as process, self.assertRaisesRegex(ValueError, "unknown release worker role"):
            release_gpu_group.check_group("ZINC_RUNNER_METAL_APPLE7", self.root, self.root / "preview-only")
        process.assert_not_called()

    def test_rocm_archive_resolves_the_standard_runtime_without_shell_setup(self):
        binary = self.root / "fixture-elf"
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        header[18:20] = (62).to_bytes(2, "little")
        binary.write_bytes(header)
        needed = "\n".join(f"(NEEDED) Shared library: [{name}]" for name in
                           ["libc.so.6", "libamdhip64.so.7", "libhiprtc.so.7", "libhipblas.so.3"])
        for paths, valid in [(None, False), ("/opt/rocm/lib", True),
                             ("$ORIGIN:/opt/rocm/lib", True), ("/tmp/toolkit/lib", False)]:
            dynamic = needed + (f"\n(RUNPATH) Library runpath: [{paths}]" if paths else "")
            def inspect(command):
                return "GLIBC_2.39" if "--version-info" in command else dynamic
            with self.subTest(paths=paths), patch.object(release, "command_output", side_effect=inspect):
                if valid:
                    self.assertEqual(release.inspect_binary(binary, target("linux-x86_64-rocm"))["runtime_paths"], [paths])
                else:
                    with self.assertRaises(ValueError):
                        release.inspect_binary(binary, target("linux-x86_64-rocm"))

    def test_missing_gpu_configuration_fails_instead_of_skipping(self):
        with patch.dict(os.environ, {"RELEASE_VARIABLES": "{}"}):
            with self.assertRaisesRegex(ValueError, "configuration missing"):
                release.matrix(type("Args", (), {"kind": "gpu"})())

    def test_package_and_verify_outside_checkout_with_real_native_executable(self):
        tree = self.tree()
        # Package the fixture as an install prefix, so the real asset lookup and
        # architecture/version checks run. It does not initialize any GPU.
        result = subprocess.run(["bash", str(ROOT / "scripts/package_release.sh"), "--version", "9.8.7",
                                 "--target", NATIVE, "--prefix", str(tree), "--dist", str(self.root / "dist"),
                                 "--commit", COMMIT], cwd=self.root,
                                env={**os.environ, "SOURCE_DATE_EPOCH": "1700000000"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        archive = self.root / "dist" / archive_name("9.8.7", NATIVE)
        extracted = extract_archive(archive, self.root / "extract", "9.8.7", NATIVE, COMMIT)
        self.assertEqual(check_metadata(extracted, "9.8.7", NATIVE)["commit"], COMMIT)
        self.assertTrue((extracted / "bin/zinc").stat().st_mode & 0o111)

    def test_package_refuses_source_identity_mismatch_and_existing_archive(self):
        tree = self.tree()
        args = ["python3", str(ROOT / "tools/release.py"), "package", "--version", "9.8.7", "--target", NATIVE,
                "--prefix", str(tree), "--dist", str(self.root / "dist"), "--commit"]
        wrong = subprocess.run([*args, "2" * 40], capture_output=True, text=True)
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("commit/backend", wrong.stderr)

    def test_archive_links_traversal_and_duplicates_are_rejected_before_extraction(self):
        tree = self.tree()
        for index, (name, kind) in enumerate([("../escape", tarfile.REGTYPE),
                                            (tree.name + "/linked", tarfile.SYMTYPE),
                                            (tree.name + "/hard", tarfile.LNKTYPE),
                                            (tree.name + "/VERSION.json", tarfile.REGTYPE)]):
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "/outside"
            member.size = 1 if kind == tarfile.REGTYPE else 0
            archive = self.bundle(tree, [member])
            dest = self.root / ("unsafe-" + str(index))
            with self.subTest(name=name), self.assertRaises(ValueError):
                extract_archive(archive, dest, "9.8.7", NATIVE)
            self.assertFalse(any(dest.iterdir()))

    def test_file_inventory_detects_missing_and_modified_runtime_assets(self):
        tree = self.tree()
        shader = next((tree / "share/zinc/shaders").rglob("fixture.*"))
        shader.write_text("modified")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            check_metadata(tree, "9.8.7", NATIVE)
        shader.unlink()
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            check_metadata(tree, "9.8.7", NATIVE)

    def test_installer_native_install_spaces_idempotence_and_bin_link_repair(self):
        tree = self.tree()
        self.bundle(tree)
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assert_installed("9.8.7")
        (self.root / "commands/zinc").unlink()
        second = self.install()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertTrue((self.root / "commands/zinc").is_symlink())
        self.assertFalse((self.root / "installation/previous").exists())

    def test_upgrade_and_explicit_rollback_preserve_model_cache(self):
        self.bundle(self.tree("9.8.7"))
        self.assertEqual(self.install("9.8.7").returncode, 0)
        cache = self.root / "models/user-model.gguf"
        cache.parent.mkdir()
        cache.write_text("untouched")
        self.bundle(self.tree("9.8.8"))
        update = self.install("9.8.8")
        self.assertEqual(update.returncode, 0, update.stderr)
        self.assert_installed("9.8.8")
        rollback = self.install("9.8.8", "--rollback")
        self.assertEqual(rollback.returncode, 0, rollback.stderr)
        self.assert_installed("9.8.7")
        self.assertEqual(cache.read_text(), "untouched")

    def test_non_executable_download_never_replaces_working_current(self):
        self.bundle(self.tree("9.8.7"))
        self.assertEqual(self.install("9.8.7").returncode, 0)
        self.bundle(self.tree("9.8.8", exit_code=42))
        failure = self.install("9.8.8")
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("--version failed", failure.stderr)
        self.assert_installed("9.8.7")
        self.assertFalse((self.root / "installation/.install.lock").exists())

    def test_wrong_archive_version_is_rejected(self):
        tree = self.tree()
        metadata = read_json(tree / "VERSION.json")
        metadata["version"] = "9.8.6"
        write_json(tree / "VERSION.json", metadata)
        self.refresh(tree)
        self.bundle(tree)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("metadata disagree", result.stderr)

    def test_missing_duplicate_or_corrupt_archive_checksums_fail(self):
        tree = self.tree()
        archive = self.bundle(tree, duplicate_checksum=True)
        duplicate = self.install()
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("duplicate checksum", duplicate.stderr)
        (tree.parent / "SHA256SUMS.txt").write_text("a" * 64 + "  other.tar.gz\n")
        self.assertNotEqual(self.install().returncode, 0)
        (tree.parent / "SHA256SUMS.txt").write_text("a" * 64 + "  " + archive.name + "\n")
        corrupt = self.install()
        self.assertNotEqual(corrupt.returncode, 0)
        self.assertIn("checksum mismatch", corrupt.stderr)

    def test_installer_rejects_an_unavailable_target_before_requesting_its_archive(self):
        folder = self.root / "release/v9.8.7"
        folder.mkdir(parents=True)
        (folder / "SHA256SUMS.txt").write_text("a" * 64 + "  zinc-v9.8.7-other-target.tar.gz\n")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not include " + NATIVE, result.stderr)
        self.assertNotIn("release download failed", result.stderr)
        self.assertFalse((self.root / "installation/current").exists())
        self.assertFalse((self.root / "installation/.install.lock").exists())

    def test_installer_rejects_archive_links_traversal_and_duplicate_members(self):
        tree = self.tree()
        for name, kind in [("../escape", tarfile.REGTYPE), (tree.name + "/share/link", tarfile.SYMTYPE),
                           (tree.name + "/share/hard", tarfile.LNKTYPE), (tree.name + "/VERSION.json", tarfile.REGTYPE)]:
            member = tarfile.TarInfo(name)
            member.type, member.linkname = kind, "/outside"
            member.size = 1 if kind == tarfile.REGTYPE else 0
            self.bundle(tree, [member])
            with self.subTest(name=name):
                self.assertNotEqual(self.install().returncode, 0)
                self.assertFalse((self.root / "installation/current").exists())
        self.assertFalse((self.root / "escape").exists())

    def test_installer_fails_when_current_points_at_an_unmanaged_directory(self):
        self.bundle(self.tree())
        (self.root / "installation/current").mkdir(parents=True)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unmanaged", result.stderr)

    def test_installer_does_not_adopt_an_unmanaged_current_symlink(self):
        self.bundle(self.tree())
        outside = self.root / "outside installation"
        outside.mkdir()
        (outside / "keep").write_text("keep this installation")
        (self.root / "installation").mkdir()
        (self.root / "installation/current").symlink_to(outside)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unmanaged", result.stderr)
        self.assertEqual((self.root / "installation/current").resolve(), outside.resolve())
        self.assertEqual((outside / "keep").read_text(), "keep this installation")

    def test_explicit_prerelease_and_build_metadata_version(self):
        self.bundle(self.tree("9.8.7-rc.1+build-01"))
        result = self.install("9.8.7-rc.1+build-01")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unconfigured_backend_and_invalid_version_do_not_execute_shell(self):
        result = self.install(ZINC_BACKEND="cuda")
        self.assertNotEqual(result.returncode, 0)
        result = self.install(ZINC_VERSION="v0.1.0;touch marker")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "marker").exists())

    def candidate(self, all_targets=False):
        directory, reports = self.root / "candidate", self.root / "reports"
        directory.mkdir()
        reports.mkdir()
        artifacts = []
        scoped = policy() if all_targets else release_lib.release_policy()
        for spec in scoped["targets"]:
            archive = directory / archive_name("9.8.7", spec["id"])
            archive.write_bytes(("CPU-only gate fixture: " + spec["id"]).encode())
            artifacts.append({"target": spec["id"], "filename": archive.name, "size_bytes": archive.stat().st_size,
                              "sha256": sha256(archive), "binary_sha256": "a" * 64,
                              "signing": {"status": "notarized" if spec["backend"] == "metal" else "unsigned"}})
        data = {"schema_version": 1, "version": "9.8.7", "commit": COMMIT, "candidate_run": "123", "workflow_sha": COMMIT,
                "policy_sha256": sha256(ROOT / "config/release/targets.json"), "artifacts": artifacts,
                "required_profiles": [{k: p[k] for k in ("id", "target", "family")} for p in scoped["gpu_profiles"]]}
        write_json(directory / "release-manifest.json", data)
        (directory / "SHA256SUMS.txt").write_text("".join(a["sha256"] + "  " + a["filename"] + "\n" for a in artifacts))
        for profile in scoped["gpu_profiles"]:
            artifact = next(a for a in artifacts if a["target"] == profile["target"])
            models = [dict(id=m["id"], sha256=m["sha256"], context=m["context"], passed=True,
                           reference_agreement=release_lib.reference_agreement([{"tokens": 16, "near_ties": []} for _ in range(3)]))
                      for m in policy()["models"] if m["id"] in profile["models"]]
            for model in models:
                pin = next(m for m in policy()["models"] if m["id"] == model["id"])
                if pin.get("sliding_window"):
                    model["sliding_window_reference"] = {"passed": True, "http_passed": True, "context": pin["context"], "window_tokens": pin["sliding_window"],
                        "minimum_prompt_tokens": 1104, "maximum_prompt_tokens": 1108,
                        "reference_agreement": release_lib.reference_agreement([{"tokens": 16, "near_ties": []} for _ in range(3)])}
            if profile["target"].endswith("vulkan"):
                for model in models:
                    if model["id"] == "qwen36-35b-a3b-q4k-xl":
                        model["prefill_cancellation"] = {"passed": True, "recovery_ms": 1}
            report = {"schema_version": 1, "profile": profile["id"], "target": profile["target"], "family": profile["family"],
                      **{key: data[key] for key in ("version", "commit", "policy_sha256")},
                      "archive_sha256": artifact["sha256"], "binary_sha256": artifact["binary_sha256"], "models": models,
                      "hardware": {"gpu": "CPU fixture, no GPU was executed", "os": "fixture", "driver": "fixture"},
                      "options": {"parallel": 1, "kv_cache": "backend-default", "mtp": "backend-default", "greedy": True},
                      "reference": {"commit": policy()["reference"]["commit"], "generation": policy()["reference"]["generation"],
                                    "device_backend": target(profile["target"])["backend"],
                                    "library_sha256": {"libggml-base.so": "d" * 64},
                                    "server_sha256": "b" * 64, "tokenizer_sha256": "c" * 64},
                      "checks": {name: True for name in ("package", "identity", "gpu", "model_hash", "tokenizer_parity", "inference_parity", "http", "cleanup", "kernel_reference")}}
            if profile["target"].endswith("rocm"):
                report["rocm_kernel_reference"] = {"passed": True, "cases": 305,
                    "kernel_source_sha256": sha256(ROOT / "src/shaders/cuda/kernels.cu"),
                    "validator_source_sha256": sha256(ROOT / "tools/validate_rocm_partial_tiles.c")}
            write_json(reports / (profile["id"] + ".json"), report)
        return directory, reports

    def aggregate(self, directory, reports):
        with contextlib.redirect_stdout(io.StringIO()):
            release.aggregate(SimpleNamespace(directory=directory, results=reports, version="9.8.7", commit=COMMIT))

    def test_every_policy_model_is_pinned_in_the_managed_catalog(self):
        catalog = (ROOT / "src/model/catalog.zig").read_text()
        for model in policy()["models"]:
            with self.subTest(model=model["id"]):
                entry = re.search(r'\.id = "' + re.escape(model["id"]) + r'",([\s\S]*?)(?=\n    \},)', catalog)
                self.assertIsNotNone(entry)
                self.assertIn('/resolve/' + model["revision"] + '/' + model["file"] + '?download=true', entry.group(1))
                self.assertIn('.sha256 = "' + model["sha256"] + '"', entry.group(1))
                self.assertIn('.size_bytes = ' + format(model["size_bytes"], '_') + ',', entry.group(1))

    def test_complete_linux_release_profile_evidence_is_accepted(self):
        directory, reports = self.candidate()
        self.aggregate(directory, reports)
        data = read_json(directory / "validation.json")
        self.assertEqual(data["status"], "passed")
        self.assertEqual(len(data["profiles"]), 3)
        self.assertEqual(len(read_json(directory / "release-manifest.json")["artifacts"]), 2)

    def test_report_cannot_mislabel_backend_default_kv_as_f32(self):
        directory, reports = self.candidate()
        path = reports / "vulkan-intel-xe2.json"
        report = read_json(path)
        report["options"]["kv_cache"] = "f32"
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, "runtime option policy"):
            self.aggregate(directory, reports)

    def test_aggregate_binds_ssh_provenance_to_the_candidate_and_rejects_private_fields(self):
        directory, reports = self.candidate()
        path = reports / "vulkan-intel-xe2.json"
        report = read_json(path)
        transport = {"kind": "ssh", "artifact_id": 7, "artifact_digest": "sha256:" + "b" * 64,
                     "controller_run": 123, "controller_attempt": 1, "validation_commit": COMMIT,
                     "helpers_sha256": release_ssh.helper_digests(), "cleanup": True}
        write_json(path, {**report, "transport": transport})
        self.aggregate(directory, reports)
        for changed in ({"controller_run": 124}, {"validation_commit": "2" * 40}, {"cleanup": False},
                        {"artifact_digest": "missing"}, {"artifact_id": True}, {"helpers_sha256": {}}, {"host": "<private-host>"}):
            write_json(path, {**report, "transport": {**transport, **changed}})
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "SSH qualification provenance"):
                self.aggregate(directory, reports)

    def test_gemma_policy_requires_an_explicit_sliding_window_boundary(self):
        data = copy.deepcopy(policy())
        self.assertEqual({m["sliding_window"] for m in data["models"] if m["id"].startswith("gemma4-")}, {1024})
        path = self.root / "window-policy.json"
        for bad in (None, True, 0, -1, 4000):
            changed = copy.deepcopy(data)
            model = next(m for m in changed["models"] if m["id"] == "gemma4-31b-q4k-m")
            if bad is None:
                model.pop("sliding_window")
            else:
                model["sliding_window"] = bad
            write_json(path, changed)
            with self.subTest(window=bad), self.assertRaisesRegex(ValueError, "sliding-window boundary"):
                policy(path)

    def test_gemma_cannot_qualify_with_only_short_reference_prompts(self):
        directory, reports = self.candidate()
        path = reports / "vulkan-intel-xe2.json"
        saved = read_json(path)
        changes = [lambda m: m.pop("sliding_window_reference"),
                   lambda m: m.update(sliding_window_reference={}),
                   lambda m: m["sliding_window_reference"].update(passed=False),
                   lambda m: m["sliding_window_reference"].pop("http_passed"),
                   lambda m: m["sliding_window_reference"].update(window_tokens=True),
                   lambda m: m["sliding_window_reference"].update(context=2048),
                   lambda m: m["sliding_window_reference"].update(minimum_prompt_tokens=1024),
                   lambda m: m["sliding_window_reference"].update(maximum_prompt_tokens=4096),
                   lambda m: m["sliding_window_reference"].update(reference_agreement=None)]
        for index, change in enumerate(changes):
            report = copy.deepcopy(saved)
            model = next(m for m in report["models"] if m["id"] == "gemma4-31b-q4k-m")
            change(model)
            write_json(path, report)
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.aggregate(directory, reports)

    def test_preview_targets_and_hardware_cannot_replace_or_join_the_release_scope(self):
        directory, reports = self.candidate()
        manifest = directory / "release-manifest.json"
        saved = manifest.read_text()
        data = json.loads(saved)
        extra = directory / archive_name("9.8.7", "macos-aarch64-metal")
        extra.write_bytes(b"CPU-only preview archive fixture")
        data["artifacts"].append({"target": "macos-aarch64-metal", "filename": extra.name,
                                  "size_bytes": extra.stat().st_size, "sha256": sha256(extra),
                                  "binary_sha256": "a" * 64, "signing": {"status": "notarized"}})
        write_json(manifest, data)
        with self.assertRaisesRegex(ValueError, "release-scope targets"):
            release.verify_candidate(directory)
        manifest.write_text(saved)
        preview = read_json(reports / "vulkan-rdna4.json")
        preview.update(profile="metal-apple7", target="macos-aarch64-metal", family="apple7")
        write_json(reports / "metal-apple7.json", preview)
        with self.assertRaisesRegex(ValueError, "unexpected GPU report"):
            self.aggregate(directory, reports)

    def test_missing_duplicate_or_skipped_gpu_reports_block_a_candidate(self):
        directory, reports = self.candidate()
        report = reports / "vulkan-rdna4.json"
        saved = report.read_text()
        report.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.aggregate(directory, reports)
        report.write_text(saved)
        duplicate = reports / "duplicate.json"
        duplicate.write_text(saved)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.aggregate(directory, reports)
        duplicate.unlink()
        data = json.loads(saved)
        data["checks"]["cleanup"] = False
        write_json(report, data)
        with self.assertRaisesRegex(ValueError, "failed or were skipped"):
            self.aggregate(directory, reports)

    def test_report_cannot_substitute_source_artifact_family_model_context_or_reference(self):
        directory, reports = self.candidate()
        path = reports / "vulkan-rdna4.json"
        saved = path.read_text()
        changes = [lambda d: d.update(commit="2" * 40), lambda d: d.update(archive_sha256="0" * 64),
                   lambda d: d.update(binary_sha256="0" * 64), lambda d: d.update(family="rdna3"),
                   lambda d: d["models"][0].update(sha256="0" * 64), lambda d: d["models"][0].update(context=1),
                   lambda d: d["models"].pop(), lambda d: d["reference"].update(commit="2" * 40),
                   lambda d: d["models"][0].pop("reference_agreement"),
                   lambda d: d["models"][0]["reference_agreement"].update(exact_tokens=1),
                   lambda d: d["reference"].pop("generation"), lambda d: d["reference"]["generation"].update(micro_batch=2),
                   lambda d: d["checks"].pop("kernel_reference"), lambda d: d["hardware"].pop("driver"),
                   lambda d: d["reference"].pop("device_backend"), lambda d: d["reference"].update(device_backend="rocm"),
                   lambda d: d["reference"].pop("library_sha256"),
                   lambda d: d["reference"].update(library_sha256={}),
                   lambda d: d["reference"].update(library_sha256={"libllama.so": "d" * 64}),
                   lambda d: d["reference"].update(library_sha256={"libggml-base.so": "not-a-digest"}),
                   lambda d: d["reference"].update(library_sha256={"<private-path>/libggml-base.so": "d" * 64})]
        for index, change in enumerate(changes):
            data = json.loads(saved)
            change(data)
            write_json(path, data)
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.aggregate(directory, reports)

    def test_aggregate_requires_complete_rocm_reference_evidence_from_this_candidate(self):
        directory, reports = self.candidate()
        path = reports / "rocm-rdna4.json"
        saved = path.read_text()
        for change in (lambda d: d.pop("rocm_kernel_reference"), lambda d: d["checks"].pop("kernel_reference"),
                       lambda d: d["rocm_kernel_reference"].update(passed=False),
                       lambda d: d["rocm_kernel_reference"].update(cases=304),
                       lambda d: d["rocm_kernel_reference"].update(kernel_source_sha256="0" * 64),
                       lambda d: d["rocm_kernel_reference"].update(validator_source_sha256="0" * 64)):
            data = json.loads(saved)
            change(data)
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "ROCm numerical reference checks"):
                self.aggregate(directory, reports)

    def test_aggregate_requires_measured_prompt_cancellation_and_bounded_recovery(self):
        directory, reports = self.candidate()
        path = reports / "vulkan-intel-xe2.json"
        saved = path.read_text()
        for proof in (None, {}, {"passed": False, "recovery_ms": 1}, {"passed": True},
                      {"passed": True, "recovery_ms": True}, {"passed": True, "recovery_ms": -1},
                      {"passed": True, "recovery_ms": 10000}):
            data = json.loads(saved)
            model = next(m for m in data["models"] if m["id"] == "qwen36-35b-a3b-q4k-xl")
            model["prefill_cancellation"] = proof
            write_json(path, data)
            with self.subTest(proof=proof), self.assertRaisesRegex(ValueError, "active-prefill cancellation"):
                self.aggregate(directory, reports)

    def test_unsigned_metal_or_changed_archive_cannot_be_promoted(self):
        directory, _ = self.candidate(all_targets=True)
        path = directory / "release-manifest.json"
        data = read_json(path)
        data["artifacts"][0]["signing"]["status"] = "unsigned"
        write_json(path, data)
        expanded = policy()
        expanded.pop("release")
        with patch.object(release, "policy", return_value=expanded), self.assertRaisesRegex(ValueError, "not notarized"):
            release.verify_candidate(directory)
        data["artifacts"][0]["signing"]["status"] = "notarized"
        write_json(path, data)
        (directory / data["artifacts"][1]["filename"]).write_bytes(b"different archive")
        with patch.object(release, "policy", return_value=expanded), self.assertRaisesRegex(ValueError, "digest"):
            release.verify_candidate(directory)

    def test_publisher_accepts_only_successful_manual_main_runs(self):
        run = {"id": 123, "path": release_publish.WORKFLOW, "event": "workflow_dispatch", "head_branch": "main",
               "head_repository": {"full_name": release_publish.REPOSITORY}, "status": "completed", "conclusion": "success", "head_sha": COMMIT}
        self.assertEqual(release_publish.validate_run(run, "123"), COMMIT)
        for key, value in [("event", "pull_request"), ("event", "schedule"), ("head_branch", "feature"), ("conclusion", "skipped"),
                           ("status", "in_progress"), ("path", ".github/workflows/test.yml"), ("head_repository", {"full_name": "someone/fork"})]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                release_publish.validate_run({**run, key: value}, "123")

    def test_unprotected_environment_is_not_an_approval(self):
        for environment in [None, {}, {"protection_rules": []}, {"protection_rules": [{"type": "wait_timer"}]},
                            {"protection_rules": [{"type": "required_reviewers", "reviewers": []}]},
                            {"protection_rules": [{"type": "required_reviewers", "reviewers": [None]}]},
                            {"protection_rules": [{"type": "required_reviewers", "reviewers": [{"type": "User"}]}]}]:
            with self.subTest(environment=environment), self.assertRaisesRegex(ValueError, "required reviewers"):
                release_publish.validate_environment(environment, None)
        github = PreflightGitHub()
        environment = github.responses["environments/release"]
        branches = github.responses["environments/release/deployment-branch-policies"][0]
        release_publish.validate_environment(environment, branches)
        environment["protection_rules"][0]["reviewers"][0]["type"] = "Team"
        release_publish.validate_environment(environment, branches)

    def test_publication_rejects_broad_branch_access_and_incomplete_rule_inventories(self):
        github = PreflightGitHub()
        environment = github.responses["environments/release"]
        correct = github.responses["environments/release/deployment-branch-policies"][0]
        with self.assertRaisesRegex(ValueError, "could not read"):
            release_publish.validate_environment(environment, None)
        for branches in ({}, {"total_count": 0, "branch_policies": []},
                         {"total_count": True, "branch_policies": correct["branch_policies"]},
                         {"total_count": 2, "branch_policies": correct["branch_policies"]},
                         {"total_count": 1, "branch_policies": [{"name": "main", "type": "tag"}]},
                         {"total_count": 1, "branch_policies": [{"name": "*", "type": "branch"}]},
                         {"total_count": 1, "branch_policies": [None]},
                         {"total_count": 1, "branch_policies": [*correct["branch_policies"], {"name": "v*", "type": "tag"}]}):
            with self.subTest(branches=branches), self.assertRaisesRegex(ValueError, "only the main branch"):
                release_publish.validate_environment(environment, branches)
        for restriction in (None, {}, {"protected_branches": True, "custom_branch_policies": False}):
            with self.subTest(restriction=restriction), self.assertRaisesRegex(ValueError, "only the main branch"):
                release_publish.validate_environment({**environment, "deployment_branch_policy": restriction}, correct)

    def test_publication_checks_current_protections_before_downloading_or_writing_a_release(self):
        github = PreflightGitHub()
        environment = github.responses["environments/release"]
        # A candidate can pass preflight and then wait for approval while the
        # environment changes. Its former configuration must not authorize it.
        release_publish.validate_environment(environment, github.responses["environments/release/deployment-branch-policies"][0])
        environment["deployment_branch_policy"] = None
        run = {"id": 123, "path": release_publish.WORKFLOW, "event": "workflow_dispatch", "head_branch": "main",
               "head_repository": {"full_name": release_publish.REPOSITORY}, "status": "completed", "conclusion": "success", "head_sha": COMMIT}
        responses = {
            "actions/runs/123": run,
            "environments/release": environment,
            "environments/release/deployment-branch-policies": github.responses["environments/release/deployment-branch-policies"][0],
        }
        for publish in (False, True):
            argv = ["release_publish.py", "--run", "123", "--directory", str(self.root / "not-downloaded")]
            if publish:
                argv.append("--publish")
            with self.subTest(publish=publish), patch.object(sys, "argv", argv), \
                    patch.dict(os.environ, {"GITHUB_REF": "refs/heads/main", "GITHUB_REPOSITORY": release_publish.REPOSITORY}), \
                    patch.object(release_publish, "api", side_effect=lambda path, **kwargs: responses[path]) as api, \
                    patch.object(release_publish, "gh") as cli, patch.object(release_publish, "publish") as write, \
                    self.assertRaisesRegex(ValueError, "only the main branch"):
                release_publish.main()
            cli.assert_not_called()
            write.assert_not_called()
            self.assertTrue(all(call.args and len(call.args) == 1 for call in api.call_args_list))

    def test_publisher_never_replaces_an_existing_published_release(self):
        directory, _ = self.candidate()
        data = read_json(directory / "release-manifest.json")
        with patch.object(release_publish, "api", side_effect=[{"object": {"type": "commit", "sha": COMMIT}}, {"draft": False}]) as api, patch.object(release_publish, "gh") as gh:
            with self.assertRaisesRegex(ValueError, "existing/published"):
                release_publish.publish(directory, data, "123")
            self.assertEqual(api.call_count, 2)
            gh.assert_not_called()

    def test_published_candidate_rerun_is_read_only_when_all_bytes_match(self):
        directory, _ = self.candidate()
        data = read_json(directory / "release-manifest.json")
        marker = f"<!-- zinc-candidate-run: 123; manifest: {sha256(directory / 'release-manifest.json')} -->"
        existing = {"draft": False, "body": marker}
        def command(args):
            self.assertEqual(args[:2], ["release", "download"])
            destination = Path(args[args.index("--dir") + 1])
            for path in directory.iterdir():
                shutil.copyfile(path, destination / path.name)
        with patch.object(release_publish, "api", side_effect=[{"object": {"type": "commit", "sha": COMMIT}}, existing]) as api, patch.object(release_publish, "gh", side_effect=command) as gh, contextlib.redirect_stdout(io.StringIO()):
            release_publish.publish(directory, data, "123")
            self.assertEqual(api.call_count, 2)
            self.assertEqual(gh.call_count, 1)

    @contextlib.contextmanager
    def interrupted_publication(self, directory):
        """A local GitHub boundary: an accepted upload loses its client reply."""
        (directory / "release-notes.md").write_text("CPU-only publication rehearsal fixture.\n")
        state = SimpleNamespace(tag=None, draft=None, assets={}, uploads=[], edits=0, interrupted=False)
        def api(path, body=None, optional=False):
            if path.startswith("git/ref/tags/"):
                return state.tag
            if path.startswith("releases/tags/"):
                return None if state.draft is None else {**state.draft, "assets": [{"name": name} for name in state.assets]}
            if path == "git/refs":
                self.assertEqual(body, {"ref": "refs/tags/v9.8.7", "sha": COMMIT})
                self.assertIsNone(state.tag)
                state.tag = {"object": {"type": "commit", "sha": COMMIT}}
                return state.tag
            self.assertEqual(path, "releases")
            self.assertIsNone(state.draft)
            self.assertTrue(body["draft"])
            state.draft = body
            return {**body, "assets": []}
        def gh(args):
            self.assertEqual(args[0], "release")
            self.assertNotIn("--clobber", args)
            if args[1] == "upload":
                path = Path(args[3])
                self.assertNotIn(path.name, state.assets)
                state.assets[path.name] = path.read_bytes()
                state.uploads.append(path.name)
                if len(state.uploads) == 2 and not state.interrupted:
                    state.interrupted = True
                    raise RuntimeError("fixture connection lost after accepted upload")
            elif args[1] == "download":
                destination = Path(args[args.index("--dir") + 1])
                for name, content in state.assets.items():
                    (destination / name).write_bytes(content)
            else:
                self.assertEqual(args[1], "edit")
                self.assertIn("--draft=false", args)
                state.edits += 1
                state.draft["draft"] = False
        with patch.object(release_publish, "api", side_effect=api), patch.object(release_publish, "gh", side_effect=gh):
            yield state

    def test_interrupted_upload_resumes_without_overwrite_and_published_retry_is_read_only(self):
        directory, _ = self.candidate()
        data = read_json(directory / "release-manifest.json")
        with self.interrupted_publication(directory) as remote, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "connection lost"):
                release_publish.publish(directory, data, "123")
            self.assertTrue(remote.draft["draft"])
            self.assertEqual(remote.edits, 0)
            accepted = dict(remote.assets)
            release_publish.publish(directory, data, "123")
            self.assertFalse(remote.draft["draft"])
            self.assertEqual(remote.edits, 1)
            self.assertEqual(set(remote.uploads), {p.name for p in directory.iterdir()})
            self.assertEqual(len(remote.uploads), len(set(remote.uploads)))
            self.assertEqual({name: remote.assets[name] for name in accepted}, accepted)
            uploads = list(remote.uploads)
            release_publish.publish(directory, data, "123")
            self.assertEqual(remote.uploads, uploads)
            self.assertEqual(remote.edits, 1)

    def test_interrupted_draft_rejects_changed_bytes_other_candidates_and_orphan_tags(self):
        directory, _ = self.candidate()
        data = read_json(directory / "release-manifest.json")
        for failure in ("changed_asset", "other_candidate", "orphan_tag"):
            with self.subTest(failure=failure), self.interrupted_publication(directory) as remote:
                with self.assertRaises(RuntimeError):
                    release_publish.publish(directory, data, "123")
                if failure == "changed_asset":
                    remote.assets[next(iter(remote.assets))] = b"changed uploaded bytes"
                elif failure == "orphan_tag":
                    remote.draft = None
                uploads = list(remote.uploads)
                with self.assertRaisesRegex(ValueError, {"changed_asset": "uploaded bytes differ", "other_candidate": "existing/published", "orphan_tag": "tag already exists"}[failure]):
                    release_publish.publish(directory, data, "456" if failure == "other_candidate" else "123")
                self.assertEqual(remote.edits, 0)
                # A matching draft can upload missing assets before the final
                # byte audit; existing assets must never be replaced.
                if failure != "changed_asset":
                    self.assertEqual(remote.uploads, uploads)

    def starter_catalog(self, status="supported", profiles='"amd-rdna4-32gb", "intel-arc"'):
        catalog = (ROOT / "src/model/catalog.zig").read_text()
        # Exercise both qualification states even after the real starter is
        # promoted; this contract must not lock the current experimental state.
        match = re.search(r'(\.id = "qwen3-0.6b-q8-0",[\s\S]*?)(?=\n    \},)', catalog)
        entry = re.sub(r'\.status = \.[a-z]+', '.status = .' + status, match.group(1))
        entry = re.sub(r'\.tested_profiles\s*=\s*&\.\{[\s\S]*?\},', '.tested_profiles = &.{' + profiles + '},', entry, count=1)
        return catalog[:match.start()] + entry + catalog[match.end():]

    def test_stable_catalog_requires_reviewed_starter_qualification(self):
        unqualified = self.starter_catalog(status="experimental", profiles="")
        release_publish.validate_catalog(unqualified, "0.1.0-rc.1")
        with self.assertRaisesRegex(ValueError, "starter model to be qualified"):
            release_publish.validate_catalog(unqualified, "0.1.0")
        release_publish.validate_catalog(self.starter_catalog(), "0.1.0")

    def test_promoted_starter_matches_both_recorded_packaged_inference_results(self):
        model = next(m for m in policy()["models"] if m["id"] == "qwen3-0.6b-q8-0")
        public = ROOT / "site/public/zinc/validation"
        rdna = read_json(public / "worker-readiness-window-gate-2026-11-02.json")
        run = next(r["report"] for r in rdna["runs"] if r["report"]["profile"] == "vulkan-rdna4")
        self.assertTrue(run["passed"] and run["models_exercised"])
        amd = next(m for m in run["models"] if m["id"] == model["id"])
        intel = read_json(public / "intel-opencode-packaged-default-2026-11-02.json")
        xe2 = next(m for m in intel["native_defaults"] if m["id"] == model["id"])
        self.assertTrue(xe2["passed"] and xe2["tokenizer_passed"] and xe2["http_passed"])
        for row, sha, agreement in ((amd, "sha256", "agreement"), (xe2, "model_sha256", "reference_agreement")):
            self.assertEqual(row[sha], model["sha256"])
            self.assertEqual(row["context"], model["context"])
            self.assertEqual((row[agreement]["exact_tokens"], row[agreement]["total_tokens"], row[agreement]["near_tie_tokens"]), (48, 48, 0))
        self.assertFalse(intel["matrix_passed"])  # The recorded coding failures remain failures.
        release_publish.validate_catalog((ROOT / "src/model/catalog.zig").read_text(), "0.1.0")

    def test_stable_starter_cannot_expand_backend_or_chat_claims_without_review(self):
        catalog = self.starter_catalog()
        for changed in (catalog.replace('.tested_backend = .vulkan,', '.tested_backend = .rocm,', 1),
                        catalog.replace('.tested_backend = .vulkan,', '', 1),
                        catalog.replace('.recommended_for_chat = false,', '.recommended_for_chat = true,', 1),
                        catalog.replace('.thinking_stable = false,', '.thinking_stable = true,', 1)):
            with self.subTest(changed=changed[:80]), self.assertRaises(ValueError):
                release_publish.validate_catalog(changed, "0.1.0")

    def test_generated_notes_match_backend_default_kv_and_inference_only_starter(self):
        directory, reports = self.candidate()
        self.aggregate(directory, reports)
        with contextlib.redirect_stdout(io.StringIO()):
            release.finalize(SimpleNamespace(directory=directory))
        notes = (directory / "release-notes.md").read_text()
        self.assertIn("each backend's default KV/MTP settings", notes)
        self.assertIn("native reference uses F32 K/V", notes)
        self.assertNotIn("one slot, f32 KV", notes)
        self.assertIn("not recommended for chat, reasoning or coding", notes)
        self.assertNotIn("qwen3-0.6b-q8-0 --chat", notes)
        self.assertIn("qwen35-9b-q4k-m --chat", notes)
        # Public notes are generated independently of the preparation brief;
        # keep its known limitation and source-frozen evidence visible there.
        self.assertIn("slower decode than llama.cpp on all seven models", notes)
        self.assertIn("https://github.com/zolotukhin/zinc/blob/" + COMMIT + "/docs/INTEL_PERFORMANCE.md", notes)

    def test_stable_starter_coverage_must_match_the_selected_linux_profiles(self):
        for profiles in ('', '"apple-silicon"', '"amd-rdna4-32gb"', '"intel-arc"', '"unknown-profile"',
                         '"amd-rdna4-32gb",\n // "intel-arc",\n'):
            for version in ("0.1.0", "0.1.0+build.7"):
                with self.subTest(profiles=profiles, version=version), self.assertRaisesRegex(ValueError, "coverage is missing selected release profiles"):
                    release_publish.validate_catalog(self.starter_catalog(profiles=profiles), version)
        # Extra documented preview coverage and the catalog's string constant
        # syntax do not change which Linux roles the stable gate requires.
        release_publish.validate_catalog(self.starter_catalog(profiles='"amd-rdna4-32gb", "intel-arc", apple_silicon_profile,'), "0.1.0")
        selected = [p for p in policy()["gpu_profiles"] if p["id"] == "vulkan-intel-xe2"]
        with patch.object(release_publish, "release_policy", return_value={"gpu_profiles": selected}):
            release_publish.validate_catalog(self.starter_catalog(profiles='"intel-arc"'), "0.1.0")
        unreviewed = [p for p in policy()["gpu_profiles"] if p["id"] == "vulkan-rdna3"]
        with patch.object(release_publish, "release_policy", return_value={"gpu_profiles": unreviewed}), self.assertRaisesRegex(ValueError, "explicit catalog mapping"):
            release_publish.validate_catalog(self.starter_catalog(), "0.1.0")

    def test_starter_catalog_comments_or_unsupported_fields_cannot_fake_qualification(self):
        commented = self.starter_catalog(status="experimental").replace(
            '.status = .experimental,', '// .status = .supported,\n        .status = .experimental,', 1)
        with self.assertRaisesRegex(ValueError, "starter model to be qualified"):
            release_publish.validate_catalog(commented, "0.1.0")
        duplicate = self.starter_catalog().replace('.status = .supported,', '.status = .supported,\n        .status = .experimental,', 1)
        with self.assertRaisesRegex(ValueError, "catalog field is missing, duplicate or unsupported"):
            release_publish.validate_catalog(duplicate, "0.1.0")
        for profiles in ('"amd-rdna4-32gb", "intel-arc", "intel-arc"',
                         '"amd-rdna4-32gb", unreviewed_profile', '"amd-rdna4-32gb", "intel-arc" ++ "extra"'):
            with self.subTest(profiles=profiles), self.assertRaisesRegex(ValueError, "starter catalog profile"):
                release_publish.validate_catalog(self.starter_catalog(profiles=profiles), "0.1.0")

    def test_shipped_catalog_defaults_stay_within_the_model_qualification(self):
        for name, value, error in (("size_bytes", "1", "model size differs"),
                                   ("default_context_length", "4096", "default context exceeds"),
                                   ("default_context_length", "0", "default context exceeds")):
            changed = re.sub(r'(\.' + name + r'\s*=\s*)[0-9_]+', lambda m: m[1] + value, self.starter_catalog(), count=1)
            for version in ("0.1.0-rc.1", "0.1.0"):
                with self.subTest(field=name, value=value, version=version), self.assertRaisesRegex(ValueError, error):
                    release_publish.validate_catalog(changed, version)

    def test_archive_decompression_limit_applies_before_tar_metadata_parsing(self):
        tree = self.tree()
        archive = self.bundle(tree)
        with patch.object(release_lib, "MAX_UNPACKED_BYTES", 512):
            with self.assertRaisesRegex(ValueError, "expands beyond"):
                extract_archive(archive, self.root / "bounded", "9.8.7", NATIVE)
        self.assertEqual(list((self.root / "bounded").iterdir()), [])

    def test_extra_executable_is_rejected_even_with_a_matching_inventory(self):
        tree = self.tree()
        (tree / "bin/unexpected").write_text("extra executable")
        self.refresh(tree)
        with self.assertRaisesRegex(ValueError, "unexpected packaged file"):
            check_metadata(tree, "9.8.7", NATIVE)

    def test_publisher_recomputes_reports_and_verifies_every_attested_file(self):
        directory, reports = self.candidate()
        self.aggregate(directory, reports)
        with contextlib.redirect_stdout(io.StringIO()):
            release.finalize(SimpleNamespace(directory=directory))
        (directory / "provenance.jsonl").write_text("CPU fixture: signature verification is mocked\n")
        run = {"id": 123, "path": release_publish.WORKFLOW, "event": "workflow_dispatch", "head_branch": "main",
               "head_repository": {"full_name": release_publish.REPOSITORY}, "status": "completed", "conclusion": "success", "head_sha": COMMIT}
        with patch.object(release_publish, "gh", return_value="") as gh, patch.object(release_publish.subprocess, "run"), \
                patch.object(release_publish, "verify_publication_sources") as sources, contextlib.redirect_stdout(io.StringIO()):
            release_publish.verify_bundle(directory, "123", run)
            sources.assert_called_once()
            expected = {p.name for p in directory.iterdir()} - {"provenance.jsonl"}
            self.assertEqual({Path(call.args[0][2]).name for call in gh.call_args_list}, expected)
            for call in gh.call_args_list:
                self.assertIn("--deny-self-hosted-runners", call.args[0])
                self.assertIn("--source-digest", call.args[0])
        notes = (directory / "release-notes.md").read_text()
        self.assertIn("Vulkan and ROCm binaries", notes)
        self.assertIn("Metal remains an unsigned preview", notes)
        self.assertNotIn("Metal, Vulkan and ROCm binaries", notes)
        validation = read_json(directory / "validation.json")
        validation["profiles"][0]["models"][0]["context"] = 1
        write_json(directory / "validation.json", validation)
        with patch.object(release_publish, "verify_publication_sources"), self.assertRaisesRegex(ValueError, "context"):
            release_publish.verify_bundle(directory, "123", run)

    def test_gpu_family_assertions_reject_software_and_wrong_devices(self):
        good = "Selected GPU: #0 AMD Radeon AI PRO R9700 (RADV GFX1201)\nGPU tuning: amd_rdna4 | 32768 MB VRAM\nVerdict: READY [OK]\n"
        self.assertIn("R9700", release_gpu.gpu_evidence(good, "rdna4"))
        rocm = "  Device: AMD Radeon AI PRO R9700 [gfx1201] [OK]\n  Compute target: gfx1201 [OK]\nVerdict: READY [OK]\n"
        self.assertEqual(release_gpu.gpu_evidence(rocm, "gfx1201"), "AMD Radeon AI PRO R9700")
        self.assertEqual(release_gpu.gpu_evidence("GPU family: apple7 [OK]\nVerdict: READY [OK]", "apple7"), "apple7")
        for text, family in [(good, "rdna3"), (good.replace("R9700", "llvmpipe"), "rdna4"),
                             (good + "requested #1 out of range", "rdna4"), (good + "[FAIL]", "rdna4"),
                             (rocm.replace("[gfx1201]", "[gfx1100]"), "gfx1201"),
                             (rocm.replace(" [gfx1201]", ""), "gfx1201"),
                             (rocm.replace("R9700", "llvmpipe"), "gfx1201"),
                             (rocm.replace("R9700", "<private-host>"), "gfx1201"),
                             (rocm.replace("Compute target: gfx1201", "Compute target: gfx1100"), "gfx1201")]:
            with self.subTest(family=family), self.assertRaises(ValueError):
                release_gpu.gpu_evidence(text, family)

    def test_api_uses_the_loaded_model_instead_of_switching_to_an_unloaded_catalog_entry(self):
        self.assertEqual(release_gpu.active_model_id([{"id": "unloaded", "active": False}, {"id": "loaded", "active": True}]), "loaded")
        self.assertEqual(release_gpu.active_model_id([{"id": "rocm-loaded.gguf"}]), "rocm-loaded.gguf")
        for data in [[], [{"id": "unloaded", "active": False}], [{"id": "one"}, {"id": "two"}]]:
            with self.assertRaises(ValueError):
                release_gpu.active_model_id(data)

    def test_reference_device_backend_accepts_explicit_native_selectors(self):
        for backend, selector in (("metal", "Metal"), ("metal", "Metal7"), ("vulkan", "Vulkan7"), ("rocm", "ROCm9")):
            for args in (["-ngl", "99", "--device", selector], ["-dev", selector], ["--device=" + selector]):
                with self.subTest(backend=backend, args=args):
                    self.assertEqual(release_gpu.reference_device_backend(args, backend), backend)

    def test_reference_device_backend_rejects_cross_backend_missing_and_ambiguous_selectors(self):
        for args in (None, [], [None], ["-ngl", "99"], ["--device"], ["--device", "Vulkan7"],
                     ["--device", "none"], ["--device", "ROCm"], ["--device", "ROCm7,ROCm9"],
                     ["--device", "ROCm7", "-dev=Vulkan9"], ["--device", "ROCm7", "--device=ROCm7"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                release_gpu.reference_device_backend(args, "rocm")

    def test_wrong_reference_backend_is_rejected_before_probing_or_starting_gpu_processes(self):
        config = {"device": 7, "reference_args": ["-ngl", "99", "--device", "Vulkan7"]}
        argv = ["release_gpu.py", "--directory", str(self.root), "--profile", "rocm-rdna4", "--output", str(self.root / "report.json")]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {"ZINC_RELEASE_WORKER_CONFIG": str(self.root / "worker.json")}), \
                patch.object(release_gpu, "verify_candidate", return_value={"artifacts": [{"target": "linux-x86_64-rocm"}]}), \
                patch.object(release_gpu, "read_json", return_value=config), patch.object(release_gpu, "output") as probe, \
                patch.object(release_gpu, "process") as launch, self.assertRaisesRegex(ValueError, "native GPU backend"):
            release_gpu.main()
        probe.assert_not_called()
        launch.assert_not_called()

    def test_reference_inventory_requires_adjacent_ggml_artifacts(self):
        (self.root / "llama-server").write_bytes(b"server")
        (self.root / "llama-tokenize").write_bytes(b"tokenizer")
        (self.root / "libllama.so").write_bytes(b"llama")
        with self.assertRaisesRegex(ValueError, "ggml libraries beside"):
            release_gpu.reference_artifacts(self.root)
        (self.root / "libggml-base.so").write_bytes(b"ggml")
        self.assertEqual(set(release_gpu.reference_artifacts(self.root)["library_sha256"]), {"libllama.so", "libggml-base.so"})

    def test_candidate_and_readiness_reject_stale_cross_backend_or_changed_references(self):
        for entrypoint in ("candidate", "readiness"):
            failures = [None, "library", "during_version_probe", "stale_server", "wrong_backend", "late_cleanup", "sliding_window"]
            if entrypoint == "readiness": failures += ["wrong_validation", "dirty_validation"]
            else: failures.append("busy_bench")
            for failure in failures:
                folder = self.root / (entrypoint + "-" + str(failure))
                reference, tree = folder / "reference", folder / "package"
                reference.mkdir(parents=True)
                (tree / "bin").mkdir(parents=True)
                for path, content in ((reference / "llama-server", b"server"), (reference / "llama-tokenize", b"tokenizer"),
                                      (reference / "libggml-hip.so", b"original library"), (reference / "notes.txt", b"private notes"),
                                      (tree / "bin/zinc", b"packaged binary"), (folder / "model.gguf", b"pinned model"),
                                      (folder / "candidate.tar.gz", b"archive"),
                                      (folder / archive_name("9.8.7", "linux-x86_64-rocm"), b"archive")):
                    path.write_bytes(content)
                before = release_gpu.reference_artifacts(reference)
                model = folder / "model.gguf"
                real = policy()
                data = {"reference": real["reference"], "gpu_profiles": [{"id": "rocm-rdna4", "target": "linux-x86_64-rocm", "family": "gfx1201", "models": ["fixture"]}],
                        "targets": [{"id": "linux-x86_64-rocm", "backend": "rocm", "interfaces": {"browser_chat": False}}],
                        "models": [{"id": "fixture", "context": 2048, "sliding_window": 1024,
                                    "size_bytes": model.stat().st_size, "sha256": sha256(model)}]}
                config = {"device": 7, "reference_args": ["-ngl", "99", "--device", "Vulkan7" if failure == "wrong_backend" else "ROCm7"],
                          "reference_bin": str(reference), "reference_source": str(folder), "lock_directory": str(folder / "reservation"), "models": {"fixture": str(model)}}
                write_json(folder / "worker.json", config)
                candidate = {"version": "9.8.7", "commit": COMMIT, "policy_sha256": "b" * 64,
                             "artifacts": [{"target": "linux-x86_64-rocm", "filename": "candidate.tar.gz"}]}
                def output(command, *args, **kwargs):
                    if command[0] == "ps" and failure == "busy_bench": return "llama-bench\n"
                    if command[-1] == "HEAD": return COMMIT if command[2] == str(release_worker.ROOT) else real["reference"]["commit"]
                    if "status" in command and command[2] == str(release_worker.ROOT) and failure == "dirty_validation": return " M tools/release_worker.py"
                    if command[-1] == "--version":
                        if failure == "during_version_probe": (reference / "libggml-hip.so").write_bytes(b"changed during version probe")
                        return "b96806d96" if failure == "stale_server" else real["reference"]["commit"][:7]
                    return ""
                def requests(*args):
                    self.assertNotIn("LD_LIBRARY_PATH", args[5])
                    self.assertNotIn("DYLD_LIBRARY_PATH", args[5])
                    if failure == "library": (reference / "libggml-hip.so").write_bytes(b"rebuilt library")
                samples = [("prompt", {"comparison": {"tokens": 16, "near_ties": []}}) for _ in range(3)]
                sliding = {"passed": True, "http_passed": True, "context": 2048, "window_tokens": 1024,
                           "minimum_prompt_tokens": 1100, "maximum_prompt_tokens": 1200,
                           "reference_agreement": {"total_tokens": 48, "exact_tokens": 48, "near_tie_tokens": 0}}
                argv = ["release_gpu.py", "--directory", str(folder), "--profile", "rocm-rdna4", "--output", str(folder / "report.json")]
                args = SimpleNamespace(profile="rocm-rdna4", directory=folder, version="9.8.7", commit=PREFLIGHT_MAIN,
                                       validation_commit="3" * 40 if failure == "wrong_validation" else COMMIT,
                                       output=folder / "report.json", exercise_models=True)
                with self.subTest(entrypoint=entrypoint, failure=failure), contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
                    stack.enter_context(patch.object(sys, "argv", argv))
                    stack.enter_context(patch.dict(os.environ, {"ZINC_RELEASE_WORKER_CONFIG": str(folder / "worker.json"),
                                                              "LD_LIBRARY_PATH": "<private-library-override>", "DYLD_LIBRARY_PATH": "<private-library-override>"}))
                    for name, value in (("verify_candidate", candidate), ("policy", data), ("extract_archive", tree),
                                        ("probe_binary", None), ("tokenizer_parity", None), ("inference_parity", samples),
                                        ("hardware_evidence", {"gpu": "CPU fixture", "os": "fixture", "driver": "fixture"})):
                        stack.enter_context(patch.object(release_gpu, name, return_value=value))
                    for name, value in (("policy", data), ("extract_archive", tree), ("probe_binary", None), ("idle", None)):
                        stack.enter_context(patch.object(release_worker, name, return_value=value))
                    stack.enter_context(patch.object(release_gpu, "output", side_effect=output))
                    stack.enter_context(patch.object(release_gpu, "api_checks", side_effect=requests))
                    kernel_check = stack.enter_context(patch.object(release_gpu, "rocm_kernel_reference", return_value={"passed": True, "cases": 305}))
                    window_check = stack.enter_context(patch.object(release_gpu, "sliding_window_reference", return_value=sliding,
                        side_effect=ValueError("fixture long-window continuation differed") if failure == "sliding_window" else None))
                    if failure == "late_cleanup":
                        original_reservation = release_worker.reservation if entrypoint == "readiness" else release_gpu.worker_reservation
                        @contextlib.contextmanager
                        def cleanup_failure(path):
                            with original_reservation(path): yield
                            raise ValueError("fixture cleanup failed")
                        stack.enter_context(patch.object(release_worker if entrypoint == "readiness" else release_gpu,
                                                        "reservation" if entrypoint == "readiness" else "worker_reservation", side_effect=cleanup_failure))
                    launch = stack.enter_context(patch.object(release_gpu, "process"))
                    module = release_gpu if entrypoint == "candidate" else release_worker
                    report = stack.enter_context(patch.object(module, "write_json"))
                    def run():
                        if entrypoint == "candidate": release_gpu.main()
                        else: release_worker.check(args)
                    if failure:
                        reason = {"library": "reference artifacts changed", "during_version_probe": "reference artifacts changed",
                                  "stale_server": "reference executable", "wrong_backend": "native GPU backend",
                                  "wrong_validation": "validation checkout differs", "dirty_validation": "validation checkout is dirty",
                                  "late_cleanup": "fixture cleanup failed", "busy_bench": "GPU worker is busy",
                                  "sliding_window": "fixture long-window continuation differed"}[failure]
                        with self.assertRaisesRegex(ValueError, reason): run()
                        if entrypoint == "candidate": report.assert_not_called()
                        else: self.assertFalse(report.call_args.args[1]["passed"])
                    else:
                        run()
                        # Exercise the real pin-to-execution wiring: all short
                        # probes pass, but a missing extended check must still fail this test.
                        window_check.assert_called_once()
                        kernel_check.assert_called_once()
                        self.assertEqual(window_check.call_args.args[3], 2048)
                        self.assertEqual(window_check.call_args.args[-1], 1024)
                        self.assertEqual(report.call_args.args[1]["models"][0]["sliding_window_reference"], sliding)
                        self.assertEqual(report.call_args.args[1]["reference"]["library_sha256"], before["library_sha256"])
                        self.assertEqual(set(before["library_sha256"]), {"libggml-hip.so"})
                        if entrypoint == "readiness":
                            evidence = report.call_args.args[1]
                            self.assertTrue(evidence["passed"])
                            self.assertEqual((evidence["source_commit"], evidence["validation_source_commit"]), (PREFLIGHT_MAIN, COMMIT))
                            self.assertEqual(evidence["validation_helpers_sha256"], {
                                name: sha256(ROOT / "tools" / name) for name in ("release_worker.py", "release_gpu.py", "release_lib.py", "release.py")})
                    launch.assert_not_called()  # Every inference boundary is mocked: these are CPU lifecycle tests.
                after = release_gpu.reference_artifacts(reference)
                self.assertEqual(before["server_sha256"], after["server_sha256"])
                self.assertEqual(before["tokenizer_sha256"], after["tokenizer_sha256"])
                self.assertFalse((folder / "reservation").exists())

    def test_release_unicode_gate_rejects_invalid_bytes_missing_usage_and_lost_text(self):
        def nonstream(port, path, body):
            text = "café 東京 🚀" if body["max_tokens"] == 32 else ""
            return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}

        def streaming(body, damage=None):
            text = nonstream(0, "", body)["choices"][0]["message"]["content"]
            event = {"choices": [{"delta": {"content": "wrong" if damage == "text" else text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20}}
            if damage == "usage":
                del event["usage"]
            payload = ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" + ("" if damage == "done" else "data: [DONE]\n\n")).encode()
            if damage == "utf8":
                payload = b"data: \xff\n\ndata: [DONE]\n\n"
            return contextlib.nullcontext(io.BytesIO(payload))

        with patch.object(release_gpu, "json_request", side_effect=nonstream) as json_request, patch.object(release_gpu, "request", side_effect=lambda port, path, body: streaming(body)) as request:
            release_gpu.unicode_chat_checks(123, "loaded")
            self.assertEqual(request.call_count, 4)
            for call in json_request.call_args_list:
                self.assertEqual(call.args[2]["messages"][0], {
                    "role": "system", "content": "Repeat the requested text exactly, preserving every character."})
        for damage in ("utf8", "usage", "done", "text"):
            with self.subTest(damage=damage), patch.object(release_gpu, "json_request", side_effect=nonstream), patch.object(release_gpu, "request", side_effect=lambda port, path, body: streaming(body, damage)):
                with self.assertRaises(ValueError):
                    release_gpu.unicode_chat_checks(123, "loaded")

        # Valid, identical JSON/SSE is insufficient if the model translates
        # away the required bytes: preserve the original Unicode coverage.
        def translated(port, path, body):
            return {"choices": [{"message": {"content": "☕ 东京咖啡 🚀"}, "finish_reason": "stop"}]}
        event = {"choices": [{"delta": {"content": "☕ 东京咖啡 🚀"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20}}
        payload = ("data: " + json.dumps(event, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode()
        with patch.object(release_gpu, "json_request", side_effect=translated), \
             patch.object(release_gpu, "request", side_effect=lambda *args: contextlib.nullcontext(io.BytesIO(payload))), \
             self.assertRaisesRegex(ValueError, "Unicode echo was incomplete"):
            release_gpu.unicode_chat_checks(123, "loaded")

    def test_http_reference_gate_rejects_early_stops_and_uses_the_exact_reference_prompt(self):
        samples = [("<rendered reference prompt>", {"content": "The complete reference continuation.\nSecond line."})]
        body = {"model": "loaded", "prompt": "unrelated", "temperature": 0}
        with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": "The complete reference"}]}) as request:
            with self.assertRaisesRegex(ValueError, "full HTTP reference"):
                release_gpu.reference_http_parity(1, body, samples)
            self.assertEqual(request.call_args.args[2]["prompt"], samples[0][0])
            self.assertEqual(request.call_args.args[2]["model"], "loaded")
            self.assertFalse(request.call_args.args[2]["stream"])
        with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": samples[0][1]["content"]}]}):
            release_gpu.reference_http_parity(1, body, samples)
        samples[0][1].update(validated_content="A qualified alternate.", comparison={"near_ties": [{"rank": 2}]})
        for content in [samples[0][1]["content"], "A qualified alternate."]:
            with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": content}]}):
                release_gpu.reference_http_parity(1, body, samples)
        with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": "A qualified"}]}), self.assertRaises(ValueError):
            release_gpu.reference_http_parity(1, body, samples)
        samples[0][1].pop("comparison")
        with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": "A qualified alternate."}]}), self.assertRaises(ValueError):
            release_gpu.reference_http_parity(1, body, samples)

    def test_http_reference_gate_distinguishes_eos_from_a_token_limit(self):
        for stop_type, finish in (("eos", "stop"), ("limit", "length")):
            samples = [("prompt", {"content": "complete", "stop_type": stop_type})]
            with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": "complete", "finish_reason": finish}]}):
                release_gpu.reference_http_parity(1, {}, samples)
            with patch.object(release_gpu, "json_request", return_value={"choices": [{"text": "complete", "finish_reason": "length" if finish == "stop" else "stop"}]}), \
                 self.assertRaisesRegex(ValueError, "finish reason"):
                release_gpu.reference_http_parity(1, {}, samples)

    def test_cancellation_closes_an_active_chat_stream_before_draining_it(self):
        class Stream(io.BytesIO):
            headers = {"Content-Type": "text/event-stream"}
            consumed = 0

            def close(self):
                if not self.closed:
                    self.consumed = self.tell()
                    super().close()

        payload = (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                   b'data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}]}\n\n'
                   b'data: {"choices":[{"delta":{"content":"still generating"}}]}\n\ndata: [DONE]\n\n')
        response = Stream(payload)
        with patch.object(release_gpu, "request", return_value=response) as request:
            release_gpu.cancel_chat_stream(1, {"max_tokens": 256})
        self.assertTrue(response.closed)
        self.assertLess(response.consumed, len(payload))
        self.assertEqual(request.call_args.args[1], "/v1/chat/completions")
        self.assertTrue(request.call_args.args[2]["stream"])
        for content_type, data in (("application/json", b'{"choices":[{}]}'),
                                   ("text/event-stream", b''),
                                   ("text/event-stream", b'data: \xff\n\n'),
                                   ("text/event-stream", b'data: [DONE]\n\n'),
                                   ("text/event-stream", b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'),
                                   ("text/event-stream", b'data: {"error":"failed"}\n\n')):
            response = Stream(data)
            response.headers = {"Content-Type": content_type}
            with self.subTest(content_type=content_type, data=data), \
                 patch.object(release_gpu, "request", return_value=response), self.assertRaises(ValueError):
                release_gpu.cancel_chat_stream(1, {})

    def test_prefill_cancellation_requires_active_work_closes_early_and_proves_recovery(self):
        class Stream(io.BytesIO):
            headers = {"Content-Type": "text/event-stream"}
            consumed = 0

            def close(self):
                if not self.closed:
                    self.consumed = self.tell()
                super().close()

        payload = (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n: prefill\n\n'
                   b'data: {"choices":[{"delta":{"content":"must not drain"}}]}\n\n')
        stream = Stream(payload)
        recovery = io.BytesIO(b'{"choices":[{"message":{"content":"OK"}}]}')
        with patch.object(release_gpu, "request", side_effect=[stream, recovery]) as request:
            result = release_gpu.prefill_cancellation_check(1, "loaded", 4096)
        self.assertTrue(result["passed"])
        self.assertTrue(stream.closed and recovery.closed)
        self.assertLess(stream.consumed, len(payload))
        self.assertEqual(request.call_count, 2)
        self.assertTrue(request.call_args_list[0].args[2]["stream"])
        self.assertFalse(request.call_args_list[1].args[2]["stream"])
        self.assertEqual(request.call_args.kwargs["timeout"], 10)
        self.assertEqual(request.call_args.args[2]["model"], "loaded")

        for failure in (b'', b'data: [DONE]\n\n',
                        b'data: {"choices":[{"delta":{"content":"already decoding"}}]}\n\n',
                        b'data: {"error":{"message":"failed"}}\n\n'):
            with self.subTest(failure=failure), patch.object(release_gpu, "request", return_value=Stream(failure)) as request:
                with self.assertRaises(ValueError):
                    release_gpu.prefill_cancellation_check(1, "loaded", 4096)
                self.assertEqual(request.call_count, 1)
        with patch.object(release_gpu, "request", side_effect=[Stream(payload), TimeoutError()]), self.assertRaises(TimeoutError):
            release_gpu.prefill_cancellation_check(1, "loaded", 4096)

    def test_cli_reference_gate_checks_all_backend_token_formats_and_complete_text(self):
        expected = {"content": "First line.\n  Second line. ",
                    "completion_probabilities": [{"id": token} for token in [791, 315, 198]]}
        for label in ["Generated tokens (3): { 791, 315, 198 }",
                      "Output tokens (3): { 791, 315, 198 }",
                      "Output tokens (3): first20={ 791, 315, 198 }"]:
            for output in ["Output (3 tokens): ", "Output text: "]:
                text = "debug(zinc): " + label + "\ninfo(zinc): " + output + expected["content"] + "\n"
                with self.subTest(label=label, output=output):
                    release_gpu.cli_reference_parity(text, expected)
                for bad in [text.replace("315, 198", "315, 199"),
                            text.replace("(3):", "(4):"), text.replace("791, 315, 198", "791"),
                            text.replace("  Second line. ", "Second line."),
                            text.replace("\n  Second line. ", ""), text.replace(expected["content"], "First")]:
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        release_gpu.cli_reference_parity(bad, expected)
        with self.assertRaises(ValueError):
            release_gpu.cli_reference_parity("topk: top=791\nOutput text: First line.\n", expected)

    def test_bounded_prefill_gate_forces_chunks_and_compares_the_frozen_continuation(self):
        commands, environments = [], []
        @contextlib.contextmanager
        def fake_process(command, log, _cwd, env):
            commands.append(command)
            environments.append(env)
            Path(log).write_text("Gemma prefill scratch: 16 tokens per chunk for 19 prompt tokens\n"
                                 "Generated tokens (2): { 760, 42 }\ninfo(zinc): Output text: The answer\n")
            yield SimpleNamespace(wait=lambda **_kwargs: 0)
        expected = {"content": "The answer", "completion_probabilities": [{"id": 760}, {"id": 42}]}
        env = {"ZINC_PREFILL_CHUNK": "999", "fixture": "preserved"}
        with patch.object(release_gpu, "process", fake_process):
            result = release_gpu.bounded_prefill_parity(self.root, self.root / "model", 4096, 3,
                                                      self.root, env, [("rendered prompt", expected)])
        self.assertEqual(result, {"passed": True, "chunk_tokens": 16, "cases": 1, "tokens": 2})
        self.assertEqual(environments, [{"ZINC_PREFILL_CHUNK": "16", "fixture": "preserved"}])
        self.assertEqual(env["ZINC_PREFILL_CHUNK"], "999")
        for flag, value in [("-d", "3"), ("-c", "4096"), ("-n", "2"), ("--prompt", "rendered prompt")]:
            self.assertEqual(commands[0][commands[0].index(flag) + 1], value)

    def test_bounded_prefill_gate_rejects_unexercised_and_incorrect_execution(self):
        expected = {"content": "The answer", "completion_probabilities": [{"id": 760}, {"id": 42}]}
        good = ("Gemma prefill scratch: 16 tokens per chunk for 19 prompt tokens\n"
                "Generated tokens (2): { 760, 42 }\ninfo(zinc): Output text: The answer\n")
        for text, status in [(good.replace("Gemma prefill scratch:", "unused:"), 0),
                             (good.replace("760, 42", "760, 43"), 0),
                             (good.replace("The answer", "The"), 0), (good, 1)]:
            @contextlib.contextmanager
            def fake_process(_command, log, _cwd, _env):
                Path(log).write_text(text)
                yield SimpleNamespace(wait=lambda **_kwargs: status)
            with self.subTest(text=text, status=status), patch.object(release_gpu, "process", fake_process), self.assertRaises(ValueError):
                release_gpu.bounded_prefill_parity(self.root, self.root / "model", 4096, 0,
                                                  self.root, {}, [("rendered prompt", expected)])

    def test_reference_precision_is_fixed_after_worker_flags_and_cli_keeps_gpu_argmax(self):
        commands, environments = [], []
        @contextlib.contextmanager
        def fake_process(command, log, _cwd, env):
            commands.append(command)
            environments.append(env)
            if Path(command[0]).name == "zinc":
                Path(log).write_text("debug(zinc): Generated tokens (1): { 760 }\ninfo(zinc): Output (1 tokens): The\n")
            yield SimpleNamespace(wait=lambda **_kwargs: 0)
        expected = {"content": "The", "completion_probabilities": [{"id": 760}]}
        worker_flags = ["--ubatch-size", "512", "--cache-type-k", "f16", "--flash-attn", "on"]
        with patch.object(release_gpu, "PROMPTS", ["fixture"]), patch.object(release_gpu, "process", fake_process), \
             patch.object(release_gpu, "ready"), patch.object(release_gpu, "json_request", side_effect=[{"prompt": "rendered"}, expected]):
            samples = release_gpu.inference_parity(self.root / "package", self.root / "reference", self.root / "model",
                                                  4096, 0, worker_flags, self.root, {})
        self.assertEqual(samples, [("rendered", expected)])
        for flag, value in [("--ubatch-size", "1"), ("--batch-size", "512"), ("--parallel", "1"),
                            ("--cache-type-k", "f32"), ("--cache-type-v", "f32"), ("--flash-attn", "off")]:
            self.assertEqual([commands[0][i + 1] for i, item in enumerate(commands[0][:-1]) if item == flag][-1], value)
        self.assertIn("--debug", commands[1])
        self.assertNotIn("ZINC_LOG_TOPK", environments[1])
        self.assertNotIn("ZINC_FORCE_CPU_ARGMAX", environments[1])
        self.assertNotIn("LD_LIBRARY_PATH", environments[1])
        if sys.platform.startswith("linux"):
            self.assertEqual(environments[0]["LD_LIBRARY_PATH"], str(self.root / "reference"))

    def test_reference_agreement_records_a_bounded_tie_without_counting_it_exact(self):
        cases = [{"tokens": 16, "near_ties": []} for _ in range(3)]
        tie = {"position": 11, "chosen_id": 36337, "reference_id": 3618, "rank": 2, "logprob_gap": 0.032}
        cases[1]["near_ties"].append(tie)
        report = release_lib.reference_agreement(cases)
        self.assertEqual((report["exact_tokens"], report["total_tokens"], report["near_tie_tokens"]), (47, 48, 1))
        release_lib.verify_reference_agreement(report)
        for change in [lambda data: data[1]["near_ties"][0].update(rank=3),
                       lambda data: data[1]["near_ties"][0].update(logprob_gap=0.051),
                       lambda data: data[1]["near_ties"][0].update(logprob_gap=float("nan")),
                       lambda data: data[1]["near_ties"][0].update(position=16),
                       lambda data: data[1]["near_ties"][0].update(chosen_id=3618),
                       lambda data: data[0]["near_ties"].append(dict(tie)),
                       lambda data: data[1]["near_ties"].append(dict(tie)),
                       lambda data: data.pop()]:
            data = json.loads(json.dumps(cases));change(data)
            with self.assertRaises(ValueError):
                release_lib.reference_agreement(data)
        report["exact_tokens"] = 48
        with self.assertRaisesRegex(ValueError, "totals"):
            release_lib.verify_reference_agreement(report)

    def test_reference_near_tie_rejects_unrelated_choices_and_large_probability_gaps(self):
        expected = {"completion_probabilities": [{"id": token} for token in range(100, 116)]}
        expected["completion_probabilities"][11]["top_logprobs"] = [{"id": 111, "logprob": -1.0}, {"id": 999, "logprob": -1.032}]
        actual = list(range(100, 116));actual[11] = 999
        tie = release_gpu.reference_near_tie(actual, expected)
        self.assertEqual((tie["position"], tie["rank"]), (11, 2))
        self.assertAlmostEqual(tie["logprob_gap"], 0.032)
        for selected, low in [(998, -1.032), (999, -1.051), (999, float("nan")), (999, -0.9)]:
            data = json.loads(json.dumps(expected));data["completion_probabilities"][11]["top_logprobs"][1]["logprob"] = low
            tokens = actual.copy();tokens[11] = selected
            with self.assertRaises(ValueError):
                release_gpu.reference_near_tie(tokens, data)
        with self.assertRaises(ValueError):
            release_gpu.reference_near_tie(actual[:-1], expected)

    def test_near_tie_branch_requires_matching_prompt_all_suffix_tokens_and_complete_detokenization(self):
        actual = list(range(100, 116));actual[11] = 999
        tie = {"position": 11, "chosen_id": 999, "reference_id": 111, "rank": 2, "logprob_gap": 0.032}
        content = "A complete alternate continuation.\n Second line. "
        log = "info(zinc): Output text: " + content + "\n"
        expected = {"content": "Original reference remains visible"}
        replies = [{"tokens": [1, 2]}, {"completion_probabilities": [{"id": token} for token in actual[12:]]}, {"content": content}]
        with patch.object(release_gpu, "output", return_value="[1, 2]\n"), \
             patch.object(release_gpu, "json_request", side_effect=replies) as request:
            release_gpu.qualify_reference_branch(1, self.root, self.root / "model", "rendered", expected, actual, log, tie, self.root, {})
        self.assertEqual(request.call_args_list[1].args[2]["prompt"], [1, 2] + actual[:12])
        self.assertEqual(request.call_args_list[1].args[2]["n_predict"], 4)
        self.assertEqual(expected["content"], "Original reference remains visible")
        self.assertEqual(expected["validated_content"], content)
        for native, wrong in [("[9]\n", replies), ("[1, 2]\n", [replies[0], {"completion_probabilities": [{"id": 7}]}]),
                              ("[1, 2]\n", [*replies[:2], {"content": content.rstrip()}])]:
            data = {"content": "original"}
            with patch.object(release_gpu, "output", return_value=native), patch.object(release_gpu, "json_request", side_effect=wrong), self.assertRaises(ValueError):
                release_gpu.qualify_reference_branch(1, self.root, self.root / "model", "rendered", data, actual, log, tie, self.root, {})
            self.assertNotIn("validated_content", data)

    def test_backend_device_indices_can_differ_but_the_physical_reservation_cannot(self):
        base = {"device": 1, "lock_directory": "<shared-reservation>", "models": {"one": "<model>"},
                "profiles": {"rocm-rdna4": {"device": 0, "reference_bin": "<HIP-reference>"}}}
        self.assertEqual(release_gpu.worker_for_profile(base, "vulkan-rdna4")["device"], 1)
        rocm = release_gpu.worker_for_profile(base, "rocm-rdna4")
        self.assertEqual(rocm["device"], 0)
        self.assertEqual(rocm["lock_directory"], base["lock_directory"])
        self.assertEqual(rocm["models"], base["models"])
        base["profiles"]["rocm-rdna4"]["lock_directory"] = "<different-reservation>"
        with self.assertRaisesRegex(ValueError, "shared reservation"):
            release_gpu.worker_for_profile(base, "rocm-rdna4")

    def test_gpu_process_is_reaped_on_test_failure(self):
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with release_gpu.process([sys.executable, "-c", "import time; time.sleep(120)"], self.root / "child.log", self.root, os.environ) as child:
                raise RuntimeError("fixture failure")
        self.assertIsNotNone(child.poll())

    def test_group_cancellation_waits_for_a_profiles_nested_server_cleanup(self):
        ready, done = self.root / "server-ready", self.root / "profile-cleanup.json"
        server, controller = self.root / "server.py", self.root / "profile.py"
        # CPU-only stand-ins reproduce the production process topology: the
        # group owns a helper, and the helper owns a separate server session.
        server.write_text("import os,pathlib,signal,time\n"
                          "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                          f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()))\n"
                          "while True: time.sleep(1)\n")
        controller.write_text("import json,os,pathlib,sys,time\n"
                              f"sys.path.insert(0,{str(ROOT / 'tools')!r})\n"
                              "import release_gpu\nfrom release_lib import install_signal_handlers\n"
                              "install_signal_handlers()\n"
                              "try:\n"
                              f" with release_gpu.process([sys.executable,{str(server)!r}],{str(self.root / 'server.log')!r},{str(self.root)!r},dict(os.environ)) as child:\n"
                              "  try:\n   while True: time.sleep(1)\n"
                              "  finally: time.sleep(2)\n"
                              "finally:\n"
                              f" pathlib.Path({str(done)!r}).write_text(json.dumps({{'child_returncode':child.poll()}}))\n")
        server_pid, reaped = None, False
        try:
            with self.assertRaisesRegex(RuntimeError, "cancel profile"):
                with release_gpu.process([sys.executable, str(controller)], self.root / "profile.log", self.root, dict(os.environ),
                                         grace_seconds=release_gpu_group.CHILD_CLEANUP_GRACE_SECONDS) as child:
                    import time
                    deadline = time.monotonic() + 10
                    while not ready.exists():
                        self.assertIsNone(child.poll())
                        self.assertLess(time.monotonic(), deadline, "CPU server fixture did not start")
                        time.sleep(0.05)
                    server_pid = int(ready.read_text())
                    raise RuntimeError("cancel profile")
            # The helper must finish its own cleanup, not merely disappear
            # while its separately grouped child is still running.
            self.assertEqual(read_json(done), {"child_returncode": -9})
            reaped = True
        finally:
            if server_pid is not None and not reaped:
                import signal
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(server_pid, signal.SIGKILL)

    def test_signing_timeouts_do_not_expose_secret_arguments(self):
        command = ["security", "-p", "synthetic-test-password"]
        with patch.object(release_sign.subprocess, "run", side_effect=subprocess.TimeoutExpired(command, 1)):
            with self.assertRaisesRegex(ValueError, "timed out: security") as caught:
                release_sign.run(command, timeout=1)
        self.assertNotIn("synthetic-test-password", str(caught.exception))

    def test_gpu_process_is_reaped_when_controller_receives_sigterm(self):
        script = ("import sys,time,os; sys.path.insert(0,sys.argv[1]); import release_gpu; "
                  "release_gpu.install_signal_handlers()\n"
                  "with release_gpu.process([sys.executable,'-c','import time;time.sleep(120)'], sys.argv[2], '.', os.environ) as child:\n"
                  " print(child.pid,flush=True)\n time.sleep(120)\n")
        controller = subprocess.Popen([sys.executable, "-c", script, str(ROOT / "tools"), str(self.root / "signal-child.log")],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            child_pid = int(controller.stdout.readline())
            controller.terminate()
            controller.communicate(timeout=10)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.communicate(timeout=10)


class ReleaseSourceFreezeTests(unittest.TestCase):
    """Source binding and publication order, including real Git histories."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="zinc-source-freeze-")
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repository"
        self.repo.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Release fixture")
        self.git("config", "user.email", "release-fixture@example.invalid")
        self.env = patch.dict(os.environ, {"GITHUB_SHA": ""})
        self.env.start()
        self.base = self.commit("tools/release.py", "qualification-v1\n")
        self.git("update-ref", "refs/remotes/origin/main", self.base)

    def tearDown(self):
        self.env.stop()
        self.temporary.cleanup()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE,
                                       text=True, timeout=10).strip()

    def commit(self, name, text):
        file = self.repo / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
        self.git("add", "--", name)
        self.git("commit", "-m", "CPU-only source-freeze fixture")
        return self.git("rev-parse", "HEAD")

    def resolver(self, ref, frozen):
        output = self.root / "outputs"
        variables = {p["runner_variable"]: '["self-hosted","zinc-fixture"]'
                     for p in release_lib.release_policy(policy())["gpu_profiles"]}
        with patch.object(release, "ROOT", self.repo), \
                patch.dict(os.environ, {"GITHUB_SHA": frozen, "GITHUB_OUTPUT": str(output),
                                        "RELEASE_VARIABLES": json.dumps(variables)}), \
                contextlib.redirect_stdout(io.StringIO()):
            release.resolve(SimpleNamespace(version="0.1.0-rc.1", ref=ref))
        return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_main_resolution_uses_the_dispatch_snapshot_even_when_origin_moves(self):
        newer = self.commit("docs/guide.md", "Later documentation\n")
        self.git("update-ref", "refs/remotes/origin/main", newer)
        self.git("checkout", "--detach", self.base)
        result = self.resolver("main", self.base)
        self.assertEqual(result["commit"], self.base)
        self.assertNotEqual(result["commit"], newer)
        with self.assertRaisesRegex(ValueError, "ancestor of the frozen workflow"):
            self.resolver(newer, self.base)

    def test_explicit_engine_and_docs_ancestors_keep_the_same_qualification_contract(self):
        engine = self.commit("src/main.zig", "// older selectable engine\n")
        newer = self.commit("docs/guide.md", "Documentation-only update\n")
        self.git("update-ref", "refs/remotes/origin/main", newer)
        release_lib.verify_release_sources(self.base, newer, self.repo)
        self.assertEqual(self.resolver(engine, newer)["commit"], engine)

    def test_changed_added_and_deleted_release_inputs_are_rejected_before_outputs(self):
        for name in ("tools/release.py", "tools/release_gpu_group.py", "tools/release_workflow.test.ts",
                     "config/release/targets.json", ".github/workflows/release-candidate.yml",
                     "tools/validate_gemma_router.c", "scripts/release_worker/launch.py",
                     "build.zig", "src/model/catalog.zig", "tools/setup_opencode.mjs",
                     ".github/workflows/new-qualification.yml", "tools/new-gate.py", "future-driver/setup.sh"):
            self.git("reset", "--hard", self.base)
            newer = self.commit(name, "changed qualification input\n")
            self.git("update-ref", "refs/remotes/origin/main", newer)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "qualification inputs changed"):
                self.resolver(self.base, newer)
            self.assertFalse((self.root / "outputs").exists())
        self.git("reset", "--hard", self.base)
        self.git("rm", "tools/release.py")
        self.git("commit", "-m", "Deleted qualification helper")
        with self.assertRaisesRegex(ValueError, "qualification inputs changed"):
            release_lib.verify_release_sources(self.base, self.git("rev-parse", "HEAD"), self.repo)

    def test_future_or_divergent_source_is_not_an_ancestor_even_with_unchanged_gates(self):
        main = self.commit("docs/main.md", "Main history\n")
        self.git("checkout", "-b", "other", self.base)
        other = self.commit("docs/other.md", "Divergent history\n")
        with self.assertRaisesRegex(ValueError, "ancestor of the frozen workflow"):
            release_lib.verify_release_sources(other, main, self.repo)

    def test_dispatch_sha_mismatch_and_dirty_checkout_fail_before_output(self):
        newer = self.commit("docs/guide.md", "Newer dispatch\n")
        self.git("checkout", "--detach", self.base)
        with patch.dict(os.environ, {"GITHUB_SHA": newer}), self.assertRaisesRegex(ValueError, "dispatched workflow SHA"):
            release_lib.frozen_checkout(self.repo)
        self.git("update-ref", "refs/remotes/origin/main", newer)
        (self.repo / "untracked-input").write_text("unreviewed\n")
        with self.assertRaisesRegex(ValueError, "checkout is dirty"):
            self.resolver("main", self.base)
        self.assertFalse((self.root / "outputs").exists())

    def test_publisher_rejects_changed_gates_before_assets_attestations_or_writes(self):
        newer = self.commit("tools/release_gpu.py", "new mandatory GPU check\n")
        self.git("update-ref", "refs/remotes/origin/main", newer)
        run = {"id": 123, "path": release_publish.WORKFLOW, "event": "workflow_dispatch", "head_branch": "main",
               "head_repository": {"full_name": release_publish.REPOSITORY}, "status": "completed",
               "conclusion": "success", "head_sha": self.base}
        data = {"candidate_run": "123", "workflow_sha": self.base, "commit": self.base}
        # Both a newer publisher and main advancing during approval must fail.
        for publisher in (self.base, newer):
            self.git("checkout", "--detach", publisher)
            with self.subTest(publisher=publisher), patch.object(release_publish, "ROOT", self.repo), \
                    patch.object(release_publish, "verify_candidate", return_value=data), \
                    patch.object(release_publish, "gh") as attest, patch.object(release_publish, "api") as writes, \
                    patch.dict(os.environ, {"GITHUB_SHA": publisher}), \
                    self.assertRaisesRegex(ValueError, "qualification inputs changed"):
                release_publish.verify_bundle(self.root / "not-yet-inspected", "123", run)
            attest.assert_not_called()
            writes.assert_not_called()

    def test_publication_refresh_detects_main_advancing_after_checkout(self):
        origin = self.root / "origin.git"
        self.git("init", "--bare", str(origin))
        self.git("remote", "add", "origin", str(origin))
        newer = self.commit("tools/release_gpu.py", "New qualification check during approval\n")
        self.git("push", "origin", "HEAD:refs/heads/main")
        self.git("update-ref", "refs/remotes/origin/main", self.base)
        self.git("checkout", "--detach", self.base)
        data = {"commit": self.base, "workflow_sha": self.base}
        with patch.object(release_publish, "ROOT", self.repo), patch.dict(os.environ, {"GITHUB_SHA": self.base}), \
                self.assertRaisesRegex(ValueError, "qualification inputs changed"):
            release_publish.verify_publication_sources(data, self.base, refresh=True)
        self.assertEqual(self.git("rev-parse", "origin/main"), newer)

    def test_publish_refresh_failure_prevents_all_tag_and_release_mutation(self):
        run = {"id": 123, "path": release_publish.WORKFLOW, "event": "workflow_dispatch", "head_branch": "main",
               "head_repository": {"full_name": release_publish.REPOSITORY}, "status": "completed",
               "conclusion": "success", "head_sha": self.base}
        data = {"version": "0.1.0-rc.1", "commit": self.base, "workflow_sha": self.base,
                "required_profiles": [], "artifacts": []}
        for error in (ValueError("qualification inputs changed"), subprocess.CalledProcessError(1, ["git", "fetch"])):
            with self.subTest(error=type(error).__name__), \
                    patch.object(sys, "argv", ["release_publish.py", "--run", "123", "--directory", str(self.root), "--publish"]), \
                    patch.dict(os.environ, {"GITHUB_REF": "refs/heads/main", "GITHUB_REPOSITORY": release_publish.REPOSITORY,
                                            "GITHUB_OUTPUT": "", "GITHUB_STEP_SUMMARY": ""}), \
                    patch.object(release_publish, "api", return_value=run), \
                    patch.object(release_publish, "validate_environment"), \
                    patch.object(release_publish, "verify_bundle", return_value=data), \
                    patch.object(release_publish.subprocess, "check_output", return_value=""), \
                    patch.object(release_publish, "validate_catalog"), \
                    patch.object(release_publish, "verify_publication_sources", side_effect=error) as refresh, \
                    patch.object(release_publish, "publish") as mutate, \
                    self.assertRaises(type(error)):
                release_publish.main()
            refresh.assert_called_once_with(data, self.base, refresh=True)
            mutate.assert_not_called()

    def test_source_inspection_timeouts_fail_closed_without_echoing_private_output(self):
        for args in (["rev-parse", "HEAD"], ["merge-base", "--is-ancestor", self.base, self.base],
                     ["diff", "--name-only", self.base, self.base], ["fetch", "origin"]):
            error = subprocess.TimeoutExpired(["git", *args], 1, output=b"<private-output>")
            with self.subTest(command=args[0]), patch.object(release_lib.subprocess, "run", side_effect=error), \
                    self.assertRaisesRegex(ValueError, "release inspection timed out") as failure:
                release_lib.git_inspection(args, "release inspection", self.repo, timeout=1)
            self.assertNotIn("private-output", str(failure.exception))


class ReleaseSSHTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="zinc-ssh-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.request = {"schema": release_ssh.SCHEMA, "mode": "candidate", "profile": "vulkan-intel-xe2",
                        "controller_run": 42, "controller_attempt": 1, "artifact_id": 7, "artifact_run": 42,
                        "validation_commit": COMMIT, "nonce": "a" * 64, "size_bytes": 10, "exercise_models": True}
        self.config = {"profiles": ["vulkan-intel-xe2"], "actors": ["maintainer"],
                       "worker_config": str(self.root / "worker.json"), "work_root": str(self.root)}
        Path(self.config["worker_config"]).write_text("{}")
        self.run = {"id": 42, "repository": {"id": 1, "full_name": "zolotukhin/zinc"},
                    "head_repository": {"id": 1, "full_name": "zolotukhin/zinc"}, "head_branch": "main", "head_sha": COMMIT,
                    "path": ".github/workflows/release-candidate.yml", "event": "workflow_dispatch",
                    "actor": {"login": "maintainer"}, "triggering_actor": {"login": "maintainer"},
                    "run_attempt": 1, "status": "in_progress", "conclusion": None}
        self.artifact = {"id": 7, "name": "candidate-inputs", "expired": False, "digest": "sha256:" + "b" * 64,
                         "size_in_bytes": 10, "workflow_run": {"id": 42, "head_branch": "main", "head_sha": COMMIT,
                                                              "repository_id": 1, "head_repository_id": 1}}

    def metadata(self, path):
        return {"actions/runs/42": self.run, "actions/artifacts/7": self.artifact}[path]

    def payload(self):
        import zipfile
        path = self.root / "inputs.zip"
        manifest = {"commit": COMMIT, "candidate_run": "42", "workflow_sha": COMMIT}
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as bundle:
            bundle.writestr("release-manifest.json", json.dumps(manifest))
            bundle.writestr("SHA256SUMS.txt", "fixture\n")
            bundle.writestr("zinc-v0.1.0-linux-x86_64-vulkan.tar.gz", b"fixture candidate bytes")
        self.artifact.update(size_in_bytes=path.stat().st_size, digest=release_ssh.digest(path))
        self.request["size_bytes"] = path.stat().st_size
        return path.read_bytes()

    def test_endpoint_uses_public_metadata_and_checks_both_actor_identities(self):
        with patch.object(release_ssh, "github_json", side_effect=self.metadata):
            self.assertEqual(release_ssh.authenticate(self.request, self.config)[0], self.artifact["digest"])
            changes = [{"head_branch": "feature"}, {"path": ".github/workflows/release-preview.yml"},
                       {"event": "pull_request"}, {"actor": {"login": "stranger"}},
                       {"triggering_actor": {"login": "stranger"}}, {"status": "completed"}, {"run_attempt": 2},
                       {"head_repository": {"id": 2, "full_name": "fork/zinc"}}]
            original = copy.deepcopy(self.run)
            for changed in changes:
                self.run = {**original, **changed}
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    release_ssh.authenticate(self.request, self.config)

    def test_artifact_cannot_be_expired_unsigned_swapped_or_from_another_run(self):
        changes = [{"expired": True}, {"digest": None}, {"name": "preview-linux-x86_64-vulkan"}, {"size_in_bytes": 11},
                   {"workflow_run": {**self.artifact["workflow_run"], "id": 43}},
                   {"workflow_run": {**self.artifact["workflow_run"], "head_repository_id": 2}},
                   {"workflow_run": {**self.artifact["workflow_run"], "head_sha": "2" * 40}}]
        original = copy.deepcopy(self.artifact)
        for changed in changes:
            self.artifact = {**original, **changed}
            with self.subTest(changed=changed), patch.object(release_ssh, "github_json", side_effect=self.metadata), self.assertRaises(ValueError):
                release_ssh.authenticate(self.request, self.config)

    def test_request_rejects_caller_supplied_commands_urls_and_skipped_candidate_checks(self):
        for changed in ({"command": "echo injected"}, {"source_url": "https://example.com/other.git"},
                        {"exercise_models": False}, {"controller_run": True}, {"profile": "vulkan-rdna4"},
                        {"validation_commit": "main"}, {"nonce": "short"}, {"size_bytes": release_ssh.MAX_PAYLOAD + 1}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                release_ssh.request_identity({**self.request, **changed}, self.config["profiles"])

    def test_ancestor_check_never_accepts_divergence_or_an_unrelated_merge_base(self):
        good = {"status": "ahead", "behind_by": 0, "base_commit": {"sha": COMMIT}, "merge_base_commit": {"sha": COMMIT}}
        with patch.object(release_ssh, "github_json", return_value=good):
            release_ssh.ancestor(COMMIT, "2" * 40)
        for changed in ({"status": "diverged"}, {"behind_by": 1}, {"merge_base_commit": {"sha": "3" * 40}}):
            with self.subTest(changed=changed), patch.object(release_ssh, "github_json", return_value={**good, **changed}), self.assertRaises(ValueError):
                release_ssh.ancestor(COMMIT, "2" * 40)

    def test_redirect_strips_the_github_token_and_rejects_plain_http(self):
        import urllib.request
        request = urllib.request.Request(release_ssh.API + "actions/artifacts/7/zip", headers={"Authorization": "Bearer <secret>"})
        redirected = release_ssh.SafeRedirect().redirect_request(request, None, 302, "Found", {}, "https://example.com/artifact.zip")
        self.assertFalse(redirected.has_header("Authorization"))
        with self.assertRaises(ValueError):
            release_ssh.SafeRedirect().redirect_request(request, None, 302, "Found", {}, "http://example.com/artifact.zip")

    def test_zip_rejects_traversal_symlinks_directories_duplicates_and_expansion(self):
        import zipfile
        for filename in ("../escape", "/escape", "nested/release-manifest.json", "extra.py", "folder/"):
            archive = self.root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(filename, b"bad")
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                release_ssh.unpack_payload(archive, self.root / ("out-" + str(len(list(self.root.iterdir())))), "candidate")
        archive = self.root / "link.zip"
        link = zipfile.ZipInfo("release-manifest.json")
        link.external_attr = 0o120777 << 16
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(link, "target")
        with self.assertRaises(ValueError):
            release_ssh.unpack_payload(archive, self.root / "symlink", "candidate")
        import warnings
        with zipfile.ZipFile(archive, "w") as bundle, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bundle.writestr("release-manifest.json", b"first")
            bundle.writestr("release-manifest.json", b"second")
        with self.assertRaisesRegex(ValueError, "ZIP members"):
            release_ssh.unpack_payload(archive, self.root / "duplicates", "candidate")
        payload = self.payload()
        with patch.object(release_ssh, "MAX_PAYLOAD", 1), self.assertRaises(ValueError):
            release_ssh.unpack_payload(self.root / "inputs.zip", self.root / "expanded", "candidate")
        self.assertTrue(payload)

    def test_truncated_header_and_partial_heartbeat_cannot_hold_a_gpu_job(self):
        import time
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "rb", buffering=0) as reader, os.fdopen(write_fd, "wb", buffering=0) as writer:
            writer.write(b'{"schema":')
            with patch.object(release_ssh, "LEASE_SECONDS", 0.05), self.assertRaisesRegex(ValueError, "header timed out"):
                release_ssh.read_header(reader)
        for data in (b"P", b"BAD\n", b""):
            read_fd, write_fd = os.pipe()
            with os.fdopen(read_fd, "rb", buffering=0) as reader, os.fdopen(write_fd, "wb", buffering=0) as writer:
                writer.write(data)
                with release_ssh.owned_process([sys.executable, "-c", "import time; time.sleep(30)"], self.root / "lease.log", dict(os.environ)) as child:
                    started = time.monotonic()
                    with self.assertRaises(ValueError):
                        release_ssh.wait_with_lease(child, reader, timeout=1, lease=0.05)
                    self.assertLess(time.monotonic() - started, 2)
                self.assertIsNotNone(child.poll())

    def test_disconnect_and_remote_failure_reap_only_the_owned_cpu_child(self):
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        with os.fdopen(read_fd, "rb", buffering=0) as reader:
            with release_ssh.owned_process([sys.executable, "-c", "import time; time.sleep(30)"], self.root / "disconnect.log", dict(os.environ)) as child:
                with self.assertRaisesRegex(ValueError, "disconnected"):
                    release_ssh.wait_with_lease(child, reader)
            self.assertIsNotNone(child.poll())
        with release_ssh.owned_process([sys.executable, "-c", "raise SystemExit(1)"], self.root / "failed.log", dict(os.environ)) as child:
            child.wait(timeout=10)
            with self.assertRaisesRegex(ValueError, "packaged checks failed"):
                release_ssh.wait_with_lease(child, None)

    def test_ssh_cleanup_includes_marked_kernel_helpers_but_excludes_other_sessions(self):
        marker = "9" + "1" * 78
        for pid, name, identity in ((123, "validate-attent", marker), (124, "zinc", "42"), (125, "python3", marker)):
            process = self.root / str(pid)
            process.mkdir()
            (process / "comm").write_text(name)
            (process / "environ").write_bytes(b"GITHUB_RUN_ID=" + identity.encode() + b"\0")
            (process / "stat").write_text(str(pid) + " (fixture) " + " ".join(["S"] + ["0"] * 18 + ["100"]))
        original = Path
        def private_proc(path):
            text = str(path)
            return self.root / text.removeprefix("/proc/") if text.startswith("/proc/") else self.root if text == "/proc" else original(path)
        with patch.object(release_worker_guard, "Path", side_effect=private_proc):
            self.assertEqual(release_worker_guard.owned_gpu_processes(marker), [])
            self.assertEqual(sorted(release_worker_guard.owned_gpu_processes(marker, include_helpers=True)), [(123, "100"), (125, "100")])
        with self.assertRaisesRegex(ValueError, "unique SSH session"):
            release_worker_guard.cleanup("42", include_helpers=True)

    def test_readiness_requires_a_successful_preview_and_identical_frozen_policy(self):
        self.request.update(mode="readiness", artifact_run=41)
        self.run["path"] = ".github/workflows/release-worker-check.yml"
        preview = {**self.run, "id": 41, "path": ".github/workflows/release-preview.yml", "status": "completed", "conclusion": "success"}
        self.artifact.update(name="preview-linux-x86_64-vulkan", workflow_run={**self.artifact["workflow_run"], "id": 41})
        def metadata(path):
            return preview if path == "actions/runs/41" else self.metadata(path)
        with patch.object(release_ssh, "github_json", side_effect=metadata):
            release_ssh.authenticate(self.request, self.config)
            preview["conclusion"] = "failure"
            with self.assertRaisesRegex(ValueError, "did not pass"):
                release_ssh.authenticate(self.request, self.config)
        source = self.root / "source"
        (source / "config/release").mkdir(parents=True)
        (source / "config/release/targets.json").write_bytes(b"current-policy")
        encoded = {"type": "file", "path": "config/release/targets.json", "encoding": "base64",
                   "content": base64.b64encode(b"different-policy").decode()}
        with patch.object(release_ssh, "github_json", return_value=encoded), self.assertRaisesRegex(ValueError, "policy differs"):
            release_ssh.preview_policy(COMMIT, source)
        encoded["content"] = base64.b64encode(b"current-policy").decode()
        with patch.object(release_ssh, "github_json", return_value=encoded):
            release_ssh.preview_policy(COMMIT, source)

    def test_controller_streams_original_zip_and_never_forwards_github_credentials(self):
        payload = self.payload()
        expected = {"profile": self.request["profile"], "archive_sha256": "c" * 64}
        report = {"schema_version": "zinc-worker-readiness-v1", "release_qualification": False, "passed": True,
                  "models_exercised": True, **expected}
        script = ("import json,os,sys\n"
                  "assert 'GH_TOKEN' not in os.environ and 'GITHUB_TOKEN' not in os.environ\n"
                  "r=json.loads(sys.stdin.buffer.readline())\n"
                  "payload=sys.stdin.buffer.read(r['size_bytes'])\n"
                  "assert len(payload)==r['size_bytes']\n"
                  "reply={k:r[k] for k in ('nonce','profile','controller_run','controller_attempt','artifact_id','validation_commit')}\n"
                  "reply.update(schema=" + repr(release_ssh.SCHEMA) + ",artifact_digest=" + repr(self.artifact["digest"]) +
                  ",helpers_sha256=" + repr(release_ssh.helper_digests()) + ",report=" + repr(report) + ")\n"
                  "print(json.dumps(reply))\n")
        def download(artifact, destination):
            destination.write_bytes(payload)
        env = {"GITHUB_RUN_ID": "42", "GITHUB_RUN_ATTEMPT": "1", "GH_TOKEN": "<secret>", "GITHUB_TOKEN": "<secret>"}
        destination = self.root / "passed.json"
        with patch.dict(os.environ, env), patch.object(release_ssh, "ssh_command", return_value=[sys.executable, "-c", script]), \
                patch.object(release_ssh, "find_artifact", return_value=self.artifact), patch.object(release_ssh, "download_artifact", side_effect=download):
            release_ssh.check(self.config, "readiness", self.request["profile"], self.root, destination, COMMIT,
                              preview_run=41, expected=expected)
        self.assertEqual(json.loads(destination.read_text())["transport"]["artifact_digest"], self.artifact["digest"])

    def test_oversized_ssh_output_is_bounded_and_the_controller_reaps_its_child(self):
        payload = self.payload()
        command = [sys.executable, "-c", "import sys,time; sys.stdout.write('x'*1024); sys.stdout.flush(); time.sleep(30)"]
        children, original = [], subprocess.Popen
        def launch(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child
        destination = self.root / "not-passed.json"
        with patch.dict(os.environ, {"GITHUB_RUN_ID": "42", "GITHUB_RUN_ATTEMPT": "1"}), \
                patch.object(release_ssh, "ssh_command", return_value=command), patch.object(release_ssh, "MAX_REPORT", 128), \
                patch.object(release_ssh, "find_artifact", return_value=self.artifact), \
                patch.object(release_ssh, "download_artifact", side_effect=lambda a, p: p.write_bytes(payload)), \
                patch.object(release_ssh.subprocess, "Popen", side_effect=launch), self.assertRaisesRegex(ValueError, "reply is too large"):
            release_ssh.check(self.config, "readiness", self.request["profile"], self.root, destination, COMMIT, preview_run=41)
        self.assertFalse(destination.exists())
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    def test_agent_checks_zip_digest_before_fetching_or_executing_code(self):
        payload = self.payload()
        self.artifact["digest"] = "sha256:" + "0" * 64
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "rb", buffering=0) as reader, os.fdopen(write_fd, "wb", buffering=0) as writer:
            writer.write(json.dumps(self.request).encode() + b"\n" + payload)
            with patch.object(release_ssh.sys, "platform", "linux"), patch.object(release_ssh, "github_json", side_effect=self.metadata), \
                    patch.object(release_ssh, "checkout_source") as checkout, self.assertRaisesRegex(ValueError, "ZIP digest differs"):
                release_ssh.agent(self.config, reader, io.BytesIO())
            checkout.assert_not_called()

    def test_agent_runs_the_existing_harness_and_emits_report_only_after_cleanup(self):
        payload = self.payload()
        report = {"schema_version": 1, "profile": self.request["profile"], "checks": {"cleanup": True}}
        def checkout(commit, destination):
            self.assertEqual(commit, COMMIT)
            (destination / "tools").mkdir(parents=True)
            # CPU fixture stands in for the unchanged packaged GPU validator.
            (destination / "tools/release_gpu.py").write_text(
                "import json, pathlib, sys\npathlib.Path(sys.argv[sys.argv.index('--output') + 1]).write_text(" + repr(json.dumps(report)) + ")\n")
        result = io.BytesIO()
        def cleanup(identifier):
            self.assertEqual(identifier, "9" + str(int(self.request["nonce"], 16)).zfill(78))
            self.assertEqual(result.getvalue(), b"")
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, "rb", buffering=0) as reader, os.fdopen(write_fd, "wb", buffering=0) as writer:
            writer.write(json.dumps(self.request).encode() + b"\n" + payload)
            with patch.object(release_ssh.sys, "platform", "linux"), patch.object(release_ssh, "github_json", side_effect=self.metadata), \
                    patch.object(release_ssh, "checkout_source", side_effect=checkout), patch.object(release_ssh, "cleanup_session", side_effect=cleanup) as cleaned:
                release_ssh.agent(self.config, reader, result)
            cleaned.assert_called_once()
        reply = json.loads(result.getvalue())
        self.assertEqual(reply["report"], report)
        self.assertEqual(reply["artifact_digest"], self.artifact["digest"])
        self.assertEqual(list(self.root.glob("qualification-*")), [])

    def test_report_replay_source_and_archive_substitution_are_rejected(self):
        expected = {"profile": self.request["profile"], "archive_sha256": "c" * 64}
        report = {"schema_version": 1, "checks": {"cleanup": True}, **expected}
        reply = {key: self.request[key] for key in ("nonce", "profile", "controller_run", "controller_attempt", "artifact_id", "validation_commit")}
        reply.update(schema=release_ssh.SCHEMA, artifact_digest=self.artifact["digest"], helpers_sha256=release_ssh.helper_digests(), report=report)
        self.assertEqual(release_ssh.verify_reply(copy.deepcopy(reply), self.request, self.artifact, expected)["transport"]["kind"], "ssh")
        for changed in ({"nonce": "d" * 64}, {"controller_attempt": 2}, {"validation_commit": "2" * 40},
                        {"artifact_digest": "sha256:" + "d" * 64}, {"helpers_sha256": {}},
                        {"report": {**report, "archive_sha256": "e" * 64}}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                release_ssh.verify_reply({**reply, **changed}, self.request, self.artifact, expected)

    def test_ssh_client_cannot_use_an_agent_forwarding_or_unverified_host_key(self):
        key, known = self.root / "key", self.root / "known_hosts"
        key.touch(mode=0o600)
        known.touch()
        config = {"kind": "ssh", "profiles": ["vulkan-intel-xe2"], "host": "example.com", "user": "fixture", "port": 22,
                  "identity_file": str(key), "known_hosts_file": str(known)}
        command = release_ssh.ssh_command(config)
        for argument in ("-oIdentityAgent=none", "-oStrictHostKeyChecking=yes", "-oClearAllForwardings=yes", "-T"):
            self.assertIn(argument, command)
        self.assertEqual(command[-1], "example.com")
        for changed in ({"host": "-oProxyCommand=echo"}, {"user": "bad;echo"}, {"port": True}, {"profiles": ["rocm-rdna4"]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                release_ssh.ssh_command({**config, **changed})


if __name__ == "__main__":
    unittest.main()
