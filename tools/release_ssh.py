#!/usr/bin/env python3
"""Restricted SSH transport for the existing packaged GPU qualification checks.

Install a reviewed copy with `agent --config <private-json>` as a forced SSH
command. The endpoint independently authenticates GitHub artifact ZIPs and
fetches validation code from public main history; neither code nor credentials
arrive through the SSH request. This is a trusted non-root compute account,
not a filesystem sandbox for untrusted pull requests.
"""

import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
import re
import secrets
import select
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY = "zolotukhin/zinc"
API = "https://api.github.com/repos/" + REPOSITORY + "/"
SOURCE_URL = "https://github.com/" + REPOSITORY + ".git"
SCHEMA = "zinc-release-ssh-v1"
MAX_PAYLOAD = 384 * 1024 * 1024
MAX_REPORT = 2 * 1024 * 1024
MAX_HEADER = 16384
LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 10
CHECK_SECONDS = 150 * 60
SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def positive(value):
    return type(value) is int and value > 0


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return "sha256:" + result.hexdigest()


def helper_digests():
    return {name: digest(Path(__file__).with_name(name)).removeprefix("sha256:")
            for name in ("release_ssh.py", "release_worker_guard.py")}


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, *args):
        redirected = super().redirect_request(request, *args)
        require(redirected is not None and redirected.full_url.startswith("https://"), "unsafe artifact redirect")
        # GitHub redirects ZIP downloads to a signed storage URL. A repository
        # token must never accompany that redirect, or be sent to the GPU host.
        redirected.remove_header("Authorization")
        return redirected


def response(path, token=None):
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "zinc-release-ssh"}
    if token:
        headers["Authorization"] = "Bearer " + token
    return urllib.request.build_opener(SafeRedirect()).open(urllib.request.Request(API + path, headers=headers), timeout=45)


def github_json(path):
    # Deliberately unauthenticated, including on the remote endpoint.
    with response(path) as stream:
        raw = stream.read(MAX_REPORT + 1)
    require(len(raw) <= MAX_REPORT, "GitHub metadata is too large")
    return json.loads(raw)


def trusted_run(run, identifier, workflow, events, actors, active=False, attempt=None):
    require(isinstance(run, dict) and run.get("id") == identifier, "workflow run identity differs")
    repository, head = run.get("repository", {}), run.get("head_repository", {})
    require(repository.get("full_name") == REPOSITORY and head.get("full_name") == REPOSITORY and
            positive(repository.get("id")) and repository.get("id") == head.get("id"), "workflow repository differs")
    require(run.get("head_branch") == "main" and run.get("path", "").split("@")[0] == ".github/workflows/" + workflow and
            run.get("event") in events and SHA.fullmatch(run.get("head_sha", "")), "workflow is not trusted main")
    require(run.get("actor", {}).get("login") in actors and run.get("triggering_actor", {}).get("login") in actors,
            "workflow actor is not allowed")
    if active:
        require(run.get("status") == "in_progress" and run.get("conclusion") is None and run.get("run_attempt") == attempt,
                "controller workflow is not the active attempt")
    else:
        require(run.get("status") == "completed" and run.get("conclusion") == "success", "preview workflow did not pass")
    return run["head_sha"]


def artifact_identity(artifact, request, run, name):
    require(isinstance(artifact, dict) and artifact.get("id") == request["artifact_id"] and artifact.get("name") == name and
            artifact.get("expired") is False and isinstance(artifact.get("digest"), str) and DIGEST.fullmatch(artifact["digest"]),
            "artifact metadata is invalid")
    workflow = artifact.get("workflow_run", {})
    require(workflow.get("id") == request["artifact_run"] == run["id"] and workflow.get("head_branch") == "main" and
            workflow.get("head_sha") == run["head_sha"] and
            workflow.get("repository_id") == workflow.get("head_repository_id") == run["repository"]["id"],
            "artifact workflow identity differs")
    require(type(artifact.get("size_in_bytes")) is int and 0 < artifact["size_in_bytes"] <= MAX_PAYLOAD and
            request["size_bytes"] == artifact["size_in_bytes"], "artifact size differs or exceeds the limit")
    return artifact["digest"]


def request_identity(request, profiles):
    keys = {"schema", "mode", "profile", "controller_run", "controller_attempt", "artifact_id", "artifact_run",
            "validation_commit", "nonce", "size_bytes", "exercise_models"}
    require(isinstance(request, dict) and set(request) == keys and request.get("schema") == SCHEMA, "unsupported SSH request")
    require(request["mode"] in ("candidate", "readiness") and request["profile"] in profiles, "SSH profile is not allowed")
    require(all(positive(request[key]) for key in ("controller_run", "controller_attempt", "artifact_id", "artifact_run", "size_bytes")) and
            request["size_bytes"] <= MAX_PAYLOAD, "invalid SSH request identity or size")
    require(isinstance(request["validation_commit"], str) and SHA.fullmatch(request["validation_commit"]) and
            isinstance(request["nonce"], str) and re.fullmatch(r"[0-9a-f]{64}", request["nonce"]), "invalid SSH source or session identity")
    require(type(request["exercise_models"]) is bool and (request["mode"] != "candidate" or request["exercise_models"]),
            "candidate checks cannot be skipped")


def ancestor(base, head):
    if base == head:
        return
    comparison = github_json("compare/" + base + "..." + head + "?per_page=1")
    require(comparison.get("status") == "ahead" and comparison.get("behind_by") == 0 and
            comparison.get("base_commit", {}).get("sha") == base and comparison.get("merge_base_commit", {}).get("sha") == base,
            "validation source is not an ancestor of trusted main")


def authenticate(request, config):
    request_identity(request, config["profiles"])
    controller = github_json("actions/runs/" + str(request["controller_run"]))
    candidate = request["mode"] == "candidate"
    workflow = "release-candidate.yml" if candidate else "release-worker-check.yml"
    head = trusted_run(controller, request["controller_run"], workflow,
                       ("workflow_dispatch", "schedule") if candidate else ("workflow_dispatch",),
                       config["actors"], active=True, attempt=request["controller_attempt"])
    if candidate:
        require(request["artifact_run"] == request["controller_run"], "candidate artifact belongs to a different run")
        artifact_run, name = controller, "candidate-inputs"
        ancestor(request["validation_commit"], head)
    else:
        require(request["validation_commit"] == head, "readiness validation source differs from the controller")
        artifact_run = github_json("actions/runs/" + str(request["artifact_run"]))
        preview = trusted_run(artifact_run, request["artifact_run"], "release-preview.yml", ("push", "workflow_dispatch"), config["actors"])
        ancestor(preview, head)
        # This transport initially serves only the approved Intel Vulkan role.
        name = "preview-linux-x86_64-vulkan"
    artifact = github_json("actions/artifacts/" + str(request["artifact_id"]))
    return artifact_identity(artifact, request, artifact_run, name), artifact_run


def unpack_payload(archive, destination, mode):
    destination.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [item.filename for item in members]
        require(0 < len(members) <= 4 and len(names) == len(set(names)), "invalid artifact ZIP members")
        require(sum(item.file_size for item in members) <= MAX_PAYLOAD, "artifact ZIP expands beyond the limit")
        for item in members:
            filename = item.filename
            allowed = re.fullmatch(r"zinc-v[0-9A-Za-z.+-]+-linux-x86_64-(?:vulkan|rocm)\.tar\.gz", filename)
            if mode == "candidate":
                allowed = allowed or filename in ("release-manifest.json", "SHA256SUMS.txt")
            kind = stat.S_IFMT(item.external_attr >> 16)
            require(allowed and kind in (0, stat.S_IFREG) and not item.is_dir() and not item.flag_bits & 1,
                    "unsafe artifact ZIP member")
            with bundle.open(item) as source, (destination / filename).open("xb") as target:
                remaining = item.file_size
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    require(block, "truncated artifact ZIP member")
                    target.write(block)
                    remaining -= len(block)
                require(not source.read(1), "artifact ZIP member exceeds its declared size")


def clean_environment():
    return {"PATH": os.defpath, "HOME": str(Path.home()), "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0"}


def checkout_source(commit, destination):
    env = clean_environment()
    for command in (["git", "init", "--quiet", str(destination)],
                    ["git", "-C", str(destination), "-c", "core.hooksPath=/dev/null", "-c", "protocol.allow=never",
                     "-c", "protocol.https.allow=always", "fetch", "--quiet", "--no-tags", "--depth=1", SOURCE_URL, commit],
                    ["git", "-C", str(destination), "-c", "core.hooksPath=/dev/null", "checkout", "--quiet", "--detach", "FETCH_HEAD"]):
        result = subprocess.run(command, env=env, capture_output=True, timeout=180)
        require(result.returncode == 0, "could not fetch the authenticated validation source")


def preview_policy(commit, source):
    data = github_json("contents/config/release/targets.json?ref=" + commit)
    require(data.get("type") == "file" and data.get("path") == "config/release/targets.json" and data.get("encoding") == "base64",
            "could not authenticate the preview policy")
    raw = base64.b64decode("".join(data["content"].split()), validate=True)
    require(hashlib.sha256(raw).hexdigest() == digest(source / "config/release/targets.json").removeprefix("sha256:"),
            "preview policy differs; build a new packaging preview")


def read_header(stream):
    deadline, raw = time.monotonic() + LEASE_SECONDS, bytearray()
    while len(raw) < MAX_HEADER:
        remaining = deadline - time.monotonic()
        require(remaining > 0 and select.select([stream], [], [], max(0, remaining))[0], "SSH request header timed out")
        block = stream.read(1)
        require(block, "SSH request header is incomplete")
        raw.extend(block)
        if block == b"\n":
            return json.loads(raw)
    raise ValueError("SSH request header is too large")


@contextlib.contextmanager
def owned_process(command, log, env):
    with log.open("wb") as stream:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield child
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)


def wait_with_lease(child, stream, timeout=CHECK_SECONDS, lease=LEASE_SECONDS):
    deadline, last = time.monotonic() + timeout, time.monotonic()
    pending = b""
    while child.poll() is None:
        now = time.monotonic()
        require(now < deadline, "remote GPU validation timed out")
        require(now - last < lease, "SSH controller heartbeat expired")
        ready, _, _ = select.select([stream], [], [], min(1, lease, deadline - now))
        if ready:
            block = stream.read(4096)
            require(block, "SSH controller disconnected")
            pending += block
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                require(line == b"PING", "SSH controller sent an invalid heartbeat")
                last = time.monotonic()
            require(b"PING".startswith(pending), "SSH controller sent an invalid heartbeat")
    require(child.returncode == 0, "remote packaged checks failed; inspect private endpoint logs")


def cleanup_session(identifier):
    # Installed beside this agent, outside the fetched checkout. The cleanup
    # hook pins UID + ownership marker + PID start time before signalling.
    import importlib.util
    spec = importlib.util.spec_from_file_location("installed_cleanup", Path(__file__).with_name("release_worker_guard.py"))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    with contextlib.redirect_stdout(io.StringIO()):
        guard.cleanup(identifier, include_helpers=True)


def agent(config, stream, output):
    require(sys.platform == "linux" and os.getuid() != 0, "SSH GPU agent requires a non-root Linux account")
    require(config.get("profiles") == ["vulkan-intel-xe2"] and isinstance(config.get("actors"), list) and config["actors"] and
            all(isinstance(actor, str) and re.fullmatch(r"[A-Za-z0-9-]+", actor) for actor in config["actors"]), "invalid SSH endpoint policy")
    worker = Path(config["worker_config"]).resolve()
    require(worker.is_file() and worker.is_absolute(), "private endpoint worker configuration is absent")
    work_root = Path(config["work_root"]).resolve()
    require(work_root.is_dir() and work_root.stat().st_uid == os.getuid() and work_root.stat().st_mode & 0o077 == 0,
            "SSH work root must be private and owned by the compute account")
    installed = helper_digests()
    request = read_header(stream)
    expected_digest, artifact_run = authenticate(request, config)
    # One endpoint lease also covers source checkout and forced cleanup. The
    # existing validator separately takes the shared physical GPU reservation.
    import fcntl
    with (work_root / "agent.flock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("SSH endpoint is already leased") from None
        with tempfile.TemporaryDirectory(prefix="qualification-", dir=work_root) as temporary:
            work = Path(temporary)
            archive = work / "artifact.zip"
            with archive.open("xb") as target:
                remaining = request["size_bytes"]
                while remaining:
                    ready, _, _ = select.select([stream], [], [], LEASE_SECONDS)
                    require(ready, "SSH artifact upload timed out")
                    block = stream.read(min(1024 * 1024, remaining))
                    require(block, "SSH artifact upload was truncated")
                    target.write(block)
                    remaining -= len(block)
            require(digest(archive) == expected_digest, "artifact ZIP digest differs from GitHub")
            inputs = work / "inputs"
            unpack_payload(archive, inputs, request["mode"])
            source = work / "source"
            checkout_source(request["validation_commit"], source)
            if request["mode"] == "candidate":
                manifest = json.loads((inputs / "release-manifest.json").read_text())
                require(manifest.get("commit") == request["validation_commit"] and
                        str(manifest.get("candidate_run")) == str(request["controller_run"]) and
                        manifest.get("workflow_sha") == artifact_run["head_sha"], "candidate manifest source or run differs")
            else:
                # The existing readiness helper verifies both the package's
                # source and the frozen policy; an older preview is diagnostic.
                preview_commit = artifact_run["head_sha"]
                preview_version = "0.1.0-dev." + str(artifact_run["id"])
                preview_policy(preview_commit, source)
            report_path = work / "report.json"
            if request["mode"] == "candidate":
                command = [sys.executable, str(source / "tools/release_gpu.py"), "--directory", str(inputs),
                           "--profile", request["profile"], "--output", str(report_path)]
            else:
                command = [sys.executable, str(source / "tools/release_worker.py"), "check", "--directory", str(inputs),
                           "--profile", request["profile"], "--output", str(report_path), "--version", preview_version,
                           "--commit", preview_commit, "--validation-commit", request["validation_commit"]]
                if request["exercise_models"]:
                    command.append("--exercise-models")
            env = clean_environment()
            allowed_env = {"XDG_CACHE_HOME", "VK_DRIVER_FILES", "VK_ICD_FILENAMES", "RADV_PERFTEST", "RADV_DEBUG", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
            overrides = config.get("environment", {})
            require(isinstance(overrides, dict) and set(overrides).issubset(allowed_env) and all(isinstance(v, str) for v in overrides.values()),
                    "invalid endpoint environment overrides")
            env.update(overrides)
            # A unique numeric cleanup identity prevents concurrent jobs from
            # sharing the Actions run's process ownership marker.
            cleanup_id = "9" + str(int(request["nonce"], 16)).zfill(78)
            env.update(ZINC_RELEASE_WORKER_CONFIG=str(worker), GITHUB_RUN_ID=cleanup_id)
            logs = work_root / "logs"
            logs.mkdir(mode=0o700, exist_ok=True)
            require(logs.stat().st_uid == os.getuid() and logs.stat().st_mode & 0o077 == 0, "endpoint log directory must be private")
            try:
                with owned_process(command, logs / (request["nonce"] + ".log"), env) as child:
                    wait_with_lease(child, stream)
            finally:
                cleanup_session(cleanup_id)
            require(report_path.stat().st_size <= MAX_REPORT, "remote GPU report is too large")
            report = json.loads(report_path.read_text())
            require(helper_digests() == installed, "installed SSH helpers changed during qualification")
            result = {"schema": SCHEMA, "nonce": request["nonce"], "profile": request["profile"], "controller_run": request["controller_run"],
                      "controller_attempt": request["controller_attempt"], "artifact_id": request["artifact_id"],
                      "artifact_digest": expected_digest, "validation_commit": request["validation_commit"],
                      "helpers_sha256": installed, "report": report}
            output.write(json.dumps(result, separators=(",", ":")).encode() + b"\n")
            output.flush()


def transport_config():
    path = os.environ.get("ZINC_RELEASE_WORKER_CONFIG")
    if not path:
        return None
    path = Path(path).resolve()
    require(Path(__file__).resolve().parent.parent not in path.parents, "worker configuration must be outside the checkout")
    config = json.loads(path.read_text())
    transport = config.get("transport")
    if transport is None:
        return None
    require(isinstance(transport, dict) and transport.get("kind") == "ssh", "unsupported worker transport")
    return transport


def ssh_command(config):
    require(config.get("profiles") == ["vulkan-intel-xe2"], "SSH controller must serve only the Intel Vulkan profile")
    require(isinstance(config.get("host"), str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", config["host"]) and
            isinstance(config.get("user"), str) and re.fullmatch(r"[a-z_][a-z0-9_-]*", config["user"]) and
            type(config.get("port")) is int and 1 <= config["port"] <= 65535, "invalid private SSH destination")
    for key in ("identity_file", "known_hosts_file"):
        require(isinstance(config.get(key), str) and Path(config[key]).is_absolute() and Path(config[key]).is_file() and
                not any(c in config[key] for c in "\r\n"), "SSH identity and known hosts files must exist outside the checkout")
        require(Path(__file__).resolve().parent.parent not in Path(config[key]).resolve().parents, "SSH credentials must be outside the checkout")
    require(Path(config["identity_file"]).stat().st_mode & 0o077 == 0, "SSH qualification key must be private")
    return ["ssh", "-F", os.devnull, "-T", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oIdentitiesOnly=yes",
            "-oIdentityAgent=none", "-oClearAllForwardings=yes", "-oGlobalKnownHostsFile=" + os.devnull, "-oUpdateHostKeys=no",
            "-oPreferredAuthentications=publickey", "-oPasswordAuthentication=no", "-oKbdInteractiveAuthentication=no",
            "-oConnectTimeout=20", "-oServerAliveInterval=10",
            "-oServerAliveCountMax=3", "-oUserKnownHostsFile=" + config["known_hosts_file"], "-i", config["identity_file"],
            "-p", str(config["port"]), "-l", config["user"], config["host"]]


def find_artifact(run, name):
    found = []
    for page in range(1, 101):
        data = github_json(f"actions/runs/{run}/artifacts?per_page=100&page={page}")
        rows = data.get("artifacts")
        require(isinstance(rows, list), "could not inspect workflow artifacts")
        found.extend(a for a in rows if a.get("name") == name and a.get("expired") is False)
        if len(rows) < 100:
            break
    else:
        raise ValueError("workflow artifact inventory exceeds the limit")
    require(len(found) == 1, "expected exactly one immutable workflow artifact")
    return found[0]


def download_artifact(artifact, destination):
    require(DIGEST.fullmatch(artifact.get("digest", "")) and positive(artifact.get("id")) and
            positive(artifact.get("size_in_bytes")) and artifact["size_in_bytes"] <= MAX_PAYLOAD, "invalid controller artifact metadata")
    with response("actions/artifacts/" + str(artifact["id"]) + "/zip", os.environ["GH_TOKEN"]) as source, destination.open("xb") as target:
        remaining = artifact["size_in_bytes"]
        while remaining:
            block = source.read(min(1024 * 1024, remaining))
            require(block, "artifact download was truncated")
            target.write(block)
            remaining -= len(block)
        require(not source.read(1), "artifact download exceeds its declared size")
    require(digest(destination) == artifact["digest"], "downloaded artifact ZIP digest differs")


def verify_reply(reply, request, artifact, expected):
    require(isinstance(reply, dict) and reply.get("schema") == SCHEMA and
            all(reply.get(key) == request[key] for key in ("nonce", "profile", "controller_run", "controller_attempt", "artifact_id", "validation_commit")) and
            reply.get("artifact_digest") == artifact["digest"], "SSH reply belongs to a different qualification attempt")
    require(reply.get("helpers_sha256") == helper_digests(), "installed SSH helpers differ from the frozen controller source")
    report = reply.get("report")
    require(isinstance(report, dict) and all(report.get(key) == value for key, value in expected.items()), "SSH report differs from the frozen local inputs")
    if request["mode"] == "readiness":
        require(report.get("schema_version") == "zinc-worker-readiness-v1" and report.get("release_qualification") is False and
                report.get("passed") is True and report.get("models_exercised") is request["exercise_models"], "SSH readiness checks did not pass")
    else:
        checks = report.get("checks", {})
        require(report.get("schema_version") == 1 and checks and all(value is True for value in checks.values()), "SSH candidate checks did not pass")
    report["transport"] = {"kind": "ssh", "artifact_id": artifact["id"], "artifact_digest": artifact["digest"],
                           "controller_run": request["controller_run"], "controller_attempt": request["controller_attempt"],
                           "validation_commit": request["validation_commit"], "helpers_sha256": reply["helpers_sha256"], "cleanup": True}
    return report


def check(config, mode, profile, directory, output, validation_commit, preview_run=None, exercise_models=True, expected=None):
    require(profile in config["profiles"], "profile is not assigned to this SSH controller")
    command = ssh_command(config)
    run = int(os.environ["GITHUB_RUN_ID"])
    artifact_run = run if mode == "candidate" else int(preview_run)
    artifact = find_artifact(artifact_run, "candidate-inputs" if mode == "candidate" else "preview-linux-x86_64-vulkan")
    request = {"schema": SCHEMA, "mode": mode, "profile": profile, "controller_run": run,
               "controller_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]), "artifact_id": artifact["id"], "artifact_run": artifact_run,
               "validation_commit": validation_commit, "nonce": secrets.token_hex(32), "size_bytes": artifact["size_in_bytes"],
               "exercise_models": exercise_models}
    request_identity(request, config["profiles"])
    if mode == "candidate":
        from release import verify_candidate
        from release_lib import sha256
        data = verify_candidate(directory, expected_commit=validation_commit)
        spec = next(a for a in data["artifacts"] if a["target"] == "linux-x86_64-vulkan")
        expected = {"profile": profile, "target": spec["target"], "family": "intel-xe2", "archive_sha256": spec["sha256"],
                    "binary_sha256": spec["binary_sha256"], **{key: data[key] for key in ("version", "commit", "policy_sha256")}}
    with tempfile.TemporaryDirectory(prefix="zinc-ssh-controller-") as temporary:
        work = Path(temporary)
        archive = work / "artifact.zip"
        download_artifact(artifact, archive)
        # Logs can contain connection details. Only the verified public JSON
        # report is allowed out of this private temporary directory.
        with (work / "reply.json").open("wb") as reply_file, (work / "private-ssh.log").open("wb") as log:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, start_new_session=True,
                                     env={"PATH": os.defpath, "HOME": str(Path.home()), "LANG": "C.UTF-8"})
            stopped = threading.Event()
            def upload():
                try:
                    child.stdin.write(json.dumps(request, separators=(",", ":")).encode() + b"\n")
                    child.stdin.flush()
                    with archive.open("rb") as source:
                        while block := source.read(1024 * 1024):
                            child.stdin.write(block)
                    child.stdin.flush()
                    while not stopped.wait(HEARTBEAT_SECONDS):
                        child.stdin.write(b"PING\n")
                        child.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            sender = threading.Thread(target=upload, daemon=True)
            sender.start()
            try:
                deadline, received = time.monotonic() + CHECK_SECONDS + 600, 0
                while True:
                    remaining = deadline - time.monotonic()
                    require(remaining > 0, "SSH qualification timed out")
                    ready, _, _ = select.select([child.stdout], [], [], min(1, remaining))
                    if ready:
                        block = os.read(child.stdout.fileno(), 65536)
                        if not block:
                            break
                        received += len(block)
                        require(received <= MAX_REPORT, "SSH reply is too large")
                        reply_file.write(block)
                result = child.wait(timeout=30)
                require(result == 0, "SSH qualification failed; inspect private endpoint configuration and connectivity")
            finally:
                stopped.set()
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                sender.join(timeout=5)
                child.stdin.close()
                child.stdout.close()
        require((work / "reply.json").stat().st_size <= MAX_REPORT, "SSH reply is too large")
        report = verify_reply(json.loads((work / "reply.json").read_text()), request, artifact, expected or {})
        destination = Path(output)
        require(not destination.exists(), "SSH report destination already exists")
        with destination.open("x") as target:
            target.write(json.dumps(report, indent=2) + "\n")
    print("Packaged SSH checks passed: " + profile, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("agent",))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        os.umask(0o077)
        config = json.loads(Path(args.config).read_text())
        # Unbuffered reads matter: select() must see every byte, including an
        # upload or heartbeat already delivered in the same network packet.
        agent(config, sys.stdin.buffer.raw, sys.stdout.buffer)
    except Exception:
        if "config" in locals():
            try:
                import traceback
                (Path(config["work_root"]) / "last-error.log").write_text(traceback.format_exc())
            except Exception:
                pass
        # Remote paths, model output, SSH identifiers and signed download URLs
        # stay private. A failed agent never returns a usable report.
        raise SystemExit("SSH qualification agent failed; inspect private endpoint configuration and logs") from None


if __name__ == "__main__":
    main()
