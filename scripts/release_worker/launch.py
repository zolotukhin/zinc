#!/usr/bin/env python3
"""Administrator-side container launcher. The private configuration stays on host."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def container_args(config):
    name = config["name"]
    require(re.fullmatch(r"zinc-release-[a-z0-9-]+", name), "use a generic release role name")
    require(type(config["uid"]) is int and config["uid"] > 0 and type(config["gid"]) is int and config["gid"] > 0,
            "a dedicated non-root UID/GID is required")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", config["image"]), "worker image must be an immutable local digest")
    command = ["docker", "run", "--rm", "--name", name, "--hostname", "zinc-release-worker",
               "--label", "org.zinc.release-worker=true", "--read-only", "--cap-drop=ALL",
               "--security-opt=no-new-privileges", "--pid=host", "--user", f"{config['uid']}:{config['gid']}",
               "--pids-limit=768", "--memory=32g", "--memory-swap=32g", "--cpus=4", "--cpu-shares=256",
               "--shm-size=1g", "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=8g,mode=1777"]
    require(isinstance(config["gpu_devices"], list) and config["gpu_devices"] and
            set(config["gpu_devices"]).issubset({"/dev/kfd", "/dev/dri"}), "invalid GPU device access")
    for device in config["gpu_devices"]:
        command.append("--device=" + device)
    for group in config["gpu_groups"]:
        require(type(group) is int and group >= 0, "invalid GPU access group")
        command.extend(["--group-add", str(group)])
    for mount in config["mounts"]:
        source, destination = mount["source"], mount["target"]
        require(isinstance(source, str) and Path(source).is_absolute() and "," not in source and "\n" not in source, "invalid mount source")
        require(Path(source).name != "docker.sock" and not Path(source).is_socket(), "daemon sockets cannot be mounted")
        source = str(Path(source).resolve())
        require(source.rstrip("/") not in ("", "/root", "/home", "/etc", "/proc", "/run", "/var/run", "/var/run/docker.sock", "/run/docker.sock"),
                "broad host directories and daemon sockets cannot be mounted")
        require(isinstance(destination, str) and re.fullmatch(r"/(?:runner|worker-reservation|worker-driver|models/[a-z0-9][a-z0-9.-]*/model\.gguf|reference/(?:source|vulkan|rocm)|opt/rocm|etc/zinc-worker|run/secrets/registration-token|usr/share/(?:libdrm|drirc\.d)|tmp/zinc-gpu-vulkan-[0-9]+\.lock)", destination),
                "mount destination is outside the worker layout")
        writable = destination in ("/runner", "/worker-reservation") or destination.startswith("/tmp/zinc-gpu-vulkan-")
        command.extend(["--mount", f"type=bind,source={source},target={destination}" + ("" if writable else ",readonly")])
    fixed_env = {
        "ZINC_RUNNER_REPOSITORY_URL": "https://github.com/" + config["repository"],
        "ZINC_RUNNER_NAME": name, "ZINC_RUNNER_LABELS": config["labels"],
        "LD_LIBRARY_PATH": "/worker-driver:/opt/rocm/lib",
        "VK_DRIVER_FILES": "/etc/zinc-worker/radeon_icd.json", "VK_ICD_FILENAMES": "/etc/zinc-worker/radeon_icd.json",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
    }
    for key, value in fixed_env.items():
        command.extend(["--env", key + "=" + value])
    return command + [config["image"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("action", choices=("run", "stop"))
    args = parser.parse_args()
    require(os.geteuid() == 0, "the host administrator must start the sandbox")
    path = Path(args.config).resolve()
    stat = path.stat()
    require(stat.st_uid == 0 and stat.st_mode & 0o077 == 0, "host configuration must be root-owned and private")
    config = json.loads(path.read_text())
    command = container_args(config)
    if args.action == "stop":
        probe = subprocess.run(["docker", "inspect", "--format", '{{ index .Config.Labels "org.zinc.release-worker" }}', config["name"]], capture_output=True, text=True)
        if probe.returncode == 0:
            require(probe.stdout.strip() == "true", "refusing to stop an unrelated container")
            subprocess.run(["docker", "stop", "--time", "45", config["name"]], check=True)
    else:
        for mount in config["mounts"]:
            require(Path(mount["source"]).exists(), "a configured worker mount is absent")
        os.execvp(command[0], command)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        reason = str(error) if type(error) is ValueError else type(error).__name__
        raise SystemExit("Release worker launcher failed: " + reason)
