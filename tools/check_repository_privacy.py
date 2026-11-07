#!/usr/bin/env python3
"""Check tracked content, compressed evidence and image metadata for private data.

Diagnostics intentionally contain only filenames, line numbers and categories.
This complements Gitleaks; it cannot decide whether a face or a serial sticker
is visible in a photo. Review screenshots and photos before publishing them.
"""

import argparse
import gzip
import ipaddress
import re
import shlex
import subprocess
from pathlib import Path
from urllib.parse import unquote

from image_metadata import clean_image

HOME_PATH = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\+Users\\+)([\w.-]+)")
GPU_ID = re.compile(r"\bGPU-[a-fA-F0-9]{8}(?:[-a-fA-F0-9]*)(?:\.\.\.|…)?")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6 = re.compile(r"\b(?:f[cd][a-f0-9]{2}|fe[89ab][a-f0-9]):[a-f0-9:]+", re.I)
PRIVATE_NETWORKS = tuple(map(ipaddress.ip_network, (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10",
    "169.254.0.0/16", "fc00::/7", "fe80::/10",
)))
GENERIC_USERS = {"...", "user", "username", "USER", "runner", "test", "example"}
FIXTURE_USERS = {"me", "alice", "tempuser"}


def inspect_text(name, text, private_hosts=()):
    # Serialized logs can quote slashes; URL paths can percent-encode them.
    text = unquote(text).replace(r"\/", "/")
    issues = set()

    def report(category, start):
        issues.add((name, text.count("\n", 0, start) + 1, category))

    for match in HOME_PATH.finditer(text):
        user = match.group(1)
        fixture = name.endswith(".test.ts") or name == "src/model/managed.zig"
        if user not in GENERIC_USERS and not (fixture and user in FIXTURE_USERS):
            report("personal home path", match.start())
    for match in GPU_ID.finditer(text):
        report("GPU identifier", match.start())
    for pattern in (IPV4, IPV6):
        for match in pattern.finditer(text):
            try:
                address = ipaddress.ip_address(match.group())
            except ValueError:
                continue
            # Standard RFC network declarations describe address classes, not
            # the address of a particular worker.
            cidr = re.match(r"/\d{1,3}", text[match.end():])
            if cidr and any(str(net) == match.group() + cidr.group() for net in PRIVATE_NETWORKS):
                continue
            if any(address.version == net.version and address in net for net in PRIVATE_NETWORKS):
                report("private network address", match.start())
    for match in re.finditer(r"\b[\w.-]+\.ts\.net\b", text, re.I):
        report("private tailnet hostname", match.start())
    for host in private_hosts:
        for match in re.finditer(r"(?<![\w.-])" + re.escape(host) + r"(?![\w.-])", text, re.I):
            report("configured private endpoint", match.start())
    return issues


def inspect_file(root, name, private_hosts=()):
    path = root / name
    if not path.is_file():
        return set()  # A tracked file can be deleted in the working tree.
    issues = set()
    private_env = path.name == ".env" or (
        path.name.startswith(".env.") and not path.name.endswith((".example", ".sample", ".template"))
    )
    if private_env or re.search(r"(?:^id_(?:rsa|ed25519)$|\.(?:p12|pfx|key)$)", path.name):
        issues.add((name, 1, "credential file"))
    data = path.read_bytes()
    if name.endswith(".gz"):
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError):
            return {(name, 1, "unreadable compressed evidence")}
    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif"):
        try:
            if clean_image(data) != data:
                issues.add((name, 1, "identifying image metadata"))
        except (ValueError, IndexError):
            issues.add((name, 1, "unreadable image metadata"))
    # Decode embedded strings too. No binary file or evidence directory is
    # exempt: compiler paths and endpoints can leak in those formats as well.
    issues.update(inspect_text(name, data.decode("utf-8", errors="replace"), private_hosts))
    return issues


def read_private_hosts(path):
    hosts = []
    for line in path.read_text().splitlines():
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        if words and words[0] == "export":
            words = words[1:]
        if len(words) != 1 or "=" not in words[0]:
            continue
        key, value = words[0].split("=", 1)
        if key.endswith("_HOST") and value and value not in ("localhost", "127.0.0.1", "0.0.0.0"):
            hosts.append(value)
    return hosts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--private-env", type=Path, help="optional ignored local .env; values are never printed")
    args = parser.parse_args()
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=args.root
    ).decode().split("\0")
    hosts = read_private_hosts(args.private_env) if args.private_env else ()
    issues = set()
    for name in filter(None, names):
        issues.update(inspect_file(args.root, name, hosts))
    for name, line, category in sorted(issues):
        print(f"{name}:{line}: {category} [value redacted]")
    print(f"Privacy check: {len(issues)} findings.")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
