#!/usr/bin/env python3
"""Sign/notarize an unsigned Metal preview before its candidate digest freezes."""

import argparse
import base64
import os
import secrets
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from release_lib import extract_archive, read_json, require, sha256, version, source_sha, write_json


def run(command, timeout=120):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # TimeoutExpired includes argv, which can contain keychain/API secrets.
        raise ValueError("signing command timed out: " + Path(command[0]).name) from None
    require(result.returncode == 0, "signing command failed: " + Path(command[0]).name)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    version(args.version)
    source_sha(args.commit)
    require(sys.platform == "darwin" and os.environ.get("GITHUB_ACTIONS") == "true", "sign only on a disposable macOS Actions worker")
    names = ("ZINC_APPLE_CERTIFICATE_BASE64", "ZINC_APPLE_CERTIFICATE_PASSWORD", "ZINC_APPLE_SIGNING_IDENTITY",
             "ZINC_APPLE_API_KEY_BASE64", "ZINC_APPLE_API_KEY_ID", "ZINC_APPLE_API_ISSUER")
    require(all(os.environ.get(name) for name in names), "Apple signing/notarization secrets are missing")
    old_keychains = shlex.split(run(["security", "list-keychains", "-d", "user"]))
    with tempfile.TemporaryDirectory(prefix="zinc-sign-") as directory:
        temporary = Path(directory)
        tree = extract_archive(args.archive, temporary / "unpack", args.version, "macos-aarch64-metal", args.commit)
        certificate, api_key = temporary / "certificate.p12", temporary / "AuthKey.p8"
        certificate.write_bytes(base64.b64decode(os.environ[names[0]], validate=True))
        api_key.write_bytes(base64.b64decode(os.environ[names[3]], validate=True))
        certificate.chmod(0o600)
        api_key.chmod(0o600)
        keychain, password = str(temporary / "signing.keychain-db"), secrets.token_hex(32)
        try:
            run(["security", "create-keychain", "-p", password, keychain])
            run(["security", "set-keychain-settings", "-lut", "21600", keychain])
            run(["security", "unlock-keychain", "-p", password, keychain])
            run(["security", "import", str(certificate), "-k", keychain, "-P", os.environ[names[1]], "-T", "/usr/bin/codesign"])
            run(["security", "set-key-partition-list", "-S", "apple-tool:,apple:,codesign:", "-s", "-k", password, keychain])
            run(["security", "list-keychains", "-d", "user", "-s", keychain, *old_keychains])
            executables = [tree / "bin/zinc", tree / "bin/zinc-tokenize"]
            for executable in executables:
                run(["codesign", "--force", "--timestamp", "--options", "runtime", "--keychain", keychain,
                     "--sign", os.environ[names[2]], str(executable)])
                run(["codesign", "--verify", "--strict", str(executable)])
            payload = temporary / "notarization.zip"
            run(["ditto", "-c", "-k", "--keepParent", str(tree / "bin"), str(payload)])
            import json
            receipt = json.loads(run(["xcrun", "notarytool", "submit", str(payload), "--key", str(api_key),
                                      "--key-id", os.environ[names[4]], "--issuer", os.environ[names[5]],
                                      "--wait", "--timeout", "20m", "--output-format", "json"], timeout=1300))
            require(receipt.get("status") == "Accepted" and receipt.get("id"), "Apple did not accept notarization")
            receipt_path = temporary / "receipt.json"
            write_json(receipt_path, {"status": "Accepted", "id": receipt["id"], "binary_sha256": sha256(tree / "bin/zinc"),
                                      "binaries": {p.name: sha256(p) for p in executables}})
            # Repackage once after signing, before any hardware test or manifest.
            run([sys.executable, str(Path(__file__).with_name("release.py")), "package", "--version", args.version,
                 "--target", "macos-aarch64-metal", "--commit", args.commit, "--prefix", str(tree),
                 "--dist", args.output, "--notary-receipt", str(receipt_path)])
        finally:
            subprocess.run(["security", "list-keychains", "-d", "user", "-s", *old_keychains], capture_output=True)
            subprocess.run(["security", "delete-keychain", keychain], capture_output=True)
    print("Signed and notarized Metal candidate packaged")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        sys.exit("release signing: " + str(error))
