# Intel release qualification over SSH

Intel can keep its existing SSH access without installing a GitHub runner,
Docker, or a sudo rule on the compute node. A separate restricted Linux runner
acts as the controller. Both worker readiness and candidate qualification use
the same packaged tests as a directly attached GPU worker.

This transport requires an online Intel endpoint and a registered controller.
Implementing it does not qualify a candidate or make an offline endpoint ready.
The release still requires R9700 Vulkan, R9700 ROCm, and Intel Vulkan reports
for the exact selected archives. See [RELEASING.md](RELEASING.md).

## Trust and authentication

The controller uses a dedicated qualification key. Its public key authorizes
only an installed agent through OpenSSH's `restrict` and `command` options.
Disable terminal allocation and forwarding; do not reuse an operator's key.
Pin the endpoint's host key through a separately verified `known_hosts` file.

The compute account must be non-root and operator-managed. Prefer a dedicated
account with access only to its private work directory, the selected render
device, pinned model files, reference tools and the shared GPU reservation.
The endpoint is a trusted compute account; SSH does not provide the read-only
filesystem isolation of the directly attached Docker worker. No untrusted PR
code is accepted. Keep credentials and unrelated workloads out of this account.

The installed agent independently reads public GitHub metadata and checks:

- The active main-branch candidate/readiness run, workflow, repository, attempt,
  event, actor and triggering actor match its administrator policy.
- The uploaded ZIP belongs to that run, or to a successful main preview for
  diagnostic readiness. Its size and SHA256 match GitHub's artifact metadata.
- Validation code comes directly from public GitHub at the authenticated full
  commit, within the trusted main run's ancestry. SSH cannot supply code, a
  command, a source URL or a model path.
- Candidate manifests bind the artifact to its source and workflow run.
  Readiness preserves separate preview/validation identities and requires
  identical release policy bytes when using an older preview.

The controller also requires the installed agent and cleanup helper hashes to
match its frozen validation checkout. Both files must stay unchanged throughout
the remote check. Update the installed pair deliberately when either changes;
documentation-only advancement does not require replacing identical helpers.

GitHub tokens stay on the controller. It needs Actions read permission to
download the original artifact ZIP; redirect requests omit authorization.
Intel reads public metadata without credentials. A missing digest, unavailable
metadata, expired artifact, divergent source or mismatched report fails closed.
See GitHub's [artifact metadata API](https://docs.github.com/en/rest/actions/artifacts)
and [artifact digest validation](https://docs.github.com/en/actions/tutorials/store-and-share-data).

## Endpoint installation

On Intel, install reviewed copies of `tools/release_ssh.py` and
`tools/release_worker_guard.py` together, outside any mutable Git checkout. The
forced command must run that installed copy, rather than a script selected by
the caller. Python 3, Git, HTTPS access to public GitHub, and the normal Vulkan
worker prerequisites are required. No ZINC build happens on this path.

Keep an endpoint configuration outside the checkout, readable only by the
compute account:

```json
{
  "profiles": ["vulkan-intel-xe2"],
  "actors": ["<authorized maintainer login>"],
  "worker_config": "<absolute private native worker JSON>",
  "work_root": "<absolute private agent work directory>",
  "environment": {
    "XDG_CACHE_HOME": "<absolute private shader cache directory>",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4"
  }
}
```

Use mode `0700` for the existing work root and `0600` for configurations. The
native worker JSON uses the device, model, pinned native Vulkan reference and
shared `lock_directory` schema in [RELEASING.md](RELEASING.md#provision-gpu-workers).
It must not contain a controller `transport` section. Select the observed GPU
index explicitly. Use the same reservation as other qualification jobs on
that physical GPU. Driver overrides, when necessary, may use `VK_DRIVER_FILES`
or `VK_ICD_FILENAMES` in `environment`; inherited shell/library overrides and
GitHub credentials are excluded.

Add only the dedicated public key to `authorized_keys`, with this form. Replace
the paths and key privately; do not put the completed line in the repository:

```text
restrict,command="/usr/bin/python3 -I <absolute installed release_ssh.py> agent --config <absolute private agent JSON>" ssh-ed25519 <qualification public key>
```

OpenSSH interprets the forced command through a shell. Use absolute paths
without spaces or shell metacharacters in this administrator-owned line.
Keep the installed agent and guard read-only and update them deliberately after
review. An ordinary SSH command with this key is rejected unless it supplies
the bounded qualification protocol on stdin.

The agent leases the endpoint, then runs the existing `release_gpu.py` or
`release_worker.py` from a fresh authenticated checkout. Those helpers retain
model hashes, tokenizer parity, 376 shader reference cases, native continuation
agreement, extended Gemma windows, HTTP/SSE/cancellation/recovery and cleanup
checks. No archived binary or shader is rebuilt or modified.

A ten-second controller heartbeat maintains a sixty-second lease. Disconnects,
partial heartbeats, timeouts and cancellation unwind the helper, then the
installed cleanup hook reaps only this session's marked GPU processes owned
by the compute UID, checking PID start times. The shared reservation prevents
overlap with other managed jobs. Cleanup finishes before a report is returned.
Private logs remain under the work root's `logs/` directory; `last-error.log`
records agent failures. Retain or rotate these locally, rather than uploading
them as public workflow artifacts.

## Controller installation

Use the [Linux worker kit](https://github.com/zolotukhin/zinc/blob/main/scripts/release_worker/README.md)
on an available administrator-managed Linux host. It may share the R9700 host
under a separate dedicated UID and runner volume. The controller itself needs
no GPU, model or reference mounts. In the private host launcher JSON, set:

```json
{
  "mode": "ssh-controller",
  "gpu_devices": [],
  "gpu_groups": []
}
```

Keep the existing immutable image, generic runner name/labels, non-root UID,
resource limits and trusted-main pre/post hooks. Add only these mounts:

| Container destination                    | Access                                                   |
| ---------------------------------------- | -------------------------------------------------------- |
| `/runner`                                | Read/write dedicated runner volume                       |
| `/etc/zinc-worker`                       | Read-only controller `worker.json` and job `policy.json` |
| `/run/secrets/registration-token`        | Read-only short-lived registration token                 |
| `/run/secrets/qualification-key`         | Read-only dedicated restricted SSH key                   |
| `/run/secrets/qualification-known-hosts` | Read-only independently verified host key                |

The key must be mode `0600` or stricter and readable by the controller UID.
Controller mode rejects local GPU access and model/reference mounts and keeps
its own PID namespace. It retains no Docker socket, host home, signing key or
administrator GitHub token.

The controller's private `worker.json` contains:

```json
{
  "transport": {
    "kind": "ssh",
    "profiles": ["vulkan-intel-xe2"],
    "host": "<host>",
    "user": "<compute user>",
    "port": 22,
    "identity_file": "/run/secrets/qualification-key",
    "known_hosts_file": "/run/secrets/qualification-known-hosts"
  }
}
```

Register it with a generic label such as `zinc-intel-ssh`. After installing and
checking the endpoint, set `ZINC_RUNNER_INTEL_XE2` to
`["self-hosted","linux","x64","zinc-intel-ssh"]`. The protected
`release-workers` environment and immutable job policy remain mandatory.
The normal configuration preflight checks that the assigned controller is
online; the readiness workflow checks endpoint reachability and actual hardware.

First run `release-worker-check.yml` for `vulkan-intel-xe2`, using a successful
main preview and `exercise_models=true`. Inspect the sanitized report and its
`transport` provenance. A readiness report cannot certify a release. Once it
passes, the ordinary candidate workflow invokes the same transport and checks
the report against its locally downloaded manifest and archive hashes.
Only complete candidate aggregation can proceed to protected publication.
