# Linux release worker

This administrator kit runs a GitHub Actions worker with a dedicated non-root
UID inside a restricted Docker container. It is intended for trusted main-branch
release jobs. Hardware and signing qualification still follow
[RELEASING.md](../../docs/RELEASING.md).

The image pins Ubuntu and the official runner archive by SHA256. Build it from
the repository root, passing the dedicated host account's numeric UID/GID:

```sh
docker build -f scripts/release_worker/Dockerfile \
  --build-arg RUNNER_UID=<uid> --build-arg RUNNER_GID=<gid> \
  -t zinc-release-worker:local .
docker image inspect --format '{{.Id}}' zinc-release-worker:local
```

Use that immutable image ID in the host configuration. Never give the worker the
Docker socket, a privileged container, host networking, a host home directory,
operator SSH keys or signing credentials. `launch.py` drops capabilities, prevents gaining
privileges, makes the image read-only and limits CPU, memory and process counts.
The host PID namespace lets the existing GPU checks detect competing workloads;
use a dedicated UID so it cannot signal services owned by other accounts.

For Intel compute over SSH, use the separate `ssh-controller` mode described in
[RELEASE_SSH_WORKER.md](../../docs/RELEASE_SSH_WORKER.md). It has no local GPU,
host PID namespace or model/reference mounts. Its sole SSH key is restricted to
the installed qualification agent; operator keys remain excluded.

## Administrator inputs

Keep the launcher and its JSON configuration root-owned, outside the checkout;
the configuration must have mode `0600`. Prepare these mounts:

| Container destination                     | Contents                                                       | Access     |
| ----------------------------------------- | -------------------------------------------------------------- | ---------- |
| `/runner`                                 | Runner registration, work and logs; owned by the dedicated UID | Read/write |
| `/worker-reservation`                     | Dedicated UID-owned parent for the shared GPU reservation      | Read/write |
| `/models/<model-id>/model.gguf`           | Each exact model file from the release policy                  | Read-only  |
| `/reference/source`                       | Clean checkout of the policy's pinned llama.cpp commit         | Read-only  |
| `/reference/vulkan`, `/reference/rocm`    | Reference `bin/` directories and matching libraries            | Read-only  |
| `/worker-driver`                          | Matching Vulkan loader/ICD and non-libc dependency closure     | Read-only  |
| `/opt/rocm`                               | Matching ROCm toolkit, when needed                             | Read-only  |
| `/etc/zinc-worker`                        | `worker.json`, `policy.json`, `radeon_icd.json`                | Read-only  |
| `/run/secrets/registration-token`         | Short-lived runner registration token                          | Read-only  |
| `/tmp/zinc-gpu-vulkan-<index>.lock`       | Same native Vulkan lock file used outside the container        | Read/write |
| `/usr/share/libdrm`, `/usr/share/drirc.d` | Driver data, when needed                                       | Read-only  |

Bind individual existing model files; do not move them or expose their parent
directories. Grant the dedicated UID read access to those files, using an ACL
when necessary. Keep host paths only in private administrator configuration.
Do not replace the container's libc with a host library bundle. Put each reference
build's own `libggml` libraries beside its tools; the harness selects that build's
loader path before executing it.

The private host JSON has these fields:

```json
{
  "name": "zinc-release-rdna4",
  "uid": 10001,
  "gid": 10001,
  "image": "sha256:<local image digest>",
  "repository": "zolotukhin/zinc",
  "labels": "zinc-rdna4",
  "gpu_devices": ["/dev/dri", "/dev/kfd"],
  "gpu_groups": ["<numeric render/video groups>"],
  "mounts": [
    { "source": "<absolute host runner directory>", "target": "/runner" },
    { "source": "<absolute private configuration directory>", "target": "/etc/zinc-worker" }
  ]
}
```

Replace placeholders and add all required mounts. Intel/Vulkan workers need
`/dev/dri`; ROCm also needs `/dev/kfd`. The supplied ICD environment targets
Radeon; adapt the administrator launcher and ICD for another vendor before use.
`radeon_icd.json` must point to the staged ICD under `/worker-driver`.
`worker.json` uses the schema in RELEASING.md, with container paths and observed
device indices. Backend profiles share one reservation. No network port is
published; inference servers stay inside the container.

Enumerate devices inside the final container: an ICD or driver mount can change
their order relative to the host. Verify that both packaged diagnostics and the
reference select the intended physical card. Remap its native lock-file mount
to the container's observed index so host and container jobs still share the
same lock.

### Driver library lookup

Packaged checks clear inherited library search overrides. If the driver's
dependency closure lives under `/worker-driver`, generate a loader cache with
the same worker base image and runtime mounts. Keep its build context outside
the checkout: the cache is deployment data, not a portable source artifact.

With the administrator's paths in the variables below, generate the cache
without exposing a GPU or changing the mounted libraries. Set `ZINC_WORKER_BASE`
to the currently configured image; verify any local build tag resolves to that
immutable image ID before using it:

```bash
docker run --rm --network=none --read-only --user=0 --cap-drop=ALL \
  --security-opt=no-new-privileges --entrypoint=/sbin/ldconfig \
  --mount "type=bind,src=$ZINC_DRIVER_DIR,dst=/worker-driver,readonly" \
  --mount "type=bind,src=$ZINC_ROCM_DIR,dst=/opt/rocm,readonly" \
  --mount "type=bind,src=$ZINC_CACHE_CONTEXT,dst=/cache" \
  "$ZINC_WORKER_BASE" -X -C /cache/ld.so.cache /worker-driver /opt/rocm/lib
docker build --network=none --pull=false \
  --build-arg "ZINC_WORKER_BASE=$ZINC_WORKER_BASE" \
  -f scripts/release_worker/loader-cache.Dockerfile "$ZINC_CACHE_CONTEXT"
```

Update the private launch configuration to the resulting immutable image ID.
Drain the runner before restarting its service, then verify driver dependencies
and both packaged `--check` probes with library overrides removed. Regenerate
the cache whenever the base image, driver closure or ROCm mount changes. The
cache layer preserves the base image's non-root user and immutable job hooks.

## Job policy and registration

`policy.json` is administrator-owned and read-only to the runner:

```json
{
  "repository": "zolotukhin/zinc",
  "actors": ["<authorized maintainer login>"],
  "workflows": {
    "zolotukhin/zinc/.github/workflows/release-worker-check.yml@refs/heads/main": {
      "events": ["workflow_dispatch"],
      "jobs": ["check"]
    },
    "zolotukhin/zinc/.github/workflows/release-candidate.yml@refs/heads/main": {
      "events": ["workflow_dispatch", "schedule"],
      "jobs": ["hardware"]
    }
  }
}
```

The pre-job hook checks the repository, main ref, actor, workflow, event and job
before checkout. The post-job hook stops only GPU processes owned by this UID
and this run, checking PID start times before signalling. The harness itself
always stops and reaps its servers. Neither mechanism interrupts another user's
workload. The harness automatically recovers a marked reservation after its
kernel lock is released by a hard crash. Unknown legacy reservations still
require inspection. Keep the persistent `.flock` file in place.

Obtain a short-lived registration token on the administrator's machine through
GitHub's runner registration API, transfer it privately to the token mount and
start the launcher:

```sh
sudo python3 /opt/zinc-release-worker/launch.py --config <private-host-json> run
```

The entrypoint reads the token without placing it in command-line arguments.
After registration succeeds, clear the token file in place. Runner credentials
remain in its private persistent volume. Do not place an administrator's GitHub
token on the worker. Use `stop` with the same configuration to stop only the
labelled worker container. A systemd service can supervise the foreground
launcher and call `stop` from `ExecStop`.

Create the `release-workers` GitHub environment restricted to main, and set the
role variable, for example `ZINC_RUNNER_RDNA4`, to
`["self-hosted","linux","x64","zinc-rdna4"]`. Leave unavailable roles unset.
Run the separate readiness workflow against a successful main preview. An older
preview must be an ancestor of the workflow's frozen main commit with identical
release policy bytes. The worker uses that frozen commit's validation code and
records its identity separately from the tested archive's source. Its reports
cannot be aggregated into release qualification.

Automatic runner updates are disabled to preserve the reviewed image. Check
official runner releases regularly and rebuild with a verified archive hash
before GitHub's 30-day update deadline. Drain and stop the worker first. Register
the new image with a fresh runner volume and token; replacing the image alone
does not replace application files already copied into the persistent volume.
Retain private logs locally; only the harness's sanitized report belongs in
public artifacts.
