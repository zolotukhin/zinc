#!/usr/bin/env python3
"""Own a bounded Linux GPU server and report model lifecycle stability.

Run on a dedicated GPU worker over SSH. Configuration stays outside Git:
{"binary":"<installed prefix>/bin/zinc","backend":"rocm","device":"<GPU index as integer>",
 "context":4096,"models":{"<catalog id>":"<verified GGUF>"}}
No weights are downloaded/copied/deleted; a temporary managed-cache namespace
contains links to operator-selected files and is removed after the child exits.
"""
import argparse
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

from model_lifecycle_soak import Client, ID, limits, read_metrics, require, run_soak


def run_owned(binary, models, backend, device, context, work, env, settings, reference_check=None, exercise=None):
    require(platform.system() == 'Linux', 'GPU soak execution requires a remote Linux worker')
    require(backend in ('rocm', 'vulkan') and type(device) is int and device >= 0,
            'invalid Linux backend/device configuration')
    require(type(context) is int and 2048 <= context <= 262144, 'invalid soak context')
    require(models and all(ID.fullmatch(identifier) and Path(path).is_file() for identifier, path in models.items()),
            'soak requires verified local catalog models')
    # Import only after the platform guard; CPU tests never start native tools.
    from release_gpu import free_port, process
    from release_lib import sha256
    identity = sha256(binary)
    work = Path(work)
    cache = work / 'cache'
    for identifier, path in models.items():
        folder = cache / 'zinc/models/models' / identifier
        folder.mkdir(parents=True, exist_ok=False)
        source = Path(path).resolve()
        (folder / 'model.gguf').symlink_to(source)
        (folder / 'manifest.json').write_text(json.dumps(dict(size_bytes=source.stat().st_size, sha256=None,
                                                           required_vram_bytes=None, offloadable_vram_bytes=None)))
    env = {**env, 'XDG_CACHE_HOME': str(cache), 'XDG_CONFIG_HOME': str(work / 'config')}
    snapshots = {identifier: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for identifier, path in models.items()}
    first = next(iter(models))
    port = free_port()
    client = Client(port, time.monotonic() + settings['deadline_seconds'])
    command = [str(Path(binary).resolve()), 'serve', '--no-browser', '--model-id', first,
               '-d', str(device), '-c', str(context), '--parallel', '1', '--port', str(port)]
    with process(command, work / 'soak-private.log', work, env) as child:
        while True:
            require(child.poll() is None, 'owned server exited during startup')
            client.remaining()
            try:
                status, health = client.request('/health')
                if status == 200 and health.get('status') == 'ok':
                    break
            except (OSError, ValueError):
                pass
            time.sleep(.2)
        def metrics():
            require(child.poll() is None, 'owned server exited during the soak')
            return read_metrics(child.pid)
        status, catalog = client.request('/v1/models')
        require(status == 200 and all(any(entry.get('id') == identifier and entry.get('size_bytes') == snapshots[identifier][0]
                                         for entry in catalog.get('data', [])) for identifier in models),
                'advertised model inventory differs from the selected files')
        if exercise:
            report = exercise(client)
        else:
            report = run_soak(client, list(models), backend, metrics, settings,
                              reference_check=reference_check,
                              progress=lambda current, total, operations: print(
                                  f'Lifecycle round {current}/{total}: {operations} model operations complete', flush=True))
    require(sha256(binary) == identity, 'installed binary changed during the soak')
    require(all((Path(path).stat().st_size, Path(path).stat().st_mtime_ns) == snapshots[identifier]
                for identifier, path in models.items()), 'model files changed during the soak')
    report.update(binary_sha256=identity, context=context, cleanup=True, telemetry='drm-process-resident')
    report['model_sizes'] = {identifier: value[0] for identifier, value in snapshots.items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Private Linux worker configuration outside the checkout')
    parser.add_argument('--output', required=True, help='Sanitized JSON report')
    parser.add_argument('--cycles', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--deadline-seconds', type=int, default=1800)
    parser.add_argument('--private-log', help='Optional private server log, excluded from the JSON report')
    args = parser.parse_args()
    settings = limits(args.cycles, args.warmup, args.deadline_seconds)
    config_path = Path(args.config).resolve()
    require(Path(__file__).resolve().parent.parent not in config_path.parents,
            'worker configuration must remain outside the checkout')
    config = json.loads(config_path.read_text())
    from release_gpu import packaged_environment
    from release_lib import install_signal_handlers
    install_signal_handlers()
    with tempfile.TemporaryDirectory(prefix='zinc-lifecycle-soak-') as directory:
        try:
            report = run_owned(Path(config['binary']), config['models'], config['backend'], config['device'],
                               config.get('context', 4096), Path(directory), packaged_environment(), settings)
        finally:
            log = Path(directory) / 'soak-private.log'
            if args.private_log and log.is_file():
                destination = Path(args.private_log)
                destination.touch(mode=0o600, exist_ok=True)
                destination.chmod(0o600)
                shutil.copyfile(log, destination)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('Model lifecycle soak: ' + report['status'] + '; ' + str(report['completed_cycles']) + ' measured cycles')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit('Model lifecycle soak failed: ' + type(error).__name__ + '; inspect private worker logs') from None
