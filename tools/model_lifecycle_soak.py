"""Bounded lifecycle checks for an owned Linux GPU server; no model downloads.

Telemetry is sampled only after unloading every model. Warm-up traverses the
whole cohort before setting the baseline, so driver/compiler caches are not
mistaken for a model leak. Reports never contain PIDs, endpoints or paths.
"""
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MIB = 1024 * 1024
ID = re.compile(r"[a-z0-9][a-z0-9.-]{0,127}")


def require(value, message):
    if not value:
        raise ValueError(message)


def limits(cycles=8, warmup=2, deadline_seconds=1800, rss_mib=128, vram_mib=64, fd_growth=0):
    for value, low, high in ((cycles, 1, 100), (warmup, 1, 10), (deadline_seconds, 60, 14400),
                             (rss_mib, 0, 4096), (vram_mib, 0, 4096), (fd_growth, 0, 32)):
        require(type(value) is int and low <= value <= high, "invalid bounded soak settings")
    return dict(cycles=cycles, warmup=warmup, deadline_seconds=deadline_seconds,
                max_growth=dict(rss_bytes=rss_mib * MIB, vram_bytes=vram_mib * MIB,
                                gpu_host_bytes=rss_mib * MIB, fds=fd_growth))


def memory_bytes(value):
    # DRM defaults to bytes and optionally emits KiB/MiB; /proc/status uses
    # kB for kibibytes. Zero-valued DRM regions commonly omit the unit.
    # https://docs.kernel.org/gpu/drm-usage-stats.html#memory
    match = re.fullmatch(r"(\d+)(?:\s+(kB|KiB|MiB))?", value.strip())
    require(match is not None, "invalid process memory telemetry")
    return int(match[1]) * {None: 1, 'kB': 1024, 'KiB': 1024, 'MiB': MIB}[match[2]]


def read_metrics(pid, proc=Path('/proc')):
    """Deduplicate DRM descriptors by client/device, retaining only byte totals."""
    folder = proc / str(pid)
    status = dict(line.split(':', 1) for line in (folder / 'status').read_text().splitlines() if ':' in line)
    clients = {}
    descriptors = list((folder / 'fd').iterdir())
    for entry in (folder / 'fdinfo').iterdir():
        try:
            fields = dict(line.split(':', 1) for line in entry.read_text().splitlines() if ':' in line)
        except FileNotFoundError:
            continue  # A completed HTTP connection can close while sampling.
        if 'drm-client-id' not in fields:
            continue
        key = tuple(fields.get(name, '').strip() for name in ('drm-driver', 'drm-pdev', 'drm-client-id'))
        # New amdgpu/xe expose resident counters; older drivers use memory-*.
        names = [name for name in fields if name.startswith('drm-resident-')]
        if not names:
            names = [name for name in fields if name.startswith('drm-memory-')]
        require(names, "GPU process does not expose DRM memory telemetry")
        # Include GTT/system allocations: watching dedicated VRAM alone would
        # miss a leak that the driver silently spills to host-visible memory.
        resident = (sum(memory_bytes(fields[name]) for name in names if name.endswith(('-vram', '-local'))),
                    sum(memory_bytes(fields[name]) for name in names if not name.endswith(('-vram', '-local'))))
        clients[key] = tuple(max(a, b) for a, b in zip(clients.get(key, (0, 0)), resident))
    require(clients, "GPU process DRM memory telemetry is unavailable")
    return dict(rss_bytes=memory_bytes(status['VmRSS']), vram_bytes=sum(value[0] for value in clients.values()),
                gpu_host_bytes=sum(value[1] for value in clients.values()), fds=len(descriptors))


class Client:
    def __init__(self, port, deadline, timeout=120):
        require(type(port) is int and 0 < port < 65536, "invalid local server port")
        self.root = 'http://127.0.0.1:' + str(port)
        self.deadline = deadline
        self.timeout = timeout

    def remaining(self):
        left = self.deadline - time.monotonic()
        require(left > 0, "soak deadline exceeded")
        return min(self.timeout, left)

    def request(self, path, body=None):
        req = urllib.request.Request(self.root + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(req, timeout=self.remaining())
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read(2 * MIB + 1)
            self.remaining()
            require(len(data) <= 2 * MIB, "server response exceeded the bounded budget")
            return response.status, json.loads(data)

    def stream(self, body, cancel=False, on_content=None):
        req = urllib.request.Request(self.root + '/v1/chat/completions', data=json.dumps({**body, 'stream': True}).encode(),
                                     headers={'Content-Type': 'application/json'})
        seen = 0
        content = False
        done = False
        with urllib.request.urlopen(req, timeout=self.remaining()) as response:
            while True:
                self.remaining()
                line = response.readline(65537)
                if not line:
                    break
                seen += len(line)
                require(len(line) <= 65536 and seen <= 2 * MIB, "stream exceeded the bounded budget")
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    done = True
                    break
                event = json.loads(payload)
                if any(choice.get('delta', {}).get('content') for choice in event.get('choices', [])):
                    content = True
                    if cancel:
                        if on_content:
                            on_content()
                        return  # Closing the response exercises transport cancellation.
        require(content and done, "chat stream was empty or incomplete")


def expect(client, path, body=None, status=200):
    actual, data = client.request(path, body)
    require(actual == status, 'unexpected HTTP status for lifecycle operation')
    return data


def active(client, expected):
    health = expect(client, '/health')
    entries = expect(client, '/v1/models')['data']
    loaded = [entry for entry in entries if entry.get('active') is True]
    require(health.get('status') == 'ok' and len(loaded) == (0 if expected is None else 1), 'active model state is inconsistent')
    if expected is not None:
        require(loaded[0]['id'] == expected, 'server activated the wrong model')
    require(health.get('active_model') == expected, 'health and model discovery disagree')
    return loaded[0] if loaded else None


def generation(client, identifier, mtp=None, check_mtp=False):
    body = dict(model=identifier, messages=[dict(role='user', content='Reply with the word READY.')],
                enable_thinking=False, max_tokens=24, temperature=0, stream=False)
    if check_mtp:
        body['enable_mtp'] = mtp
    data = expect(client, '/v1/chat/completions', body)
    require(data.get('choices') and data['choices'][0].get('message', {}).get('content') and
            data.get('usage', {}).get('completion_tokens', 0) > 0, 'generation did not complete after a lifecycle change')
    if check_mtp:
        require(data.get('mtp', {}).get('requested') is mtp, 'queued MTP choices crossed request boundaries')
    return data


def unload(client, identifier):
    # Generation can release its lease just after its last response byte.
    end = time.monotonic() + min(30, client.remaining())
    while True:
        code, data = client.request('/v1/models/unload', {'model': identifier})
        if code == 200:
            break
        require(code == 409 and time.monotonic() < end, 'unload did not recover after cancellation')
        time.sleep(.05)
    require(data.get('deleted_model') is False, 'unload did not confirm cached file preservation')
    active(client, None)


def exercise_model(client, identifier, backend, reference_check=None):
    expect(client, '/v1/models/activate', {'model': identifier})
    entry = active(client, identifier)
    controllable = entry.get('supports_mtp') is True and entry.get('mtp_request_control') is True
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(generation, client, identifier, setting, controllable) for setting in (False, True)]
        for job in jobs:
            job.result(timeout=client.remaining())
    body = dict(model=identifier, messages=[dict(role='user', content='List three uses of copper.')],
                enable_thinking=False, max_tokens=32, temperature=0)
    client.stream(body)
    def busy():
        if backend == 'rocm':
            expect(client, '/v1/models/unload', {'model': identifier}, status=409)
    client.stream({**body, 'messages': [dict(role='user', content='Count upward from 1 to 1000, one number per line.')],
                   'max_tokens': 1024}, cancel=True, on_content=busy)
    generation(client, identifier)
    if reference_check:
        reference_check(client, identifier)
    expect(client, '/v1/models/activate', {'model': 'zinc-soak-unknown'}, status=400)
    active(client, identifier)
    unload(client, identifier)
    require(any(m.get('id') == identifier and m.get('installed') is True for m in expect(client, '/v1/models')['data']),
            'unload removed cached model installation')
    expect(client, '/v1/models/activate', {'model': identifier})
    restored = active(client, identifier)
    require(all(restored.get(key) == entry.get(key) for key in ('supports_mtp', 'mtp_default_enabled', 'mtp_request_control')),
            'reload changed the active MTP capability or policy')
    generation(client, identifier)
    # Leave this model resident. The next activation must exercise replacement
    # of live weights, rather than only loading into an already empty device.


def run_soak(client, identifiers, backend, metrics, settings, settle=lambda: time.sleep(.5), progress=None, reference_check=None):
    require(identifiers and len(identifiers) == len(set(identifiers)) and all(ID.fullmatch(x) for x in identifiers),
            'soak requires unique advertised catalog identifiers')
    require(backend in ('rocm', 'vulkan'), 'soak requires a Linux release backend')
    report = dict(schema_version=1, status='failed', backend=backend, models=list(identifiers),
                  options=settings, samples=[], completed_cycles=0, operations=0, direct_switches=0,
                  reference_cases=0, failure_stage='discovery')
    try:
        health = expect(client, '/health')
        require(health.get('backend') == backend and health.get('supports_model_management') is True and
                health.get('supports_model_unload') is True, 'server does not advertise required lifecycle support')
        entries = expect(client, '/v1/models')['data']
        for identifier in identifiers:
            require(any(m.get('id') == identifier and m.get('installed') is True and m.get('can_activate') is not False and
                        m.get('supported_on_current_gpu') is not False and m.get('fits_current_gpu') is not False for m in entries),
                    'a required installed model is not an activation candidate')
        for cycle in range(settings['warmup'] + settings['cycles']):
            for index, identifier in enumerate(identifiers):
                report['failure_stage'] = 'model_lifecycle'
                report['failure_model'] = identifier
                def check_reference(current, model):
                    cases = reference_check(current, model)
                    require(type(cases) is int and cases > 0, 'missing repeated native reference cases')
                    report['reference_cases'] += cases
                exercise_model(client, identifier, backend, check_reference if reference_check else None)
                report['operations'] += 1
                if index:
                    report['direct_switches'] += 1
            report['failure_stage'] = 'round_unload'
            unload(client, identifiers[-1])
            report['failure_stage'] = 'telemetry'
            report['failure_model'] = None
            settle()
            client.remaining()
            sample = metrics()
            require(set(sample) == {'rss_bytes', 'vram_bytes', 'gpu_host_bytes', 'fds'} and
                    all(type(value) is int and value >= 0 for value in sample.values()), 'invalid lifecycle telemetry sample')
            if cycle + 1 >= settings['warmup']:
                report['samples'].append(sample)
            if cycle >= settings['warmup']:
                report['completed_cycles'] += 1
            if progress:
                progress(cycle + 1, settings['warmup'] + settings['cycles'], report['operations'])
        report['failure_stage'] = 'memory_plateau'
        baseline = report['samples'][0]
        growth = {key: max(0, max(s[key] for s in report['samples'][1:]) - baseline[key]) for key in baseline}
        report['max_growth'] = growth
        require(all(growth[key] <= settings['max_growth'][key] for key in growth), 'unloaded process exceeded its post-warmup memory/handle growth budget')
        report.update(status='passed', failure_stage=None, failure_model=None)
    except Exception as error:
        # Exception text, generated content and HTTP bodies may contain local
        # details. Only the category and completed bounded checks are public.
        report['error_category'] = type(error).__name__
    return report


def policy_settings(spec):
    keys = {'cycles', 'warmup', 'deadline_seconds', 'context', 'max_rss_growth_mib', 'max_vram_growth_mib', 'max_fd_growth'}
    require(isinstance(spec, dict) and set(spec) == keys and type(spec['context']) is int and
            2048 <= spec['context'] <= 8192, 'invalid release lifecycle soak policy')
    return limits(spec['cycles'], spec['warmup'], spec['deadline_seconds'], spec['max_rss_growth_mib'],
                  spec['max_vram_growth_mib'], spec['max_fd_growth'])


def verify_report(report, identifiers, backend, spec, binary_sha256, expected_sizes=None, reference_cases_per_model=0):
    settings = policy_settings(spec)
    require(isinstance(report, dict) and report.get('schema_version') == 1 and report.get('status') == 'passed' and
            report.get('failure_stage') is None and report.get('backend') == backend and
            report.get('binary_sha256') == binary_sha256 and report.get('cleanup') is True and
            report.get('telemetry') == 'drm-process-resident' and report.get('context') == spec['context'] and
            report.get('options') == settings, 'missing, failed or mismatched lifecycle soak evidence')
    models = report.get('models')
    require(isinstance(models, list) and len(models) == len(identifiers) and set(models) == set(identifiers),
            'lifecycle soak omitted a required model')
    require(isinstance(report.get('model_sizes'), dict) and set(report['model_sizes']) == set(identifiers) and
            all(type(size) is int and size > 0 for size in report['model_sizes'].values()) and
            (expected_sizes is None or report['model_sizes'] == expected_sizes), 'lifecycle soak selected different model files')
    require(type(report.get('completed_cycles')) is int and report['completed_cycles'] == settings['cycles'] and
            type(report.get('operations')) is int and report['operations'] == (settings['cycles'] + settings['warmup']) * len(models),
            'lifecycle soak did not complete the required rounds')
    require(type(report.get('direct_switches')) is int and report['direct_switches'] ==
            (settings['cycles'] + settings['warmup']) * (len(models) - 1), 'lifecycle soak omitted resident model replacement')
    require(type(report.get('reference_cases')) is int and report['reference_cases'] ==
            report['operations'] * reference_cases_per_model, 'lifecycle soak omitted repeated native reference checks')
    samples = report.get('samples')
    require(isinstance(samples, list) and len(samples) == settings['cycles'] + 1 and
            all(isinstance(s, dict) and set(s) == set(settings['max_growth']) and
                all(type(v) is int and v >= 0 for v in s.values()) for s in samples), 'invalid lifecycle telemetry evidence')
    growth = {key: max(0, max(s[key] for s in samples[1:]) - samples[0][key]) for key in settings['max_growth']}
    require(report.get('max_growth') == growth and all(growth[key] <= settings['max_growth'][key] for key in growth),
            'lifecycle telemetry exceeded its frozen growth budget')


def allocation_failure_probe(client, initial, replacement, context):
    """The profile chooses a valid model/context that cannot allocate on its GPU.

    HTTP 400 is insufficient: the probe must reach the actual load, fail with
    HTTP 500, restore the original model and reproduce its greedy continuation.
    """
    report = dict(status='failed', initial_model=initial, replacement_model=replacement, context=context)
    try:
        active(client, initial)
        before = generation(client, initial)
        status, response = client.request('/v1/models/activate', {'model': replacement})
        report['response_status'] = status
        require(status == 500 and response.get('error', {}).get('type') == 'model_load_error',
                'allocation probe did not exercise a failed model load')
        active(client, initial)
        after = generation(client, initial)
        require(before['choices'][0]['message']['content'] == after['choices'][0]['message']['content'],
                'restored model continuation changed after allocation failure')
        report.update(status='passed', restored=True, continuation_matches=True)
    except Exception as error:
        report['error_category'] = type(error).__name__
    return report


def verify_allocation_failures(reports, specs, binary_sha256):
    require(isinstance(reports, list) and len(reports) == len(specs), 'missing allocation failure recovery evidence')
    for report, spec in zip(reports, specs):
        require(isinstance(report, dict) and report.get('status') == 'passed' and report.get('restored') is True and
                report.get('continuation_matches') is True and report.get('response_status') == 500 and
                report.get('binary_sha256') == binary_sha256 and report.get('cleanup') is True and
                all(report.get(key) == spec[key] for key in ('initial_model', 'replacement_model', 'context')),
                'failed or mismatched allocation failure recovery evidence')
