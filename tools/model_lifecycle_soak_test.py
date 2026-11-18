import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import model_lifecycle_soak as soak


class FakeClient:
    def __init__(self, **faults):
        self.loaded = 'alpha'
        self.faults = faults
        self.calls = []
        self.busy = False

    def remaining(self):
        return 5

    def request(self, path, body=None):
        self.calls.append((path, body))
        if path == '/health':
            return 200, dict(status='ok', backend='rocm', active_model=self.loaded,
                             supports_model_management=True, supports_model_unload=True)
        if path == '/v1/models':
            return 200, dict(data=[dict(id=name, active=self.loaded == name, installed=not self.faults.get('deleted'),
                                        can_activate=True, supports_mtp=True, mtp_request_control=True) for name in ['alpha', 'beta']])
        if path.endswith('/activate'):
            if body['model'] == 'zinc-soak-unknown':
                return (200 if self.faults.get('accept_unknown') else 400), {}
            self.loaded = 'wrong' if self.faults.get('wrong_model') else body['model']
            return 200, {}
        if path.endswith('/unload'):
            if self.busy and not self.faults.get('busy_unload'):
                return 409, {}
            if not self.faults.get('retain_loaded'):
                self.loaded = None
            return 200, dict(deleted_model=False)
        if path.endswith('/completions'):
            return 200, dict(choices=[dict(message=dict(content='' if self.faults.get('empty') else 'READY'))],
                             usage=dict(completion_tokens=1), mtp=dict(requested=True if self.faults.get('mtp_bleed') else body.get('enable_mtp')))
        raise AssertionError(path)

    def stream(self, body, cancel=False, on_content=None):
        if self.faults.get('broken_stream'):
            raise ValueError('DO_NOT_PUBLISH /fixtures/private.gguf')
        if cancel:
            self.busy = True
            try:
                on_content()
            finally:
                self.busy = False


def sample(**extra):
    return dict(rss_bytes=1024, vram_bytes=2048, gpu_host_bytes=512, fds=4, **extra)


def run(client=None, samples=None):
    metrics = iter(samples or [sample() for _ in range(3)])
    return soak.run_soak(client or FakeClient(), ['alpha', 'beta'], 'rocm', lambda: next(metrics),
                         soak.limits(2, 1, 60), settle=lambda: None)


class LifecycleTests(unittest.TestCase):
    def test_full_cycle_exercises_each_model_and_keeps_equivalent_unloaded_samples(self):
        client = FakeClient()
        report = run(client)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['operations'], 6)
        self.assertEqual(report['completed_cycles'], 2)
        self.assertEqual(len(report['samples']), 3)
        self.assertIsNone(client.loaded)
        self.assertEqual(report['max_growth'], {key: 0 for key in sample()})
        self.assertFalse(any('pull' in path or 'remove' in path for path, _ in client.calls))

    def test_generation_identity_cancellation_and_installation_regressions_fail(self):
        for fault in ['accept_unknown', 'wrong_model', 'retain_loaded', 'empty', 'mtp_bleed', 'broken_stream', 'deleted', 'busy_unload']:
            with self.subTest(fault=fault):
                report = run(FakeClient(**{fault: True}))
                self.assertEqual(report['status'], 'failed')
                self.assertNotIn('DO_NOT_PUBLISH', json.dumps(report))
                self.assertNotIn('/fixtures/', json.dumps(report))

    def test_each_memory_domain_and_file_descriptors_is_gated_after_warmup(self):
        for key, delta in [('rss_bytes', 128 * soak.MIB + 1), ('vram_bytes', 64 * soak.MIB + 1),
                           ('gpu_host_bytes', 128 * soak.MIB + 1), ('fds', 1)]:
            values = [sample(), sample(), sample()]
            values[-1][key] += delta
            report = run(samples=values)
            self.assertEqual(report['status'], 'failed', key)
            self.assertEqual(report['failure_stage'], 'memory_plateau')

    def test_budgets_are_inclusive_and_warmup_growth_is_excluded(self):
        values = [sample(), sample(), sample()]
        values[-1]['vram_bytes'] += 64 * soak.MIB
        self.assertEqual(run(samples=values)['status'], 'passed')
        values = iter([sample(), sample(), sample()])
        report = soak.run_soak(FakeClient(), ['alpha'], 'rocm', lambda: next(values), soak.limits(1, 2, 60), settle=lambda: None)
        self.assertEqual(len(report['samples']), 2)
        self.assertEqual(report['completed_cycles'], 1)

    def test_invalid_settings_and_ids_fail_before_mutation(self):
        for kwargs in [dict(cycles=0), dict(warmup=0), dict(deadline_seconds=59), dict(fd_growth=-1), dict(cycles=True)]:
            with self.assertRaises(ValueError):
                soak.limits(**kwargs)
        client = FakeClient()
        with self.assertRaises(ValueError):
            soak.run_soak(client, ['../model'], 'rocm', sample, soak.limits(), settle=lambda: None)
        self.assertEqual(client.calls, [])

    def test_drm_counters_deduplicate_handles_and_never_publish_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            folder = proc / '7'
            (folder / 'fd').mkdir(parents=True)
            (folder / 'fdinfo').mkdir()
            (folder / 'status').write_text('Name:\tzinc\nVmRSS:\t4096 kB\n')
            for descriptor in ['3', '4']:
                (folder / 'fd' / descriptor).touch()
                (folder / 'fdinfo' / descriptor).write_text('drm-driver: amdgpu\ndrm-client-id: 77\ndrm-pdev: private-device\ndrm-resident-vram: 1024 KiB\ndrm-resident-gtt: 2048 KiB\n')
            metrics = soak.read_metrics(7, proc)
            self.assertEqual(metrics, dict(rss_bytes=4 * soak.MIB, vram_bytes=soak.MIB, gpu_host_bytes=2 * soak.MIB, fds=2))
            self.assertNotIn('private-device', json.dumps(metrics))
            for descriptor in ['3', '4']:
                (folder / 'fdinfo' / descriptor).write_text('pos: 0\n')
            with self.assertRaises(ValueError):
                soak.read_metrics(7, proc)

    def test_client_enforces_stream_terminal_marker_and_deadline(self):
        client = soak.Client(1, time.monotonic() + 5)
        for payload, valid in [(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: [DONE]\n', True),
                               (b'data: {"choices":[{"delta":{"content":"x"}}]}\n', False)]:
            with patch.object(soak.urllib.request, 'urlopen', return_value=io.BytesIO(payload)):
                if valid:
                    client.stream({})
                else:
                    with self.assertRaises(ValueError):
                        client.stream({})
        client.deadline = time.monotonic() - 1
        with self.assertRaises(ValueError):
            client.remaining()

    def test_workstation_guard_precedes_any_server_start(self):
        from run_model_lifecycle_soak import run_owned
        with patch('run_model_lifecycle_soak.platform.system', return_value='Darwin'):
            with self.assertRaisesRegex(ValueError, 'remote Linux'):
                run_owned(None, {}, 'rocm', 0, 4096, None, {}, soak.limits())

    def test_release_evidence_recomputes_growth_and_binds_settings_and_files(self):
        import copy
        spec = dict(cycles=2, warmup=1, deadline_seconds=60, context=4096,
                    max_rss_growth_mib=128, max_vram_growth_mib=64, max_fd_growth=0)
        report = run()
        report.update(binary_sha256='a' * 64, cleanup=True, context=4096, telemetry='drm-process-resident',
                      model_sizes=dict(alpha=10, beta=20))
        soak.verify_report(report, ['alpha', 'beta'], 'rocm', spec, 'a' * 64, dict(alpha=10, beta=20))
        for field, value in [('completed_cycles', 1), ('operations', 1), ('cleanup', False),
                             ('binary_sha256', 'b' * 64), ('models', ['alpha']), ('samples', []),
                             ('model_sizes', dict(alpha=10, beta=21)), ('options', {}), ('status', 'failed')]:
            changed = copy.deepcopy(report)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                soak.verify_report(changed, ['alpha', 'beta'], 'rocm', spec, 'a' * 64, dict(alpha=10, beta=20))
        report['samples'][-1]['fds'] += 1
        with self.assertRaises(ValueError):
            soak.verify_report(report, ['alpha', 'beta'], 'rocm', spec, 'a' * 64)


if __name__ == '__main__':
    unittest.main()
