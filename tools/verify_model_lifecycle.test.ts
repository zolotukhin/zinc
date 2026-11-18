import { expect, test } from 'bun:test';
import { parseArgs, runVerification } from './verify_model_lifecycle.mjs';

const options = (extra: Record<string, any> = {}) => ({
  base: 'http://localhost:9090',
  timeoutMs: 1000,
  allowModelReload: false,
  switchModel: null,
  ...extra,
});
const mtp = { supports_mtp: true, mtp_default_enabled: true, mtp_request_control: true };
const lifecycle = {
  supports_model_management: true,
  supports_model_unload: true,
  supports_model_pull: false,
};

function fixture({
  legacy = false,
  failSwitch = false,
  corruptReload = false,
  retainActive = false,
  loseInstallation = false,
  thirdParty = false,
  unloadDisconnect = false,
} = {}) {
  let active: string | null = 'original',
    reloaded = false;
  const calls: Array<{ path: string; body: any }> = [];
  const entries = [
    {
      id: 'original',
      installed: true,
      can_activate: true,
      managed: false,
      supported_on_current_gpu: true,
      fits_current_gpu: true,
    },
    {
      id: 'alternative',
      installed: true,
      can_activate: true,
      managed: false,
      supported_on_current_gpu: true,
      fits_current_gpu: true,
    },
    { id: 'not-installed', installed: false, can_activate: true, managed: true },
    { id: 'unsupported', installed: true, can_activate: false, managed: false },
  ];
  return {
    calls,
    active: () => active,
    fetch: async (url: string, init: any = {}) => {
      const path = new URL(url).pathname;
      const body = init.body ? JSON.parse(init.body) : undefined;
      calls.push({ path, body });
      if (path === '/health')
        return Response.json({
          status: 'ok',
          backend: 'rocm',
          ...(legacy ? {} : lifecycle),
          ...(active
            ? mtp
            : { supports_mtp: false, mtp_default_enabled: false, mtp_request_control: true }),
        });
      if (path === '/v1/models')
        return Response.json({
          ...(legacy ? {} : lifecycle),
          data: entries.map((entry) => ({
            ...entry,
            ...(entry.id === 'original' && loseInstallation && reloaded
              ? { installed: false }
              : {}),
            active: entry.id === active,
            ...(entry.id === active
              ? corruptReload && reloaded
                ? { ...mtp, mtp_default_enabled: false }
                : mtp
              : {}),
          })),
        });
      if (path === '/v1/chat/reset') return Response.json({ reset: true });
      if (path === '/v1/models/unload') {
        expect(body).toEqual({ model: 'original' });
        active = retainActive ? 'original' : null;
        reloaded = true;
        if (thirdParty) {
          active = 'third-party';
          entries.push({ ...entries[0], id: 'third-party' });
        }
        if (unloadDisconnect) throw new Error('Connection failed for ' + url);
        return Response.json({ unloaded: true });
      }
      if (path === '/v1/models/activate') {
        if (body.model === 'alternative' && failSwitch) {
          active = null;
          return Response.json({ error: { message: 'Model loading failed' } }, { status: 500 });
        }
        active = body.model;
        return Response.json({ activated: true });
      }
      throw new Error('Unexpected test endpoint');
    },
  };
}

test('model lifecycle CLI is read-only by default and requires explicit reload for switching', () => {
  expect(parseArgs([]).allowModelReload).toBe(false);
  expect(parseArgs(['--base', 'http://localhost:9090/v1/']).base).toBe('http://localhost:9090');
  expect(() => parseArgs(['--switch-model', 'alternative'])).toThrow('requires');
  expect(() => parseArgs(['--timeout-ms', '1'])).toThrow();
  expect(() => parseArgs(['--base', 'http://user:secret@localhost:9090'])).toThrow();
});

test('discovery alone never resets, activates, unloads, pulls, or removes models', async () => {
  const fake = fixture();
  const report = await runVerification(options(), fake.fetch);
  expect(report.status).toBe('passed');
  expect(report.mode).toBe('discovery');
  expect(fake.calls.map((item) => item.path)).toEqual(['/health', '/v1/models']);
  expect(report.restored_original).toBeNull();
});

test('reload checks clear chat separately, retain installed files, and restore MTP policy', async () => {
  const fake = fixture();
  const report = await runVerification(options({ allowModelReload: true }), fake.fetch);
  expect(report.status).toBe('passed');
  expect(report.restored_original).toBe(true);
  expect(report.original_mtp).toEqual(mtp);
  expect(fake.active()).toBe('original');
  const mutations = fake.calls.filter((item) => item.body);
  expect(mutations.map((item) => item.path)).toEqual([
    '/v1/chat/reset',
    '/v1/models/unload',
    '/v1/models/activate',
  ]);
  expect(mutations[0].body.session_id).toBe('zinc-lifecycle-verification');
  expect(report.checks.map((item) => item.name)).toContain('reload_restores_model_and_mtp');
});

test('an installed alternate model switches successfully then restores the original', async () => {
  const fake = fixture();
  const report = await runVerification(
    options({ allowModelReload: true, switchModel: 'alternative' }),
    fake.fetch,
  );
  expect(report.status).toBe('passed');
  expect(
    fake.calls.filter((item) => item.path.endsWith('/activate')).map((item) => item.body.model),
  ).toEqual(['original', 'alternative', 'original']);
  expect(fake.active()).toBe('original');
});

test('unsupported discovery and uninstalled or unknown switch targets stop before mutation', async () => {
  const old = fixture({ legacy: true });
  expect((await runVerification(options({ allowModelReload: true }), old.fetch)).status).toBe(
    'failed',
  );
  expect(old.calls.some((item) => item.body)).toBe(false);
  for (const switchModel of ['unknown', 'not-installed', 'unsupported', 'original']) {
    const fake = fixture();
    expect(
      (await runVerification(options({ allowModelReload: true, switchModel }), fake.fetch)).status,
    ).toBe('failed');
    expect(fake.calls.some((item) => item.body)).toBe(false);
  }
});

test('failed switch still restores the original before reporting failure', async () => {
  const fake = fixture({ failSwitch: true });
  const report = await runVerification(
    options({ allowModelReload: true, switchModel: 'alternative' }),
    fake.fetch,
  );
  expect(report.status).toBe('failed');
  expect(report.restored_original).toBe(true);
  expect(fake.active()).toBe('original');
  expect(
    report.checks.find((item) => item.name === 'switch_activates_selected_model')?.passed,
  ).toBe(false);
});

test('an unload response lost in transit still triggers a restoration attempt', async () => {
  const fake = fixture({ unloadDisconnect: true });
  const report = await runVerification(options({ allowModelReload: true }), fake.fetch);
  expect(report.status).toBe('failed');
  expect(report.restored_original).toBe(true);
  expect(fake.active()).toBe('original');
  expect(JSON.stringify(report)).not.toContain('localhost');
});

test('unload and reload regressions fail honestly without deleting model files', async () => {
  for (const config of [
    { retainActive: true },
    { loseInstallation: true },
    { corruptReload: true },
  ]) {
    const fake = fixture(config);
    const report = await runVerification(options({ allowModelReload: true }), fake.fetch);
    expect(report.status).toBe('failed');
    expect(
      fake.calls.some((item) => item.path.includes('remove') || item.path.includes('pull')),
    ).toBe(false);
  }
});

test("restoration preserves a third party's newly active model", async () => {
  const fake = fixture({ thirdParty: true });
  const report = await runVerification(options({ allowModelReload: true }), fake.fetch);
  expect(report.status).toBe('failed');
  expect(report.restored_original).toBe(false);
  expect(fake.active()).toBe('third-party');
  expect(fake.calls.filter((item) => item.path.endsWith('/activate'))).toHaveLength(0);
});
