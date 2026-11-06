import { expect, test } from 'bun:test';

import { managedModelRoot } from './test_qwen_smoke';

test("managed model root mirrors the engine's cache root on each platform", () => {
  expect(managedModelRoot({ XDG_CACHE_HOME: '/xdg' }, 'linux', '/home/example')).toBe(
    '/xdg/zinc/models/models',
  );
  expect(managedModelRoot({ XDG_CACHE_HOME: '/xdg' }, 'darwin', '/Users/example')).toBe(
    '/xdg/zinc/models/models',
  );
  expect(managedModelRoot({}, 'darwin', '/Users/example')).toBe(
    '/Users/example/Library/Caches/zinc/models/models',
  );
  expect(managedModelRoot({}, 'linux', '/root')).toBe('/root/.cache/zinc/models/models');
});
