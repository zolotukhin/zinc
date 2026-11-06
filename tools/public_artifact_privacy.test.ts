import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

test('published benchmark measurements and public/loopback URLs remain valid', () => {
  const artifact = JSON.parse(
    readFileSync(new URL('../site/src/data/zinc-performance.json', import.meta.url), 'utf8'),
  );
  expect(() => assertPublicArtifact(artifact, {})).not.toThrow();
  expect(() =>
    assertPublicArtifact(
      {
        decode_ms: 2222.06,
        url: 'https://github.com/zolotukhin/zinc',
        base: 'http://127.0.0.1:9090',
      },
      {},
    ),
  ).not.toThrow();
});

test('publication rejects private endpoints, paths and identifiers without echoing values', () => {
  const privateIp = [10, 23, 45, 67].join('.');
  const gpu = 'GPU-' + '01234567-89ab-cdef-0123-456789abcdef';
  const values = [
    privateIp,
    'worker.example.' + 'ts.net',
    '/Users/' + 'private-person/work',
    'C:\\Users\\' + 'private-person\\work',
    '/home%2F' + 'private-person/work',
    'fd12' + ':3456::1',
    gpu,
  ];
  for (const value of values) {
    try {
      assertPublicArtifact({ error: value }, {});
      throw new Error('expected a privacy rejection');
    } catch (error) {
      expect(String(error)).toContain('Refusing to publish');
      expect(String(error)).not.toContain(value);
    }
  }
  expect(() =>
    assertPublicArtifact(
      { error: 'worker.example.invalid' },
      { ZINC_HOST: 'worker.example.invalid' },
    ),
  ).toThrow('Refusing to publish');
  expect(() => assertPublicArtifact({ nested: { hostname: 'example' } }, {})).toThrow(
    'Refusing to publish',
  );
  expect(() => assertPublicArtifact({ api_key: 'example' }, {})).toThrow('Refusing to publish');
});
