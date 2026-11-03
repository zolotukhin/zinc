import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../src/server/chat.html', import.meta.url), 'utf8');
const source = html.match(/function modelQualityHint\(model\) \{[\s\S]*?\n      \}/)?.[0];
if (!source) throw new Error('Chat model recommendation helper is missing');
const hint = new Function(`return (${source});`)() as (model: Record<string, unknown>) => string;

test('supported inference does not become a chat recommendation in the picker', () => {
  expect(hint({ status: 'supported', recommended_for_chat: false })).toContain(
    'Chat not recommended',
  );
  expect(hint({ status: 'supported', recommended_for_chat: true })).toBe('');
  expect(html).toContain('modelQualityHint(m)');
});

test('missing recommendation metadata from older servers remains unknown', () => {
  expect(hint({ status: 'supported' })).toBe('');
  expect(hint({ status: 'experimental' })).toBe('');
});
