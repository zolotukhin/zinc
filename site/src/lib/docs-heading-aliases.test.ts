import { expect, test } from 'bun:test';
import { rehypeDocHeadingAliases } from './docs-heading-aliases';

test('renamed backend headings retain incoming links without changing canonical navigation', () => {
  const text = { type: 'text', value: 'Backend coverage' };
  const heading = {
    type: 'element',
    tagName: 'h2',
    properties: { id: 'backend-coverage' },
    children: [text],
  };
  rehypeDocHeadingAliases({ slug: 'api' })({ children: [heading] });
  expect(heading.properties.id).toBe('backend-coverage');
  expect(heading.children[0]).toEqual(text);
  expect(heading.children[1]).toEqual({
    type: 'element',
    tagName: 'span',
    properties: { id: 'compatibility-notes' },
    children: [],
  });
});

test('heading aliases are scoped to their guide and do not add IDs to unrelated content', () => {
  const tree = {
    children: [
      { type: 'element', tagName: 'h2', properties: { id: 'backend-coverage' }, children: [] },
      { type: 'element', tagName: 'p', properties: { id: 'kv-cache-configuration' }, children: [] },
      null,
    ],
  };
  const original = structuredClone(tree);
  rehypeDocHeadingAliases({ slug: 'running-zinc' })(tree);
  expect(tree).toEqual(original);
});
