import { expect, test } from 'bun:test';
import { createDocLinkResolver, rehypeDocLinks } from '../site/src/lib/docs-links';

const options = {
  sourcePath: 'docs/OPENCODE.md',
  docs: [
    { sourcePath: 'docs/GETTING_STARTED.md', slug: 'getting-started' },
    { sourcePath: 'docs/OPENCODE.md', slug: 'opencode' },
    { sourcePath: 'docs/ROCM.md', slug: 'rocm' },
    { sourcePath: 'docs/Guide With Spaces.md', slug: 'guide-with-spaces' },
  ],
};
const resolve = createDocLinkResolver(options);

test('relative Markdown docs use their published route and retain query and anchor', () => {
  expect(resolve('GETTING_STARTED.md')).toBe('/zinc/docs/getting-started/');
  expect(resolve('./ROCM.md?plain=1#long-coding-sessions')).toBe(
    '/zinc/docs/rocm/?plain=1#long-coding-sessions',
  );
  expect(resolve('../docs/OPENCODE.md#try-a-real-task-in-a-worktree')).toBe(
    '/zinc/docs/opencode/#try-a-real-task-in-a-worktree',
  );
  expect(resolve('Guide%20With%20Spaces.md')).toBe('/zinc/docs/guide-with-spaces/');
  expect(resolve('docs/ROCM.md#context')).toBe('/zinc/docs/rocm/#context');
});

test('external, absolute, section and non-repository asset links stay unchanged', () => {
  for (const href of [
    'https://example.com/guide.md',
    '//example.com/guide.md',
    'mailto:hello@example.com',
    '/zinc/docs/rocm/',
    '#examples',
    '../assets/graph.svg',
    'setup.mjs',
    '',
  ])
    expect(resolve(href)).toBe(href);
  expect(resolve('invalid%zz.md')).toBe('invalid%zz.md');
});

test('relative and bare repository sources never become fake documentation URLs', () => {
  for (const [href, destination] of [
    ['../src/server/model_manager_metal.zig', 'src/server/model_manager_metal.zig'],
    ['src/server/runtime.zig', 'src/server/runtime.zig'],
    ['./src/model/loader_metal.zig', 'src/model/loader_metal.zig'],
    ['../src/compute/forward_metal.zig', 'src/compute/forward_metal.zig'],
    ['../src/metal/buffer.zig', 'src/metal/buffer.zig'],
    ['../src/metal/device.zig', 'src/metal/device.zig'],
    ['../src/metal/shim.h', 'src/metal/shim.h'],
    ['../benchmarks/metal_inference.zig', 'benchmarks/metal_inference.zig'],
    ['../tools/benchmark_api.mjs', 'tools/benchmark_api.mjs'],
    ['../build.zig', 'build.zig'],
    ['build.zig', 'build.zig'],
  ])
    expect(resolve(href), href).toBe(`https://github.com/zolotukhin/zinc/blob/main/${destination}`);
  expect(resolve('../src/server/runtime.zig?plain=1#L20')).toBe(
    'https://github.com/zolotukhin/zinc/blob/main/src/server/runtime.zig?plain=1#L20',
  );
  expect(resolve('../src/metal/')).toBe('https://github.com/zolotukhin/zinc/tree/main/src/metal/');
});

test('repository Markdown outside the published docs points to its source', () => {
  expect(resolve('../AGENTS.md#boundaries')).toBe(
    'https://github.com/zolotukhin/zinc/blob/main/AGENTS.md#boundaries',
  );
  expect(resolve('../README.md')).toBe('https://github.com/zolotukhin/zinc/blob/main/README.md');
});

test('the HTML transform rewrites anchors without changing text, code or images', () => {
  const tree: any = {
    type: 'root',
    children: [
      {
        type: 'element',
        tagName: 'p',
        children: [
          { type: 'element', tagName: 'a', properties: { href: 'ROCM.md' }, children: [] },
          {
            type: 'element',
            tagName: 'a',
            properties: { href: '../src/server/runtime.zig' },
            children: [],
          },
          { type: 'element', tagName: 'code', children: [{ type: 'text', value: 'ROCM.md' }] },
          { type: 'element', tagName: 'img', properties: { src: 'ROCM.md' } },
        ],
      },
    ],
  };
  rehypeDocLinks(options)(tree);
  const nodes = tree.children[0].children;
  expect(nodes[0].properties.href).toBe('/zinc/docs/rocm/');
  expect(nodes[1].properties.href).toBe(
    'https://github.com/zolotukhin/zinc/blob/main/src/server/runtime.zig',
  );
  expect(nodes[2].children[0].value).toBe('ROCM.md');
  expect(nodes[3].properties.src).toBe('ROCM.md');
});
