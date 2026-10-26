import { expect, test } from 'bun:test';
import { parseArgs } from './performance_suite.mjs';

function parse(argv: string[]) {
  const previous = process.env.ZINC_RDNA_VK_DEVICE;
  delete process.env.ZINC_RDNA_VK_DEVICE;
  try {
    return parseArgs(argv);
  } finally {
    if (previous === undefined) delete process.env.ZINC_RDNA_VK_DEVICE;
    else process.env.ZINC_RDNA_VK_DEVICE = previous;
  }
}

const flags = [
  ['--runs', 'runs'],
  ['--warmup', 'warmupRuns'],
  ['--timeout-ms', 'timeoutMs'],
  ['--rdna-vk-device', 'rdnaVkDevice'],
] as const;

for (const [flag, key] of flags) {
  test(flag + ' accepts complete decimal safe integers', () => {
    for (const input of ['1', '12', '0003', '9007199254740991'])
      expect(parse([flag, input])[key]).toBe(Number(input));
  });

  test(flag + ' rejects partial parses, nondecimal syntax, and unsafe integers', () => {
    for (const input of [
      '3junk',
      '3.5',
      '1e3',
      '0x10',
      '+3',
      '-0',
      '-1',
      ' 3',
      '3 ',
      '',
      'Infinity',
      'NaN',
      '9007199254740992',
      '9007199254740993',
    ])
      expect(() => parse([flag, input])).toThrow(flag);
    expect(() => parse([flag])).toThrow(flag);
  });
}

test('integer validation preserves each flag zero rule', () => {
  expect(() => parse(['--runs', '0'])).toThrow('--runs');
  for (const [flag, key] of flags.slice(1)) expect(parse([flag, '0'])[key]).toBe(0);
});

test('environment device selectors use the same strict integer contract', () => {
  const previous = process.env.ZINC_RDNA_VK_DEVICE;
  try {
    for (const value of ['3junk', '3.5', '1e3', '9007199254740993']) {
      process.env.ZINC_RDNA_VK_DEVICE = value;
      expect(() => parseArgs([])).toThrow('ZINC_RDNA_VK_DEVICE');
    }
    process.env.ZINC_RDNA_VK_DEVICE = '0002';
    expect(parseArgs([]).rdnaVkDevice).toBe(2);
  } finally {
    if (previous === undefined) delete process.env.ZINC_RDNA_VK_DEVICE;
    else process.env.ZINC_RDNA_VK_DEVICE = previous;
  }
});
