#!/usr/bin/env bun
/**
 * Render the README's benchmark chart from the published artifact.
 *
 * The chart used to be hand-drawn SVG, so its numbers drifted from
 * site/src/data/zinc-performance.json every time the suite ran. This reads the
 * artifact and draws the core-scenario comparison for one target:
 *
 *   bun tools/render_benchmark_svg.mjs --target rdna-rocm --out assets/rocm-r9700-benchmark.svg
 *
 *   bun tools/render_benchmark_svg.mjs --readme
 *
 * Each phase has a shared absolute tokens/s scale across the models. Paired
 * orange/blue bars show the two engines directly; badges show ZINC / llama.cpp.
 * --readme writes light/dark and wide/compact variants for GitHub's <picture>.
 *
 * Only explicit MTP-off/off pairs are eligible. Missing settings, mismatched
 * settings and correctness cautions never become comparison bars.
 */
import { speculationPair } from './benchmark_speculation.mjs';
import { outputQualityStatus } from './benchmark_quality.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_DATA = path.join(ROOT, 'site/src/data/zinc-performance.json');

const THEMES = {
  light: {
    bg: '#ffffff',
    ink: '#1f2328',
    muted: '#59636e',
    faint: '#d1d9e0',
    panel: '#f6f8fa',
    track: '#e5eaf0',
    zinc: '#bc4c00',
    zincTint: '#fff1e5',
    llama: '#2864b4',
    llamaTint: '#eaf2ff',
  },
  dark: {
    bg: '#0d1117',
    ink: '#f0f6fc',
    muted: '#9da7b3',
    faint: '#30363d',
    panel: '#151b23',
    track: '#252e3b',
    zinc: '#ffa657',
    zincTint: '#38271c',
    llama: '#79b8ff',
    llamaTint: '#1a2c44',
  },
};

function metric(block, key) {
  const value = block?.[key];
  if (!value) return null;
  const n = value.median ?? value.avg ?? null;
  return Number.isFinite(n) ? n : null;
}

function escapeXml(text) {
  return String(text).replace(
    /[<>&"]/g,
    (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' })[c],
  );
}

function shortModelLabel(model) {
  const label = model.label ?? model.id;
  return label.replace(/\s*(Q4_K_M|Q4_K_XL|UD Q4_K_XL)\s*$/i, '').trim();
}

/** Core-scenario rows for a target, strongest prompt-processing ratio first. */
export function collectRows(data, targetId) {
  const target = data.targets?.find((t) => t.id === targetId);
  if (!target) throw new Error(`No target '${targetId}' in the artifact`);
  const rows = [];
  for (const model of target.models ?? []) {
    const core = (model.scenarios ?? []).find((s) => s.id === 'core');
    if (!core) continue;
    const selected = core.zinc?.speculative_decoding?.enabled ? core.variants?.mtp_off : core;
    if (!selected) continue;
    const baseline = selected.baseline ?? core.baseline;
    if (
      !speculationPair(selected.zinc, baseline).matched ||
      selected.zinc?.speculative_decoding?.enabled !== false
    )
      continue;
    if (
      [model, selected].some(
        (row) =>
          row.correctness_review?.tone === 'caution' ||
          row.output_quality?.tone === 'caution' ||
          outputQualityStatus(row.zinc?.output_preview, row.zinc?.generated_tokens).tone ===
            'caution',
      )
    )
      continue;
    const zincPrefill = metric(selected.zinc, 'prefill_tps');
    const zincDecode = metric(selected.zinc, 'decode_tps');
    const basePrefill = metric(baseline, 'prefill_tps');
    const baseDecode = metric(baseline, 'decode_tps');
    if (!zincPrefill || !zincDecode || !basePrefill || !baseDecode) continue;
    rows.push({
      id: model.id,
      label: shortModelLabel(model),
      quant: model.quant ?? 'GGUF',
      prefillPct: (zincPrefill / basePrefill) * 100,
      decodePct: (zincDecode / baseDecode) * 100,
      zincPrefill,
      zincDecode,
      basePrefill,
      baseDecode,
      comparable: true,
      zincVersion:
        selected.provenance?.zinc?.version ??
        selected.provenance?.version ??
        model.provenance?.zinc?.version ??
        target.provenance?.zinc?.version ??
        'unknown',
      llamaCommit:
        selected.provenance?.llama_cpp?.commit ??
        model.provenance?.llama_cpp?.commit ??
        target.provenance?.llama_cpp?.commit ??
        'unknown',
      measuredAt:
        selected.measured_at ?? core.measured_at ?? model.generated_at ?? target.generated_at ?? '',
    });
  }
  // Ordered by prompt processing: it is the axis with the widest spread, and
  // each model's generation bar sits directly beneath its own prompt bar.
  rows.sort((a, b) => b.prefillPct - a.prefillPct);
  return { target, rows };
}

function rect(x, y, width, height, fill, radius = 0) {
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}"/>`;
}

function text(x, y, value, size, fill, attributes = '') {
  return `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" ${attributes}>${escapeXml(value)}</text>`;
}

function dateLabel(iso) {
  const value = new Date(iso);
  return Number.isFinite(value.valueOf())
    ? new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        timeZone: 'UTC',
      }).format(value)
    : 'Date not recorded';
}

export function renderSvg(data, targetId, { theme = 'light', layout = 'wide' } = {}) {
  const colors = THEMES[theme];
  if (!colors) throw new Error(`Unknown theme '${theme}'`);
  if (!['wide', 'compact'].includes(layout)) throw new Error(`Unknown layout '${layout}'`);
  const { target, rows } = collectRows(data, targetId);
  if (rows.length === 0) throw new Error(`Target '${targetId}' has no complete core rows`);

  const compact = layout === 'compact';
  const width = compact ? 420 : 1080;
  const pad = compact ? 24 : 32;
  const contentW = width - pad * 2;
  const headerH = compact ? 176 : 210;
  const rowH = compact ? 228 : 100;
  const rowsBottom = headerH + rows.length * rowH;
  const gpu = target.machine?.gpu ?? 'GPU';
  const backendNames = {
    'rdna-rocm': 'ROCm / HIP',
    rdna: 'Vulkan',
    intel: 'Vulkan',
    metal: 'Metal',
    cuda: 'CUDA',
  };
  const backend =
    backendNames[targetId] ?? target.methodology?.zinc_backend ?? 'Backend not recorded';
  const sources = [];
  const labelledRows = rows.map((row) => {
    let index = sources.findIndex(
      (source) => source.zincVersion === row.zincVersion && source.llamaCommit === row.llamaCommit,
    );
    if (index < 0) index = sources.push(row) - 1;
    return { ...row, ref: index + 1 };
  });
  const sourceLines = compact ? sources.length : Math.ceil(sources.length / 2);
  const height = rowsBottom + 172 + sourceLines * 20;
  const prefillMax = Math.max(...rows.flatMap((row) => [row.zincPrefill, row.basePrefill]));
  const decodeMax = Math.max(...rows.flatMap((row) => [row.zincDecode, row.baseDecode]));
  const out = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">`,
    `<title id="title">ZINC vs llama.cpp · ${escapeXml(gpu)} · ${escapeXml(backend)}</title>`,
    `<desc id="desc">Quick-chat medians in tokens per second; higher is faster. Orange bars are ZINC, blue bars are llama.cpp. Both engines use the same GGUF and prompt with MTP off. Each phase uses one absolute scale across all models. A badge is ZINC speed divided by llama.cpp speed, not an end-to-end speedup. Mixed measurement dates and revisions are shown per model. ${rows
      .map(
        (row) =>
          `${row.label}: ${dateLabel(row.measuredAt)}; ZINC ${row.zincVersion}; llama.cpp ${row.llamaCommit}.`,
      )
      .map(escapeXml)
      .join(' ')}</desc>`,
    `<rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" rx="16" fill="${colors.bg}" stroke="${colors.faint}"/>`,
    `<g font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif" fill="${colors.ink}" style="font-variant-numeric:tabular-nums">`,
  ];
  const label = (x, y, value, size = 14, fill = colors.ink, attributes = '') =>
    out.push(text(x, y, value, size, fill, attributes));
  const line = (y) => out.push(`<path d="M${pad} ${y}H${width - pad}" stroke="${colors.faint}"/>`);

  // The same palette identifies engines in both phases; the bars show actual
  // speed, so a slower ZINC result remains as visible as a faster one.
  function phase(name, x, y, w, zinc, llama, maximum, ratio, showHeading) {
    if (showHeading) label(x, y + 14, name, 14, colors.ink, 'font-weight="600"');
    const pillW = 72;
    const pillX = x + w - pillW;
    const ahead = ratio >= 1;
    out.push(rect(pillX, y, pillW, 24, ahead ? colors.zincTint : colors.llamaTint, 6));
    label(
      pillX + pillW / 2,
      y + 17,
      `${ratio.toFixed(2)}×`,
      15,
      ahead ? colors.zinc : colors.llama,
      'font-weight="700" text-anchor="middle"',
    );
    const barX = x + 65;
    const barW = w - 65 - 62;
    for (const [engine, value, color, offset] of [
      ['ZINC', zinc, colors.zinc, 43],
      ['llama.cpp', llama, colors.llama, 68],
    ]) {
      label(x, y + offset, engine, 12, color, 'font-weight="600"');
      out.push(rect(barX, y + offset - 10, barW, 10, colors.track, 3));
      out.push(
        rect(barX, y + offset - 10, Number(((value / maximum) * barW).toFixed(2)), 10, color, 3),
      );
      label(
        x + w,
        y + offset + 1,
        value.toFixed(1),
        16,
        colors.ink,
        'text-anchor="end" font-weight="600"',
      );
    }
  }

  label(pad, 35, 'ZINC  /  BENCHMARKS', 11, colors.zinc, 'font-weight="700" letter-spacing="1.6"');
  label(
    pad,
    compact ? 73 : 82,
    'ZINC vs llama.cpp',
    compact ? 30 : 36,
    colors.ink,
    'font-weight="700" letter-spacing="-0.8"',
  );
  label(pad, compact ? 101 : 113, gpu, compact ? 16 : 18, colors.muted);
  label(
    pad,
    compact ? 126 : 143,
    `${backend} · Quick chat · MTP off for both engines`,
    compact ? 12 : 14,
    colors.muted,
  );
  if (!compact) {
    label(
      width - pad,
      37,
      `${rows.length} MODELS`,
      11,
      colors.muted,
      'text-anchor="end" font-weight="600" letter-spacing="1.2"',
    );
    label(pad, 188, 'MODEL / MEASURED', 11, colors.muted, 'font-weight="600" letter-spacing="1"');
    label(328, 182, 'Prompt processing', 18, colors.ink, 'font-weight="600"');
    label(714, 182, 'Token generation', 18, colors.ink, 'font-weight="600"');
    label(328, 199, 'Tokens/s · higher is faster', 12, colors.muted);
    label(714, 199, 'Tokens/s · higher is faster', 12, colors.muted);
  } else {
    label(pad, 152, 'Tokens/s · higher is faster', 13, colors.muted);
  }
  line(headerH);

  labelledRows.forEach((row, i) => {
    const top = headerH + i * rowH;
    if (i % 2 === 0) out.push(rect(12, top + 1, width - 24, rowH - 2, colors.panel, 8));
    label(
      pad,
      top + (compact ? 30 : 34),
      row.label,
      compact ? 19 : 18,
      colors.ink,
      'font-weight="600"',
    );
    label(
      pad,
      top + (compact ? 52 : 59),
      `${row.quant} · ${dateLabel(row.measuredAt)} · [${row.ref}]`,
      compact ? 12 : 12.5,
      colors.muted,
    );
    const x1 = compact ? pad : 328;
    const x2 = compact ? pad : 714;
    const w = compact ? contentW : 334;
    phase(
      'Prompt processing',
      x1,
      top + (compact ? 70 : 12),
      w,
      row.zincPrefill,
      row.basePrefill,
      prefillMax,
      row.prefillPct / 100,
      compact,
    );
    phase(
      'Token generation',
      x2,
      top + (compact ? 148 : 12),
      w,
      row.zincDecode,
      row.baseDecode,
      decodeMax,
      row.decodePct / 100,
      compact,
    );
  });

  line(rowsBottom);
  let footY = rowsBottom + 30;
  label(
    pad,
    footY,
    compact
      ? '× = ZINC speed ÷ llama.cpp speed'
      : '× = ZINC speed ÷ llama.cpp speed · Each column uses a shared absolute scale.',
    compact ? 12 : 13,
    colors.ink,
    'font-weight="600"',
  );
  footY += 22;
  label(
    pad,
    footY,
    compact
      ? 'Shared scale within each phase; medians after warmup.'
      : 'Medians of repeated requests after warmup. Same model files, prompt tokens and reusable servers.',
    compact ? 11.5 : 12.5,
    colors.muted,
  );
  footY += 22;
  label(
    pad,
    footY,
    'Mixed measurement dates and revisions; sources by row:',
    compact ? 11.5 : 12.5,
    colors.muted,
  );
  footY += 27;
  sources.forEach((row, i) => {
    label(
      pad + (!compact && i % 2 ? 514 : 0),
      footY + (compact ? i : Math.floor(i / 2)) * 20,
      `[${i + 1}] ZINC ${row.zincVersion.slice(0, 9)} / llama.cpp ${row.llamaCommit.slice(0, 9)}`,
      compact ? 11.5 : 12,
      colors.muted,
      'font-family="ui-monospace, SFMono-Regular, Consolas, monospace"',
    );
  });
  footY += sourceLines * 20;
  footY += compact ? 10 : 5;
  label(
    pad,
    footY,
    compact
      ? 'All workloads, settings & raw samples →'
      : 'All GPUs, workloads, matched MTP-on pairs, settings & raw samples →',
    compact ? 13 : 14,
    colors.ink,
    'font-weight="600"',
  );
  label(pad, footY + 23, 'zolotukhin.ai/zinc/benchmarks', compact ? 13 : 14, colors.zinc);
  out.push('</g></svg>');
  return out.join('\n');
}

function parseArgs(argv) {
  const args = {
    target: 'rdna-rocm',
    data: DEFAULT_DATA,
    out: null,
    theme: 'light',
    layout: 'wide',
    readme: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--target') args.target = argv[++i];
    else if (argv[i] === '--data') args.data = argv[++i];
    else if (argv[i] === '--out') args.out = argv[++i];
    else if (argv[i] === '--theme') args.theme = argv[++i];
    else if (argv[i] === '--layout') args.layout = argv[++i];
    else if (argv[i] === '--readme') args.readme = true;
    else throw new Error(`Unknown argument '${argv[i]}'`);
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const data = JSON.parse(fs.readFileSync(args.data, 'utf8'));
  if (args.readme) {
    for (const layout of ['wide', 'compact']) {
      for (const theme of ['light', 'dark']) {
        const suffix = `${layout === 'compact' ? '-mobile' : ''}${theme === 'dark' ? '-dark' : ''}`;
        const output = path.join(ROOT, `assets/rocm-r9700-benchmark${suffix}.svg`);
        fs.writeFileSync(output, `${renderSvg(data, 'rdna-rocm', { theme, layout })}\n`);
        console.log(`Rendered ${path.relative(ROOT, output)}`);
      }
    }
    return;
  }
  const svg = renderSvg(data, args.target, args);
  if (!args.out) {
    process.stdout.write(`${svg}\n`);
    return;
  }
  fs.writeFileSync(args.out, `${svg}\n`);
  const { rows } = collectRows(data, args.target);
  console.log(`Rendered ${rows.length} model rows to ${path.relative(ROOT, args.out)}`);
  for (const row of rows) {
    console.log(
      `  ${row.label.padEnd(26)} prefill ${Math.round(row.prefillPct)}%  decode ${Math.round(row.decodePct)}%${row.speculative ? '  (speculative)' : ''}`,
    );
  }
}

if (import.meta.main) {
  await main();
}
