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
 * Bars are ZINC as a percentage of the same-machine llama.cpp run, so models
 * with very different absolute speeds stay comparable in one picture; the
 * absolute tok/s sit next to each bar.
 *
 * Only explicit MTP-off/off pairs are eligible. Missing settings, mismatched
 * settings and correctness cautions never become percentage bars.
 */
import { speculationPair } from './benchmark_speculation.mjs';
import { outputQualityStatus } from './benchmark_quality.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_DATA = path.join(ROOT, 'site/src/data/zinc-performance.json');

const THEME = {
  bg: '#faf8f4',
  card: '#ffffff',
  ink: '#1f2428',
  muted: '#5c6670',
  faint: '#e3ded4',
  track: '#e8e4db',
  zinc: '#d35400',
  zincSoft: '#f0a06a',
  baseline: '#9aa4ae',
  rule: '#d7d2c8',
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

function bar(x, y, width, height, fill, radius = 3) {
  return `<rect x="${x.toFixed(1)}" y="${y}" width="${Math.max(width, 2).toFixed(1)}" height="${height}" rx="${radius}" fill="${fill}"/>`;
}

export function renderSvg(data, targetId) {
  const { target, rows } = collectRows(data, targetId);
  if (rows.length === 0) throw new Error(`Target '${targetId}' has no complete core rows`);

  const width = 1040;
  const padX = 40;
  const cardX = padX;
  const cardW = width - padX * 2;
  const labelW = 250;
  const barX = cardX + labelW + 24;
  const barW = cardW - labelW - 130; // leaves room for the percentage label
  const rowH = 78;
  const headerH = 150;
  const footerH = 90;
  const height = headerH + rows.length * rowH + footerH;
  // The axis follows the data (rounded up to a clean step) so no bar is clipped
  // into looking the same length as a slower one.
  const axisMax = Math.max(
    150,
    Math.ceil(Math.max(...rows.flatMap((r) => [r.prefillPct, r.decodePct])) / 50) * 50,
  );
  const pctToX = (pct) => barX + (Math.min(pct, axisMax) / axisMax) * barW;
  const hundred = pctToX(100);

  const rowVersions = [...new Set(rows.map((row) => row.zincVersion))];
  const llamaVersions = [...new Set(rows.map((row) => row.llamaCommit))];
  const measuredDates = [...new Set(rows.map((row) => row.measuredAt.slice(0, 10)))];
  const mixedSources =
    rowVersions.length > 1 || llamaVersions.length > 1 || measuredDates.length > 1;
  const zincVersion = rowVersions[0];
  const llamaCommit = llamaVersions[0];
  const measured = measuredDates[0];
  const gpu = target.machine?.gpu ?? 'GPU';
  const backend = /rocm/i.test(targetId) ? 'ROCm' : (target.methodology?.zinc_backend ?? '');
  const llamaBackend = target.methodology?.llama_backend ?? null;
  const backends =
    llamaBackend && llamaBackend !== backend
      ? `ZINC on ${backend}, llama.cpp on ${llamaBackend}`
      : `both on ${backend}`;

  const out = [];
  out.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">`,
  );
  out.push(
    `<title id="title">ZINC compared with llama.cpp on ${escapeXml(gpu)} using ${escapeXml(backend)}</title>`,
  );
  out.push(
    `<desc id="desc">For each model, two bars show ZINC's prompt-processing and token-generation speed as a percentage of a llama.cpp run on the same GPU, model file and prompt. The dashed line marks llama.cpp's own speed.</desc>`,
  );
  out.push(`<rect width="${width}" height="${height}" fill="${THEME.bg}"/>`);
  out.push(
    `<rect x="${cardX - 16}" y="24" width="${cardW + 32}" height="${height - 48}" rx="18" fill="${THEME.card}" stroke="${THEME.faint}"/>`,
  );
  out.push(
    `<g font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" fill="${THEME.ink}">`,
  );

  // Header
  out.push(
    `<text x="${cardX}" y="72" font-size="27" font-weight="700">ZINC vs llama.cpp on one Radeon</text>`,
  );
  out.push(
    `<text x="${cardX}" y="99" font-size="14.5" fill="${THEME.muted}">${escapeXml(gpu)} · ${escapeXml(backends)} · same GGUF, same prompt tokens, reusable servers</text>`,
  );

  // Legend
  const legendY = 122;
  out.push(bar(cardX, legendY - 9, 26, 11, THEME.zinc, 3));
  out.push(
    `<text x="${cardX + 34}" y="${legendY}" font-size="13" fill="${THEME.muted}">prompt processing</text>`,
  );
  out.push(bar(cardX + 168, legendY - 9, 26, 11, THEME.zincSoft, 3));
  out.push(
    `<text x="${cardX + 202}" y="${legendY}" font-size="13" fill="${THEME.muted}">token generation</text>`,
  );
  out.push(
    `<text x="${cardX + cardW}" y="${legendY}" font-size="13" text-anchor="end" fill="${THEME.muted}">bar length = ZINC speed ÷ llama.cpp speed</text>`,
  );

  // Axis
  const axisTop = headerH - 12;
  const axisBottom = headerH + rows.length * rowH - 18;
  out.push(
    `<line x1="${cardX}" y1="${axisTop - 14}" x2="${cardX + cardW}" y2="${axisTop - 14}" stroke="${THEME.rule}"/>`,
  );
  out.push(
    `<line x1="${hundred}" y1="${axisTop}" x2="${hundred}" y2="${axisBottom}" stroke="${THEME.baseline}" stroke-dasharray="4 5"/>`,
  );
  out.push(
    `<text x="${hundred}" y="${axisBottom + 18}" font-size="11.5" text-anchor="middle" fill="${THEME.muted}">llama.cpp speed</text>`,
  );

  // Rows
  rows.forEach((row, i) => {
    const top = headerH + i * rowH;
    out.push(
      `<text x="${cardX}" y="${top + 20}" font-size="16" font-weight="600">${escapeXml(row.label)}</text>`,
    );
    out.push(
      `<text x="${cardX}" y="${top + 40}" font-size="12.5" fill="${THEME.muted}">${Math.round(row.zincPrefill)} vs ${Math.round(row.basePrefill)} tok/s · ${row.zincDecode.toFixed(1)} vs ${row.baseDecode.toFixed(1)} tok/s</text>`,
    );

    out.push(
      `<text x="${cardX}" y="${top + 61}" font-size="10.5" fill="${THEME.muted}">${escapeXml(row.measuredAt.slice(0, 10))} · ZINC ${escapeXml(row.zincVersion.slice(0, 9))} / llama.cpp ${escapeXml(row.llamaCommit.slice(0, 9))}</text>`,
    );

    const bars = [
      { pct: row.prefillPct, fill: THEME.zinc, y: top + 4 },
      { pct: row.decodePct, fill: THEME.zincSoft, y: top + 27 },
    ];
    for (const b of bars) {
      out.push(bar(barX, b.y, barW, 17, THEME.track, 4));
      out.push(bar(barX, b.y, pctToX(b.pct) - barX, 17, b.fill, 4));
      out.push(
        `<text x="${pctToX(b.pct) + 10}" y="${b.y + 13}" font-size="13.5" font-weight="700" fill="${THEME.ink}">${Math.round(b.pct)}%</text>`,
      );
    }
  });

  // Footnotes
  let footY = headerH + rows.length * rowH + 22;
  out.push(
    `<text x="${cardX}" y="${footY}" font-size="12" fill="${THEME.muted}">${mixedSources ? 'Mixed measurement dates and revisions; see each row. Medians of repeated requests after warmup.' : `Measured ${escapeXml(measured)} · ZINC ${escapeXml(zincVersion)} · llama.cpp ${escapeXml(llamaCommit)} · medians of repeated runs after warmup`}</text>`,
  );
  footY += 20;
  out.push(
    `<text x="${cardX}" y="${footY}" font-size="12" fill="${THEME.muted}">MTP off for both engines. Matched MTP-on pairs, per-model dates, settings and raw samples: zolotukhin.ai/zinc/benchmarks</text>`,
  );
  out.push(`</g></svg>`);
  return out.join('\n');
}

function parseArgs(argv) {
  const args = { target: 'rdna-rocm', data: DEFAULT_DATA, out: null };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--target') args.target = argv[++i];
    else if (argv[i] === '--data') args.data = argv[++i];
    else if (argv[i] === '--out') args.out = argv[++i];
    else throw new Error(`Unknown argument '${argv[i]}'`);
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const data = JSON.parse(fs.readFileSync(args.data, 'utf8'));
  const svg = renderSvg(data, args.target);
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
