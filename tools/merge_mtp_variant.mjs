#!/usr/bin/env bun
/**
 * Merge an MTP-off benchmark artifact into the published site data.
 *
 * Both engines support NextN/MTP. Use explicit --mtp off when collecting the
 * off variant. A fresh artifact's baseline takes precedence; a historical
 * baseline only produces a ratio when both MTP modes were recorded and match.
 *
 *   bun tools/performance_suite.mjs --target rdna --mtp off \
 *     --models qwen38-27b-q4k-m --phase all --no-site-write \
 *     --output /tmp/mtp-off.json
 *   bun tools/merge_mtp_variant.mjs --off /tmp/mtp-off.json
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildComparison } from './performance_suite.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_SITE_DATA = path.join(ROOT, 'site/src/data/zinc-performance.json');

/**
 * Attach every measured MTP-off scenario in `offArtifact` to the matching site
 * scenario as `variants.mtp_off`.
 * @returns {{merged: string[], skipped: string[]}}
 */
export function mergeMtpVariant(siteData, offArtifact, targetId = 'rdna') {
  const offTarget = offArtifact?.targets?.find((target) => target.id === targetId);
  if (!offTarget) throw new Error(`MTP-off artifact has no target '${targetId}'`);
  const siteTarget = siteData?.targets?.find((target) => target.id === targetId);
  if (!siteTarget) throw new Error(`Site data has no target '${targetId}'`);

  const merged = [];
  const skipped = [];
  for (const offModel of offTarget.models ?? []) {
    const siteModel = (siteTarget.models ?? []).find((model) => model.id === offModel.id);
    if (!siteModel) {
      skipped.push(`${offModel.id} (no published row)`);
      continue;
    }
    for (const offScenario of offModel.scenarios ?? []) {
      const siteScenario = (siteModel.scenarios ?? []).find(
        (scenario) => scenario.id === offScenario.id,
      );
      const zinc = offScenario?.zinc;
      if (!siteScenario || !zinc || zinc.unavailable_reason) {
        skipped.push(`${offModel.id}/${offScenario.id}`);
        continue;
      }
      if (zinc.speculative_decoding?.enabled !== false) {
        throw new Error(
          `${offModel.id}/${offScenario.id} in the MTP-off artifact still reports speculative decoding or lacks explicit MTP-off telemetry; rerun with --mtp off`,
        );
      }
      const baseline = offScenario.baseline ?? siteScenario.baseline;
      const comparison =
        baseline && !baseline.unavailable_reason ? buildComparison(zinc, baseline) : null;
      siteScenario.variants = {
        ...(siteScenario.variants ?? {}),
        mtp_off: {
          zinc,
          baseline,
          comparison,
          measured_at: offTarget.generated_at ?? null,
          provenance: offTarget.provenance ?? null,
        },
      };
      merged.push(`${offModel.id}/${offScenario.id}`);
    }
  }
  return { merged, skipped };
}

function parseArgs(argv) {
  const args = { off: null, site: DEFAULT_SITE_DATA, target: 'rdna', dryRun: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--off') args.off = argv[++i];
    else if (arg === '--site') args.site = argv[++i];
    else if (arg === '--target') args.target = argv[++i];
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '--help' || arg === '-h') args.help = true;
    else throw new Error(`Unknown argument '${arg}'`);
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.off) {
    console.log(
      'Usage: bun tools/merge_mtp_variant.mjs --off <artifact.json> [--site <path>] [--target rdna] [--dry-run]',
    );
    process.exit(args.help ? 0 : 1);
  }
  const siteData = JSON.parse(fs.readFileSync(args.site, 'utf8'));
  const offArtifact = JSON.parse(fs.readFileSync(args.off, 'utf8'));
  const { merged, skipped } = mergeMtpVariant(siteData, offArtifact, args.target);
  if (merged.length === 0) throw new Error('No MTP-off scenarios matched the published rows');

  for (const id of merged) console.log(`  merged mtp_off: ${id}`);
  for (const id of skipped) console.log(`  skipped: ${id}`);
  if (args.dryRun) {
    console.log('--dry-run: site data left unchanged');
    return;
  }
  fs.writeFileSync(args.site, `${JSON.stringify(siteData, null, 2)}\n`);
  console.log(`Wrote ${merged.length} MTP-off variant(s) to ${path.relative(ROOT, args.site)}`);
}

if (import.meta.main) {
  await main();
}
