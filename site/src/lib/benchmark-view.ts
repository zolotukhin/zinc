import { outputQualityStatus } from '../../../tools/benchmark_quality.mjs';

// Presentation helpers. Published measurements stay immutable; each view selects
// a complete recorded run before deriving any chart, summary, or timing.
export type BenchmarkRecord = Record<string, any>;
export type MtpMode = 'off' | 'on';

export const scenarioOrder = ['core', 'context-medium', 'context-long', 'decode-extended'];
export const scenarioNames: Record<string, string> = {
  core: 'Quick chat',
  'context-medium': 'Coding review',
  'context-long': 'Incident context',
  'decode-extended': 'Long coding draft',
};

export function metric(summary: any): number | null {
  const value = summary?.median ?? summary?.avg;
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function formatNumber(value: number | null | undefined, digits = 1) {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(digits)
    : 'Not recorded';
}

export function formatRatio(value: number | null) {
  return value != null && Number.isFinite(value) ? `${value.toFixed(2)}×` : 'Not comparable';
}

export function formatDate(iso: string | null | undefined) {
  if (!iso || !Number.isFinite(Date.parse(iso))) return 'Not recorded';
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(iso));
}

export function sourceUrl(commit: string | null | undefined, path: string) {
  return `https://github.com/zolotukhin/zinc/blob/${commit || 'main'}/${path}`;
}

export function commitUrl(project: 'zinc' | 'llama.cpp', commit: string) {
  const repo = project === 'zinc' ? 'zolotukhin/zinc' : 'ggml-org/llama.cpp';
  return `https://github.com/${repo}/commit/${commit}`;
}

export function scenarios(model: BenchmarkRecord): BenchmarkRecord[] {
  return model.scenarios?.length ? model.scenarios : [{ ...model, id: 'core' }];
}

export function hasMtpVariants(target: BenchmarkRecord) {
  return target.models.some((model: BenchmarkRecord) =>
    scenarios(model).some(
      (scenario) =>
        scenario.zinc?.speculative_decoding?.enabled &&
        scenario.variants?.mtp_off?.zinc &&
        !scenario.variants.mtp_off.zinc.unavailable_reason,
    ),
  );
}

export function backendLabel(target: BenchmarkRecord) {
  const labels: Record<string, string> = {
    'rdna-rocm': 'ROCm',
    rdna: 'Vulkan',
    'rdna-zinc-rt': 'ZINC_RT',
    intel: 'Vulkan',
    metal: 'Metal',
    cuda: 'CUDA',
  };
  return labels[target.id] ?? target.methodology?.zinc_backend ?? 'Not recorded';
}

export function platformLabel(target: BenchmarkRecord) {
  if (target.id.startsWith('rdna')) return 'AMD Radeon';
  return (
    ({ intel: 'Intel Arc', metal: 'Apple Silicon', cuda: 'NVIDIA RTX' } as Record<string, string>)[
      target.id
    ] ?? target.label
  );
}

export function rowsFor(target: BenchmarkRecord, scenarioId: string, mode: MtpMode) {
  return target.models
    .flatMap((model: BenchmarkRecord) => {
      const original = scenarios(model).find((scenario) => scenario.id === scenarioId);
      if (!original) return [];
      const variant =
        mode === 'off' && original.zinc?.speculative_decoding?.enabled
          ? original.variants?.mtp_off
          : null;
      const missingOff =
        mode === 'off' && original.zinc?.speculative_decoding?.enabled && !variant?.zinc;
      const baseProvenance = { ...target.provenance, ...model.provenance };
      const variantProvenance = variant?.provenance;
      const provenance = variantProvenance
        ? {
            ...baseProvenance,
            ...(variantProvenance.zinc ? variantProvenance : { zinc: variantProvenance }),
          }
        : baseProvenance;
      const selected = variant
        ? {
            ...original,
            ...variant,
            baseline: variant.baseline ?? original.baseline,
            output_quality: variant.output_quality,
          }
        : original;
      return [
        {
          ...selected,
          ...(missingOff
            ? {
                zinc: { unavailable_reason: 'A run with MTP off was not recorded.' },
                comparison: null,
              }
            : {}),
          model,
          provenance,
          measured_at:
            variant?.measured_at ??
            original.measured_at ??
            model.generated_at ??
            target.generated_at,
        },
      ];
    })
    .sort((a: BenchmarkRecord, b: BenchmarkRecord) =>
      a.model.label.localeCompare(b.model.label, undefined, { numeric: true }),
    );
}

export function phaseSeconds(engine: BenchmarkRecord | undefined): number | null {
  const prompt = engine?.prefill_tokens ?? engine?.prompt_tokens;
  const generated = engine?.generated_tokens;
  const prefill = metric(engine?.prefill_tps);
  const decode = metric(engine?.decode_tps);
  if (
    !Number.isFinite(prompt) ||
    !Number.isFinite(generated) ||
    prompt < 0 ||
    generated <= 0 ||
    (prompt > 0 && (prefill == null || prefill <= 0)) ||
    decode == null ||
    decode <= 0
  )
    return null;
  return (prompt > 0 ? prompt / prefill! : 0) + generated / decode;
}

export function promptParity(row: BenchmarkRecord) {
  const zinc = row.zinc?.prompt_tokens;
  const llama = row.baseline?.prompt_tokens;
  if (!Number.isFinite(zinc) || !Number.isFinite(llama)) return null;
  // Match the harness: a one-token BOS accounting difference is allowed.
  return Math.abs(zinc - llama) <= 1;
}

export function overallRatio(row: BenchmarkRecord): number | null {
  const zincSeconds = phaseSeconds(row.zinc);
  const llamaSeconds = phaseSeconds(row.baseline);
  const generated = row.zinc?.generated_tokens;
  if (
    promptParity(row) !== true ||
    qualityStatus(row).tone === 'caution' ||
    !Number.isFinite(generated) ||
    generated !== row.baseline?.generated_tokens ||
    !Number.isFinite(row.max_tokens) ||
    generated < row.max_tokens ||
    zincSeconds == null ||
    llamaSeconds == null
  )
    return null;
  return llamaSeconds / zincSeconds;
}

export function rateRatio(row: BenchmarkRecord, phase: 'prefill_tps' | 'decode_tps') {
  const zinc = metric(row.zinc?.[phase]);
  const llama = metric(row.baseline?.[phase]);
  return zinc != null && zinc > 0 && llama != null && llama > 0 ? zinc / llama : null;
}

export function aggregate(rows: BenchmarkRecord[]) {
  const completed = rows.filter(
    (row) => (metric(row.zinc?.decode_tps) ?? 0) > 0 && qualityStatus(row).tone !== 'caution',
  );
  const comparable = completed.filter((row) => overallRatio(row) != null);
  const mean = (phase: string) => {
    const values = completed
      .map((row) => metric(row.zinc?.[phase]))
      .filter((value): value is number => value != null);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const zincSeconds = comparable.reduce((sum, row) => sum + phaseSeconds(row.zinc)!, 0);
  const llamaSeconds = comparable.reduce((sum, row) => sum + phaseSeconds(row.baseline)!, 0);
  return {
    prefill: mean('prefill_tps'),
    decode: mean('decode_tps'),
    overall: zincSeconds > 0 ? llamaSeconds / zincSeconds : null,
    comparable: comparable.length,
    completed: completed.length,
    total: rows.length,
  };
}

export function sampleCount(summary: any): number | null {
  return Array.isArray(summary?.samples) ? summary.samples.length : null;
}

export function barWidth(value: number | null, maximum: number) {
  return value != null && value > 0 && maximum > 0 ? (100 * value) / maximum : 0;
}

export function qualityStatus(row: BenchmarkRecord) {
  const observed = outputQualityStatus(row.zinc?.output_preview, row.zinc?.generated_tokens);
  return row.output_quality?.tone === 'caution' ? row.output_quality : observed;
}
