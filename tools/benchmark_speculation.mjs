// Missing historical telemetry is unknown, never evidence of matched settings.
export function speculationMode(engine) {
  const value = engine?.speculative_decoding?.enabled;
  return typeof value === 'boolean' ? (value ? 'on' : 'off') : 'unknown';
}

export function speculationPair(zinc, baseline) {
  const a = speculationMode(zinc);
  const b = speculationMode(baseline);
  const matched = a !== 'unknown' && a === b;
  return {
    matched,
    label: `ZINC MTP ${a} · llama.cpp MTP ${b === 'unknown' ? 'not recorded' : b}`,
    note: matched
      ? null
      : a === 'unknown' || b === 'unknown'
        ? 'MTP settings were not recorded for both engines. Excluded from comparisons.'
        : 'The engines used different MTP settings. Excluded from comparisons.',
  };
}

// With MTP, some engines exclude draft priming from their prefill timer.
// Equal mode flags alone cannot make those phase totals comparable.
export function prefillTimingsMatch(zinc, baseline) {
  if (!speculationPair(zinc, baseline).matched) return false;
  if (speculationMode(zinc) === 'off') return true;
  const scope = zinc?.prefill_timing_scope;
  return (
    ['target-prefill', 'target-and-draft-prefill'].includes(scope) &&
    scope === baseline?.prefill_timing_scope
  );
}

export function requireBenchmarkReference(observed, expected) {
  if (!observed?.commit || !expected.startsWith(observed.commit) || observed.commit.length < 7)
    throw new Error(`Benchmark llama.cpp must be built from pinned commit ${expected}.`);
  if (!/^[a-f0-9]{64}$/.test(observed.sha256 ?? ''))
    throw new Error('Benchmark llama.cpp needs an exact executable SHA256.');
}
