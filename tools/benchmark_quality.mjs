// Shared by the benchmark harness and dashboard so flagged output uses one policy.
export function outputQualityStatus(preview, generatedTokens = null) {
  const text = `${preview ?? ''}`.trim();
  if (!text) {
    if (generatedTokens != null && generatedTokens <= 0) {
      return {
        tone: 'caution',
        label: 'Generation failed',
        note: 'The request returned without generating a token.',
      };
    }
    return {
      tone: 'neutral',
      label: 'No Preview',
      note: 'No output preview was captured for this workload.',
    };
  }

  if (
    /^<\|(?:im_end|endoftext|end)\|>$/.test(text) ||
    (generatedTokens != null && generatedTokens <= 4 && text.includes('<|im_end|>'))
  ) {
    return {
      tone: 'caution',
      label: 'Preview flagged',
      note: 'ZINC completed numerically, but the preview is only a stop token.',
    };
  }

  if (/<\|[^>\n]{1,64}\|>/.test(text)) {
    return {
      tone: 'caution',
      label: 'Preview flagged',
      note: 'ZINC completed numerically, but the preview contains raw chat control tokens.',
    };
  }

  if (/<pad>/i.test(text)) {
    return {
      tone: 'caution',
      label: 'Preview flagged',
      note: 'ZINC completed numerically, but the preview contains raw padding tokens.',
    };
  }

  if ((text.match(/<think>/g) ?? []).length > 1 || /<\/?parameter>/.test(text)) {
    return {
      tone: 'caution',
      label: 'Preview flagged',
      note: 'ZINC completed numerically, but the preview contains repeated reasoning/control tags.',
    };
  }

  if (
    /(?:^|[\s:])(?:\d+\.){12,}/.test(text) ||
    /(?:^|\n)\s*(\d+[.)])\s*(?:\n\s*\1\s*){11,}/.test(text) ||
    hasRepeatedNumberedItemText(text) ||
    hasRepeatedLineText(text) ||
    /([#*_=-])\1{31,}/.test(text) ||
    hasRepeatedTokenPattern(text) ||
    /\b([A-Za-z]{3,})\1{8,}\b/.test(text)
  ) {
    return {
      tone: 'caution',
      label: 'Preview flagged',
      note: 'ZINC completed numerically, but the preview shows a repeated pattern.',
    };
  }

  return {
    tone: 'positive',
    label: 'Preview OK',
    note: 'The captured preview does not show an obvious stop-token, control-token, or repetition failure.',
  };
}

function hasRepeatedTokenPattern(text) {
  const tokens = text.toLowerCase().match(/[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu) ?? [];
  for (let width = 1; width <= 8; width += 1) {
    if (tokens.length < width * 6) continue;
    for (let start = 0; start + width * 6 <= tokens.length; start += 1) {
      const pattern = tokens.slice(start, start + width).join('\u0000');
      let repeats = 1;
      while (
        start + (repeats + 1) * width <= tokens.length &&
        tokens.slice(start + repeats * width, start + (repeats + 1) * width).join('\u0000') ===
          pattern
      )
        repeats += 1;
      if (repeats >= 6) return true;
    }
  }
  return false;
}

function hasRepeatedNumberedItemText(text) {
  const items = [...text.matchAll(/(?:^|\s)(\d{1,3})[.)]\s+([\s\S]*?)(?=(?:\s+\d{1,3}[.)]\s+)|$)/g)]
    .map((match) => `${match[2] ?? ''}`.toLowerCase().replace(/\s+/g, ' ').trim())
    .filter((item) => item.length >= 24);
  let last = '';
  let run = 0;
  for (const item of items) {
    const key = item.slice(0, 160);
    if (key === last) {
      run += 1;
    } else {
      last = key;
      run = 1;
    }
    if (run >= 4) return true;
  }
  return false;
}

function hasRepeatedLineText(text) {
  const counts = new Map();
  let previous = '';
  let run = 0;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine
      .toLowerCase()
      .replace(/^\s*(?:[-*\u2022]|\d{1,3}[.)])\s+/, '')
      .replace(/\s+/g, ' ')
      .trim();
    if (line.length < 18) continue;

    if (line === previous) {
      run += 1;
    } else {
      previous = line;
      run = 1;
    }
    if (run >= 3) return true;

    const nextCount = (counts.get(line) ?? 0) + 1;
    if (nextCount >= 4) return true;
    counts.set(line, nextCount);
  }
  return false;
}
