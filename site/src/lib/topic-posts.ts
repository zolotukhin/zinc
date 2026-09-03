const topicTagRules: Record<string, string[]> = {
  'amd-rdna4-llm-inference': ['rdna4', 'r9700', 'rx-9070-xt'],
  'gemma-local-inference': ['gemma', 'gemma4'],
  'kv-cache-quantization': ['kv-cache', 'turboquant'],
  'opencode-local-coding': ['opencode'],
  'qwen3-6-local-inference': ['qwen3-6'],
};

export function getTopicSlugsForTags(tags: string[] = []): string[] {
  const normalizedTags = new Set(tags.map((tag) => tag.toLowerCase()));

  return Object.entries(topicTagRules)
    .filter(([, matchingTags]) => matchingTags.some((tag) => normalizedTags.has(tag)))
    .map(([slug]) => slug);
}

export function postMatchesTopic(topicSlug: string, tags: string[] = []): boolean {
  return getTopicSlugsForTags(tags).includes(topicSlug);
}
