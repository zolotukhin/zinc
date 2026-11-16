import { describe, expect, it } from 'bun:test';
import { getTopicSlugsForTags, postMatchesTopic } from './topic-posts';

describe('topic post matching', () => {
  it('connects RDNA4 and KV cache posts to both relevant hubs', () => {
    expect(getTopicSlugsForTags(['zinc', 'rdna4', 'kv-cache'])).toEqual([
      'amd-rdna4-llm-inference',
      'kv-cache-quantization',
    ]);
  });

  it('does not classify generic quantization as KV cache content', () => {
    expect(getTopicSlugsForTags(['quantization', 'q4-k'])).toEqual([]);
  });

  it('matches topic slugs case-insensitively', () => {
    expect(postMatchesTopic('qwen3-8-local-inference', ['QWEN3-8'])).toBe(true);
  });

  it('includes both published Qwen 3.8 tag spellings and excludes older model notes', () => {
    expect(postMatchesTopic('qwen3-8-local-inference', ['qwen3.8'])).toBe(true);
    expect(postMatchesTopic('qwen3-8-local-inference', ['qwen3-6'])).toBe(false);
  });
});
