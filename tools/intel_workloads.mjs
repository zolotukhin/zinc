import policy from '../config/release/targets.json' with { type: 'json' };
import {
  buildZincOpenAiPayload,
  defaultPromptForModelId,
  defaultScenarioDefsForModel,
} from './performance_suite.mjs';

export const intelModelIds = policy.gpu_profiles.find(
  (row) => row.id === 'vulkan-intel-xe2',
).models;

export function intelWorkloads(modelId) {
  if (!intelModelIds.includes(modelId)) throw new Error('Unknown Intel qualification model');
  return defaultScenarioDefsForModel(modelId, 'chat', defaultPromptForModelId(modelId)).map(
    (scenario) => ({
      ...scenario,
      body: { ...buildZincOpenAiPayload(scenario), top_p: 1, seed: 0, cache_prompt: false },
    }),
  );
}

if (import.meta.main) process.stdout.write(JSON.stringify(intelWorkloads(process.argv[2])));
