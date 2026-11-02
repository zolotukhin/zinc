import {
  buildZincOpenAiPayload,
  defaultPromptForModelId,
  defaultScenarioDefsForModel,
} from './performance_suite.mjs';

export const gemmaModelId = 'gemma4-26b-a4b-q4k-m';
export function gemmaWorkloadsForModel(modelId = gemmaModelId) {
  if (![gemmaModelId, 'gemma4-31b-q4k-m'].includes(modelId)) throw new Error('Unknown Gemma model');
  return defaultScenarioDefsForModel(modelId, 'chat', defaultPromptForModelId(modelId)).map(
    (scenario) => ({
      ...scenario,
      body: { ...buildZincOpenAiPayload(scenario), top_p: 1, seed: 0, cache_prompt: false },
    }),
  );
}
export const gemmaWorkloads = gemmaWorkloadsForModel();

if (import.meta.main) process.stdout.write(JSON.stringify(gemmaWorkloadsForModel(process.argv[2])));
