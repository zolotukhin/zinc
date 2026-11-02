import {
  buildZincOpenAiPayload,
  defaultPromptForModelId,
  defaultScenarioDefsForModel,
} from './performance_suite.mjs';

export const gemmaModelId = 'gemma4-26b-a4b-q4k-m';
export const gemmaWorkloads = defaultScenarioDefsForModel(
  gemmaModelId,
  'chat',
  defaultPromptForModelId(gemmaModelId),
).map((scenario) => ({
  ...scenario,
  body: { ...buildZincOpenAiPayload(scenario), top_p: 1, seed: 0, cache_prompt: false },
}));

if (import.meta.main) process.stdout.write(JSON.stringify(gemmaWorkloads));
