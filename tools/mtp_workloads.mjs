import {
  buildZincOpenAiPayload,
  defaultPromptForModelId,
  defaultScenarioDefsForModel,
} from './performance_suite.mjs';

export const mtpModelId = 'qwen38-27b-q4k-m';
export const mtpWorkloads = defaultScenarioDefsForModel(
  mtpModelId,
  'chat',
  defaultPromptForModelId(mtpModelId),
).map((scenario) => ({
  ...scenario,
  body: { ...buildZincOpenAiPayload(scenario), top_p: 1, seed: 0, cache_prompt: false },
}));

if (import.meta.main) process.stdout.write(JSON.stringify(mtpWorkloads));
