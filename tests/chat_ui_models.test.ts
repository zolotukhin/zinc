import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

function loadModelUi(fetchImpl: (...args: any[]) => Promise<any>) {
  const html = readFileSync(new URL('../src/server/chat.html', import.meta.url), 'utf8');
  const script = html.match(/<script>([\s\S]*)<\/script>\s*<\/body>/)?.[1];
  if (!script) throw new Error('Chat UI script not found');
  const elements = new Map<string, any>();
  function element() {
    let markup = '';
    return {
      value: '',
      textContent: '',
      hidden: false,
      disabled: false,
      title: '',
      href: '',
      style: {},
      children: [] as any[],
      writes: 0,
      get innerHTML() {
        return markup;
      },
      set innerHTML(value: string) {
        markup = value;
        this.writes++;
        this.children = [];
      },
      classList: { toggle() {}, add() {}, remove() {} },
      addEventListener() {},
      focus() {},
      appendChild(child: any) {
        this.children.push(child);
      },
      querySelector(selector: string) {
        if (selector === '.model-picker')
          return markup.includes('class="model-picker"') ? {} : null;
        if (selector === '#w.empty .starters')
          return markup.includes('class="starters"') ? {} : null;
        return null;
      },
      querySelectorAll() {
        return [];
      },
    };
  }
  const document = {
    getElementById(id: string) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    createElement: element,
  };
  const delays: number[] = [];
  const factory = new Function(
    'document',
    'fetch',
    'location',
    'setTimeout',
    'clearTimeout',
    `
    ${script.replace(/init\(\);\s*setGenerating\(false\);\s*I\.focus\(\);/, '')}
    return { refreshModels, performModelAction, renderModelPicker, send, setGenerating,
      setModels(value) { models=value; modelsReady=true; }, setModelTag,
      setHistory(value) { H=value; }, readHistory() { return H; } };
  `,
  );
  const api = factory(
    document,
    fetchImpl,
    { origin: 'http://localhost:8080' },
    (_callback: any, delay: number) => {
      delays.push(delay);
      return delays.length;
    },
    () => {},
  );
  return { api, elements, delays };
}

const model = {
  id: 'qwen-test',
  display_name: 'Qwen test',
  managed: true,
  installed: true,
  supported_on_current_gpu: true,
  fits_current_gpu: true,
  active: false,
  required_vram_bytes: 8 * 1024 ** 3,
};

test('unchanged catalog refresh preserves model selection and picker DOM', async () => {
  const { api, elements } = loadModelUi(async () => ({
    ok: true,
    json: async () => ({ data: [model] }),
  }));
  await api.refreshModels();
  const select = elements.get('ms');
  const messages = elements.get('messages');
  const selectWrites = select.writes;
  const pickerWrites = messages.writes;
  await api.refreshModels();
  expect(select.value).toBe(model.id);
  expect(select.writes).toBe(selectWrites);
  expect(messages.writes).toBe(pickerWrites);
  expect(messages.innerHTML).toContain('<button type="button" class="model-card"');
});

test('catalog HTTP failures show a reconnect notice and retry', async () => {
  const { api, elements, delays } = loadModelUi(async () => ({ ok: false }));
  await api.refreshModels();
  expect(elements.get('ui-notice').textContent).toContain('Could not reach ZINC');
  expect(elements.get('ms').disabled).toBe(true);
  expect(delays).toContain(5000);
});

test('model activation errors stay inline and leave retry available', async () => {
  const { api, elements } = loadModelUi(async () => ({
    ok: false,
    json: async () => ({ error: { message: 'Not enough GPU memory' } }),
  }));
  api.setModels([model]);
  await api.performModelAction(model);
  expect(elements.get('ui-notice').textContent).toBe('Could not load model. Not enough GPU memory');
  expect(elements.get('mbtn').disabled).toBe(false);
});

test('model names are escaped in header and keyboard accessible picker', () => {
  const { api, elements } = loadModelUi(async () => ({}));
  api.setModels([{ ...model, id: 'model"test', display_name: '<img src=x onerror=alert(1)>' }]);
  api.renderModelPicker();
  api.setModelTag('<img src=x>');
  expect(elements.get('mn').innerHTML).toBe('&lt;img src=x&gt;');
  expect(elements.get('messages').innerHTML).toContain('data-model-id="model&quot;test"');
  expect(elements.get('messages').innerHTML).toContain('&lt;img src=x onerror=alert(1)&gt;');
});

test('pending model activation keeps generation and new chat locked', async () => {
  let complete!: (response: any) => void;
  let requests = 0;
  const { api, elements } = loadModelUi(async () => {
    requests++;
    return await new Promise((resolve) => {
      complete = resolve;
    });
  });
  api.setModels([model]);
  elements.get('i').value = 'Keep this draft';
  const activation = api.performModelAction(model);
  expect(elements.get('b').disabled).toBe(true);
  expect(elements.get('cb').disabled).toBe(true);
  await api.send();
  expect(requests).toBe(1);
  api.setGenerating(false);
  expect(elements.get('ms').disabled).toBe(true);
  expect(elements.get('cb').disabled).toBe(true);
  expect(elements.get('b').disabled).toBe(true);
  complete({ ok: false, json: async () => ({ error: { message: 'Try again' } }) });
  await activation;
  expect(elements.get('b').disabled).toBe(false);
  expect(elements.get('cb').disabled).toBe(false);
  expect(elements.get('i').value).toBe('Keep this draft');
});

test('active model replaces the startup placeholder with welcome prompts once', async () => {
  const { api, elements } = loadModelUi(async () => ({
    ok: true,
    json: async () => ({ data: [{ ...model, active: true }] }),
  }));
  const messages = elements.get('messages');
  messages.innerHTML = '<div class="empty" id="w"><h2>Getting things ready</h2></div>';
  await api.refreshModels();
  expect(messages.innerHTML).toContain('class="starters"');
  expect(messages.innerHTML).not.toContain('Getting things ready');
  const writes = messages.writes;
  await api.refreshModels();
  expect(messages.writes).toBe(writes);
});

test('activation awaiting catalog reconnect preserves conversation and draft', async () => {
  const { api, elements } = loadModelUi(async (_url, options) => ({
    ok: options?.method === 'POST',
    json: async () => ({}),
  }));
  const history = [{ role: 'user', content: 'Keep my existing conversation' }];
  api.setModels([{ ...model, id: 'previous', active: true }, model]);
  api.setHistory(history);
  elements.get('i').value = 'Keep my next draft';
  await api.performModelAction(model);
  expect(api.readHistory()).toEqual(history);
  expect(elements.get('i').value).toBe('Keep my next draft');
  expect(elements.get('stats').textContent).toBe('Waiting for model status');
  expect(elements.get('ui-notice').textContent).toContain('Reconnecting');
  expect(elements.get('ui-notice').hidden).toBe(false);
});
