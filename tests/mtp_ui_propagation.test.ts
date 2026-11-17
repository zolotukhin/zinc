import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

class Element {
  value = '';
  className = '';
  disabled = false;
  hidden = false;
  checked = false;
  indeterminate = false;
  open = false;
  title = '';
  href = '';
  ariaLabel = '';
  scrollTop = 0;
  scrollHeight = 100;
  clientHeight = 100;
  style: Record<string, string> = {};
  dataset: Record<string, string> = {};
  parentNode: Element | null = null;
  children: Element[] = [];
  listeners = new Map<string, Array<(event: any) => void>>();
  private text = '';
  private html = '';
  classList = { add() {}, remove() {}, toggle() {} };
  get parentElement() {
    return this.parentNode;
  }
  get textContent() {
    return this.text;
  }
  set textContent(value: string) {
    this.text = value;
    this.children = [];
  }
  get innerHTML() {
    return this.html;
  }
  set innerHTML(value: string) {
    this.html = value;
    this.children = [];
    const names = value.includes('class="reasoning-summary"')
      ? ['reasoning-state', 'reasoning-body', 'stream-markdown']
      : value.includes('class="av"')
        ? ['av', 'bbl']
        : [];
    for (const name of names) {
      const child = new Element();
      child.className = name;
      this.appendChild(child);
    }
  }
  appendChild(child: Element) {
    child.remove();
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  insertBefore(child: Element, reference: Element) {
    child.remove();
    child.parentNode = this;
    this.children.splice(this.children.indexOf(reference), 0, child);
    return child;
  }
  remove() {
    if (this.parentNode)
      this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
    this.parentNode = null;
  }
  querySelector(selector: string): Element | null {
    for (const child of this.children) {
      if (child.className.split(' ').includes(selector.slice(1))) return child;
      const nested = child.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }
  querySelectorAll() {
    return [];
  }
  insertAdjacentHTML(_position: string, html: string) {
    this.html += html;
  }
  addEventListener(type: string, handler: (event: any) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) || []), handler]);
  }
  dispatch(type: string, event: any = {}) {
    for (const handler of this.listeners.get(type) || []) handler.call(this, event);
  }
  focus() {}
}

type MtpResult = {
  requested: boolean | null;
  enabled: boolean;
  used: boolean;
  drafted: number;
  accepted: number;
  cycles: number;
  fallback_reason: string | null;
};
const used: MtpResult = {
  requested: null,
  enabled: true,
  used: true,
  drafted: 6,
  accepted: 4,
  cycles: 3,
  fallback_reason: null,
};
const off: MtpResult = {
  requested: false,
  enabled: false,
  used: false,
  drafted: 0,
  accepted: 0,
  cycles: 0,
  fallback_reason: null,
};
const capable = {
  supports_mtp: true,
  mtp_default_enabled: true,
  mtp_request_control: true,
  mtp_unavailable_reason: '',
};
const model = {
  id: 'qwen38-27b-q4k-m',
  display_name: 'Qwen3.8 27B',
  active: true,
  installed: true,
  managed: false,
  supported_on_current_gpu: true,
  fits_current_gpu: true,
};

function stream(content: string, metadata?: any, finishReason = 'stop') {
  const packets = [
    { choices: [{ delta: { content }, finish_reason: null }] },
    { choices: [{ delta: {}, finish_reason: finishReason }] },
    ...(metadata === undefined ? [] : [{ choices: [], mtp: metadata }]),
  ];
  return new Response(
    packets.map((packet) => 'data: ' + JSON.stringify(packet) + '\n\n').join('') +
      'data: [DONE]\n\n',
  );
}

type Options = {
  capability?: Record<string, any>;
  healthCapability?: Record<string, any>;
  storage?: Map<string, string>;
  reply?: (body: any, index: number) => Promise<Response> | Response;
};

function harness(options: Options = {}) {
  const html = readFileSync(new URL('../src/server/chat.html', import.meta.url), 'utf8');
  const source = html
    .match(/<script>([\s\S]*)<\/script>\s*<\/body>/)![1]
    .replace(/init\(\);\s*setGenerating\(false\);\s*I\.focus\(\);/, '');
  const elements = new Map<string, Element>();
  const element = (id: string) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id)!;
  };
  const storage = options.storage || new Map<string, string>();
  const requests: any[] = [];
  const timers = new Map<number, () => void>();
  let timer = 0,
    clock = 0;
  const fetchMock = async (url: string, init: any = {}) => {
    if (url.endsWith('/models'))
      return Response.json({ data: [{ ...model, ...(options.capability || {}) }] });
    if (url.endsWith('/health'))
      return Response.json({
        status: 'ok',
        model: model.id,
        ...(options.healthCapability || options.capability || {}),
      });
    if (url.endsWith('/chat/reset')) return Response.json({});
    if (!url.endsWith('/chat/completions')) throw new Error('Unexpected endpoint ' + url);
    const body = JSON.parse(init.body);
    requests.push(body);
    return options.reply
      ? options.reply(body, requests.length - 1)
      : stream('A useful answer.', body.enable_mtp === false ? off : used);
  };
  const document = {
    getElementById: element,
    createElement: () => new Element(),
    querySelectorAll: () => [],
    activeElement: element('i'),
  };
  const factory = new Function(
    'document',
    'fetch',
    'location',
    'localStorage',
    'window',
    'performance',
    'navigator',
    'setTimeout',
    'clearTimeout',
    'requestAnimationFrame',
    'cancelAnimationFrame',
    `${source}
    return { send, refreshModels, refreshHealth, restoreChatState, updateMtpUi, setGenerating, mtpRequestSetting, normalizeMtpResult,
      history(){return H}, setModels(value){models=value;runtimeMtpCapability=null;updateMtpUi()} };
  `,
  );
  const schedule = (callback: () => void) => {
    const id = ++timer;
    timers.set(id, callback);
    return id;
  };
  const cancel = (id: number) => timers.delete(id);
  const api = factory(
    document,
    fetchMock,
    { origin: 'http://localhost:8080' },
    {
      getItem: (key: string) => storage.get(key) || null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
    },
    { addEventListener() {}, crypto: { randomUUID: () => 'mtp-ui-test' } },
    { now: () => (clock += 10) },
    { clipboard: { writeText: async () => {} } },
    schedule,
    cancel,
    schedule,
    cancel,
  );
  return {
    api,
    element,
    requests,
    storage,
    async discover() {
      await api.refreshModels();
    },
    async send(text = 'Describe your current execution path') {
      element('i').value = text;
      await api.send();
    },
    toggle(enabled: boolean) {
      element('mtp').checked = enabled;
      element('mtp').dispatch('change');
    },
  };
}

test('MTP engine default survives discovery and is sent as null until the user explicitly overrides it', async () => {
  const ui = harness({ capability: capable });
  await ui.discover();
  expect(ui.element('mtp').checked).toBe(true);
  expect(ui.element('mtp').disabled).toBe(false);
  expect(ui.api.mtpRequestSetting()).toBeNull();
  await ui.send();
  expect(ui.requests[0].enable_mtp).toBeNull();
  ui.toggle(false);
  await ui.send('Now use ordinary decoding');
  expect(ui.requests[1].enable_mtp).toBe(false);
  ui.toggle(true);
  await ui.send('Now use MTP again');
  expect(ui.requests[2].enable_mtp).toBe(true);
  expect(ui.requests.map((request) => request.enable_mtp)).toEqual([null, false, true]);
});

test('MTP default-off is a capability, not an unsupported model, and explicit opt-in reaches the request', async () => {
  const ui = harness({ capability: { ...capable, mtp_default_enabled: false } });
  await ui.discover();
  expect(ui.element('mtp').checked).toBe(false);
  expect(ui.element('mtp').disabled).toBe(false);
  await ui.send();
  expect(ui.requests[0].enable_mtp).toBeNull();
  ui.toggle(true);
  await ui.send();
  expect(ui.requests[1].enable_mtp).toBe(true);
});

test('explicit MTP preference persists across reload and Default restores automatic policy', async () => {
  const storage = new Map<string, string>();
  const first = harness({ capability: capable, storage });
  await first.discover();
  first.toggle(false);
  const reloaded = harness({ capability: capable, storage });
  await reloaded.discover();
  expect(reloaded.element('mtp').checked).toBe(false);
  await reloaded.send();
  expect(reloaded.requests[0].enable_mtp).toBe(false);
  reloaded.element('mtp-reset').dispatch('click');
  expect(reloaded.element('mtp').checked).toBe(true);
  await reloaded.send();
  expect(reloaded.requests[1].enable_mtp).toBeNull();
  expect(JSON.parse(storage.get('zinc.chat.prefs.v1')!).mtpPreference).toBe('auto');
});

test('old false preferences migrate to automatic so they cannot silently disable an MTP-on server', async () => {
  const storage = new Map([
    ['zinc.chat.prefs.v1', JSON.stringify({ thinkingEnabled: true, mtpEnabled: false })],
  ]);
  const ui = harness({ capability: capable, storage });
  await ui.discover();
  await ui.send();
  expect(ui.requests[0].enable_mtp).toBeNull();
  expect(ui.element('mtp').checked).toBe(true);
});

test('unsupported, unreported, and engine-managed MTP never send ineffective request overrides', async () => {
  for (const capability of [
    {},
    {
      supports_mtp: false,
      mtp_request_control: false,
      mtp_unavailable_reason: 'This model lacks NextN weights',
    },
    { supports_mtp: true, mtp_default_enabled: true, mtp_request_control: false },
  ]) {
    const ui = harness({ capability, reply: () => stream('A useful answer.') });
    await ui.discover();
    ui.toggle(true);
    expect(ui.element('mtp').disabled).toBe(true);
    await ui.send();
    expect(ui.requests[0]).not.toHaveProperty('enable_mtp');
    expect(ui.element('mtp-last-status').textContent).not.toContain('MTP used');
  }
});

test('health capability overrides stale model capability and updates request eligibility', async () => {
  const ui = harness({
    capability: capable,
    healthCapability: {
      supports_mtp: false,
      mtp_request_control: false,
      mtp_unavailable_reason: 'MTP requires a single request slot',
    },
  });
  await ui.discover();
  expect(ui.element('mtp').disabled).toBe(false);
  await ui.api.refreshHealth();
  expect(ui.element('mtp').disabled).toBe(true);
  expect(ui.element('mtp-control').title).toContain('single request slot');
  await ui.send();
  expect(ui.requests[0]).not.toHaveProperty('enable_mtp');
});

test('health-first discovery retains capability when an older model list omits it', async () => {
  const ui = harness({ capability: {}, healthCapability: capable });
  await ui.api.refreshHealth();
  await ui.discover();
  expect(ui.element('mtp').disabled).toBe(false);
  expect(ui.element('mtp').checked).toBe(true);
  await ui.send();
  expect(ui.requests[0].enable_mtp).toBeNull();
});

test('terminal SSE execution metadata confirms actual MTP use and persists with the assistant reply', async () => {
  const ui = harness({ capability: capable, reply: () => stream('The verified answer.', used) });
  await ui.discover();
  await ui.send();
  expect(ui.element('mtp-last-status').textContent).toBe(
    'Last reply: MTP used · 4/6 drafts accepted',
  );
  expect(ui.api.history().at(-1).mtp).toEqual(used);
  const restored = harness({ capability: capable, storage: ui.storage });
  restored.api.restoreChatState();
  expect(restored.api.history().at(-1).mtp).toEqual(used);
});

test('inline model stop markers do not discard final MTP execution metadata', async () => {
  for (const chunks of [
    ['The answer is **ready**.<|im_end|>'],
    ['The answer is **ready**.<|im_', 'end|>'],
  ]) {
    const packets = [
      ...chunks.map((content) => ({ choices: [{ delta: { content }, finish_reason: null }] })),
      { choices: [{ delta: {}, finish_reason: 'stop' }], mtp: used },
    ];
    const response = () =>
      new Response(
        packets.map((packet) => 'data: ' + JSON.stringify(packet) + '\n\n').join('') +
          'data: [DONE]\n\n',
      );
    const ui = harness({ capability: capable, reply: response });
    await ui.discover();
    await ui.send();
    expect(ui.element('mtp-last-status').textContent).toContain('MTP used');
    expect(ui.api.history().at(-1).mtp).toEqual(used);
    expect(ui.api.history().at(-1).content).toBe('The answer is **ready**.');
    expect(ui.api.history().at(-1).incomplete).not.toBe(true);
  }
});

test('MTP preparation fallback is shown as ordinary decoding despite an enabled preference', async () => {
  const fallback = {
    ...used,
    used: false,
    drafted: 0,
    accepted: 0,
    cycles: 0,
    fallback_reason: 'Draft priming failed; used ordinary decoding.',
  };
  const ui = harness({ capability: capable, reply: () => stream('An ordinary answer.', fallback) });
  await ui.discover();
  await ui.send();
  expect(ui.element('mtp').checked).toBe(true);
  expect(ui.element('mtp-last-status').textContent).toContain('MTP fallback');
  expect(ui.element('mtp-last-status').textContent).toContain('ordinary decoding');
  expect(ui.api.history().at(-1).mtp.used).toBe(false);
});

test('an enabled one-token reply reports that no draft cycle ran', async () => {
  const no_cycle = { ...used, used: false, drafted: 0, accepted: 0, cycles: 0 };
  const ui = harness({ capability: capable, reply: () => stream('Yes.', no_cycle) });
  await ui.discover();
  await ui.send();
  expect(ui.element('mtp-last-status').textContent).toBe(
    'Last reply: MTP enabled · no draft cycle needed',
  );
  expect(ui.api.history().at(-1).mtp.used).toBe(false);
});

test('actual execution proof can be shown even when an older model endpoint omits capability metadata', async () => {
  const ui = harness({ capability: {}, reply: () => stream('The verified answer.', used) });
  await ui.discover();
  await ui.send();
  expect(ui.requests[0]).not.toHaveProperty('enable_mtp');
  expect(ui.element('mtp-label').textContent).toBe('MTP status unreported');
  expect(ui.element('mtp-last-status').textContent).toContain('MTP used');
});

test('opposite MTP choices from concurrent UI clients remain attached to their own replies', async () => {
  let complete_on!: (response: Response) => void;
  let complete_off!: (response: Response) => void;
  const on_ui = harness({
    capability: capable,
    reply: () =>
      new Promise((resolve) => {
        complete_on = resolve;
      }),
  });
  const off_ui = harness({
    capability: capable,
    reply: () =>
      new Promise((resolve) => {
        complete_off = resolve;
      }),
  });
  await on_ui.discover();
  await off_ui.discover();
  on_ui.toggle(true);
  off_ui.toggle(false);
  const on_turn = on_ui.send('Use MTP for this request');
  const off_turn = off_ui.send('Use ordinary decoding for this request');
  expect(on_ui.requests[0].enable_mtp).toBe(true);
  expect(off_ui.requests[0].enable_mtp).toBe(false);
  complete_off(stream('Ordinary reply.', off));
  await off_turn;
  complete_on(stream('Speculative reply.', { ...used, requested: true }));
  await on_turn;
  expect(off_ui.api.history().at(-1).mtp.used).toBe(false);
  expect(on_ui.api.history().at(-1).mtp.used).toBe(true);
  expect(off_ui.element('mtp-last-status').textContent).toBe('Last reply: MTP off');
  expect(on_ui.element('mtp-last-status').textContent).toContain('MTP used');
});

test("missing or invalid execution metadata cannot retain a previous reply's MTP-used claim", async () => {
  const ui = harness({
    capability: capable,
    reply: (_body, index) =>
      stream(
        'An answer.',
        index === 0 ? used : index === 1 ? undefined : { ...used, accepted: 99 },
      ),
  });
  await ui.discover();
  await ui.send();
  expect(ui.element('mtp-last-status').textContent).toContain('MTP used');
  await ui.send();
  expect(ui.element('mtp-last-status').textContent).not.toContain('MTP used');
  expect(ui.api.history().at(-1).mtp).toBeUndefined();
  await ui.send();
  expect(ui.element('mtp-last-status').textContent).not.toContain('MTP used');
  expect(ui.api.history().at(-1).mtp).toBeUndefined();
});

test("non-streaming answer recovery preserves the request's latched MTP choice and reads actual result", async () => {
  let ui: ReturnType<typeof harness>;
  ui = harness({
    capability: { ...capable, supports_thinking_toggle: true },
    reply: (body, index) => {
      if (index === 0) {
        // Simulate a preference change while the first request is already running.
        ui.toggle(true);
        return stream(
          '<think>I need to inspect both matrices.\nThen I should verify their dimensions.',
          undefined,
          'length',
        );
      }
      expect(body.stream).toBe(false);
      return Response.json({
        choices: [{ message: { content: 'The recovered answer.' }, finish_reason: 'stop' }],
        mtp: off,
      });
    },
  });
  await ui.discover();
  ui.toggle(false);
  await ui.send();
  expect(ui.requests).toHaveLength(2);
  expect(ui.requests[0].enable_mtp).toBe(false);
  expect(ui.requests[1].enable_mtp).toBe(false);
  expect(ui.api.history().at(-1).mtp).toEqual(off);
  expect(ui.element('mtp-last-status').textContent).toBe('Last reply: MTP off');
});

test('SSE errors before choices surface an error rather than a successful MTP reply', async () => {
  const ui = harness({
    capability: capable,
    reply: () =>
      new Response('data: {"error":{"message":"MTP verification failed"}}\n\ndata: [DONE]\n\n'),
  });
  await ui.discover();
  await ui.send();
  expect(ui.element('stats').textContent).toBe('request failed');
  expect(ui.element('mtp-last-status').textContent).not.toContain('MTP used');
  expect(ui.api.history().filter((message: any) => message.role === 'assistant')).toHaveLength(0);
});
