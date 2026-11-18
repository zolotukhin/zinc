import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

function loadModelUi(fetchImpl: (...args: any[]) => Promise<any>) {
  const html = readFileSync(new URL("../src/server/chat.html", import.meta.url), "utf8");
  const script = html.match(/<script>([\s\S]*)<\/script>\s*<\/body>/)?.[1];
  if (!script) throw new Error("Chat UI script not found");
  const elements = new Map<string, any>();
  function element() {
    let markup = "";
    return {
      value: "", textContent: "", hidden: false, disabled: false, title: "", href: "",
      style: {}, children: [] as any[], writes: 0,
      listeners: new Map<string, Array<() => void>>(),
      get innerHTML() { return markup; },
      set innerHTML(value: string) { markup = value; this.writes++; this.children = []; },
      classList: { toggle() {}, add() {}, remove() {} },
      addEventListener(type: string, listener: () => void) {
        this.listeners.set(type, [...(this.listeners.get(type) || []), listener]);
      },
      dispatch(type: string) { for (const listener of this.listeners.get(type) || []) listener(); },
      focus() {},
      appendChild(child: any) { this.children.push(child); },
      querySelector(selector: string) {
        if (selector === ".model-picker") return markup.includes('class="model-picker"') ? {} : null;
        if (selector === "#w.empty .starters") return markup.includes('class="starters"') ? {} : null;
        return null;
      },
      querySelectorAll() { return []; },
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
  const storage = new Map<string, string>();
  let nextSession = 0;
  const factory = new Function("document", "fetch", "location", "setTimeout", "clearTimeout", "localStorage", "window", `
    ${script.slice(0, script.lastIndexOf("\ninit();"))}
    return { refreshModels, refreshHealth, performModelAction, unloadModel, renderModelPicker, send, setGenerating,
      setModelCapabilities, updateModelAction, activeModel, mtpRequestSetting, setRuntimeMtpCapability,
      setModels(value) { models=value; modelsReady=true; }, setModelTag,
      setHistory(value) { H=value; }, readHistory() { return H; },
      setSession(value) { chatSessionId=value; }, session() { return chatSessionId; } };
  `);
  const api = factory(document, fetchImpl, { origin: "http://localhost:8080" },
    (_callback: any, delay: number) => { delays.push(delay); return delays.length; }, () => {},
    { getItem: (key: string) => storage.get(key) || null, setItem: (key: string, value: string) => storage.set(key, value), removeItem: (key: string) => storage.delete(key) },
    { addEventListener() {}, crypto: { randomUUID: () => `chat-test-${++nextSession}` } });
  return { api, elements, delays, storage, getElement: document.getElementById };
}

const model = {
  id: "qwen-test", display_name: "Qwen test", managed: true, installed: true,
  supported_on_current_gpu: true, fits_current_gpu: true, active: false,
  required_vram_bytes: 8 * 1024 ** 3,
};

test("unchanged catalog refresh preserves model selection and picker DOM", async () => {
  const { api, elements } = loadModelUi(async () => ({ ok: true, json: async () => ({ data: [model] }) }));
  await api.refreshModels();
  const select = elements.get("ms");
  const messages = elements.get("messages");
  const selectWrites = select.writes;
  const pickerWrites = messages.writes;
  await api.refreshModels();
  expect(select.value).toBe(model.id);
  expect(select.writes).toBe(selectWrites);
  expect(messages.writes).toBe(pickerWrites);
  expect(messages.innerHTML).toContain('<button type="button" class="model-card"');
});

test("catalog HTTP failures show a reconnect notice and retry", async () => {
  const { api, elements, delays } = loadModelUi(async () => ({ ok: false }));
  await api.refreshModels();
  expect(elements.get("ui-notice").textContent).toContain("Could not reach ZINC");
  expect(elements.get("ms").disabled).toBe(true);
  expect(delays).toContain(5000);
});

test("model activation errors stay inline and leave retry available", async () => {
  const { api, elements } = loadModelUi(async () => ({
    ok: false, json: async () => ({ error: { message: "Not enough GPU memory" } }),
  }));
  api.setModels([model]);
  await api.performModelAction(model);
  expect(elements.get("ui-notice").textContent).toBe("Could not load model. Not enough GPU memory");
  expect(elements.get("mbtn").disabled).toBe(false);
});

test("model names are escaped in header and keyboard accessible picker", () => {
  const { api, elements } = loadModelUi(async () => ({}));
  api.setModels([{ ...model, id: 'model"test', display_name: '<img src=x onerror=alert(1)>' }]);
  api.renderModelPicker();
  api.setModelTag("<img src=x>");
  expect(elements.get("mn").innerHTML).toBe("&lt;img src=x&gt;");
  expect(elements.get("messages").innerHTML).toContain('data-model-id="model&quot;test"');
  expect(elements.get("messages").innerHTML).toContain("&lt;img src=x onerror=alert(1)&gt;");
});


test("pending model activation keeps generation and clear chat locked", async () => {
  let complete!: (response: any) => void;
  let requests = 0;
  const { api, elements } = loadModelUi(async (_url, options) => {
    if (!options?.method) return { ok: true, json: async () => ({ data: [model] }) };
    requests++;
    return await new Promise(resolve => { complete = resolve; });
  });
  api.setModels([model]);
  elements.get("i").value = "Keep this draft";
  const activation = api.performModelAction(model);
  expect(elements.get("b").disabled).toBe(true);
  expect(elements.get("cb").disabled).toBe(true);
  await api.send();
  expect(requests).toBe(1);
  api.setGenerating(false);
  expect(elements.get("ms").disabled).toBe(true);
  expect(elements.get("cb").disabled).toBe(true);
  expect(elements.get("b").disabled).toBe(true);
  complete({ ok: false, json: async () => ({ error: { message: "Try again" } }) });
  await activation;
  // An activation failure leaves the server unloaded; a draft must not silently
  // trigger model loading through the completions endpoint.
  expect(elements.get("b").disabled).toBe(true);
  expect(elements.get("cb").disabled).toBe(false);
  expect(elements.get("i").value).toBe("Keep this draft");
});

const lifecycleCapabilities = {
  supports_model_management: true, supports_model_unload: true, supports_model_pull: false,
};

test("Clear chat clears local history and the old session without unloading the model", async () => {
  const requests: Array<{url: string, body: any}> = [];
  const { api, elements, storage, getElement } = loadModelUi(async (url, options) => {
    requests.push({ url, body: JSON.parse(options.body) });
    return { ok: true };
  });
  api.setModels([{ ...model, active: true }]);
  api.setModelCapabilities(lifecycleCapabilities);
  api.setHistory([{ role: "user", content: "An old question" }]);
  api.setSession("previous-session");
  elements.get("i").value = "Old draft";
  getElement("mtp-last-status").textContent = "Previous reply used MTP";
  storage.set("zinc.chat.state.v3", JSON.stringify({ messages: api.readHistory(), input: "Old draft" }));
  elements.get("cb").dispatch("click");
  await Promise.resolve();
  expect(api.readHistory()).toEqual([]);
  expect(elements.get("i").value).toBe("");
  expect(api.session()).not.toBe("previous-session");
  const saved = JSON.parse(storage.get("zinc.chat.state.v3")!);
  expect(saved).toEqual({ messages: [], input: "", sessionId: api.session() });
  expect(elements.get("mtp-last-status").textContent).toBe("");
  expect(api.activeModel()?.id).toBe(model.id);
  expect(requests).toEqual([{ url: "http://localhost:8080/v1/chat/reset", body: { session_id: "previous-session" } }]);
  const html = readFileSync(new URL("../src/server/chat.html", import.meta.url), "utf8");
  expect(html).toContain('aria-label="Clear chat"');
});

test("Clear chat cannot erase the active conversation while generation is running", () => {
  let requests = 0;
  const { api, elements } = loadModelUi(async () => { requests++; return { ok: true }; });
  api.setModels([{ ...model, active: true }]);
  const history = [{ role: "user", content: "Current question" }];
  api.setHistory(history);
  elements.get("i").value = "Next draft";
  api.setSession("current-session");
  api.setGenerating(true);
  elements.get("cb").dispatch("click");
  expect(api.readHistory()).toEqual(history);
  expect(elements.get("i").value).toBe("Next draft");
  expect(api.session()).toBe("current-session");
  expect(requests).toBe(0);
});

test("unload keeps the conversation, draft and installed model while blocking sends until reload", async () => {
  const requests: Array<{url: string, body?: any}> = [];
  const { api, elements } = loadModelUi(async (url, options) => {
    requests.push({ url, body: options?.body ? JSON.parse(options.body) : undefined });
    return { ok: true, json: async () => ({ ...lifecycleCapabilities, data: [{ ...model, managed: false, active: false }] }) };
  });
  api.setModels([{ ...model, managed: false, active: true }]);
  api.setModelCapabilities(lifecycleCapabilities);
  const history = [{ role: "user", content: "Keep this conversation" }];
  api.setHistory(history);
  elements.get("i").value = "Keep my next question";
  api.updateModelAction();
  expect(elements.get("mbtn").textContent).toBe("Unload model");
  expect(elements.get("mbtn").hidden).toBe(false);
  await api.unloadModel();
  expect(requests).toEqual([
    { url: "http://localhost:8080/v1/models/unload", body: { model: model.id } },
    { url: "http://localhost:8080/v1/models", body: undefined },
  ]);
  expect(api.readHistory()).toEqual(history);
  expect(elements.get("i").value).toBe("Keep my next question");
  expect(api.activeModel()).toBe(null);
  expect(elements.get("mbtn").textContent).toBe("Load model");
  expect(elements.get("b").disabled).toBe(true);
  expect(elements.get("hint").textContent).toContain("Load a model");
  expect(elements.get("stats").textContent).toContain("files kept");
  await api.send();
  expect(requests.length).toBe(2);
});

test("a busy unload leaves the active model and conversation available for retry", async () => {
  let complete!: (response: any) => void;
  const { api, elements } = loadModelUi(async (_url, options) => {
    if (!options?.method) return { ok: true, json: async () => ({ ...lifecycleCapabilities, data: [{ ...model, active: true }] }) };
    return new Promise(resolve => { complete = resolve; });
  });
  api.setModels([{ ...model, active: true }]);
  api.setModelCapabilities(lifecycleCapabilities);
  const history = [{ role: "user", content: "Keep this conversation" }];
  api.setHistory(history);
  elements.get("i").value = "Keep this draft";
  const pending = api.unloadModel();
  expect(elements.get("cb").disabled).toBe(true);
  expect(elements.get("b").disabled).toBe(true);
  expect(elements.get("mbtn").disabled).toBe(true);
  complete({ ok: false, json: async () => ({ error: { message: "Model is busy" } }) });
  await pending;
  expect(api.activeModel()?.id).toBe(model.id);
  expect(api.readHistory()).toEqual(history);
  expect(elements.get("i").value).toBe("Keep this draft");
  expect(elements.get("ui-notice").textContent).toBe("Could not unload model. Model is busy");
  expect(elements.get("mbtn").textContent).toBe("Unload model");
  expect(elements.get("mbtn").disabled).toBe(false);
  expect(elements.get("b").disabled).toBe(false);
});

test("installed unmanaged models can switch on a lifecycle server and refresh MTP capability", async () => {
  const next = { ...model, id: "other-gguf", managed: false, supports_mtp: false };
  const previous = { ...model, id: "previous-gguf", managed: false, active: true, supports_mtp: true, mtp_request_control: true, mtp_default_enabled: true };
  const requests: Array<{url: string, body?: any}> = [];
  const { api, elements } = loadModelUi(async (url, options) => {
    requests.push({ url, body: options?.body ? JSON.parse(options.body) : undefined });
    return { ok: true, json: async () => ({ ...lifecycleCapabilities, data: [{ ...previous, active: false }, { ...next, active: true }] }) };
  });
  api.setModels([previous, next]);
  api.setModelCapabilities(lifecycleCapabilities);
  api.setRuntimeMtpCapability(previous);
  expect(api.mtpRequestSetting()).toBe(null);
  elements.get("ms").value = next.id;
  api.updateModelAction();
  expect(elements.get("mbtn").textContent).toBe("Load model");
  expect(elements.get("mbtn").disabled).toBe(false);
  await api.performModelAction(next);
  expect(requests[0]).toEqual({ url: "http://localhost:8080/v1/models/activate", body: { model: next.id } });
  expect(api.activeModel()?.id).toBe(next.id);
  expect(api.mtpRequestSetting()).toBeUndefined();
  expect(elements.get("mtp").disabled).toBe(true);
  expect(elements.get("mbtn").textContent).toBe("Unload model");
});

test("explicit fixed-model capability disables lifecycle controls despite managed catalog entries", async () => {
  let requests = 0;
  const { api, elements } = loadModelUi(async () => { requests++; return { ok: true }; });
  api.setModels([{ ...model, id: "active", active: true }, model]);
  api.setModelCapabilities({ supports_model_management: false, supports_model_unload: false, supports_model_pull: false });
  elements.get("ms").value = model.id;
  api.setGenerating(false);
  expect(elements.get("ms").disabled).toBe(true);
  expect(elements.get("mbtn").disabled).toBe(true);
  await api.performModelAction(model);
  await api.unloadModel();
  expect(requests).toBe(0);
});

test("a catalog poll started before unload cannot restore the evicted model", async () => {
  let finishOldPoll!: (response: any) => void;
  let catalogRequests = 0;
  const { api, elements } = loadModelUi(async (_url, options) => {
    if (options?.method === "POST") return { ok: true };
    catalogRequests++;
    if (catalogRequests === 1) return new Promise(resolve => { finishOldPoll = resolve; });
    return { ok: true, json: async () => ({ ...lifecycleCapabilities, data: [model] }) };
  });
  api.setModels([{ ...model, active: true }]);
  api.setModelCapabilities(lifecycleCapabilities);
  elements.get("i").value = "Keep my question";
  const oldPoll = api.refreshModels();
  await api.unloadModel();
  expect(api.activeModel()).toBe(null);
  finishOldPoll({ ok: true, json: async () => ({ ...lifecycleCapabilities, data: [{ ...model, active: true }] }) });
  await oldPoll;
  expect(api.activeModel()).toBe(null);
  expect(elements.get("b").disabled).toBe(true);
  expect(elements.get("mbtn").textContent).toBe("Load model");
});

test("a catalog poll started during a slow unload cannot restore the model after confirmation", async () => {
  let finishUnload!: (response: any) => void;
  let finishPollDuringUnload!: (response: any) => void;
  let catalogRequests = 0;
  const { api, elements } = loadModelUi(async (_url, options) => {
    if (options?.method === "POST") return new Promise(resolve => { finishUnload = resolve; });
    catalogRequests++;
    if (catalogRequests === 1) return new Promise(resolve => { finishPollDuringUnload = resolve; });
    return { ok: true, json: async () => ({ ...lifecycleCapabilities, data: [model] }) };
  });
  api.setModels([{ ...model, active: true }]);
  api.setModelCapabilities(lifecycleCapabilities);
  elements.get("i").value = "Keep my next question";
  const unload = api.unloadModel();
  const pollDuringUnload = api.refreshModels();
  finishUnload({ ok: true });
  await unload;
  expect(api.activeModel()).toBe(null);
  finishPollDuringUnload({ ok: true, json: async () => ({ ...lifecycleCapabilities, data: [{ ...model, active: true }] }) });
  await pollDuringUnload;
  expect(api.activeModel()).toBe(null);
  expect(elements.get("b").disabled).toBe(true);
  expect(elements.get("mbtn").textContent).toBe("Load model");
  expect(elements.get("i").value).toBe("Keep my next question");
});

test("a health poll started before unload cannot restore the old model name or MTP status", async () => {
  let finishOldHealth!: (response: any) => void;
  const previous = { ...model, active: true, supports_mtp: true, mtp_request_control: true, mtp_default_enabled: true };
  const { api, elements } = loadModelUi(async (url, options) => {
    if (url.endsWith("/health")) return new Promise(resolve => { finishOldHealth = resolve; });
    if (options?.method === "POST") return { ok: true };
    return { ok: true, json: async () => ({ ...lifecycleCapabilities, data: [model] }) };
  });
  api.setModels([previous]);
  api.setModelCapabilities(lifecycleCapabilities);
  const oldHealth = api.refreshHealth();
  await api.unloadModel();
  finishOldHealth({ ok: true, json: async () => ({ status: "ok", model: model.display_name, ...previous }) });
  await oldHealth;
  expect(elements.get("mn").innerHTML).toBe("No model loaded");
  expect(api.mtpRequestSetting()).toBeUndefined();
  expect(elements.get("mtp").disabled).toBe(true);
  expect(elements.get("mtp-control").title).toContain("Load a model");
});

test("an old health failure cannot mark a successfully unloaded server as reconnecting", async () => {
  let failOldHealth!: (error: Error) => void;
  const { api, elements } = loadModelUi(async (url, options) => {
    if (url.endsWith("/health")) return new Promise((_resolve, reject) => { failOldHealth = reject; });
    if (options?.method === "POST") return { ok: true };
    return { ok: true, json: async () => ({ ...lifecycleCapabilities, data: [model] }) };
  });
  api.setModels([{ ...model, active: true }]);
  api.setModelCapabilities(lifecycleCapabilities);
  elements.get("st").textContent = "Connected";
  const oldHealth = api.refreshHealth();
  await api.unloadModel();
  failOldHealth(new Error("Old connection closed"));
  await oldHealth;
  expect(elements.get("st").textContent).toBe("Connected");
  expect(api.activeModel()).toBe(null);
});

test("failed switch refreshes an unloaded server and preserves the existing conversation and draft", async () => {
  const previous = { ...model, id: "previous", active: true };
  const { api, elements } = loadModelUi(async (_url, options) => options?.method === "POST"
    ? { ok: false, json: async () => ({ error: { message: "Model loading failed" } }) }
    : { ok: true, json: async () => ({ ...lifecycleCapabilities, data: [{ ...previous, active: false }, model] }) });
  api.setModels([previous, model]);
  api.setModelCapabilities(lifecycleCapabilities);
  const history = [{ role: "user", content: "Keep my conversation after a load failure" }];
  api.setHistory(history);
  elements.get("i").value = "Keep my draft";
  elements.get("ms").value = model.id;
  await api.performModelAction(model);
  expect(api.activeModel()).toBe(null);
  expect(api.readHistory()).toEqual(history);
  expect(elements.get("i").value).toBe("Keep my draft");
  expect(elements.get("b").disabled).toBe(true);
  expect(elements.get("mbtn").textContent).toBe("Load model");
  expect(elements.get("mbtn").disabled).toBe(false);
  expect(elements.get("ui-notice").textContent).toBe("Could not load model. Model loading failed");
});


test("active model replaces the startup placeholder with welcome prompts once", async () => {
  const { api, elements } = loadModelUi(async () => ({
    ok: true, json: async () => ({ data: [{ ...model, active: true }] }),
  }));
  const messages = elements.get("messages");
  messages.innerHTML = '<div class="empty" id="w"><h2>Getting things ready</h2></div>';
  await api.refreshModels();
  expect(messages.innerHTML).toContain('class="starters"');
  expect(messages.innerHTML).not.toContain("Getting things ready");
  const writes = messages.writes;
  await api.refreshModels();
  expect(messages.writes).toBe(writes);
});


test("activation awaiting catalog reconnect preserves conversation and draft", async () => {
  const { api, elements } = loadModelUi(async (_url, options) => ({
    ok: options?.method === "POST", json: async () => ({}),
  }));
  const history = [{ role: "user", content: "Keep my existing conversation" }];
  api.setModels([{ ...model, id: "previous", active: true }, model]);
  api.setHistory(history);
  elements.get("i").value = "Keep my next draft";
  await api.performModelAction(model);
  expect(api.readHistory()).toEqual(history);
  expect(elements.get("i").value).toBe("Keep my next draft");
  expect(elements.get("stats").textContent).toBe("Waiting for model status");
  expect(elements.get("ui-notice").textContent).toContain("Reconnecting");
  expect(elements.get("ui-notice").hidden).toBe(false);
});

async function flushModelAction() {
  // The browser change handler starts the action without blocking the native
  // picker. Flush its fetch, JSON and catalog promises before reading the UI.
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

test("choosing an installed compatible model immediately activates it", async () => {
  const previous = { ...model, id: "previous", active: true };
  const next = { ...model, id: "next", managed: false, can_activate: true };
  const requests: Array<{ url: string, body?: any }> = [];
  const { api, elements } = loadModelUi(async (url, options) => {
    requests.push({ url, body: options?.body ? JSON.parse(options.body) : undefined });
    return { ok: true, json: async () => ({ ...lifecycleCapabilities, data: [{ ...previous, active: false }, { ...next, active: true }] }) };
  });
  api.setModels([previous, next]);
  api.setModelCapabilities(lifecycleCapabilities);
  elements.get("i").value = "Keep my next question";
  elements.get("ms").value = next.id;
  elements.get("ms").dispatch("change");
  await flushModelAction();
  expect(requests).toEqual([
    { url: "http://localhost:8080/v1/models/activate", body: { model: next.id } },
    { url: "http://localhost:8080/v1/models", body: undefined },
  ]);
  expect(api.activeModel()?.id).toBe(next.id);
  expect(elements.get("stats").textContent).toBe("Model ready");
  expect(elements.get("mbtn").textContent).toBe("Unload model");
  expect(elements.get("i").value).toBe("Keep my next question");
});

test("choosing a download candidate never starts a download automatically", async () => {
  let requests = 0;
  const { api, elements } = loadModelUi(async () => { requests++; return { ok: true }; });
  const next = { ...model, id: "not-installed", installed: false };
  api.setModels([{ ...model, active: true }, next]);
  api.setModelCapabilities({ ...lifecycleCapabilities, supports_model_pull: true });
  elements.get("ms").value = next.id;
  elements.get("ms").dispatch("change");
  await flushModelAction();
  expect(requests).toBe(0);
  expect(elements.get("mbtn").textContent).toBe("Download");
  expect(elements.get("mbtn").disabled).toBe(false);
});

test("unsupported picker entries cannot automatically replace the active model", async () => {
  for (const unsupported of [{ can_activate: false }, { supported_on_current_gpu: false }, { fits_current_gpu: false }]) {
    let requests = 0;
    const { api, elements } = loadModelUi(async () => { requests++; return { ok: true }; });
    const next = { ...model, id: "unsupported", ...unsupported };
    api.setModels([{ ...model, active: true }, next]);
    api.setModelCapabilities(lifecycleCapabilities);
    elements.get("ms").value = next.id;
    elements.get("ms").dispatch("change");
    await flushModelAction();
    expect(requests).toBe(0);
    expect(api.activeModel()?.id).toBe(model.id);
    expect(elements.get("mbtn").disabled).toBe(true);
  }
});

test("legacy fixed-model capability explains why the picker is disabled", async () => {
  let requests = 0;
  const { api, elements, getElement } = loadModelUi(async () => { requests++; return { ok: true }; });
  api.setModels([{ ...model, active: true }, { ...model, id: "another" }]);
  api.setModelCapabilities({ capabilities: { model_management: false, model_switching: false } });
  api.setGenerating(false);
  expect(elements.get("ms").disabled).toBe(true);
  expect(elements.get("ms").title).toContain("server update");
  expect(getElement("model-hint").textContent).toContain("server update");
  const html = readFileSync(new URL("../src/server/chat.html", import.meta.url), "utf8");
  expect(html).toContain('id="model-hint"');
  elements.get("ms").value = "another";
  elements.get("ms").dispatch("change");
  await flushModelAction();
  expect(requests).toBe(0);
});

test("explicit lifecycle capability takes precedence over older nested capability fields", () => {
  const { api, elements } = loadModelUi(async () => ({ ok: true }));
  api.setModels([{ ...model, active: true }, { ...model, id: "another" }]);
  api.setModelCapabilities({ ...lifecycleCapabilities, capabilities: { model_management: false, model_switching: false } });
  api.setGenerating(false);
  expect(elements.get("ms").disabled).toBe(false);
});

test("a singleton catalog explains why no other model can be chosen", () => {
  const { api, elements, getElement } = loadModelUi(async () => ({ ok: true }));
  api.setModels([{ ...model, active: true }]);
  api.setModelCapabilities(lifecycleCapabilities);
  api.setGenerating(false);
  expect(elements.get("ms").disabled).toBe(true);
  expect(elements.get("ms").title).toContain("Only one model");
  expect(getElement("model-hint").textContent).toContain("Only one model");
});

test("missing fit metadata does not disable an otherwise eligible model", () => {
  const { api, elements } = loadModelUi(async () => ({ ok: true }));
  const next: any = { ...model, managed: false, can_activate: true };
  delete next.fits_current_gpu;
  api.setModels([next]);
  api.setModelCapabilities(lifecycleCapabilities);
  api.updateModelAction();
  api.renderModelPicker();
  expect(elements.get("mbtn").textContent).toBe("Load model");
  expect(elements.get("mbtn").disabled).toBe(false);
  expect(elements.get("messages").innerHTML).toContain('class="model-card"');
  expect(elements.get("messages").innerHTML).not.toContain("Needs more GPU memory");
});

test("failed activation from a picker change preserves the unsent draft and conversation", async () => {
  const previous = { ...model, id: "previous", active: true };
  const next = { ...model, id: "next" };
  const { api, elements } = loadModelUi(async (_url, options) => options?.method === "POST"
    ? { ok: false, json: async () => ({ error: { message: "Model is busy" } }) }
    : { ok: true, json: async () => ({ ...lifecycleCapabilities, data: [previous, next] }) });
  api.setModels([previous, next]);
  api.setModelCapabilities(lifecycleCapabilities);
  const history = [{ role: "user", content: "Keep our existing conversation" }];
  api.setHistory(history);
  elements.get("i").value = "Do not lose this draft";
  elements.get("ms").value = next.id;
  elements.get("ms").dispatch("change");
  await flushModelAction();
  expect(api.activeModel()?.id).toBe(previous.id);
  expect(api.readHistory()).toEqual(history);
  expect(elements.get("i").value).toBe("Do not lose this draft");
  expect(elements.get("ui-notice").textContent).toBe("Could not load model. Model is busy");
  expect(elements.get("mbtn").disabled).toBe(false);
});
