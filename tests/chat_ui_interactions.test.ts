import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

class Element {
  value = "";
  className = "";
  disabled = false;
  hidden = false;
  open = false;
  checked = false;
  indeterminate = false;
  title = "";
  ariaLabel = "";
  scrollTop = 0;
  scrollHeight = 100;
  clientHeight = 100;
  style: Record<string, string> = {};
  parentNode: Element | null = null;
  children: Element[] = [];
  listeners = new Map<string, Array<(event: any) => void>>();
  htmlWrites = 0;
  private text = "";
  private html = "";
  get parentElement() { return this.parentNode; }
  classList = {
    toggle() {},
    add() {},
    remove() {},
  };
  get textContent() { return this.text; }
  set textContent(value: string) { this.text = value; this.children = []; }
  get innerHTML() { return this.html; }
  set innerHTML(value: string) {
    this.htmlWrites++;
    this.html = value;
    this.children = [];
    if (this.className !== "reasoning-block" && value.includes('class="reasoning-block"')) {
      const child = new Element();
      child.className = "reasoning-block";
      child.open = /<details\b[^>]*\bopen\b/.test(value);
      child.innerHTML = value.slice(value.indexOf("<summary>"), value.indexOf("</details>"));
      this.appendChild(child);
    } else if (value.includes('class="reasoning-summary"')) {
      for (const name of ["reasoning-state", "reasoning-body", "stream-markdown"]) {
        const child = new Element();
        child.className = name;
        this.appendChild(child);
      }
    } else if (value.includes('class="av"')) {
      for (const name of ["av", "bbl"]) {
        const child = new Element();
        child.className = name;
        this.appendChild(child);
      }
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
    if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(child => child !== this);
    this.parentNode = null;
  }
  replaceWith(replacement: Element) {
    const parent = this.parentNode;
    if (parent) { parent.insertBefore(replacement, this); this.remove(); }
  }
  querySelector(selector: string): Element | null {
    for (const child of this.children) {
      if (child.className.split(" ").includes(selector.slice(1))) return child;
      const nested = child.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }
  querySelectorAll() { return []; }
  addEventListener(type: string, listener: (event: any) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) || []), listener]);
  }
  dispatch(type: string, event: any = {}) {
    for (const listener of this.listeners.get(type) || []) listener.call(this, event);
  }
  focus() {}
  insertAdjacentHTML(_position: string, html: string) { this.html += html; }
}

function harness(fetchMock: (url: string, options?: any) => Promise<any> = async () => {throw new Error("offline")}, storage = new Map<string, string>()) {
  const source = readFileSync(resolve(import.meta.dir, "../src/server/chat.html"), "utf8")
    .match(/<script>([\s\S]*)<\/script>\s*<\/body>/)![1]
    .replace("init();setGenerating(false);I.focus();", "");
  const elements = new Map<string, Element>();
  const getElement = (id: string) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id)!;
  };
  const timers = new Map<number, () => void>();
  const frames = new Map<number, () => void>();
  let nextTimer = 1;
  let writes = 0;
  const factory = new Function("document", "location", "localStorage", "window", "setTimeout", "clearTimeout", "requestAnimationFrame", "cancelAnimationFrame", "fetch", `${source}
    return {md, send, stopGeneration, setGenerating, updateComposerState, updateMtpUi, mtpEnabledForRequest, updateLiveAssistant, completeAssistantHtml, assistantHtmlFromRaw, restoreChatState, updateStreamMarkdown, markdownCommitBoundary, scheduleRender, flushPendingRender, scheduleChatStateSave, saveChatState, autoScroll,
      replaceSend(fn){send=fn}, replaceStop(fn){stopGeneration=fn}, setHistory(value){H=value}, setModels(value){models=value}, generating(){return isGenerating}, history(){return H}};`);
  const api = factory(
    {getElementById: getElement, createElement: () => new Element(), querySelectorAll: () => []},
    {origin: "http://localhost:8080"},
    {getItem: (key: string) => storage.get(key) || null, setItem: (key: string, value: string) => {writes++;storage.set(key, value);}, removeItem: (key: string) => storage.delete(key)},
    {addEventListener() {}, crypto: {randomUUID: () => "chat-test"}},
    (fn: () => void) => { const id = nextTimer++; timers.set(id, fn); return id; },
    (id: number) => timers.delete(id),
    (fn: () => void) => { const id = nextTimer++; frames.set(id, fn); return id; },
    (id: number) => frames.delete(id),
    fetchMock,
  );
  return {api, getElement, timers, frames, storage, writes: () => writes};
}

test("Enter preserves the next draft during generation and ignores IME composition", () => {
  const {api, getElement} = harness();
  const input = getElement("i");
  let sent = 0, stopped = 0, prevented = 0;
  api.replaceSend(() => sent++);
  api.replaceStop(() => stopped++);
  const key = (key: string, extra = {}) => ({key, preventDefault: () => prevented++, ...extra});
  input.value = "Hello";
  api.setGenerating(false);
  input.dispatch("keydown", key("Enter", {isComposing: true}));
  input.dispatch("keydown", key("Enter", {keyCode: 229}));
  expect(sent).toBe(0);
  input.dispatch("keydown", key("Enter"));
  expect(sent).toBe(1);
  expect(prevented).toBe(1);
  api.setGenerating(true);
  input.dispatch("keydown", key("Enter"));
  expect(sent).toBe(1);
  expect(stopped).toBe(0);
  expect(prevented).toBe(1);
  input.dispatch("keydown", key("Escape"));
  expect(stopped).toBe(1);
});

test("send readiness and accessible action reflect the composer state", () => {
  const {api, getElement} = harness();
  const input = getElement("i"), button = getElement("b");
  api.setGenerating(false);
  expect(button.disabled).toBe(true);
  input.value = "  ";
  input.dispatch("input");
  expect(button.disabled).toBe(true);
  input.value = "Hello";
  input.dispatch("input");
  expect(button.disabled).toBe(false);
  expect(button.ariaLabel).toBe("Send message");
  input.value = "";
  api.setGenerating(true);
  expect(button.disabled).toBe(false);
  expect(button.ariaLabel).toBe("Stop generating");
});

test("typing coalesces draft persistence and final saves cancel pending writes", () => {
  const {api, getElement, timers, writes} = harness();
  const input = getElement("i");
  input.value = "H";
  input.dispatch("input");
  input.value = "Hello";
  input.dispatch("input");
  expect(writes()).toBe(0);
  expect(timers.size).toBe(1);
  api.saveChatState();
  expect(writes()).toBe(1);
  expect(timers.size).toBe(0);
});

test("stream renders coalesce to one animation frame and flush uses the newest update", () => {
  const {api, frames} = harness();
  const rendered: number[] = [];
  api.scheduleRender(() => rendered.push(1));
  api.scheduleRender(() => rendered.push(2));
  expect(frames.size).toBe(1);
  expect(rendered).toEqual([]);
  api.flushPendingRender();
  expect(rendered).toEqual([2]);
  expect(frames.size).toBe(0);
});

test("live Markdown retains containers and the reader's reasoning disclosure choice", () => {
  const {api} = harness();
  const bubble = new Element(), cursor = new Element();
  api.updateLiveAssistant(bubble, "<think>Compare the first result", true, cursor);
  const details = bubble.querySelector(".reasoning-block")!;
  const text = details.querySelector(".stream-markdown")!;
  expect(details.open).toBe(true);
  details.open = false;
  api.updateLiveAssistant(bubble, "<think>Compare the first result and check its dimensions", true, cursor);
  expect(bubble.querySelector(".reasoning-block")).toBe(details);
  expect(details.querySelector(".stream-markdown")).toBe(text);
  expect(text.querySelector(".markdown-tail")!.innerHTML).toContain("check its dimensions");
  expect(details.open).toBe(false);
  api.updateLiveAssistant(bubble, "<think>Compare the first result and check its dimensions</think>\nThe result is correct.", false, cursor);
  expect(details.open).toBe(false);
  const answer = bubble.children.find(child => child.className === "stream-markdown")!;
  expect(answer.querySelector(".markdown-tail")!.innerHTML).toBe("<p>The result is correct.</p>");
  api.updateLiveAssistant(bubble, "<think>Compare the first result and check its dimensions</think>\nThe result is correct. Here is why.", false, cursor);
  expect(bubble.children.find(child => child.className === "stream-markdown")).toBe(answer);
  expect(bubble.htmlWrites).toBe(0);
});

test("finishing a reply retains the reasoning disclosure and the reader's scroll position", () => {
  const {api} = harness();
  const bubble = new Element();
  const reasoning = "I need to compare the first result and then verify the dimensions.";
  api.updateLiveAssistant(bubble, "<think>" + reasoning, true);
  const details = bubble.querySelector(".reasoning-block")!;
  const body = details.querySelector(".reasoning-body")!;
  details.open = false;
  body.scrollHeight = 1000;
  body.clientHeight = 100;
  body.scrollTop = 120;
  const raw = "<think>" + reasoning + "</think>\nThe dimensions match.";
  api.updateLiveAssistant(bubble, raw, false);
  expect(bubble.querySelector(".reasoning-block")).toBe(details);
  expect(body.scrollTop).toBe(120);
  api.completeAssistantHtml(bubble, api.assistantHtmlFromRaw(raw, false).html, false);
  expect(bubble.querySelector(".reasoning-block")).toBe(details);
  expect(details.open).toBe(false);
  expect(details.querySelector(".reasoning-body")).toBe(body);
  expect(body.scrollTop).toBe(120);
  expect(bubble.htmlWrites).toBe(0);
});

test("the closing think token keeps all streamed planning paragraphs in the same disclosure", () => {
  const {api} = harness();
  const bubble = new Element();
  const reasoning = "Check the allocation sizes and compare each pointer offset against the required alignment before deciding.\n\nThe user is asking whether the memory layout is safe.\n\nI need to verify the final offset and explain the result.";
  const live = api.updateLiveAssistant(bubble, "<think>" + reasoning, true);
  const details = bubble.querySelector(".reasoning-block")!;
  details.open = false;
  expect(live.reasoning).toBe(reasoning);
  const completed = api.updateLiveAssistant(bubble, "<think>" + reasoning + "</think>\nThe layout is aligned correctly.", false);
  expect(completed.reasoning).toBe(reasoning);
  expect(completed.answer).toBe("The layout is aligned correctly.");
  expect(bubble.querySelector(".reasoning-block")).toBe(details);
  expect(details.open).toBe(false);
});

test("short reasoning survives reply completion, persisted history, and restore", async () => {
  const reasoning = "I need to answer directly.";
  const raw = "<think>" + reasoning + "</think>\nThe result is 408.";
  const storage = new Map<string, string>();
  const ui = harness(async () => new Response(
    'data: ' + JSON.stringify({choices: [{delta: {content: raw}, finish_reason: null}]}) + '\n\n' +
    'data: ' + JSON.stringify({choices: [{delta: {}, finish_reason: "stop"}]}) + '\n\ndata: [DONE]\n\n'
  ), storage);
  ui.api.setModels([{id: "test-model", active: true, supports_thinking_toggle: true}]);
  ui.getElement("i").value = "Calculate the result";
  await ui.api.send();
  const assistant = ui.api.history().find((message: any) => message.role === "assistant");
  expect(assistant.content).toBe("The result is 408.");
  expect(assistant.raw).toBe(raw);
  const completedBubble = ui.getElement("messages").children[1].querySelector(".bbl")!;
  const completedDetails = completedBubble.querySelector(".reasoning-block")!;
  completedDetails.open = true;
  completedDetails.dispatch("toggle");
  ui.api.saveChatState();
  expect(assistant.reasoning_open).toBe(true);
  const stored = JSON.parse(storage.get("zinc.chat.state.v3")!);
  expect(stored.messages.find((message: any) => message.role === "assistant").raw).toBe(raw);
  const restored = harness(undefined, storage);
  restored.api.restoreChatState();
  expect(restored.api.history().find((message: any) => message.role === "assistant").raw).toBe(raw);
  const rendered = restored.getElement("messages").children.map(message => message.querySelector(".bbl")?.innerHTML || "").join("");
  expect(rendered).toContain('class="reasoning-block"');
  expect(rendered).toContain(reasoning);
  expect(rendered).toContain("The result is 408.");
  const restoredBubble = restored.getElement("messages").children[1].querySelector(".bbl")!;
  expect(restoredBubble.querySelector(".reasoning-block")!.open).toBe(true);
});

test("a reader's collapsed reasoning choice survives final rendering and restore", async () => {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const raw = "<think>Check the first value, then compare the dimensions.";
  const storage = new Map<string, string>();
  const ui = harness(async () => new Response(new ReadableStream<Uint8Array>({start(value) {
    controller = value;
    value.enqueue(new TextEncoder().encode('data: ' + JSON.stringify({choices: [{delta: {content: raw}}]}) + '\n\n'));
  }})), storage);
  ui.api.setModels([{id: "test-model", active: true, supports_thinking_toggle: true}]);
  ui.getElement("i").value = "Compare the values";
  const turn = ui.api.send();
  for (let step = 0; step < 30 && ui.frames.size === 0; step++) await Promise.resolve();
  expect(ui.frames.size).toBe(1);
  ui.api.flushPendingRender();
  const bubble = ui.getElement("messages").children[1].querySelector(".bbl")!;
  const details = bubble.querySelector(".reasoning-block")!;
  expect(details.open).toBe(true);
  details.open = false;
  details.dispatch("toggle");
  controller.enqueue(new TextEncoder().encode('data: ' + JSON.stringify({choices: [{delta: {content: "</think>\nThe values match."}, finish_reason: "stop"}]}) + '\n\ndata: [DONE]\n\n'));
  controller.close();
  await turn;
  expect(bubble.querySelector(".reasoning-block")).toBe(details);
  expect(details.open).toBe(false);
  expect(ui.api.history().find((message: any) => message.role === "assistant").reasoning_open).toBe(false);
  const restored = harness(undefined, storage);
  restored.api.restoreChatState();
  const restoredBubble = restored.getElement("messages").children[1].querySelector(".bbl")!;
  expect(restoredBubble.querySelector(".reasoning-block")!.open).toBe(false);
});

test("plain Thinking Process reasoning survives completion and restored history", async () => {
  const raw = "Thinking Process:\n\n1. Compare the first value.\n2. Verify the dimensions.";
  const storage = new Map<string, string>();
  const ui = harness(async () => new Response(
    'data: ' + JSON.stringify({choices: [{delta: {content: raw}, finish_reason: null}]}) + '\n\n' +
    'data: ' + JSON.stringify({choices: [{delta: {}, finish_reason: "stop"}]}) + '\n\ndata: [DONE]\n\n'
  ), storage);
  ui.api.setModels([{id: "test-model", active: true, supports_thinking_toggle: true}]);
  ui.getElement("i").value = "Compare these values";
  await ui.api.send();
  const assistant = ui.api.history().find((message: any) => message.role === "assistant");
  expect(assistant.raw).toBe(raw);
  expect(assistant.content).toBe(raw);
  const restored = harness(undefined, storage);
  restored.api.restoreChatState();
  const bubble = restored.getElement("messages").children[1].querySelector(".bbl")!;
  expect(bubble.querySelector(".reasoning-block")).not.toBeNull();
  expect(bubble.innerHTML).toContain("Compare the first value.");
});

test("stopping before the final answer leaves streamed reasoning inspectable", async () => {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const reasoning = "I need to inspect the dimensions before selecting the operation.";
  const ui = harness(async (_url, options) => new Response(new ReadableStream<Uint8Array>({start(value) {
    controller = value;
    value.enqueue(new TextEncoder().encode('data: ' + JSON.stringify({choices: [{delta: {content: "<think>" + reasoning}}]}) + '\n\n'));
    options.signal.addEventListener("abort", () => controller.error(new DOMException("Stopped", "AbortError")));
  }})));
  ui.api.setModels([{id: "test-model", active: true, supports_thinking_toggle: true}]);
  ui.getElement("i").value = "Compare the dimensions";
  const turn = ui.api.send();
  for (let step = 0; step < 30 && ui.frames.size === 0; step++) await Promise.resolve();
  expect(ui.frames.size).toBe(1);
  ui.api.flushPendingRender();
  const bubble = ui.getElement("messages").children[1].querySelector(".bbl")!;
  const details = bubble.querySelector(".reasoning-block")!;
  ui.api.stopGeneration();
  await turn;
  expect(bubble.querySelector(".reasoning-block")).toBe(details);
  expect(details.querySelector(".markdown-tail")!.innerHTML).toContain(reasoning);
  const assistant = ui.api.history().find((message: any) => message.role === "assistant");
  expect(assistant.raw).toContain(reasoning);
  expect(assistant.incomplete).toBe(true);
});

test("live Markdown formats headings and emphasis before completion", () => {
  const {api} = harness();
  const bubble = new Element();
  api.updateLiveAssistant(bubble, "# Zig Programming Language\n\nZig is **simple, fast, safe**.", false);
  const answer = bubble.querySelector(".stream-markdown")!;
  expect(answer.querySelector(".markdown-committed")!.innerHTML).toContain("<h1>Zig Programming Language</h1>");
  expect(answer.querySelector(".markdown-tail")!.innerHTML).toContain("<strong>simple, fast, safe</strong>");
});

test("live Markdown leaves completed blocks unchanged while the formatted tail grows", () => {
  const {api} = harness();
  const element = new Element();
  api.updateStreamMarkdown(element, "# Title\n\nFirst **paragraph**.\n\nSecond");
  const first = element.children[0];
  const writes = first.htmlWrites;
  api.updateStreamMarkdown(element, "# Title\n\nFirst **paragraph**.\n\nSecond paragraph.");
  expect(element.children[0]).toBe(first);
  expect(first.htmlWrites).toBe(writes);
  expect(element.querySelector(".markdown-tail")!.innerHTML).toContain("Second paragraph.");
});

test("live Markdown preserves blank-separated list items and fenced code as whole blocks", () => {
  const {api} = harness();
  const list = new Element();
  api.updateStreamMarkdown(list, "1. First\n\n2. Second\n\n");
  expect(list.children.length).toBe(1);
  expect(list.children[0].innerHTML.match(/<ol>/g)?.length).toBe(1);
  api.updateStreamMarkdown(list, "1. First\n\n2. Second\n\nNext paragraph.\n");
  expect(list.children[0].className).toBe("markdown-committed");
  expect(list.children[0].innerHTML).toContain("<li>Second</li>");
  const code = new Element();
  api.updateStreamMarkdown(code, "```zig\nconst a = 1;\n\nconst b = 2;");
  expect(code.children.length).toBe(1);
  expect(code.children[0].innerHTML).toContain("const b = 2;");
  expect(code.children[0].innerHTML).toContain("code-block");
  api.updateStreamMarkdown(code, "```zig\nconst a = 1;\n\nconst b = 2;\n```\nNext line");
  expect(code.children[0].className).toBe("markdown-committed");
  expect(code.children[0].innerHTML).toContain("Copy</button>");
});

test("streaming through individual character chunks matches completed Markdown", () => {
  const {api} = harness();
  for (const source of [
    "# Zig Programming Language\n\nZig is **simple, fast, safe**.\n\n## Features\n\n- **Fast** compilation\n\n- No hidden allocations\n\n| Feature | Status |\n| --- | --- |\n| Native | Yes |\n\n> A useful **quote**\n> with a second line\n\n```zig\nconst n = 1;\n\nconst x = 2;\n```\n\nDone.\n",
    "Paragraph\n```c++\nint main() {}\n```\n## Heading\nBody.\n",
    "1. First item\n\n   continuation\n\n2. Second item\n\nA new paragraph.\n",
    "A | B\n--- | ---\n> quote | data\n- item | data\n# heading | data\na | b\n",
  ]) {
    const element = new Element();
    for (let end = 1; end <= source.length; end++) api.updateStreamMarkdown(element, source.slice(0, end));
    expect(element.children.map(child => child.innerHTML).join("")).toBe(api.md(source));
  }
});

test("stream scrolling respects the reader and the latest button resumes following", () => {
  const {api, getElement} = harness();
  const messages = getElement("messages"), jump = getElement("jump-latest");
  api.setHistory([{role: "user", content: "Hello"}]);
  messages.scrollHeight = 1000;
  messages.clientHeight = 400;
  messages.scrollTop = 200;
  messages.dispatch("scroll");
  api.autoScroll();
  expect(messages.scrollTop).toBe(200);
  expect(jump.hidden).toBe(false);
  jump.dispatch("click");
  expect(messages.scrollTop).toBe(1000);
  expect(jump.hidden).toBe(true);
});

test("MTP stays visible with an explanation and cannot activate on an unsupported engine", () => {
  const {api, getElement} = harness();
  api.setModels([{id: "rdna-model", active: true, supports_mtp: false, mtp_unavailable_reason: "Unavailable in this runtime"}]);
  getElement("mtp").checked = true;
  getElement("mtp").dispatch("change");
  expect(api.mtpEnabledForRequest()).toBe(false);
  expect(getElement("mtp-control").hidden).toBe(false);
  expect(getElement("mtp").disabled).toBe(true);
  expect(getElement("mtp-label").textContent).toBe("MTP unavailable");
  expect(getElement("mtp-control").title).toBe("Unavailable in this runtime");
  expect(getElement("mtp-status").title).toBe("Unavailable in this runtime");
});

test("missing MTP metadata is unknown rather than unsupported", () => {
  const {api, getElement} = harness();
  api.setModels([{id: "qwen38-27b-q4k-m", active: true}]);
  api.updateMtpUi();
  expect(getElement("mtp-label").textContent).toBe("MTP status unreported");
  expect(getElement("mtp").indeterminate).toBe(true);
  expect(getElement("mtp-control").title).toContain("may enable MTP automatically");
  expect(getElement("mtp-status").textContent).not.toContain("unavailable");
});

test("MTP opt-in routes to the engine only when capability is advertised", async () => {
  let body: any;
  const {api, getElement} = harness(async (_url, options) => {
    body = JSON.parse(options.body);
    return {ok: true, body: new ReadableStream({start(controller) {
      controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Ready."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'));
      controller.close();
    }})};
  });
  api.setModels([{id: "mtp-model", active: true, supports_mtp: true, mtp_default_enabled: false, mtp_request_control: true}]);
  getElement("mtp").checked = true;
  getElement("mtp").dispatch("change");
  expect(getElement("mtp-control").hidden).toBe(false);
  api.setGenerating(true);
  expect(getElement("mtp").disabled).toBe(true);
  api.setGenerating(false);
  getElement("i").value = "Hello";
  await api.send();
  expect(body.enable_mtp).toBe(true);
  api.setModels([{id: "other-model", active: true, supports_mtp: false}]);
  api.updateMtpUi();
  expect(api.mtpEnabledForRequest()).toBe(false);
  api.setModels([{id: "automatic-model", active: true}]);
  api.updateMtpUi();
  getElement("i").value = "Use your engine defaults";
  await api.send();
  expect(body).not.toHaveProperty("enable_mtp");
});

test("final answer recovery stays locked and can be stopped without losing the next draft", async () => {
  let markRecoveryStarted!: () => void;
  const recoveryStarted = new Promise<void>(resolve => {markRecoveryStarted = resolve;});
  let recoverySignal: AbortSignal | undefined;
  let requests = 0;
  const {api, getElement} = harness(async (_url, options) => {
    requests++;
    if (requests === 1) {
      const chunks = [
        {choices: [{delta: {content: "<think>I need to compare the matrix dimensions.\nThen check whether the inner dimensions match before multiplying."}}]},
        {choices: [{delta: {}, finish_reason: "length"}]},
      ].map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n";
      return new Response(chunks);
    }
    recoverySignal = options.signal;
    markRecoveryStarted();
    return new Promise((_resolve, reject) => {
      recoverySignal!.addEventListener("abort", () => reject(new DOMException("Stopped", "AbortError")));
    });
  });
  api.setModels([{id: "test-model", active: true, supports_thinking_toggle: true}]);
  const input = getElement("i");
  input.value = "Explain matrix multiplication";
  api.setGenerating(false);
  const turn = api.send();
  await recoveryStarted;
  expect(api.generating()).toBe(true);
  expect(getElement("b").ariaLabel).toBe("Stop generating");
  expect(getElement("cb").disabled).toBe(true);
  const recoveryBubble = getElement("messages").children[1].querySelector(".bbl")!;
  const recoveryReasoning = recoveryBubble.querySelector(".reasoning-block")!;
  recoveryReasoning.open = false;
  recoveryReasoning.dispatch("toggle");
  input.value = "My next question";
  api.stopGeneration();
  await turn;
  expect(recoverySignal!.aborted).toBe(true);
  expect(api.generating()).toBe(false);
  expect(input.value).toBe("My next question");
  expect(api.history().find((message: any) => message.role === "assistant").incomplete).toBe(true);
  expect(api.history().find((message: any) => message.role === "assistant").reasoning_open).toBe(false);
  expect(recoveryBubble.querySelector(".reasoning-block")).toBe(recoveryReasoning);
  expect(recoveryReasoning.open).toBe(false);
});
