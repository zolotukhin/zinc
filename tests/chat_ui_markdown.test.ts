import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

type ChatUiApi = {
  md: (text: string) => string;
  renderReasoning: (text: string, isThinking: boolean) => string;
  stripInlineTransportArtifacts: (text: string) => string;
  requestMessagesFromHistory: () => Array<{ role: string; content: string }>;
  splitDisplayContent: (raw: string, isThinking: boolean) => {
    reasoning: string;
    answer: string;
    showReasoning: boolean;
    isThinking: boolean;
  };
};

function makeElement(): any {
  return {
    value: "",
    textContent: "",
    innerHTML: "",
    href: "",
    title: "",
    disabled: false,
    className: "",
    scrollTop: 0,
    scrollHeight: 0,
    style: {},
    classList: {
      toggle() {},
      add() {},
      remove() {},
    },
    addEventListener() {},
    appendChild() {},
    querySelector() {
      return makeElement();
    },
    remove() {},
    focus() {},
    parentElement: { appendChild() {} },
  };
}

function loadChatUiApi(): ChatUiApi {
  const chatHtml = readFileSync(resolve(import.meta.dir, "../src/server/chat.html"), "utf8");
  const match = chatHtml.match(/<script>([\s\S]*)<\/script>\s*<\/body>/);
  if (!match) throw new Error("Chat UI script not found");

  const elements = new Map<string, any>();
  const documentMock = {
    getElementById(id: string) {
      let element = elements.get(id);
      if (!element) {
        element = makeElement();
        elements.set(id, element);
      }
      return element;
    },
    createElement() {
      return makeElement();
    },
    querySelectorAll() {
      return [];
    },
  };

  const factory = new Function(
    "document",
    "fetch",
    "location",
    "performance",
    "navigator",
    "hljs",
    "alert",
    `${match[1]}\nreturn { md, renderReasoning, stripInlineTransportArtifacts, requestMessagesFromHistory, splitDisplayContent };`,
  );

  return factory(
    documentMock,
    async () => {
      throw new Error("offline");
    },
    { origin: "http://localhost:8080" },
    { now: () => 0 },
    { clipboard: { writeText: async () => {} } },
    { highlightElement() {} },
    () => {},
  ) as ChatUiApi;
}

test("chat UI markdown renders ordered lists", () => {
  const { md } = loadChatUiApi();
  const html = md("Thinking Process:\n\n1. **Analyze the Request:** identify the topic\n2. **Summarize:** answer briefly");

  expect(html).toContain("<p>Thinking Process:</p>");
  expect(html).toContain("<ol>");
  expect(html).toContain("<li><strong>Analyze the Request:</strong> identify the topic</li>");
  expect(html).toContain("<li><strong>Summarize:</strong> answer briefly</li>");
});

test("chat UI markdown keeps ordered lists together across blank lines", () => {
  const { md } = loadChatUiApi();
  const html = md("1. First item\n\n2. Second item\n\n3. Third item");

  expect((html.match(/<ol>/g) || []).length).toBe(1);
  expect(html).toContain("<li>First item</li>");
  expect(html).toContain("<li>Second item</li>");
  expect(html).toContain("<li>Third item</li>");
});

test("chat UI markdown keeps ordered lists together when an item has blank-line continuation text", () => {
  const { md } = loadChatUiApi();
  const html = md("1. First item\n\n   continuation line\n\n2. Second item");

  expect((html.match(/<ol>/g) || []).length).toBe(1);
  expect(html).toContain("<li>First item<br>continuation line</li>");
  expect(html).toContain("<li>Second item</li>");
});

test("chat UI markdown keeps ordered lists together when an item has nested bullets after a blank line", () => {
  const { md } = loadChatUiApi();
  const html = md("1. First item\n\n- nested alpha\n- nested beta\n\n2. Second item");

  expect((html.match(/<ol>/g) || []).length).toBe(1);
  expect(html).toContain("<li>First item<ul>");
  expect(html).toContain("<li>nested alpha</li>");
  expect(html).toContain("<li>nested beta</li>");
  expect(html).toContain("<li>Second item</li>");
});

test("chat UI markdown renders unordered lists", () => {
  const { md } = loadChatUiApi();
  const html = md("Issues:\n\n- preserve newlines\n- render bullet lists");

  expect(html).toContain("<p>Issues:</p>");
  expect(html).toContain("<ul>");
  expect(html).toContain("<li>preserve newlines</li>");
  expect(html).toContain("<li>render bullet lists</li>");
});

test("chat UI markdown renders inline links", () => {
  const { md } = loadChatUiApi();
  const html = md("Use [vulkan.zig](https://github.com/ziglibs/vulkan.zig) for bindings.");

  expect(html).toContain('<a href="https://github.com/ziglibs/vulkan.zig" target="_blank" rel="noopener noreferrer">vulkan.zig</a>');
});

test("chat UI markdown keeps unordered lists together across blank lines", () => {
  const { md } = loadChatUiApi();
  const html = md("- alpha\n\n- beta\n\n- gamma");

  expect((html.match(/<ul>/g) || []).length).toBe(1);
  expect(html).toContain("<li>alpha</li>");
  expect(html).toContain("<li>beta</li>");
  expect(html).toContain("<li>gamma</li>");
});

test("chat UI markdown promotes headings that appear mid-block", () => {
  const { md } = loadChatUiApi();
  const html = md("Intro line\n### Heading\nBody line");

  expect(html).toContain("<p>Intro line</p>");
  expect(html).toContain("<h3>Heading</h3>");
  expect(html).toContain("<p>Body line</p>");
});

test("reasoning blocks use markdown formatting", () => {
  const { renderReasoning } = loadChatUiApi();
  const html = renderReasoning("Thinking Process:\n\n1. First step\n2. Second step", false);

  expect(html).toContain('<details open class="reasoning-block">');
  expect(html).toContain('class="reasoning-state"');
  expect(html).toContain("Reasoning");
  expect(html).toContain("<ol>");
  expect(html).toContain("<li>First step</li>");
  expect(html).toContain("<li>Second step</li>");
});

test("reasoning blocks render nested bullets inside numbered steps", () => {
  const { renderReasoning } = loadChatUiApi();
  const html = renderReasoning(
    "Thinking Process:\n\n1. **Analyze the Request:**\n   * **User Input:** question about Saint Petersburg\n   * **Intent:** historical overview\n2. **Summarize:** answer concisely",
    false,
  );

  expect(html).toContain("<ol>");
  expect(html).toContain("<li><strong>Analyze the Request:</strong><ul>");
  expect(html).toContain("<li><strong>User Input:</strong> question about Saint Petersburg</li>");
  expect(html).toContain("<li><strong>Intent:</strong> historical overview</li>");
  expect(html).toContain("<li><strong>Summarize:</strong> answer concisely</li>");
});

test("completed structured reasoning opens by default instead of collapsing to 1. 1. 1. preview", () => {
  const { renderReasoning } = loadChatUiApi();
  const html = renderReasoning("1. First step\n2. Second step\n3. Third step", false);

  expect(html).toContain("<details open");
  expect(html).toContain("<ol>");
  expect(html).not.toContain('class="reasoning-preview"');
});

test("splitDisplayContent strips trailing standalone quote from answers", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("Hey, I'm doing well.\n\"\n\n", false);

  expect(view.answer).toBe("Hey, I'm doing well.");
  expect(view.showReasoning).toBe(false);
});

test("splitDisplayContent strips unmatched trailing quote after punctuation", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("Hello! How can I help you today?\"\n\n", false);

  expect(view.answer).toBe("Hello! How can I help you today?");
  expect(view.showReasoning).toBe(false);
});

test("splitDisplayContent strips trailing endoftext artifacts", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("Hello! \uFFFD\uFFFD<|endoftext|>\n", false);

  expect(view.answer).toBe("Hello!");
  expect(view.showReasoning).toBe(false);
});

test("splitDisplayContent strips trailing replacement junk plus dangling quote", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("Hey there! How can I help you today? \uFFFD\uFFFD\"\n", false);

  expect(view.answer).toBe("Hey there! How can I help you today?");
  expect(view.showReasoning).toBe(false);
});

test("splitDisplayContent strips unmatched trailing quote after emoji", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("Hey there! How can I help you today? 😊\"\n", false);

  expect(view.answer).toBe("Hey there! How can I help you today? 😊");
  expect(view.showReasoning).toBe(false);
});

test("splitDisplayContent strips dangling heading markers", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("Hello\n\n###\n", false);

  expect(view.answer).toBe("Hello");
  expect(view.showReasoning).toBe(false);
});

test("splitDisplayContent strips leading standalone quote before the answer", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("\"\n\nVulcan is likely a typo for Vulkan.", false);

  expect(view.answer).toBe("Vulcan is likely a typo for Vulkan.");
  expect(view.showReasoning).toBe(false);
});

test("splitDisplayContent treats plain Thinking Process output as reasoning while thinking is active", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("Thinking Process:\n\n1. First\n2. Second", true);

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toContain("Thinking Process:");
  expect(view.answer).toBe("");
  expect(view.isThinking).toBe(true);
});

test("splitDisplayContent falls back to the full answer when plain reasoning has no final boundary", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const raw = "Thinking Process:\n\n1. First\n2. Second";
  const view = splitDisplayContent(raw, false);

  expect(view.showReasoning).toBe(false);
  expect(view.reasoning).toBe("");
  expect(view.answer).toBe(raw);
  expect(view.isThinking).toBe(false);
});

test("splitDisplayContent infers reasoning when only </think> is emitted", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("17 * 24 = 408\n</think>\n408", false);

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toBe("17 * 24 = 408");
  expect(view.answer).toBe("408");
  expect(view.isThinking).toBe(false);
});

test("stripInlineTransportArtifacts preserves real text while removing chat transport markers", () => {
  const { stripInlineTransportArtifacts } = loadChatUiApi();
  const cleaned = stripInlineTransportArtifacts("More detail here<|im_end|>");

  expect(cleaned).toBe("More detail here");
});

test("requestMessagesFromHistory excludes partial assistant entries", () => {
  const chatHtml = readFileSync(resolve(import.meta.dir, "../src/server/chat.html"), "utf8");
  const match = chatHtml.match(/<script>([\s\S]*)<\/script>\s*<\/body>/);
  if (!match) throw new Error("Chat UI script not found");

  const factory = new Function(
    "document",
    "fetch",
    "location",
    "performance",
    "navigator",
    "hljs",
    "alert",
    `${match[1]}
H = [
  { role: 'user', content: 'hello' },
  { role: 'assistant', content: 'half answer', partial: true },
  { role: 'assistant', content: 'final answer' },
];
return requestMessagesFromHistory();`,
  );

  const messages = factory(
    {
      getElementById() { return makeElement(); },
      createElement() { return makeElement(); },
      querySelectorAll() { return []; },
    },
    async () => { throw new Error("offline"); },
    { origin: "http://localhost:8080" },
    { now: () => 0 },
    { clipboard: { writeText: async () => {} } },
    { highlightElement() {} },
    () => {},
  ) as Array<{ role: string; content: string }>;

  expect(messages).toEqual([
    { role: "user", content: "hello" },
    { role: "assistant", content: "final answer" },
  ]);
});
