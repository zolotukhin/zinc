import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

type ChatUiApi = {
  md: (text: string) => string;
  renderReasoning: (text: string, isThinking: boolean) => string;
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
    `${match[1]}\nreturn { md, renderReasoning, splitDisplayContent };`,
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

test("chat UI markdown renders unordered lists", () => {
  const { md } = loadChatUiApi();
  const html = md("Issues:\n\n- preserve newlines\n- render bullet lists");

  expect(html).toContain("<p>Issues:</p>");
  expect(html).toContain("<ul>");
  expect(html).toContain("<li>preserve newlines</li>");
  expect(html).toContain("<li>render bullet lists</li>");
});

test("reasoning blocks use markdown formatting", () => {
  const { renderReasoning } = loadChatUiApi();
  const html = renderReasoning("Thinking Process:\n\n1. First step\n2. Second step", false);

  expect(html).toContain('<details class="reasoning-block">');
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

test("splitDisplayContent treats plain Thinking Process output as reasoning", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("Thinking Process:\n\n1. First\n2. Second", false);

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toContain("Thinking Process:");
  expect(view.answer).toBe("");
  expect(view.isThinking).toBe(true);
});
