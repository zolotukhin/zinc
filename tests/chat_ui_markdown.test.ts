import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

type ChatUiApi = {
  md: (text: string) => string;
  renderReasoning: (text: string, isThinking: boolean) => string;
  stripInlineTransportArtifacts: (text: string) => string;
  findInlineStopArtifactStart: (text: string) => number;
  hasInlineStopArtifact: (text: string) => boolean;
  looksLikeRepeatedParagraphLoop: (text: string) => boolean;
  findRestartedAnswerStart: (text: string) => number;
  historyTransportContent: (answer: string, raw: string, enableThinking: boolean, supportsThinking: boolean) => string;
  chatMaxTokens: (enableThinking: boolean) => number;
  assistantHtmlFromRaw: (raw: string, isThinking: boolean) => {
    html: string;
    view: {
      reasoning: string;
      answer: string;
      showReasoning: boolean;
      isThinking: boolean;
      suppressedReasoning?: boolean;
    };
  };
  reasoningIsLowSignal: (text: string) => boolean;
  reasoningDuplicatesAnswer: (reasoning: string, answer: string) => boolean;
  normalizeDisplayView: (view: {
    reasoning: string;
    answer: string;
    showReasoning: boolean;
    isThinking: boolean;
  }) => {
    reasoning: string;
    answer: string;
    showReasoning: boolean;
    isThinking: boolean;
    suppressedReasoning?: boolean;
  };
  completionNotice: (state: {
    hasAnswer: boolean;
    finishReason: string;
    errorMessage: string;
    missingFinishReason: boolean;
    showReasoning: boolean;
    suppressedReasoning?: boolean;
    hadOutput: boolean;
  }) => { kind: string; title: string; message: string } | null;
  requestMessagesFromHistory: () => Array<{ role: string; content: string }>;
  splitDisplayContent: (raw: string, isThinking: boolean) => {
    reasoning: string;
    answer: string;
    showReasoning: boolean;
    isThinking: boolean;
  };
  findUnexpectedThinkingTailStart: (text: string) => number;
  hasRepeatedPhraseLoop: (text: string) => boolean;
  startsWithLeakedReasoning: (text: string) => boolean;
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
    querySelectorAll() {
      return [];
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
    `${match[1]}\nreturn { md, renderReasoning, stripInlineTransportArtifacts, findInlineStopArtifactStart, hasInlineStopArtifact, looksLikeRepeatedParagraphLoop, findRestartedAnswerStart, historyTransportContent, chatMaxTokens, assistantHtmlFromRaw, reasoningIsLowSignal, reasoningDuplicatesAnswer, normalizeDisplayView, completionNotice, requestMessagesFromHistory, splitDisplayContent, findUnexpectedThinkingTailStart, hasRepeatedPhraseLoop, startsWithLeakedReasoning };`,
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

test("live reasoning renders substantive streamed content", () => {
  const { renderReasoning } = loadChatUiApi();
  const html = renderReasoning("1. Check the pointer math\n2. Compare against the C ABI", true);

  expect(html).toContain("Thinking");
  expect(html).toContain("<ol>");
  expect(html).toContain("<li>Check the pointer math</li>");
  expect(html).not.toContain("<em>Thinking...</em>");
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

test("splitDisplayContent strips dangling list marker", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent("Cons:\n- Ecosystem is less mature\n-", false);

  expect(view.answer).toBe("Cons:\n- Ecosystem is less mature");
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

test("splitDisplayContent strips leaked planning from thinking answer tails", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const raw = "<think>\nReasoning.\n</think>\nZig is increasingly being considered for kernel work.\n\nHowever, looking at the prompt structure, it seems I am generating the next turn.";
  const view = splitDisplayContent(raw, false);

  expect(view.reasoning).toBe("Reasoning.");
  expect(view.answer).toBe("Zig is increasingly being considered for kernel work.");
});

test("splitDisplayContent strips reopened think block from answer tail", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const raw = "<think>\nReasoning.\n</think>\nZig is promising for kernel programming.\n\n<think>\nThinking Process:\n1. Analyze the request.";
  const view = splitDisplayContent(raw, false);

  expect(view.reasoning).toBe("Reasoning.");
  expect(view.answer).toBe("Zig is promising for kernel programming.");
});

test("splitDisplayContent strips restarted answer duplicates", () => {
  const { splitDisplayContent, findRestartedAnswerStart } = loadChatUiApi();
  const repeated = [
    "Zig is a modern systems programming language known for its simplicity, safety, and performance.",
    "It has gained attention for systems programming and low-level work.",
    "",
    "Zig is a modern systems programming language known for its simplicity, safety, and performance.",
    "It has gained attention for systems programming and low-level work.",
  ].join("\n");
  const view = splitDisplayContent(repeated, false);

  expect(findRestartedAnswerStart(repeated)).toBeGreaterThan(0);
  expect(view.answer).toBe([
    "Zig is a modern systems programming language known for its simplicity, safety, and performance.",
    "It has gained attention for systems programming and low-level work.",
  ].join("\n"));
});

test("splitDisplayContent strips leaked planning suffix before repeated answer restart", () => {
  const { splitDisplayContent } = loadChatUiApi();
  const raw = [
    "Kernel development requires:",
    "- No standard library",
    "- Direct hardware access",
    "",
    "The user is asking about writing kernel programs in Zig.",
    "Here is the response:",
    "Kernel development requires:",
    "- No standard library",
    "</think>",
    "",
    "Kernel development requires:",
    "- No standard library",
  ].join("\n");
  const view = splitDisplayContent(raw, false);

  expect(view.showReasoning).toBe(false);
  expect(view.reasoning).toBe("");
  expect(view.answer).toBe("Kernel development requires:\n- No standard library\n- Direct hardware access");
});

test("stripInlineTransportArtifacts preserves real text while removing chat transport markers", () => {
  const { stripInlineTransportArtifacts, findInlineStopArtifactStart, hasInlineStopArtifact } = loadChatUiApi();
  const cleaned = stripInlineTransportArtifacts("More detail here<|endoftext|>assistant");

  expect(cleaned).toBe("More detail here");
  expect(findInlineStopArtifactStart("More detail here<|end")).toBe(-1);
  expect(findInlineStopArtifactStart("More detail here<|endoftext|><|im_start|>assistant")).toBe("More detail here".length);
  expect(hasInlineStopArtifact("More detail here<|endoftext|>assistant")).toBe(true);
});

test("looksLikeRepeatedParagraphLoop catches repeated long answer blocks", () => {
  const { looksLikeRepeatedParagraphLoop } = loadChatUiApi();
  const looping = [
    "Zig is a systems programming language designed to be a modern replacement for C. It offers several advantages over C, including better memory safety, simpler syntax, and built-in support for error handling.",
    "Zig is a systems programming language designed to be a direct replacement for C. It offers several advantages over C, including better memory safety, simpler syntax, and built-in support for error handling.",
    "Zig is a systems programming language designed to be a modern replacement for C. It offers several advantages over C, generated by the model, including better memory safety, simpler syntax, and built-in support for error handling.",
  ].join("\n");

  expect(looksLikeRepeatedParagraphLoop(looping)).toBe(true);
  expect(looksLikeRepeatedParagraphLoop("Zig is good for systems work, but the ecosystem is still maturing.")).toBe(false);
});

test("historyTransportContent adds hidden think scaffold for non-thinking Qwen history", () => {
  const { historyTransportContent } = loadChatUiApi();
  const transport = historyTransportContent("Kernel work needs explicit memory control.", "", false, true);

  expect(transport).toBe("<think>\n\n</think>\n\nKernel work needs explicit memory control.");
});

test("historyTransportContent strips visible reasoning from assistant history", () => {
  const { historyTransportContent } = loadChatUiApi();
  const raw = "<think>\nreasoning\n</think>\nKernel work needs explicit memory control.";
  const transport = historyTransportContent("Kernel work needs explicit memory control.", raw, true, true);

  expect(transport).toBe("<think>\n\n</think>\n\nKernel work needs explicit memory control.");
});

test("historyTransportContent compacts long assistant history", () => {
  const { historyTransportContent } = loadChatUiApi();
  const longAnswer = "Kernel programming in Zig gives you explicit control. ".repeat(40);
  const transport = historyTransportContent(longAnswer, "", false, true);

  expect(transport.startsWith("<think>\n\n</think>\n\nKernel programming in Zig")).toBe(true);
  expect(transport.length).toBeLessThan(longAnswer.length);
});

test("chatMaxTokens gives longer budget to non-thinking replies", () => {
  const { chatMaxTokens } = loadChatUiApi();

  expect(chatMaxTokens(false)).toBe(128);
  expect(chatMaxTokens(true)).toBe(256);
});

test("reasoningIsLowSignal flags trivial meta reasoning", () => {
  const { reasoningIsLowSignal } = loadChatUiApi();

  expect(reasoningIsLowSignal("I need to complete the response.")).toBe(true);
  expect(reasoningIsLowSignal("I need to determine if Zig is a programming language or framework relevant to kernel programming, then answer accordingly.")).toBe(true);
  expect(reasoningIsLowSignal("1. Check the pointer math\n2. Compare against the C ABI")).toBe(false);
});

test("reasoningDuplicatesAnswer flags repeated answer content", () => {
  const { reasoningDuplicatesAnswer } = loadChatUiApi();
  const text = "Whether Zig is better than C depends on your priorities. Here is a comparison of memory safety, tooling, and portability.";

  expect(reasoningDuplicatesAnswer(text, text)).toBe(true);
  expect(reasoningDuplicatesAnswer(text, "Completely different content")).toBe(false);
});

test("normalizeDisplayView hides trivial reasoning stubs but keeps the answer", () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const view = normalizeDisplayView({
    reasoning: "I need to complete the response.",
    answer: "Kernel programming in Zig is possible.",
    showReasoning: true,
    isThinking: false,
  });

  expect(view.showReasoning).toBe(false);
  expect(view.answer).toBe("Kernel programming in Zig is possible.");
  expect(view.suppressedReasoning).toBe(true);
});

test("normalizeDisplayView promotes reasoning to answer when it duplicates", () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const text = "Whether Zig is better than C depends on your priorities. Here is a comparison of memory safety, tooling, and portability.";
  const view = normalizeDisplayView({
    reasoning: text,
    answer: text,
    showReasoning: true,
    isThinking: false,
  });

  expect(view.showReasoning).toBe(false);
  expect(view.answer).toBe(text);
  expect(view.promotedReasoning).toBe(true);
});

test("normalizeDisplayView hides empty completed reasoning blocks", () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const view = normalizeDisplayView({
    reasoning: "",
    answer: "Kernel programming in Zig is possible.",
    showReasoning: true,
    isThinking: false,
  });

  expect(view.showReasoning).toBe(false);
  expect(view.answer).toBe("Kernel programming in Zig is possible.");
  expect(view.suppressedReasoning).toBe(true);
});

test("normalizeDisplayView suppresses trivial live reasoning while thinking is active", () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const view = normalizeDisplayView({
    reasoning: "I need to complete the response.",
    answer: "",
    showReasoning: true,
    isThinking: true,
  });

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toBe("");
  expect(view.deferredReasoning).toBe(true);
  expect(view.suppressedReasoning).toBe(true);
});

test("normalizeDisplayView keeps substantive live reasoning visible", () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const view = normalizeDisplayView({
    reasoning: "1. Check the pointer math\n2. Compare against the C ABI",
    answer: "",
    showReasoning: true,
    isThinking: true,
  });

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toContain("Check the pointer math");
  expect(view.suppressedReasoning).toBeUndefined();
});

test("normalizeDisplayView defers answer-like live thinking prose", () => {
  const { normalizeDisplayView, assistantHtmlFromRaw } = loadChatUiApi();
  const raw = "<think>Zig is a modern systems programming language known for its simplicity, safety, and performance. It has gained attention for low-level systems work, including kernel development. However, there are several tradeoffs to consider before using it for production kernels.";
  const view = normalizeDisplayView({
    reasoning: "Zig is a modern systems programming language known for its simplicity, safety, and performance. It has gained attention for low-level systems work, including kernel development. However, there are several tradeoffs to consider before using it for production kernels.",
    answer: "",
    showReasoning: true,
    isThinking: true,
  });
  const rendered = assistantHtmlFromRaw(raw, true);

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toBe("");
  expect(view.deferredReasoning).toBe(true);
  expect(rendered.html).toContain("Thinking...");
  expect(rendered.html).not.toContain("Zig is a modern systems programming language");
});

test("normalizeDisplayView promotes answer-like reasoning when a final answer exists", () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const view = normalizeDisplayView({
    reasoning: "Zig is a modern systems programming language known for its simplicity, safety, and performance. It has gained attention for low-level systems work, including kernel development. However, there are several tradeoffs to consider before using it for production kernels.",
    answer: "Zig is increasingly being considered for kernel programming, but it is not yet as mature or widely adopted as C.",
    showReasoning: true,
    isThinking: false,
  });

  // Reasoning looks like an answer, so it's promoted — reasoning wrapper evaporated
  expect(view.showReasoning).toBe(false);
  expect(view.promotedReasoning).toBe(true);
  // The reasoning is promoted to become the answer
  expect(view.answer).toContain("simplicity, safety, and performance");
});

test("assistantHtmlFromRaw does not leak suppressed live reasoning into the answer bubble", () => {
  const { assistantHtmlFromRaw } = loadChatUiApi();
  const rendered = assistantHtmlFromRaw("<think>I need to complete the response.", true);

  expect(rendered.view.suppressedReasoning).toBe(true);
  expect(rendered.html).toContain("Thinking...");
  expect(rendered.html).not.toContain("I need to complete the response.");
});

test("completionNotice surfaces token-limit hits clearly", () => {
  const { completionNotice } = loadChatUiApi();
  const notice = completionNotice({
    hasAnswer: true,
    finishReason: "length",
    errorMessage: "",
    missingFinishReason: false,
    showReasoning: false,
    hadOutput: true,
  });

  expect(notice.kind).toBe("warn");
  expect(notice.title).toBe("Token limit reached");
  expect(notice.message).toContain("Ask to continue");
});

test("completionNotice reports interrupted streams with partial answers", () => {
  const { completionNotice } = loadChatUiApi();
  const notice = completionNotice({
    hasAnswer: true,
    finishReason: "stop",
    errorMessage: "",
    missingFinishReason: true,
    showReasoning: false,
    hadOutput: true,
  });

  expect(notice.kind).toBe("warn");
  expect(notice.title).toBe("Response interrupted");
  expect(notice.message).toContain("clean finish");
});

test("completionNotice reports backend errors clearly", () => {
  const { completionNotice } = loadChatUiApi();
  const notice = completionNotice({
    hasAnswer: false,
    finishReason: "stop",
    errorMessage: "socket closed",
    missingFinishReason: false,
    showReasoning: false,
    hadOutput: false,
  });

  expect(notice.kind).toBe("error");
  expect(notice.title).toBe("Request failed");
  expect(notice.message).toBe("socket closed");
});

test("completionNotice reports suppressed reasoning without a final answer", () => {
  const { completionNotice } = loadChatUiApi();
  const notice = completionNotice({
    hasAnswer: false,
    finishReason: "stop",
    errorMessage: "",
    missingFinishReason: false,
    showReasoning: false,
    suppressedReasoning: true,
    hadOutput: true,
  });

  expect(notice?.kind).toBe("warn");
  expect(notice?.title).toBe("No final answer");
  expect(notice?.message).toContain("trivial reasoning stub");
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
return requestMessagesFromHistory(false);`,
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

test("requestMessagesFromHistory prefers transport content for assistant history", () => {
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
  { role: 'assistant', content: 'clean answer', transport: '<think>\\n\\n</think>\\n\\nclean answer' },
];
return requestMessagesFromHistory(true);`,
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
    { role: "assistant", content: "<think>\n\n</think>\n\nclean answer" },
  ]);
});

test("requestMessagesFromHistory excludes incomplete assistant entries", () => {
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
  { role: 'assistant', content: 'cut off answer', incomplete: true, notice_title: 'Token limit reached' },
  { role: 'assistant', content: 'final answer' },
];
return requestMessagesFromHistory(false);`,
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

test("findUnexpectedThinkingTailStart detects reopened think after initial thinking block", () => {
  const { findUnexpectedThinkingTailStart } = loadChatUiApi();

  // Reopened <think> after a full <think>...</think> block and answer — should detect it
  const raw = "<think>\nReasoning.\n</think>\nZig is promising.\n<think>\nMore thinking";
  expect(findUnexpectedThinkingTailStart(raw)).toBeGreaterThan(0);

  // Still inside initial <think> block (no </think> yet) — no detection
  expect(findUnexpectedThinkingTailStart("<think>\nStill thinking...")).toBe(-1);

  // Normal completed thinking — no reopened block
  expect(findUnexpectedThinkingTailStart("<think>\nReasoning.\n</think>\nAnswer here.")).toBe(-1);

  // No <think> at all
  expect(findUnexpectedThinkingTailStart("Just a plain answer.")).toBe(-1);

  // Reopened <think> without initial thinking block (original behavior)
  expect(findUnexpectedThinkingTailStart("Some answer.<think>\nNew thinking")).toBeGreaterThan(0);
});

test("reasoningDuplicatesAnswer detects streaming prefix duplicates early", () => {
  const { reasoningDuplicatesAnswer } = loadChatUiApi();
  const reasoning = "Zig is a modern systems programming language known for its simplicity, safety, and performance. It has been gaining attention in the systems programming community.";

  // Short streaming answer (40+ chars) that matches start of reasoning — should detect
  const shortPrefix = "Zig is a modern systems programming language known";
  expect(reasoningDuplicatesAnswer(reasoning, shortPrefix)).toBe(true);

  // Too short (< 40 chars) — should not detect
  expect(reasoningDuplicatesAnswer(reasoning, "Zig is a modern")).toBe(false);

  // Different content — should not detect
  expect(reasoningDuplicatesAnswer(reasoning, "Python is a great language for data science and ML work.")).toBe(false);
});

test("assistantHtmlFromRaw promotes reasoning to answer when answer duplicates it", () => {
  const { assistantHtmlFromRaw } = loadChatUiApi();

  // Simulate streaming state: reasoning visible, answer starts matching
  const raw = "<think>\nZig is a modern systems programming language known for its simplicity, safety, and performance.\n</think>\nZig is a modern systems programming language known for its simplicity";
  const result = assistantHtmlFromRaw(raw, false);

  // Reasoning is promoted to answer — no reasoning wrapper
  expect(result.view.showReasoning).toBe(false);
  expect(result.view.promotedReasoning).toBe(true);
  expect(result.view.answer.length).toBeGreaterThan(0);
  // No reasoning block in output
  expect(result.html).not.toContain("reasoning-block");
});

test("assistantHtmlFromRaw promotes reasoning to answer when model puts everything in think tags", () => {
  const { assistantHtmlFromRaw } = loadChatUiApi();

  // Model put entire answer in <think> with no content after </think>
  const raw = "<think>\nZig is a modern systems programming language known for its simplicity, safety, and performance. It has gained attention for kernel development due to its low-level control.\n</think>\n";
  const result = assistantHtmlFromRaw(raw, false);

  // Should promote reasoning to answer — no "No final answer" scenario
  expect(result.view.answer.length).toBeGreaterThan(0);
  expect(result.view.answer).toContain("Zig is a modern");
  expect(result.view.showReasoning).toBe(false);
});

test("hasRepeatedPhraseLoop detects sentence-level repetition", () => {
  const { hasRepeatedPhraseLoop } = loadChatUiApi();
  expect(hasRepeatedPhraseLoop("I should also mention type safety. I should also mention type safety. I should also mention type safety. I should also mention type safety.")).toBe(true);
  expect(hasRepeatedPhraseLoop("Zig is a systems programming language. It features manual memory management. It compiles to native code.")).toBe(false);
});

test("startsWithLeakedReasoning detects meta-commentary", () => {
  const { startsWithLeakedReasoning } = loadChatUiApi();
  expect(startsWithLeakedReasoning("The user is asking about C types.")).toBe(true);
  expect(startsWithLeakedReasoning("I need to provide a clear explanation.")).toBe(true);
  expect(startsWithLeakedReasoning("Zig is a modern systems programming language.")).toBe(false);
});
