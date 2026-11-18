import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

type ChatUiApi = {
  md: (text: string) => string;
  renderReasoning: (text: string, isThinking: boolean, collapsed?: boolean) => string;
  stripInlineTransportArtifacts: (text: string) => string;
  findInlineStopArtifactStart: (text: string) => number;
  hasInlineStopArtifact: (text: string) => boolean;
  looksLikeRepeatedParagraphLoop: (text: string) => boolean;
  findRestartedAnswerStart: (text: string) => number;
  historyTransportContent: (
    answer: string,
    raw: string,
    enableThinking: boolean,
    supportsThinking: boolean,
  ) => string;
  chatMaxTokens: (enableThinking: boolean) => number;
  assistantHtmlFromRaw: (
    raw: string,
    isThinking: boolean,
  ) => {
    html: string;
    view: {
      reasoning: string;
      answer: string;
      showReasoning: boolean;
      isThinking: boolean;
      suppressedReasoning?: boolean;
    };
  };
  assistantLiveHtmlFromRaw: (
    raw: string,
    isThinking: boolean,
  ) => {
    html: string;
    view: {
      reasoning: string;
      answer: string;
      showReasoning: boolean;
      isThinking: boolean;
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
  splitDisplayContent: (
    raw: string,
    isThinking: boolean,
  ) => {
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
    value: '',
    textContent: '',
    innerHTML: '',
    href: '',
    title: '',
    disabled: false,
    className: '',
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
  const chatHtml = readFileSync(resolve(import.meta.dir, '../src/server/chat.html'), 'utf8');
  const match = chatHtml.match(/<script>([\s\S]*)<\/script>\s*<\/body>/);
  if (!match) throw new Error('Chat UI script not found');

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
    'document',
    'fetch',
    'location',
    'performance',
    'navigator',
    'hljs',
    'alert',
    `${match[1]}\nreturn { md, renderReasoning, stripInlineTransportArtifacts, findInlineStopArtifactStart, hasInlineStopArtifact, looksLikeRepeatedParagraphLoop, findRestartedAnswerStart, historyTransportContent, chatMaxTokens, assistantHtmlFromRaw, assistantLiveHtmlFromRaw, reasoningIsLowSignal, reasoningDuplicatesAnswer, normalizeDisplayView, completionNotice, requestMessagesFromHistory, splitDisplayContent, findUnexpectedThinkingTailStart, hasRepeatedPhraseLoop, startsWithLeakedReasoning };`,
  );

  return factory(
    documentMock,
    async () => {
      throw new Error('offline');
    },
    { origin: 'http://localhost:8080' },
    { now: () => 0 },
    { clipboard: { writeText: async () => {} } },
    { highlightElement() {} },
    () => {},
  ) as ChatUiApi;
}

test('chat UI markdown renders ordered lists', () => {
  const { md } = loadChatUiApi();
  const html = md(
    'Thinking Process:\n\n1. **Analyze the Request:** identify the topic\n2. **Summarize:** answer briefly',
  );

  expect(html).toContain('<p>Thinking Process:</p>');
  expect(html).toContain('<ol>');
  expect(html).toContain('<li><strong>Analyze the Request:</strong> identify the topic</li>');
  expect(html).toContain('<li><strong>Summarize:</strong> answer briefly</li>');
});

test('chat UI markdown keeps ordered lists together across blank lines', () => {
  const { md } = loadChatUiApi();
  const html = md('1. First item\n\n2. Second item\n\n3. Third item');

  expect((html.match(/<ol>/g) || []).length).toBe(1);
  expect(html).toContain('<li>First item</li>');
  expect(html).toContain('<li>Second item</li>');
  expect(html).toContain('<li>Third item</li>');
});

test('chat UI markdown keeps ordered lists together when an item has blank-line continuation text', () => {
  const { md } = loadChatUiApi();
  const html = md('1. First item\n\n   continuation line\n\n2. Second item');

  expect((html.match(/<ol>/g) || []).length).toBe(1);
  expect(html).toContain('<li>First item<br>continuation line</li>');
  expect(html).toContain('<li>Second item</li>');
});

test('chat UI markdown keeps ordered lists together when an item has nested bullets after a blank line', () => {
  const { md } = loadChatUiApi();
  const html = md('1. First item\n\n- nested alpha\n- nested beta\n\n2. Second item');

  expect((html.match(/<ol>/g) || []).length).toBe(1);
  expect(html).toContain('<li>First item<ul>');
  expect(html).toContain('<li>nested alpha</li>');
  expect(html).toContain('<li>nested beta</li>');
  expect(html).toContain('<li>Second item</li>');
});

test('chat UI markdown renders unordered lists', () => {
  const { md } = loadChatUiApi();
  const html = md('Issues:\n\n- preserve newlines\n- render bullet lists');

  expect(html).toContain('<p>Issues:</p>');
  expect(html).toContain('<ul>');
  expect(html).toContain('<li>preserve newlines</li>');
  expect(html).toContain('<li>render bullet lists</li>');
});

test('chat UI markdown renders inline links', () => {
  const { md } = loadChatUiApi();
  const html = md('Use [vulkan.zig](https://github.com/ziglibs/vulkan.zig) for bindings.');

  expect(html).toContain(
    '<a href="https://github.com/ziglibs/vulkan.zig" target="_blank" rel="noopener noreferrer">vulkan.zig</a>',
  );
});

test('chat UI markdown keeps unordered lists together across blank lines', () => {
  const { md } = loadChatUiApi();
  const html = md('- alpha\n\n- beta\n\n- gamma');

  expect((html.match(/<ul>/g) || []).length).toBe(1);
  expect(html).toContain('<li>alpha</li>');
  expect(html).toContain('<li>beta</li>');
  expect(html).toContain('<li>gamma</li>');
});

test('chat UI markdown promotes headings that appear mid-block', () => {
  const { md } = loadChatUiApi();
  const html = md('Intro line\n### Heading\nBody line');

  expect(html).toContain('<p>Intro line</p>');
  expect(html).toContain('<h3>Heading</h3>');
  expect(html).toContain('<p>Body line</p>');
});

test('reasoning blocks use markdown formatting', () => {
  const { renderReasoning } = loadChatUiApi();
  const html = renderReasoning('Thinking Process:\n\n1. First step\n2. Second step', false);

  expect(html).toContain('<details open class="reasoning-block">');
  expect(html).toContain('class="reasoning-state"');
  expect(html).toContain('Reasoning');
  expect(html).toContain('<ol>');
  expect(html).toContain('<li>First step</li>');
  expect(html).toContain('<li>Second step</li>');
});

test('reasoning blocks render nested bullets inside numbered steps', () => {
  const { renderReasoning } = loadChatUiApi();
  const html = renderReasoning(
    'Thinking Process:\n\n1. **Analyze the Request:**\n   * **User Input:** question about Saint Petersburg\n   * **Intent:** historical overview\n2. **Summarize:** answer concisely',
    false,
  );

  expect(html).toContain('<ol>');
  expect(html).toContain('<li><strong>Analyze the Request:</strong><ul>');
  expect(html).toContain('<li><strong>User Input:</strong> question about Saint Petersburg</li>');
  expect(html).toContain('<li><strong>Intent:</strong> historical overview</li>');
  expect(html).toContain('<li><strong>Summarize:</strong> answer concisely</li>');
});

test('completed structured reasoning opens by default instead of collapsing to 1. 1. 1. preview', () => {
  const { renderReasoning } = loadChatUiApi();
  const html = renderReasoning('1. First step\n2. Second step\n3. Third step', false);

  expect(html).toContain('<details open');
  expect(html).toContain('<ol>');
  expect(html).not.toContain('class="reasoning-preview"');
});

test('live reasoning renders substantive streamed content', () => {
  const { renderReasoning } = loadChatUiApi();
  const html = renderReasoning('1. Check the pointer math\n2. Compare against the C ABI', true);

  expect(html).toContain('Thinking');
  expect(html).toContain('<ol>');
  expect(html).toContain('<li>Check the pointer math</li>');
  expect(html).not.toContain('<em>Thinking...</em>');
});

test('splitDisplayContent strips trailing standalone quote from answers', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent('Hey, I\'m doing well.\n"\n\n', false);

  expect(view.answer).toBe("Hey, I'm doing well.");
  expect(view.showReasoning).toBe(false);
});

test('splitDisplayContent strips unmatched trailing quote after punctuation', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent('Hello! How can I help you today?"\n\n', false);

  expect(view.answer).toBe('Hello! How can I help you today?');
  expect(view.showReasoning).toBe(false);
});

test('splitDisplayContent strips trailing endoftext artifacts', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent('Hello! \uFFFD\uFFFD<|endoftext|>\n', false);

  expect(view.answer).toBe('Hello!');
  expect(view.showReasoning).toBe(false);
});

test('splitDisplayContent strips trailing replacement junk plus dangling quote', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent('Hey there! How can I help you today? \uFFFD\uFFFD"\n', false);

  expect(view.answer).toBe('Hey there! How can I help you today?');
  expect(view.showReasoning).toBe(false);
});

test('splitDisplayContent strips unmatched trailing quote after emoji', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent('Hey there! How can I help you today? 😊"\n', false);

  expect(view.answer).toBe('Hey there! How can I help you today? 😊');
  expect(view.showReasoning).toBe(false);
});

test('splitDisplayContent strips dangling heading markers', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent('Hello\n\n###\n', false);

  expect(view.answer).toBe('Hello');
  expect(view.showReasoning).toBe(false);
});

test('splitDisplayContent strips dangling list marker', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent('Cons:\n- Ecosystem is less mature\n-', false);

  expect(view.answer).toBe('Cons:\n- Ecosystem is less mature');
  expect(view.showReasoning).toBe(false);
});

test('splitDisplayContent strips leading standalone quote before the answer', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent('"\n\nVulcan is likely a typo for Vulkan.', false);

  expect(view.answer).toBe('Vulcan is likely a typo for Vulkan.');
  expect(view.showReasoning).toBe(false);
});

test('splitDisplayContent treats plain Thinking Process output as reasoning while thinking is active', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent('Thinking Process:\n\n1. First\n2. Second', true);

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toContain('Thinking Process:');
  expect(view.answer).toBe('');
  expect(view.isThinking).toBe(true);
});

test('splitDisplayContent retains plain reasoning alongside its full-answer fallback', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const raw = 'Thinking Process:\n\n1. First\n2. Second';
  const view = splitDisplayContent(raw, false);

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toBe(raw);
  expect(view.answer).toBe(raw);
  expect(view.isThinking).toBe(false);
});

test('splitDisplayContent infers reasoning when only </think> is emitted', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const view = splitDisplayContent('17 * 24 = 408\n</think>\n408', false);

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toBe('17 * 24 = 408');
  expect(view.answer).toBe('408');
  expect(view.isThinking).toBe(false);
});

test('splitDisplayContent strips leaked planning from thinking answer tails', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const raw =
    '<think>\nReasoning.\n</think>\nZig is increasingly being considered for kernel work.\n\nHowever, looking at the prompt structure, it seems I am generating the next turn.';
  const view = splitDisplayContent(raw, false);

  expect(view.reasoning).toBe('Reasoning.');
  expect(view.answer).toBe('Zig is increasingly being considered for kernel work.');
});

test('splitDisplayContent strips reopened think block from answer tail', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const raw =
    '<think>\nReasoning.\n</think>\nZig is promising for kernel programming.\n\n<think>\nThinking Process:\n1. Analyze the request.';
  const view = splitDisplayContent(raw, false);

  expect(view.reasoning).toBe('Reasoning.');
  expect(view.answer).toBe('Zig is promising for kernel programming.');
});

test('splitDisplayContent strips restarted answer duplicates', () => {
  const { splitDisplayContent, findRestartedAnswerStart } = loadChatUiApi();
  const repeated = [
    'Zig is a modern systems programming language known for its simplicity, safety, and performance.',
    'It has gained attention for systems programming and low-level work.',
    '',
    'Zig is a modern systems programming language known for its simplicity, safety, and performance.',
    'It has gained attention for systems programming and low-level work.',
  ].join('\n');
  const view = splitDisplayContent(repeated, false);

  expect(findRestartedAnswerStart(repeated)).toBeGreaterThan(0);
  expect(view.answer).toBe(
    [
      'Zig is a modern systems programming language known for its simplicity, safety, and performance.',
      'It has gained attention for systems programming and low-level work.',
    ].join('\n'),
  );
});

test('splitDisplayContent cleans planning and repeated restarts from a plain answer without reasoning tags', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const raw = [
    'Kernel development requires:',
    '- No standard library',
    '- Direct hardware access',
    '',
    'The user is asking about writing kernel programs in Zig.',
    'Here is the response:',
    'Kernel development requires:',
    '- No standard library',
    '',
    'Kernel development requires:',
    '- No standard library',
  ].join('\n');
  const view = splitDisplayContent(raw, false);

  expect(view.showReasoning).toBe(false);
  expect(view.reasoning).toBe('');
  expect(view.answer).toBe(
    'Kernel development requires:\n- No standard library\n- Direct hardware access',
  );
});

test('an explicit closing think tag preserves its full reasoning prefix before a repeated answer', () => {
  const { splitDisplayContent } = loadChatUiApi();
  const reasoning = [
    'Kernel development requires:',
    '- No standard library',
    '- Direct hardware access',
    '',
    'The user is asking about writing kernel programs in Zig.',
    'Here is the response:',
    'Kernel development requires:',
    '- No standard library',
  ].join('\n');
  const answer = 'Kernel development requires:\n- No standard library';
  const view = splitDisplayContent(reasoning + '\n</think>\n\n' + answer, false);
  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toBe(reasoning);
  expect(view.answer).toBe(answer);
});

test('stripInlineTransportArtifacts preserves real text while removing chat transport markers', () => {
  const { stripInlineTransportArtifacts, findInlineStopArtifactStart, hasInlineStopArtifact } =
    loadChatUiApi();
  const cleaned = stripInlineTransportArtifacts('More detail here<|endoftext|>assistant');

  expect(cleaned).toBe('More detail here');
  expect(findInlineStopArtifactStart('More detail here<|end')).toBe(-1);
  expect(findInlineStopArtifactStart('More detail here<|endoftext|><|im_start|>assistant')).toBe(
    'More detail here'.length,
  );
  expect(hasInlineStopArtifact('More detail here<|endoftext|>assistant')).toBe(true);
});

test('looksLikeRepeatedParagraphLoop catches repeated long answer blocks', () => {
  const { looksLikeRepeatedParagraphLoop } = loadChatUiApi();
  const looping = [
    'Zig is a systems programming language designed to be a modern replacement for C. It offers several advantages over C, including better memory safety, simpler syntax, and built-in support for error handling.',
    'Zig is a systems programming language designed to be a direct replacement for C. It offers several advantages over C, including better memory safety, simpler syntax, and built-in support for error handling.',
    'Zig is a systems programming language designed to be a modern replacement for C. It offers several advantages over C, generated by the model, including better memory safety, simpler syntax, and built-in support for error handling.',
  ].join('\n');

  expect(looksLikeRepeatedParagraphLoop(looping)).toBe(true);
  expect(
    looksLikeRepeatedParagraphLoop(
      'Zig is good for systems work, but the ecosystem is still maturing.',
    ),
  ).toBe(false);
});

test('historyTransportContent adds hidden think scaffold for non-thinking Qwen history', () => {
  const { historyTransportContent } = loadChatUiApi();
  const transport = historyTransportContent(
    'Kernel work needs explicit memory control.',
    '',
    false,
    true,
  );

  expect(transport).toBe('<think>\n\n</think>\n\nKernel work needs explicit memory control.');
});

test('historyTransportContent strips visible reasoning from assistant history', () => {
  const { historyTransportContent } = loadChatUiApi();
  const raw = '<think>\nreasoning\n</think>\nKernel work needs explicit memory control.';
  const transport = historyTransportContent(
    'Kernel work needs explicit memory control.',
    raw,
    true,
    true,
  );

  expect(transport).toBe('<think>\n\n</think>\n\nKernel work needs explicit memory control.');
});

test('historyTransportContent compacts long assistant history', () => {
  const { historyTransportContent } = loadChatUiApi();
  const longAnswer = 'Kernel programming in Zig gives you explicit control. '.repeat(40);
  const transport = historyTransportContent(longAnswer, '', false, true);

  expect(transport.startsWith('<think>\n\n</think>\n\nKernel programming in Zig')).toBe(true);
  expect(transport.length).toBeLessThan(longAnswer.length);
});

test('chatMaxTokens gives thinking enough room to finish', () => {
  const { chatMaxTokens } = loadChatUiApi();

  expect(chatMaxTokens(false)).toBe(2048);
  expect(chatMaxTokens(true)).toBe(16384);
});

test('reasoningIsLowSignal flags trivial meta reasoning', () => {
  const { reasoningIsLowSignal } = loadChatUiApi();

  expect(reasoningIsLowSignal('I need to complete the response.')).toBe(true);
  expect(
    reasoningIsLowSignal(
      'I need to determine if Zig is a programming language or framework relevant to kernel programming, then answer accordingly.',
    ),
  ).toBe(true);
  expect(reasoningIsLowSignal('1. Check the pointer math\n2. Compare against the C ABI')).toBe(
    false,
  );
});

test('reasoningDuplicatesAnswer flags repeated answer content', () => {
  const { reasoningDuplicatesAnswer } = loadChatUiApi();
  const text =
    'Whether Zig is better than C depends on your priorities. Here is a comparison of memory safety, tooling, and portability.';

  expect(reasoningDuplicatesAnswer(text, text)).toBe(true);
  expect(reasoningDuplicatesAnswer(text, 'Completely different content')).toBe(false);
});

test('normalizeDisplayView keeps short reasoning inspectable with the answer', () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const view = normalizeDisplayView({
    reasoning: 'I need to complete the response.',
    answer: 'Kernel programming in Zig is possible.',
    showReasoning: true,
    isThinking: false,
  });

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toBe('I need to complete the response.');
  expect(view.answer).toBe('Kernel programming in Zig is possible.');
  expect(view.suppressedReasoning).not.toBe(true);
});

test('normalizeDisplayView keeps duplicate reasoning in its disclosure', () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const text =
    'Whether Zig is better than C depends on your priorities. Here is a comparison of memory safety, tooling, and portability.';
  const view = normalizeDisplayView({
    reasoning: text,
    answer: text,
    showReasoning: true,
    isThinking: false,
  });

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toBe(text);
  expect(view.answer).toBe(text);
});

test('normalizeDisplayView hides empty completed reasoning blocks', () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const view = normalizeDisplayView({
    reasoning: '',
    answer: 'Kernel programming in Zig is possible.',
    showReasoning: true,
    isThinking: false,
  });

  expect(view.showReasoning).toBe(false);
  expect(view.answer).toBe('Kernel programming in Zig is possible.');
  expect(view.suppressedReasoning).toBe(true);
});

test('normalizeDisplayView keeps short live reasoning visible while thinking is active', () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const view = normalizeDisplayView({
    reasoning: 'I need to complete the response.',
    answer: '',
    showReasoning: true,
    isThinking: true,
  });

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toBe('I need to complete the response.');
  expect(view.suppressedReasoning).not.toBe(true);
});

test('normalizeDisplayView keeps substantive live reasoning visible', () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const view = normalizeDisplayView({
    reasoning: '1. Check the pointer math\n2. Compare against the C ABI',
    answer: '',
    showReasoning: true,
    isThinking: true,
  });

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toContain('Check the pointer math');
  expect(view.suppressedReasoning).toBeUndefined();
});

test('normalizeDisplayView keeps answer-like live thinking prose visible', () => {
  const { normalizeDisplayView, assistantHtmlFromRaw } = loadChatUiApi();
  const raw =
    '<think>Zig is a modern systems programming language known for its simplicity, safety, and performance. It has gained attention for low-level systems work, including kernel development. However, there are several tradeoffs to consider before using it for production kernels.';
  const view = normalizeDisplayView({
    reasoning:
      'Zig is a modern systems programming language known for its simplicity, safety, and performance. It has gained attention for low-level systems work, including kernel development. However, there are several tradeoffs to consider before using it for production kernels.',
    answer: '',
    showReasoning: true,
    isThinking: true,
  });
  const rendered = assistantHtmlFromRaw(raw, true);

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toContain('Zig is a modern systems programming language');
  expect(rendered.html).toContain('reasoning-block');
  expect(rendered.html).toContain('Zig is a modern systems programming language');
});

test('normalizeDisplayView preserves the final answer separately from answer-like reasoning', () => {
  const { normalizeDisplayView } = loadChatUiApi();
  const view = normalizeDisplayView({
    reasoning:
      'Zig is a modern systems programming language known for its simplicity, safety, and performance. It has gained attention for low-level systems work, including kernel development. However, there are several tradeoffs to consider before using it for production kernels.',
    answer:
      'Zig is increasingly being considered for kernel programming, but it is not yet as mature or widely adopted as C.',
    showReasoning: true,
    isThinking: false,
  });

  expect(view.showReasoning).toBe(true);
  expect(view.reasoning).toContain('simplicity, safety, and performance');
  expect(view.answer).toBe(
    'Zig is increasingly being considered for kernel programming, but it is not yet as mature or widely adopted as C.',
  );
});

test('assistantHtmlFromRaw retains short live reasoning inside its disclosure', () => {
  const { assistantHtmlFromRaw } = loadChatUiApi();
  const rendered = assistantHtmlFromRaw('<think>I need to complete the response.', true);

  expect(rendered.view.showReasoning).toBe(true);
  expect(rendered.html).toContain('reasoning-block');
  expect(rendered.html).toContain('I need to complete the response.');
});

test('assistantLiveHtmlFromRaw formats lists and incomplete code fences while streaming', () => {
  const { assistantLiveHtmlFromRaw } = loadChatUiApi();
  const rendered = assistantLiveHtmlFromRaw('1. **Fast**\n```zig\nconst x = 1;', false);

  expect(rendered.html).toContain('class="stream-markdown"');
  expect(rendered.html).toContain('<ol><li><strong>Fast</strong></li></ol>');
  expect(rendered.html).toContain('class="code-block"');
  expect(rendered.html).toContain('<code class="language-zig">const x = 1;</code>');
  expect(rendered.html).not.toContain('```zig');
});

test('assistantLiveHtmlFromRaw formats live reasoning inside its disclosure', () => {
  const { assistantLiveHtmlFromRaw } = loadChatUiApi();
  const rendered = assistantLiveHtmlFromRaw('<think>1. **Analyze**\n- item', true);

  expect(rendered.html).toContain('class="reasoning-block"');
  expect(rendered.html).toContain('class="stream-markdown"');
  expect(rendered.html).toContain('<strong>Analyze</strong>');
  expect(rendered.html).toContain('<ul><li>item</li></ul>');
});

test("assistantLiveHtmlFromRaw formats the screenshot's heading and emphasis before completion", () => {
  const { assistantLiveHtmlFromRaw } = loadChatUiApi();
  const rendered = assistantLiveHtmlFromRaw(
    '<think>Consider the language features.</think>\n# Zig Programming Language\n\nZig is designed to be **simple, fast, safe**, and **zero-cost**.\n\n## Core Philosophy\n\n- **Simplicity** — no hidden control flow.',
    false,
  );
  expect(rendered.html).toContain('<h1>Zig Programming Language</h1>');
  expect(rendered.html).toContain('<strong>simple, fast, safe</strong>');
  expect(rendered.html).toContain('<h2>Core Philosophy</h2>');
  expect(rendered.html).toContain('<li><strong>Simplicity</strong> — no hidden control flow.</li>');
  expect(rendered.html).not.toContain('# Zig Programming Language');
  expect(rendered.html).not.toContain('**simple, fast, safe**');
});

test('completionNotice surfaces token-limit hits clearly', () => {
  const { completionNotice } = loadChatUiApi();
  const notice = completionNotice({
    hasAnswer: true,
    finishReason: 'length',
    errorMessage: '',
    missingFinishReason: false,
    showReasoning: false,
    hadOutput: true,
  });

  expect(notice.kind).toBe('warn');
  expect(notice.title).toBe('Token limit reached');
  expect(notice.message).toContain('Ask to continue');
});

test('completionNotice reports interrupted streams with partial answers', () => {
  const { completionNotice } = loadChatUiApi();
  const notice = completionNotice({
    hasAnswer: true,
    finishReason: 'stop',
    errorMessage: '',
    missingFinishReason: true,
    showReasoning: false,
    hadOutput: true,
  });

  expect(notice.kind).toBe('warn');
  expect(notice.title).toBe('Response interrupted');
  expect(notice.message).toContain('clean finish');
});

test('completionNotice reports backend errors clearly', () => {
  const { completionNotice } = loadChatUiApi();
  const notice = completionNotice({
    hasAnswer: false,
    finishReason: 'stop',
    errorMessage: 'socket closed',
    missingFinishReason: false,
    showReasoning: false,
    hadOutput: false,
  });

  expect(notice.kind).toBe('error');
  expect(notice.title).toBe('Request failed');
  expect(notice.message).toBe('socket closed');
});

test('completionNotice reports suppressed reasoning without a final answer', () => {
  const { completionNotice } = loadChatUiApi();
  const notice = completionNotice({
    hasAnswer: false,
    finishReason: 'stop',
    errorMessage: '',
    missingFinishReason: false,
    showReasoning: false,
    suppressedReasoning: true,
    hadOutput: true,
  });

  expect(notice?.kind).toBe('warn');
  expect(notice?.title).toBe('No final answer');
  expect(notice?.message).toContain('trivial reasoning stub');
});

test('requestMessagesFromHistory excludes partial assistant entries', () => {
  const chatHtml = readFileSync(resolve(import.meta.dir, '../src/server/chat.html'), 'utf8');
  const match = chatHtml.match(/<script>([\s\S]*)<\/script>\s*<\/body>/);
  if (!match) throw new Error('Chat UI script not found');

  const factory = new Function(
    'document',
    'fetch',
    'location',
    'performance',
    'navigator',
    'hljs',
    'alert',
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
      getElementById() {
        return makeElement();
      },
      createElement() {
        return makeElement();
      },
      querySelectorAll() {
        return [];
      },
    },
    async () => {
      throw new Error('offline');
    },
    { origin: 'http://localhost:8080' },
    { now: () => 0 },
    { clipboard: { writeText: async () => {} } },
    { highlightElement() {} },
    () => {},
  ) as Array<{ role: string; content: string }>;

  expect(messages).toEqual([
    { role: 'user', content: 'hello' },
    { role: 'assistant', content: 'final answer' },
  ]);
});

test('requestMessagesFromHistory prefers transport content for assistant history', () => {
  const chatHtml = readFileSync(resolve(import.meta.dir, '../src/server/chat.html'), 'utf8');
  const match = chatHtml.match(/<script>([\s\S]*)<\/script>\s*<\/body>/);
  if (!match) throw new Error('Chat UI script not found');

  const factory = new Function(
    'document',
    'fetch',
    'location',
    'performance',
    'navigator',
    'hljs',
    'alert',
    `${match[1]}
H = [
  { role: 'user', content: 'hello' },
  { role: 'assistant', content: 'clean answer', transport: '<think>\\n\\n</think>\\n\\nclean answer' },
];
return requestMessagesFromHistory(true);`,
  );

  const messages = factory(
    {
      getElementById() {
        return makeElement();
      },
      createElement() {
        return makeElement();
      },
      querySelectorAll() {
        return [];
      },
    },
    async () => {
      throw new Error('offline');
    },
    { origin: 'http://localhost:8080' },
    { now: () => 0 },
    { clipboard: { writeText: async () => {} } },
    { highlightElement() {} },
    () => {},
  ) as Array<{ role: string; content: string }>;

  expect(messages).toEqual([
    { role: 'user', content: 'hello' },
    { role: 'assistant', content: '<think>\n\n</think>\n\nclean answer' },
  ]);
});

test('requestMessagesFromHistory excludes incomplete assistant entries', () => {
  const chatHtml = readFileSync(resolve(import.meta.dir, '../src/server/chat.html'), 'utf8');
  const match = chatHtml.match(/<script>([\s\S]*)<\/script>\s*<\/body>/);
  if (!match) throw new Error('Chat UI script not found');

  const factory = new Function(
    'document',
    'fetch',
    'location',
    'performance',
    'navigator',
    'hljs',
    'alert',
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
      getElementById() {
        return makeElement();
      },
      createElement() {
        return makeElement();
      },
      querySelectorAll() {
        return [];
      },
    },
    async () => {
      throw new Error('offline');
    },
    { origin: 'http://localhost:8080' },
    { now: () => 0 },
    { clipboard: { writeText: async () => {} } },
    { highlightElement() {} },
    () => {},
  ) as Array<{ role: string; content: string }>;

  expect(messages).toEqual([
    { role: 'user', content: 'hello' },
    { role: 'assistant', content: 'final answer' },
  ]);
});

test('findUnexpectedThinkingTailStart detects reopened think after initial thinking block', () => {
  const { findUnexpectedThinkingTailStart } = loadChatUiApi();

  // Reopened <think> after a full <think>...</think> block and answer — should detect it
  const raw = '<think>\nReasoning.\n</think>\nZig is promising.\n<think>\nMore thinking';
  expect(findUnexpectedThinkingTailStart(raw)).toBeGreaterThan(0);

  // Still inside initial <think> block (no </think> yet) — no detection
  expect(findUnexpectedThinkingTailStart('<think>\nStill thinking...')).toBe(-1);

  // Normal completed thinking — no reopened block
  expect(findUnexpectedThinkingTailStart('<think>\nReasoning.\n</think>\nAnswer here.')).toBe(-1);

  // No <think> at all
  expect(findUnexpectedThinkingTailStart('Just a plain answer.')).toBe(-1);

  // Reopened <think> without initial thinking block (original behavior)
  expect(findUnexpectedThinkingTailStart('Some answer.<think>\nNew thinking')).toBeGreaterThan(0);
});

test('reasoningDuplicatesAnswer detects streaming prefix duplicates early', () => {
  const { reasoningDuplicatesAnswer } = loadChatUiApi();
  const reasoning =
    'Zig is a modern systems programming language known for its simplicity, safety, and performance. It has been gaining attention in the systems programming community.';

  // Short streaming answer (40+ chars) that matches start of reasoning — should detect
  const shortPrefix = 'Zig is a modern systems programming language known';
  expect(reasoningDuplicatesAnswer(reasoning, shortPrefix)).toBe(true);

  // Too short (< 40 chars) — should not detect
  expect(reasoningDuplicatesAnswer(reasoning, 'Zig is a modern')).toBe(false);

  // Different content — should not detect
  expect(
    reasoningDuplicatesAnswer(
      reasoning,
      'Python is a great language for data science and ML work.',
    ),
  ).toBe(false);
});

test('assistantHtmlFromRaw keeps reasoning when the answer begins with the same words', () => {
  const { assistantHtmlFromRaw } = loadChatUiApi();

  // Simulate streaming state: reasoning visible, answer starts matching
  const raw =
    '<think>\nZig is a modern systems programming language known for its simplicity, safety, and performance.\n</think>\nZig is a modern systems programming language known for its simplicity';
  const result = assistantHtmlFromRaw(raw, false);

  expect(result.view.showReasoning).toBe(true);
  expect(result.view.reasoning).toContain('simplicity, safety, and performance');
  expect(result.view.answer.length).toBeGreaterThan(0);
  expect(result.html).toContain('reasoning-block');
});

test('assistantHtmlFromRaw promotes reasoning to answer when model puts everything in think tags', () => {
  const { assistantHtmlFromRaw } = loadChatUiApi();

  // Model put entire answer in <think> with no content after </think>
  const raw =
    '<think>\nZig is a modern systems programming language known for its simplicity, safety, and performance. It has gained attention for kernel development due to its low-level control.\n</think>\n';
  const result = assistantHtmlFromRaw(raw, false);

  // Keep a usable answer fallback without discarding the reasoning disclosure.
  expect(result.view.answer.length).toBeGreaterThan(0);
  expect(result.view.answer).toContain('Zig is a modern');
  expect(result.view.showReasoning).toBe(true);
  expect(result.html).toContain('reasoning-block');
});

test('reasoning remains inspectable from live tokens through the completed answer', () => {
  const { assistantLiveHtmlFromRaw, assistantHtmlFromRaw } = loadChatUiApi();
  const cases = [
    { reasoning: 'Let me think.', answer: 'The answer is 408.' },
    { reasoning: 'The answer is 408.', answer: 'The answer is 408.' },
    {
      reasoning: '1. Multiply 17 by 20.\n2. Multiply 17 by 4.\n3. Add both results.',
      answer: '408',
    },
    {
      reasoning:
        'Zig is a modern systems programming language with explicit allocation and compile-time execution. It supports low-level code with predictable control over memory and runtime behavior.',
      answer: 'Use Zig when explicit control is useful.',
    },
  ];
  for (const { reasoning, answer } of cases) {
    const live = assistantLiveHtmlFromRaw('<think>' + reasoning, true);
    expect(live.view.showReasoning).toBe(true);
    expect(live.view.reasoning).toBe(reasoning);
    const complete = assistantHtmlFromRaw('<think>' + reasoning + '</think>\n' + answer, false);
    expect(complete.view.showReasoning).toBe(true);
    expect(complete.view.reasoning).toBe(reasoning);
    expect(complete.view.answer).toBe(answer);
    expect(complete.html).toContain('class="reasoning-block"');
  }
});

test('implicit closing-tag reasoning survives normalization at completion', () => {
  const { assistantHtmlFromRaw } = loadChatUiApi();
  const reasoning = 'I need to compare the values before giving the result.';
  const result = assistantHtmlFromRaw(reasoning + '</think>\nThe result is 408.', false);
  expect(result.view.reasoning).toBe(reasoning);
  expect(result.view.showReasoning).toBe(true);
  expect(result.view.answer).toBe('The result is 408.');
  expect(result.html).toContain('class="reasoning-block"');
});

test('answer cleanup does not truncate planning paragraphs inside completed reasoning', () => {
  const { assistantLiveHtmlFromRaw, assistantHtmlFromRaw } = loadChatUiApi();
  const reasoning =
    'Check the allocation sizes and compare each pointer offset against the required alignment before deciding.\n\nThe user is asking whether the memory layout is safe.\n\nI need to verify the final offset and then explain the result.';
  const live = assistantLiveHtmlFromRaw('<think>' + reasoning, true);
  expect(live.view.reasoning).toBe(reasoning);
  const completed = assistantHtmlFromRaw(
    '<think>' + reasoning + '</think>\nThe allocation layout is aligned correctly.',
    false,
  );
  expect(completed.view.showReasoning).toBe(true);
  expect(completed.view.reasoning).toBe(reasoning);
  expect(completed.view.answer).toBe('The allocation layout is aligned correctly.');
  expect(completed.html).toContain('The user is asking whether the memory layout is safe.');
  expect(completed.html).toContain('I need to verify the final offset');
});

test('answer cleanup preserves implicit closing-tag reasoning that begins with planning', () => {
  const { assistantHtmlFromRaw } = loadChatUiApi();
  const result = assistantHtmlFromRaw('I need to answer directly.\n</think>\n408', false);
  expect(result.view.showReasoning).toBe(true);
  expect(result.view.reasoning).toBe('I need to answer directly.');
  expect(result.view.answer).toBe('408');
  expect(result.html).toContain('I need to answer directly.');
});

test('hasRepeatedPhraseLoop detects sentence-level repetition', () => {
  const { hasRepeatedPhraseLoop } = loadChatUiApi();
  expect(
    hasRepeatedPhraseLoop(
      'I should also mention type safety. I should also mention type safety. I should also mention type safety. I should also mention type safety.',
    ),
  ).toBe(true);
  expect(
    hasRepeatedPhraseLoop(
      'Zig is a systems programming language. It features manual memory management. It compiles to native code.',
    ),
  ).toBe(false);
});

test('startsWithLeakedReasoning detects meta-commentary', () => {
  const { startsWithLeakedReasoning } = loadChatUiApi();
  expect(startsWithLeakedReasoning('The user is asking about C types.')).toBe(true);
  expect(startsWithLeakedReasoning('I need to provide a clear explanation.')).toBe(true);
  expect(startsWithLeakedReasoning('Zig is a modern systems programming language.')).toBe(false);
});

test("chat UI markdown formats the screenshot's heading, bold prose, and bullets", () => {
  const { md } = loadChatUiApi();
  const html = md(
    '# Zig Programming Language\n\nZig is designed to be **simple, fast, safe**, and **zero-cost**.\n\n## Core Philosophy\n\n- **Simplicity** — no hidden control flow.\n- **Composable** — cross-compilation support.',
  );
  expect(html).toContain('<h1>Zig Programming Language</h1>');
  expect(html).toContain('<strong>simple, fast, safe</strong>');
  expect(html).toContain('<h2>Core Philosophy</h2>');
  expect(html).toContain('<li><strong>Simplicity</strong> — no hidden control flow.</li>');
});

test('chat UI markdown renders all heading levels, rules, and blockquotes', () => {
  const { md } = loadChatUiApi();
  const html = md(
    '##### Small heading\n###### Smallest heading\n\n---\n\n> **A useful note**\n> With a second line.',
  );
  expect(html).toContain('<h5>Small heading</h5>');
  expect(html).toContain('<h6>Smallest heading</h6>');
  expect(html).toContain('<hr>');
  expect(html).toContain(
    '<blockquote><p><strong>A useful note</strong><br>With a second line.</p></blockquote>',
  );
});

test('chat UI markdown renders aligned tables with inline code and escaped pipes', () => {
  const { md } = loadChatUiApi();
  const html = md(
    '| Model | Speed | Note |\n| :--- | ---: | :---: |\n| **ZINC** | 12 | `a|b` |\n| Other | 3 | a\\|b |',
  );
  expect(html).toContain('<div class="table-scroll"><table><thead>');
  expect(html).toContain('<th style="text-align:left">Model</th>');
  expect(html).toContain('<th style="text-align:right">Speed</th>');
  expect(html).toContain('<td style="text-align:center"><code>a|b</code></td>');
  expect(html).toContain('<td style="text-align:center">a|b</td>');
  expect(html).toContain('<strong>ZINC</strong>');
});

test('chat UI markdown keeps code fences outside paragraphs and recognizes language names', () => {
  const { md } = loadChatUiApi();
  const html = md(
    'Before\n```c++\n#include <vector>\n// **literal**\n```\nAfter\n\n~~~objective-c\n[object call];\n~~~',
  );
  expect(html).toContain('<p>Before</p><div class="code-block">');
  expect(html).toContain(
    '<code class="language-c++">#include &lt;vector&gt;\n// **literal**\n</code>',
  );
  expect(html).toContain('</div><p>After</p>');
  expect(html).toContain('<code class="language-objective-c">');
  expect(html).not.toContain('<strong>literal</strong>');
});

test('chat UI markdown safely handles incomplete and unsafe fence info', () => {
  const { md } = loadChatUiApi();
  const html = md('```js"onload="alert(1)\n<img src=x>');
  expect(html).toContain('class="language-plaintext"');
  expect(html).toContain('&lt;img src=x&gt;');
  expect(html).not.toContain('class="language-js"onload=');
  expect(html).not.toContain('class="copy-btn"');
});

test('chat UI markdown protects code, escape sequences, and link targets from emphasis', () => {
  const { md } = loadChatUiApi();
  const html = md(
    '`**literal**` and \\*literal\\* plus [`api_name`](https://example.test/api_name) and ~~old~~.',
  );
  expect(html).toContain('<code>**literal**</code>');
  expect(html).toContain('and *literal* plus');
  expect(html).toContain(
    '<a href="https://example.test/api_name" target="_blank" rel="noopener noreferrer"><code>api_name</code></a>',
  );
  expect(html).toContain('<del>old</del>');
});

test('chat UI markdown escapes quoted link targets and leaves unsafe schemes inert', () => {
  const { md } = loadChatUiApi();
  const html = md('[safe](https://example.test/"onmouseover="bad) [unsafe](javascript:alert)');
  expect(html).toContain('href="https://example.test/&quot;onmouseover=&quot;bad"');
  expect(html).not.toContain('href="javascript:');
  expect(html).not.toContain('"onmouseover="bad');
});
