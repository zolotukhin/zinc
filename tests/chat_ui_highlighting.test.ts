import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

function element(language: string, text: string, liveTail = false) {
  const classes: string[] = [];
  return {
    className: "language-" + language,
    textContent: text,
    innerHTML: "",
    dataset: {} as Record<string, string>,
    classes,
    classList: { add(name: string) { classes.push(name); } },
    closest(selector: string) { return liveTail && [".markdown-tail", ".streaming"].includes(selector) ? {} : null; },
  };
}

function harness(blocks: ReturnType<typeof element>[] = []) {
  const html = readFileSync(new URL("../src/server/chat.html", import.meta.url), "utf8");
  const start = html.indexOf("function highlightZigCode(");
  const end = html.indexOf("TH.addEventListener('change'", start);
  const scripts: any[] = [];
  const document = {
    querySelectorAll: () => blocks,
    head: { appendChild: (script: any) => scripts.push(script) },
    createElement: () => ({ src: "", async: false, onload: null, onerror: null }),
  };
  const factory = new Function("document", "hljs", "esc", `${html.slice(start, end)}
    return {highlightZigCode, highlightAll, setHighlighter(value){hljs=value}};`);
  const api = factory(document, undefined, (text: string) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"));
  return { api, scripts, blocks };
}

test("Zig code colors real tokens locally without requiring a CDN", () => {
  const source = 'const std = @import("std");\npub fn main() !void {\n    const n: u32 = 42;\n    const enabled = true;\n    return;\n}\n';
  const block = element("zig", source);
  const { api, scripts } = harness([block]);
  api.highlightAll();
  expect(block.innerHTML).toContain('<span class="hljs-keyword">const</span>');
  expect(block.innerHTML).toContain('<span class="hljs-built_in">@import</span>');
  expect(block.innerHTML).toContain('<span class="hljs-string">"std"</span>');
  expect(block.innerHTML).toContain('<span class="hljs-title function_">main</span>');
  expect(block.innerHTML).toContain('<span class="hljs-type">u32</span>');
  expect(block.innerHTML).toContain('<span class="hljs-number">42</span>');
  expect(block.innerHTML).toContain('<span class="hljs-literal">true</span>');
  expect(block.dataset.highlighted).toBe("yes");
  expect(block.classes).toContain("hljs");
  expect(scripts).toHaveLength(0);
});

test("Zig comments and strings protect embedded keyword-like text", () => {
  const { api } = harness();
  const html = api.highlightZigCode('// const fn @import\n"fn @import //"\n\\\\const multiline\n\'x\'');
  expect(html).toContain('<span class="hljs-comment">// const fn @import</span>');
  expect(html).toContain('<span class="hljs-string">"fn @import //"</span>');
  expect(html).toContain('<span class="hljs-string">\\\\const multiline</span>');
  expect(html).toContain('<span class="hljs-string">\'x\'</span>');
  expect(html).not.toContain('hljs-keyword');
});

test("Zig numeric formats, enum literals, and source HTML remain safe", () => {
  const { api } = harness();
  const html = api.highlightZigCode('0xff 0b1010 0o755 1_024 1.5e-2 0x1.fp3 .ready <script>& "<b>"');
  for (const token of ["0xff", "0b1010", "0o755", "1_024", "1.5e-2", "0x1.fp3"]) {
    expect(html).toContain('<span class="hljs-number">' + token + '</span>');
  }
  expect(html).toContain('<span class="hljs-symbol">.ready</span>');
  expect(html).toContain('&lt;script&gt;&amp;');
  expect(html).toContain('<span class="hljs-string">"&lt;b&gt;"</span>');
  expect(html).not.toContain('<script>');
});

test("optional common-language loading coalesces requests and onload highlights all pending scopes", () => {
  const first = element("javascript", "const first = 1;"), second = element("python", "print(2)");
  const { api, scripts } = harness([first, second]);
  api.highlightAll({ querySelectorAll: () => [first] });
  api.highlightAll({ querySelectorAll: () => [second] });
  expect(scripts).toHaveLength(1);
  expect(scripts[0].async).toBe(true);
  const highlighted: any[] = [];
  api.setHighlighter({ getLanguage: () => true, highlightElement: (block: any) => { highlighted.push(block); block.dataset.highlighted = "yes"; } });
  scripts[0].onload({ type: "load" });
  expect(highlighted).toEqual([first, second]);
  api.highlightAll();
  expect(highlighted).toHaveLength(2);
});

test("offline library failure never loops or prevents Zig highlighting", () => {
  const python = element("python", "print(2)"), zig = element("zig", "const x = 2;");
  const { api, scripts, blocks } = harness([python]);
  api.highlightAll();
  scripts[0].onerror();
  blocks.push(zig);
  api.highlightAll();
  api.highlightAll();
  expect(scripts).toHaveLength(1);
  expect(zig.innerHTML).toContain('hljs-keyword');
  expect(python.dataset.highlighted).toBeUndefined();
});

test("unknown languages remain plain and active streamed fences wait until committed", () => {
  const unknown = element("unknown-language", "unknown code"), zig = element("zig", "const x = 2;", true);
  const { api, scripts } = harness([unknown, zig]);
  let attempts = 0;
  api.setHighlighter({ getLanguage: () => false, highlightElement: () => { attempts++; } });
  api.highlightAll();
  expect(attempts).toBe(0);
  expect(zig.dataset.highlighted).toBeUndefined();
  expect(scripts).toHaveLength(0);
});
