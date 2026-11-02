import { expect, test } from 'bun:test';
import { protectCodeHtml } from './code-html.mjs';

test('rendered package commands retain their exact escaped code and copy text', async () => {
  const code =
    'bun add --global --trust @opencode/cli@2.0.21\n./zinc --prompt &quot;東京 &amp; 🚀&quot;';
  const source = `<pre><code class="language-bash">${code}</code></pre>`;
  const result = await protectCodeHtml(source);
  expect(result.codeCount).toBe(1);
  expect(result.html).toBe(
    `<pre><code class="language-bash"><!--email_off-->${code}<!--/email_off--></code></pre>`,
  );
  // Comments add no DOM text, so display/copy retain the original command.
  expect(result.html.replace(/<!--\/?email_off-->/g, '')).toBe(source);
});

test('protects inline and block code while leaving real contact links outside code alone', async () => {
  const contact = '<a href="mailto:contact@example.com">contact@example.com</a>';
  const result = await protectCodeHtml(
    `${contact}<code>user@host.example</code><pre><code>opencode-ai@1.19.4</code></pre>`,
  );
  expect(result.codeCount).toBe(2);
  expect(result.html).toStartWith(contact);
  expect(result.html).toContain('<code><!--email_off-->user@host.example<!--/email_off--></code>');
  expect(result.html).toContain('<code><!--email_off-->opencode-ai@1.19.4<!--/email_off--></code>');
});

test('HTML parser preserves code-like strings in scripts and unrelated page markup', async () => {
  const source =
    '<!DOCTYPE html><p title="東京">Hello</p><script>const example = "<code>not an element</code>";</script>';
  expect(await protectCodeHtml(source)).toEqual({ html: source, codeCount: 0 });
});
