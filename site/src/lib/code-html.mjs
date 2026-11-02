// Cloudflare's documented exclusion markers keep package pins and literal
// addresses in code from becoming email-protection links. Run after rendering
// so Astro's HTML compression cannot discard the markers.
// https://developers.cloudflare.com/waf/tools/scrape-shield/email-address-obfuscation/
export async function protectCodeHtml(html) {
  let codeCount = 0;
  const response = new HTMLRewriter()
    .on('code', {
      element(element) {
        codeCount += 1;
        element.prepend('<!--email_off-->', { html: true });
        element.append('<!--/email_off-->', { html: true });
      },
    })
    .transform(new Response(html, { headers: { 'content-type': 'text/html; charset=utf-8' } }));
  return { html: await response.text(), codeCount };
}
