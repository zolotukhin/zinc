// Cloudflare's documented exclusion markers keep package pins and literal
// addresses in code from becoming email-protection links. Run after rendering
// so Astro's HTML compression cannot discard the markers.
// https://developers.cloudflare.com/waf/tools/scrape-shield/email-address-obfuscation/
export async function protectCodeHtml(html) {
  let codeCount = 0;
  let codeDepth = 0;
  const response = new HTMLRewriter()
    .on('code', {
      element(element) {
        const nested = codeDepth > 0;
        codeDepth += 1;
        element.onEndTag(() => {
          codeDepth -= 1;
        });
        if (nested) return;
        codeCount += 1;
        element.prepend('<!--email_off-->', { html: true });
        element.append('<!--/email_off-->', { html: true });
      },
    })
    .transform(new Response(html, { headers: { 'content-type': 'text/html; charset=utf-8' } }));
  const protectedHtml = await response.text();
  const withoutMarkers = (value) => value.replace(/<!--\/?email_off-->/g, '');
  // Enforce byte preservation on every built page, beyond the unit fixtures.
  if (withoutMarkers(protectedHtml) !== withoutMarkers(html)) {
    throw new Error('Code protection changed HTML beyond its exclusion comments');
  }
  return { html: protectedHtml, codeCount };
}
