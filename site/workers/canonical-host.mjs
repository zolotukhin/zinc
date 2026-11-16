// This Worker owns the www.zolotukhin.ai custom domain. Pages serves the apex.
export default {
  fetch(request) {
    const url = new URL(request.url);
    if (url.hostname !== 'www.zolotukhin.ai') {
      return new Response('Not found', { status: 404 });
    }
    url.protocol = 'https:';
    url.hostname = 'zolotukhin.ai';
    url.port = '';
    return Response.redirect(url.href, 301);
  },
};
