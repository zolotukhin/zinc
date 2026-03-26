import type { APIRoute } from 'astro';
import { loadZigApi, renderZigApiAgentText } from '../../../lib/zig-api-loader';

export const GET: APIRoute = async ({ site }) => {
  const zigApi = await loadZigApi();
  const siteUrl = site?.toString() ?? 'https://zolotukhin.ai';

  return new Response(renderZigApiAgentText(zigApi, siteUrl), {
    headers: {
      'content-type': 'text/plain; charset=utf-8',
      'cache-control': 'public, max-age=0, must-revalidate',
    },
  });
};
