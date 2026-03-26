import type { APIRoute } from 'astro';
import { createZigApiAgentPayload, loadZigApi } from '../../../lib/zig-api-loader';

export const GET: APIRoute = async ({ site }) => {
  const zigApi = await loadZigApi();
  const siteUrl = site?.toString() ?? 'https://zolotukhin.ai';
  const payload = createZigApiAgentPayload(zigApi, siteUrl);

  return new Response(JSON.stringify(payload, null, 2), {
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'public, max-age=0, must-revalidate',
    },
  });
};
