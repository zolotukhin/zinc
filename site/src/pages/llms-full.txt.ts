import type { APIRoute } from 'astro';
import { loadDocs } from '../lib/docs-loader';
import { renderLlmsFullTxt } from '../lib/agent-docs';
import { loadZigApi } from '../lib/zig-api-loader';

export const GET: APIRoute = async ({ site }) => {
  const [zigApi, docs] = await Promise.all([
    loadZigApi(),
    loadDocs(),
  ]);
  const siteUrl = site?.toString() ?? 'https://zolotukhin.ai';

  return new Response(renderLlmsFullTxt(zigApi, docs, siteUrl), {
    headers: {
      'content-type': 'text/plain; charset=utf-8',
      'cache-control': 'public, max-age=0, must-revalidate',
    },
  });
};
