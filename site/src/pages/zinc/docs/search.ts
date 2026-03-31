import type { APIRoute } from 'astro';
import { loadDocs } from '../../../lib/docs-loader';
import { buildDocsSearchIndex } from '../../../lib/docs-search';
import { loadZigApi } from '../../../lib/zig-api-loader';

export const GET: APIRoute = async () => {
  const [docs, zigApi] = await Promise.all([
    loadDocs(),
    loadZigApi(),
  ]);

  const entries = buildDocsSearchIndex(docs, zigApi);

  return new Response(JSON.stringify({ generated_at: new Date().toISOString(), entries }, null, 2), {
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'public, max-age=0, must-revalidate',
    },
  });
};
