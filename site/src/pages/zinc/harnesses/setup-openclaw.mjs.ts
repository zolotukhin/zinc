import source from '../../../../../tools/setup_openclaw.mjs?raw';

export const prerender = true;

export function GET() {
  return new Response(source, {
    headers: {
      'Content-Type': 'text/javascript; charset=utf-8',
      'Content-Disposition': 'attachment; filename="setup-openclaw.mjs"',
    },
  });
}
