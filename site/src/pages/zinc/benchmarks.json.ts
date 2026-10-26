import benchmarkData from '../../data/zinc-performance.json';

export function GET() {
  return new Response(JSON.stringify(benchmarkData, null, 2), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
