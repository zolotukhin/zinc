import net from 'node:net';

const PRIVATE_FIELDS =
  /^(?:hostname|host_name|ssh_host|ssh_port|machine_id|gpu_uuid|device_uuid|serial_number|private_key|api_key|access_token)$/i;

export function assertPublicArtifact(value, env = process.env) {
  const text = JSON.stringify(value).replace(/%2f|%5c/gi, (encoded) => decodeURIComponent(encoded));
  let blocked = false;
  function visit(entry) {
    if (!entry || typeof entry !== 'object') return;
    for (const [key, child] of Object.entries(entry)) {
      if (PRIVATE_FIELDS.test(key)) blocked = true;
      visit(child);
    }
  }
  visit(value);
  blocked ||=
    /(?:\/Users\/|\/home\/|[A-Za-z]:\\+Users\\+)[^/\s"\\]+|GPU-[a-f\d]{8}|\b[\w.-]+\.ts\.net\b/i.test(
      text,
    );
  for (const candidate of text.match(/\b(?:f[cd][a-f\d]{2}|fe[89ab][a-f\d]):[a-f\d:]+/gi) ?? []) {
    blocked ||= net.isIP(candidate) === 6;
  }
  for (const candidate of text.match(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g) ?? []) {
    if (net.isIP(candidate) !== 4) continue;
    const [a, b] = candidate.split('.').map(Number);
    blocked ||=
      a === 10 ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 168) ||
      (a === 100 && b >= 64 && b <= 127) ||
      (a === 169 && b === 254);
  }
  for (const [key, host] of Object.entries(env)) {
    if (!/(?:^|_)HOST$/.test(key) || !host || ['localhost', '127.0.0.1', '0.0.0.0'].includes(host))
      continue;
    const escaped = host.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    blocked ||= new RegExp(`(?<![\\w.-])${escaped}(?![\\w.-])`, 'i').test(text);
  }
  if (blocked) {
    // Do not repeat the offending value: CI output can itself become public.
    throw new Error(
      'Refusing to publish artifact: private machine information detected. Sanitize it before publishing.',
    );
  }
}
