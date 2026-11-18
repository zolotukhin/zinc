import { expect, test } from "bun:test";
import { parseArgs, readStreamingCompletion, runVerification, validateMtp } from "./verify_mtp.mjs";

const options = (extra: Record<string, unknown> = {}) => ({ base: "http://127.0.0.1:9090", model: null, maxTokens: 32, timeoutMs: 1000, strictParity: false, ...extra });

function fixture({ supported = true, defaultEnabled = true, corrupt = "", divergent = false, legacy = false } = {}) {
  const requests: any[] = [];
  const capability = { supports_mtp: supported, mtp_default_enabled: defaultEnabled && supported, mtp_request_control: true, mtp_unavailable_reason: supported ? "" : "No compatible NextN weights" };
  return {
    requests,
    fetch: async (url: string, init: any = {}) => {
      if (url.endsWith("/health")) return Response.json({ status: "ok", backend: "rocm", ...(legacy ? {} : capability) });
      if (url.endsWith("/v1/models")) return Response.json({ data: [{ id: "qwen-test", active: true, ...(legacy ? {} : capability) }] });
      const body = JSON.parse(init.body);
      requests.push(body);
      if (!supported && body.enable_mtp === true) return Response.json({ error: { message: "MTP unavailable" } }, { status: 400 });
      const enabled = supported && (body.enable_mtp ?? capability.mtp_default_enabled);
      const used = enabled && body.max_tokens > 1;
      const mtp = { requested: body.enable_mtp ?? null, enabled, used, drafted: used ? Math.min(4, body.max_tokens) : 0, accepted: used ? Math.min(3, body.max_tokens - 1) : 0, cycles: used ? Math.min(2, body.max_tokens - 1) : 0, fallback_reason: null };
      if (corrupt === "ignored_false" && body.enable_mtp === false) Object.assign(mtp, { enabled: true, used: true, drafted: 1, accepted: 1, cycles: 1 });
      if (corrupt === "acceptance") Object.assign(mtp, { drafted: 1, accepted: 2 });
      const text = divergent && enabled ? "2, 3, 4" : "1, 2, 3";
      if (body.stream) {
        const payload = [
          { choices: [{ delta: { content: text }, finish_reason: null }] },
          { choices: [{ delta: {}, finish_reason: "length" }], mtp },
        ].map(item => `data: ${JSON.stringify(item)}\r\n\r\n`).join("") + "data: [DONE]\r\n\r\n";
        // Split every byte so stream parsing cannot depend on transport chunks.
        return new Response(new ReadableStream({ start(controller) {
          for (const byte of new TextEncoder().encode(payload)) controller.enqueue(new Uint8Array([byte]));
          controller.close();
        }}));
      }
      return Response.json({ choices: [{ message: { content: text }, finish_reason: "length" }], mtp });
    },
  };
}

test("runner normalizes base and bounds the inference workload", () => {
  expect(parseArgs(["--base", "http://127.0.0.1:9090/v1/", "--max-tokens", "64"]).base).toBe("http://127.0.0.1:9090");
  expect(() => parseArgs(["--max-tokens", "1000"])).toThrow();
  expect(() => parseArgs(["--base", "http://user:secret@localhost:9090"])).toThrow();
});

test("supported server exercises actual on/off/default metadata across stream and queued requests", async () => {
  const fake = fixture();
  const report = await runVerification(options(), fake.fetch);
  expect(report.status).toBe("passed");
  expect(report.cases).toHaveLength(13);
  expect(fake.requests.filter(body => body.stream)).toHaveLength(7);
  expect(report.cases.find(item => item.name === "queued_off")?.mtp.enabled).toBe(false);
  expect(report.cases.find(item => item.name === "queued_on")?.mtp.used).toBe(true);
  expect(report.cases.find(item => item.name === "default_after_overrides")?.mtp.enabled).toBe(true);
  expect(report.output_parity?.greedy_text_equal).toBe(true);
  expect(report.reference_correctness_validated).toBe(false);
});

test("default-off capability is respected and explicit on still verifies speculative work", async () => {
  const fake = fixture({ defaultEnabled: false });
  const report = await runVerification(options(), fake.fetch);
  expect(report.status).toBe("passed");
  expect(report.cases.find(item => item.name === "omitted_json")?.mtp.enabled).toBe(false);
  expect(report.cases.find(item => item.name === "null_stream")?.mtp.enabled).toBe(false);
  expect(report.cases.find(item => item.name === "on_json")?.mtp.used).toBe(true);
});

test("legacy server stops before any inference rather than silently trusting ignored controls", async () => {
  const fake = fixture({ legacy: true });
  const report = await runVerification(options(), fake.fetch);
  expect(report.status).toBe("failed");
  expect(report.checks[0].error).toContain("does not report");
  expect(fake.requests).toHaveLength(0);
});

test("unsupported server must reject explicit opt-in and preserve ordinary decoding", async () => {
  const fake = fixture({ supported: false });
  const report = await runVerification(options(), fake.fetch);
  expect(report.status).toBe("passed");
  expect(report.cases).toHaveLength(9);
  expect(report.cases.filter(item => item.rejected)).toHaveLength(2);
  expect(report.cases.filter(item => item.mtp).every(item => !item.mtp.enabled)).toBe(true);
});

test("runner detects ignored off controls and impossible acceptance counters", async () => {
  for (const corrupt of ["ignored_false", "acceptance"]) {
    const fake = fixture({ corrupt });
    const report = await runVerification(options(), fake.fetch);
    expect(report.status).toBe("failed");
    expect(report.cases.some(item => !item.passed)).toBe(true);
  }
});

test("enabled fallback needs an explanation and cannot claim unexecuted MTP", () => {
  const capability = { supports_mtp: true, mtp_default_enabled: true };
  const result = { requested: true, enabled: true, used: false, drafted: 0, accepted: 0, cycles: 0, fallback_reason: null };
  expect(() => validateMtp(result, true, capability, 32)).toThrow("neither ran");
  expect(validateMtp({ ...result, fallback_reason: "Priming unavailable" }, true, capability, 32).used).toBe(false);
  expect(validateMtp(result, true, capability, 32, "stop").used).toBe(false);
  expect(() => validateMtp({ ...result, used: true }, true, capability, 32)).toThrow("no actual");
});

test("parity differences are disclosed and strict mode makes them fail", async () => {
  const observational = fixture({ divergent: true });
  const report = await runVerification(options(), observational.fetch);
  expect(report.status).toBe("passed");
  expect(report.output_parity?.greedy_text_equal).toBe(false);
  const strict = fixture({ divergent: true });
  expect((await runVerification(options({ strictParity: true }), strict.fetch)).status).toBe("failed");
});

test("stream parser rejects errors and missing completion markers", async () => {
  await expect(readStreamingCompletion(new Response('data: {"error":{"message":"bad request"}}\n\n'))).rejects.toThrow("API error");
  await expect(readStreamingCompletion(new Response('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'))).rejects.toThrow("DONE");
});

test("reports omit private endpoint details, paths, and generated text", async () => {
  const fake = fixture();
  const report = await runVerification(options({ base: "http://private-node.internal:54321" }), fake.fetch);
  const serialized = JSON.stringify(report);
  expect(serialized).not.toContain("private-node");
  expect(serialized).not.toContain("54321");
  expect(serialized).not.toContain('"1, 2, 3"');
  expect(report.cases[0].output_sha256).toMatch(/^[0-9a-f]{64}$/);
});
