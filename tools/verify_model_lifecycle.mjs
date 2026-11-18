#!/usr/bin/env node
/**
 * Bounded model-lifecycle checks against an already-running ZINC server.
 * Discovery is read-only by default. --allow-model-reload explicitly enables
 * unload/reload, retains cached files, and restores the initially active model.
 * This never pulls/removes models, restarts a server, or changes environment.
 */
import { writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { serverRoot } from "./verify_mtp.mjs";

const HELP = `Usage: node tools/verify_model_lifecycle.mjs [options]
  --base <url>           Running server root or /v1 URL (default: http://127.0.0.1:9090)
  --output <path>        Optional JSON report (omits endpoint and filesystem paths)
  --timeout-ms <ms>      Per-request timeout, 1000..600000 (default: 120000)
  --allow-model-reload   Test reset, unload, and reload of the initially active model
  --switch-model <id>    Also switch to this advertised installed model, then restore
  -h, --help             Show help

Without --allow-model-reload, only health/catalog discovery is checked.
Reload checks require an initially active model and a server advertising model
management/unload. They never download or delete weights. Run on an idle server.`;

export function parseArgs(argv) {
  const options = { base: "http://127.0.0.1:9090", output: null, timeoutMs: 120_000, allowModelReload: false, switchModel: null, help: false };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (key === "--help" || key === "-h") { options.help = true; continue; }
    if (key === "--allow-model-reload") { options.allowModelReload = true; continue; }
    const name = { "--base": "base", "--output": "output", "--timeout-ms": "timeoutMs", "--switch-model": "switchModel" }[key];
    if (!name) throw new Error("Unknown option; use --help.");
    const value = argv[++i];
    if (!value || value.startsWith("--")) throw new Error("Missing option value; use --help.");
    options[name] = name === "timeoutMs" ? Number(value) : value;
  }
  options.base = serverRoot(options.base);
  if (!Number.isInteger(options.timeoutMs) || options.timeoutMs < 1_000 || options.timeoutMs > 600_000) throw new Error("--timeout-ms must be an integer from 1000 to 600000.");
  if (options.switchModel && !options.allowModelReload) throw new Error("--switch-model requires --allow-model-reload.");
  return options;
}

function requireValue(condition, message) { if (!condition) throw new Error(message); }
function safeText(value) {
  return String(value || "")
    .replace(/https?:\/\/[^\s"<>]+/gi, "[endpoint]")
    .replace(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, "[address]")
    .replace(/\b[\w.-]+\.(?:local|internal|lan)\b/gi, "[host]")
    .replace(/(?:[A-Za-z]:[\\/]|\/(?:Users|home|root|tmp|var|opt)\/)[^\s"<>]*/g, "[path]")
    .slice(0, 400);
}
function label(value) { return safeText(String(value || "").split(/[\\/]/).pop()); }
function active(catalog) {
  requireValue(Array.isArray(catalog?.data), "Model catalog did not contain an entries array.");
  const entries = catalog.data.filter(item => item.active === true);
  requireValue(entries.length <= 1, "More than one active model was reported.");
  return entries[0] || null;
}
function mtpCapability(entry) {
  return Object.fromEntries(["supports_mtp", "mtp_default_enabled", "mtp_request_control"].filter(key => typeof entry?.[key] === "boolean").map(key => [key, entry[key]]));
}
function sameCapability(expected, actual) {
  for (const key of Object.keys(expected)) requireValue(expected[key] === actual?.[key], "Reload did not restore the original MTP capability/default policy.");
}

export async function runVerification(options, fetchImpl = fetch) {
  const root = serverRoot(options.base);
  const report = { schema_version: 1, checked_at: new Date().toISOString(), status: "failed", mode: options.allowModelReload ? "reload" : "discovery", backend: null, original_model: null, switch_model: null, original_mtp: null, restored_original: null, checks: [] };
  const request = async (path, body) => {
    let response;
    try {
      response = await fetchImpl(root + path, { ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }), signal: AbortSignal.timeout(options.timeoutMs) });
    } catch { throw new Error("Server request failed or timed out."); }
    requireValue(response.ok, "Server rejected a lifecycle request (HTTP " + response.status + ").");
    let value;
    try { value = await response.json(); } catch { throw new Error("Server returned invalid JSON."); }
    requireValue(!value.error, "Server returned a lifecycle error.");
    return value;
  };
  const check = async (name, fn) => {
    try { const value = await fn(); report.checks.push({ name, passed: true }); return value; }
    catch (error) { report.checks.push({ name, passed: false, error: safeText(error.message) }); throw error; }
  };
  const catalog = () => request("/v1/models");
  const verifyActive = async (id) => {
    const list = await catalog(), current = active(list);
    requireValue(current?.id === id, "The expected model is not active.");
    return current;
  };
  let original, expectedMtp, attemptedMutation = false;
  const ownedIds = new Set();
  try {
    const initial = await check("discovery", async () => {
      const health = await request("/health"), list = await catalog();
      requireValue(health.status === "ok", "Server is not healthy.");
      original = active(list);
      requireValue(original && typeof original.id === "string" && original.id.length > 0, "Verification requires an initially active model.");
      for (const info of [health, list]) {
        requireValue(info.supports_model_management === true && info.supports_model_unload === true, "Server does not advertise model switching and unload; deploy the updated server first.");
      }
      requireValue(original.installed === true && original.can_activate !== false, "The original model is not advertised as an installed reload candidate.");
      report.backend = ["vulkan", "metal", "cuda", "rocm", "zinc_rt"].includes(health.backend) ? health.backend : "reported";
      report.original_model = label(original.id);
      expectedMtp = mtpCapability(original);
      report.original_mtp = expectedMtp;
      ownedIds.add(original.id);
      return list;
    });
    let alternative = null;
    if (options.switchModel) {
      alternative = await check("switch_preflight", () => {
        const candidate = initial.data.find(entry => entry.id === options.switchModel);
        requireValue(candidate && candidate.id !== original.id, "Switch target must be a different advertised model.");
        requireValue(candidate.installed === true && candidate.can_activate !== false && candidate.supported_on_current_gpu !== false && candidate.fits_current_gpu !== false, "Switch target is not an installed compatible activation candidate.");
        ownedIds.add(candidate.id);report.switch_model = label(candidate.id);
        return candidate;
      });
    }
    if (options.allowModelReload) {
      await check("clear_chat_keeps_model_loaded", async () => {
        const result = await request("/v1/chat/reset", { session_id: "zinc-lifecycle-verification" });
        requireValue(result.reset === true, "Chat reset was not acknowledged.");
        sameCapability(expectedMtp, await verifyActive(original.id));
      });
      attemptedMutation = true;
      await check("unload_keeps_model_installed", async () => {
        await request("/v1/models/unload", { model: original.id });
        const list = await catalog();
        requireValue(active(list) === null, "Unload left a model marked active.");
        requireValue(list.data.find(entry => entry.id === original.id)?.installed === true, "Unload removed the original model from the installed catalog.");
        const health = await request("/health");
        requireValue(health.status === "ok", "Server stopped responding after unload.");
        requireValue(health.supports_mtp === false, "Unloaded server retained an active MTP capability.");
      });
      await check("reload_restores_model_and_mtp", async () => {
        await request("/v1/models/activate", { model: original.id });
        sameCapability(expectedMtp, await verifyActive(original.id));
      });
      if (alternative) await check("switch_activates_selected_model", async () => {
        await request("/v1/models/activate", { model: alternative.id });
        await verifyActive(alternative.id);
      });
    }
  } catch { /* The failing step is already retained; restoration still runs. */ }
  finally {
    if (attemptedMutation) {
      try {
        await check("restore_original", async () => {
          const current = active(await catalog());
          requireValue(!current || ownedIds.has(current.id), "Another client changed the active model; restoration stopped to preserve its choice.");
          if (current?.id !== original.id) await request("/v1/models/activate", { model: original.id });
          sameCapability(expectedMtp, await verifyActive(original.id));
          const health = await request("/health");
          requireValue(health.status === "ok", "Restored server is not healthy.");
          sameCapability(expectedMtp, health);
          report.restored_original = true;
        });
      } catch { report.restored_original = false; }
    }
  }
  report.status = report.checks.length > 0 && report.checks.every(item => item.passed) ? "passed" : "failed";
  return report;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) { console.log(HELP); return; }
  const report = await runVerification(options);
  if (options.output) await writeFile(options.output, JSON.stringify(report, null, 2) + "\n");
  console.log(`Model lifecycle ${report.status}: ${report.checks.filter(item => item.passed).length}/${report.checks.length} checks passed (${report.mode}).`);
  for (const failure of report.checks.filter(item => !item.passed)) console.error(`${failure.name}: ${failure.error}`);
  if (report.status !== "passed") process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main().catch(() => { console.error("Model lifecycle verification could not complete; use --help and check local configuration.");process.exitCode = 1; });
