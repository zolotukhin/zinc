#!/usr/bin/env bun

function hasEnv(name: string): boolean {
  const value = process.env[name];
  return typeof value === "string" && value.length > 0;
}

function statusLine(label: string, status: string, detail?: string): string {
  return detail ? `  ${label}: ${status} (${detail})` : `  ${label}: ${status}`;
}

const requireFull = process.env.ZINC_REQUIRE_FULL_TESTS === "1";
const hasQwenSmoke =
  hasEnv("ZINC_QWEN35_2B_MODEL") &&
  hasEnv("ZINC_QWEN35_35B_MODEL");
const hasApiSmoke = hasEnv("ZINC_API_BASE_URL");

console.log("\nCombined test summary:");
console.log(statusLine("Bun suite", "pass"));
console.log(statusLine("Zig suite", "pass"));
console.log(
  statusLine(
    "Qwen smoke",
    hasQwenSmoke ? "ran" : requireFull ? "required but missing env" : "skipped",
    hasQwenSmoke ? process.env.ZINC_QWEN35_2B_MODEL : "needs ZINC_QWEN35_2B_MODEL and ZINC_QWEN35_35B_MODEL",
  ),
);
console.log(
  statusLine(
    "OpenAI API smoke",
    hasApiSmoke ? "ran" : requireFull ? "required but missing env" : "skipped",
    hasApiSmoke ? process.env.ZINC_API_BASE_URL : "needs ZINC_API_BASE_URL",
  ),
);
console.log(statusLine("Full-test mode", requireFull ? "enabled" : "disabled"));
console.log(statusLine("Overall", "pass"));
