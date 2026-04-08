import { test } from "bun:test";
import { hasSmokeEnv, runSmokeSuite } from "./test_qwen_smoke";

const binary = process.env.ZINC_CLI_BIN ?? "./zig-out/bin/zinc";
const requireFull = process.env.ZINC_REQUIRE_FULL_TESTS === "1";

if (!hasSmokeEnv()) {
  if (requireFull) {
    test("Qwen CLI smoke", () => {
      throw new Error("Missing ZINC_QWEN3_8B_MODEL and/or ZINC_QWEN35_35B_MODEL");
    });
  } else {
    test.skip("Qwen smoke requires ZINC_QWEN3_8B_MODEL and ZINC_QWEN35_35B_MODEL", () => {});
  }
} else {
  test("Qwen CLI smoke", async () => {
    await runSmokeSuite(binary);
  }, 300_000);
}
