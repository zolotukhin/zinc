import { test } from "bun:test";
import { runSuite } from "./test_openai_sdk";

const baseUrl = process.env.ZINC_API_BASE_URL;
const requireFull = process.env.ZINC_REQUIRE_FULL_TESTS === "1";

if (!baseUrl) {
  if (requireFull) {
    test("OpenAI API smoke", () => {
      throw new Error("Missing ZINC_API_BASE_URL");
    });
  } else {
    test.skip("OpenAI API smoke requires ZINC_API_BASE_URL", () => {});
  }
} else {
  test("OpenAI API smoke", async () => {
    await runSuite(baseUrl);
  }, 180_000);
}
