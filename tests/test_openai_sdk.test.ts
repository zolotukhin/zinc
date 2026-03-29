import { test } from "bun:test";
import { runSuite } from "./test_openai_sdk";

const baseUrl = process.env.ZINC_API_BASE_URL;

if (!baseUrl) {
  test.skip("OpenAI API smoke requires ZINC_API_BASE_URL", () => {});
} else {
  test("OpenAI API smoke", async () => {
    await runSuite(baseUrl);
  }, 180_000);
}
