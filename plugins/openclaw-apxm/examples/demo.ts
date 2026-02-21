import { createOpenClawApxmAdapter } from "../src/index.js";

async function main(): Promise<void> {
  const adapter = createOpenClawApxmAdapter({
    baseUrl: "http://127.0.0.1:18800",
    defaultModel: "claude-sonnet-4-20250514"
  });

  const result = await adapter.executeMessage({
    sessionKey: "wa:dm:+1234567890",
    userMessage: "What's our deploy server?"
  });

  console.log(result.content);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
