# openclaw-apxm

OpenClaw plugin adapter that executes APXM graphs over `apxm-server` HTTP API.

## What it provides

- OpenClaw -> APXM execution bridge (`/v1/execute`)
- Skill markdown -> AIS graph transpiler
- Capability registration bridge (`/v1/capabilities/register`)
- Session ID mapping (`channel:conversation:user` -> APXM `session_id`)

## Configuration

```json
{
  "apxm": {
    "baseUrl": "http://127.0.0.1:18800",
    "defaultModel": "claude-sonnet-4-20250514",
    "systemPrompt": "You are a helpful assistant.",
    "tools": ["bash", "read", "write", "search_web"]
  }
}
```

## Usage sketch

```ts
import { createOpenClawApxmAdapter } from "./dist/index.js";

const adapter = createOpenClawApxmAdapter({
  baseUrl: "http://127.0.0.1:18800",
  defaultModel: "claude-sonnet-4-20250514"
});

const result = await adapter.executeMessage({
  sessionKey: "wa:dm:+1234567890",
  userMessage: "What's our deploy server?"
});

console.log(result.content);
```

## Endpoints used

- `POST /v1/execute`
- `POST /v1/execute/stream`
- `POST /v1/memory/facts/store`
- `POST /v1/memory/facts/search`
- `POST /v1/memory/facts/delete`
- `POST /v1/capabilities/register`
