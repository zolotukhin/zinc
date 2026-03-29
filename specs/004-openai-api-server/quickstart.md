# Quickstart: OpenAI API Server

## Start
```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc --model /path/to/model.gguf --port 8080 --parallel 4
```

## Test
```bash
# Streaming chat
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Health
curl http://localhost:8080/health

# Models
curl http://localhost:8080/v1/models
```

## OpenAI SDK
```typescript
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://localhost:8080/v1", apiKey: "unused" });
const r = await client.chat.completions.create({
  model: "qwen",
  messages: [{ role: "user", content: "Hi" }],
});
console.log(r.choices[0]?.message?.content);
```
