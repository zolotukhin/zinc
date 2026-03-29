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
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
r = client.chat.completions.create(model="qwen", messages=[{"role":"user","content":"Hi"}])
print(r.choices[0].message.content)
```
