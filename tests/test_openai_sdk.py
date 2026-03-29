#!/usr/bin/env python3
"""
Integration test for ZINC's OpenAI-compatible API server.
Run with: python tests/test_openai_sdk.py [--base-url http://localhost:8080/v1]

Requires: pip install openai requests
"""
import argparse
import json
import sys
import time
import requests

def test_health(base):
    """Test GET /health returns ok status."""
    r = requests.get(f"{base.rstrip('/v1')}/health", timeout=5)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    data = r.json()
    assert data["status"] == "ok", f"Health status: {data}"
    print(f"  PASS: /health → {data['status']}, model={data.get('model', '?')}")

def test_models(base):
    """Test GET /v1/models returns model list."""
    r = requests.get(f"{base}/models", timeout=5)
    assert r.status_code == 200, f"Models failed: {r.status_code}"
    data = r.json()
    assert data["object"] == "list", f"Expected list, got: {data['object']}"
    assert len(data["data"]) > 0, "No models returned"
    model = data["data"][0]
    assert "id" in model, "Model missing id"
    print(f"  PASS: /v1/models → {model['id']}")
    return model["id"]

def test_chat_completion_nonstreaming(base, model_id):
    """Test POST /v1/chat/completions (non-streaming)."""
    r = requests.post(f"{base}/chat/completions", json={
        "model": model_id,
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
        "max_tokens": 16,
        "temperature": 0,
    }, timeout=30)
    assert r.status_code == 200, f"Chat failed: {r.status_code} {r.text[:200]}"
    data = r.json()
    assert data["object"] == "chat.completion", f"Wrong object: {data['object']}"
    assert "choices" in data and len(data["choices"]) > 0, "No choices"
    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0, "Empty response content"
    assert "usage" in data, "Missing usage"
    print(f"  PASS: chat (non-streaming) → '{content[:50]}' ({data['usage']['completion_tokens']} tokens)")

def test_chat_completion_streaming(base, model_id):
    """Test POST /v1/chat/completions (streaming SSE)."""
    r = requests.post(f"{base}/chat/completions", json={
        "model": model_id,
        "messages": [{"role": "user", "content": "Count to 3."}],
        "stream": True,
        "max_tokens": 32,
        "temperature": 0,
    }, stream=True, timeout=30)
    assert r.status_code == 200, f"Stream failed: {r.status_code}"
    assert "text/event-stream" in r.headers.get("content-type", ""), "Not SSE"

    chunks = []
    got_done = False
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                got_done = True
                break
            chunk = json.loads(payload)
            assert chunk["object"] == "chat.completion.chunk"
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                chunks.append(delta["content"])

    assert got_done, "Never received [DONE]"
    full_text = "".join(chunks)
    assert len(full_text) > 0, "Empty streaming response"
    print(f"  PASS: chat (streaming) → '{full_text[:50]}' ({len(chunks)} chunks)")

def test_completion(base, model_id):
    """Test POST /v1/completions (non-streaming)."""
    r = requests.post(f"{base}/completions", json={
        "model": model_id,
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "temperature": 0,
    }, timeout=30)
    assert r.status_code == 200, f"Completion failed: {r.status_code}"
    data = r.json()
    assert data["object"] == "text_completion"
    text = data["choices"][0]["text"]
    assert len(text) > 0, "Empty completion"
    print(f"  PASS: completion → '{text[:50]}'")

def test_error_handling(base):
    """Test error responses match OpenAI format."""
    # Missing messages
    r = requests.post(f"{base}/chat/completions", json={"model": "x"}, timeout=5)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    data = r.json()
    assert "error" in data, "Missing error field"
    print(f"  PASS: error format → {data['error']['message'][:60]}")

    # Unknown endpoint
    r = requests.get(f"{base}/unknown", timeout=5)
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print(f"  PASS: 404 for unknown endpoint")

def test_openai_sdk(base, model_id):
    """Test with the official OpenAI Python SDK."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  SKIP: openai package not installed (pip install openai)")
        return

    client = OpenAI(base_url=base, api_key="unused")

    # Non-streaming
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=8,
    )
    assert resp.choices[0].message.content, "SDK non-streaming empty"
    print(f"  PASS: OpenAI SDK (non-streaming) → '{resp.choices[0].message.content[:40]}'")

    # Streaming
    chunks = []
    for chunk in client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Say hi."}],
        max_tokens=8,
        stream=True,
    ):
        c = chunk.choices[0].delta.content
        if c:
            chunks.append(c)
    assert len(chunks) > 0, "SDK streaming empty"
    print(f"  PASS: OpenAI SDK (streaming) → '{''.join(chunks)[:40]}'")

def main():
    parser = argparse.ArgumentParser(description="Test ZINC OpenAI API")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    args = parser.parse_args()
    base = args.base_url

    print(f"Testing ZINC API at {base}\n")

    try:
        test_health(base)
        model_id = test_models(base)
        test_chat_completion_nonstreaming(base, model_id)
        test_chat_completion_streaming(base, model_id)
        test_completion(base, model_id)
        test_error_handling(base)
        test_openai_sdk(base, model_id)
        print(f"\nAll tests passed!")
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
