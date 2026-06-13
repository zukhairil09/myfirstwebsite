"""
Ayie AI Platform — Python Flask Backend
Model: gpt-4.1-mini (OpenAI)
Streaming support via SSE
"""

import os
import json
import time
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app, origins="*")

# ──────────────────────────────────────────────
# OpenAI client — reads key from env or request
# ──────────────────────────────────────────────
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "platform": "Ayie", "version": "1.0.0"})


# ──────────────────────────────────────────────
# Non-streaming chat endpoint
# ──────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json()
        api_key   = body.get("api_key", "")
        messages  = body.get("messages", [])
        model     = body.get("model", "gpt-4.1-mini")
        max_toks  = body.get("max_tokens", 4096)
        system_p  = body.get("system_prompt", "You are Ayie, a helpful AI assistant. Format code using markdown code blocks with language labels.")

        if not api_key:
            return jsonify({"error": "API key required"}), 400
        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        client = get_client(api_key)

        full_messages = [{"role": "system", "content": system_p}] + messages

        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=max_toks,
            temperature=0.7,
        )

        content = response.choices[0].message.content
        usage   = {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }

        return jsonify({"content": content, "usage": usage, "model": model})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Streaming chat endpoint (SSE)
# ──────────────────────────────────────────────
@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    try:
        body = request.get_json()
        api_key   = body.get("api_key", "")
        messages  = body.get("messages", [])
        model     = body.get("model", "gpt-4.1-mini")
        max_toks  = body.get("max_tokens", 4096)
        system_p  = body.get("system_prompt", "You are Ayie, a helpful AI assistant. Format code using markdown code blocks with language labels.")

        if not api_key:
            def err():
                yield f"data: {json.dumps({'error': 'API key required'})}\n\n"
            return Response(stream_with_context(err()), mimetype="text/event-stream")

        client = get_client(api_key)
        full_messages = [{"role": "system", "content": system_p}] + messages

        def generate():
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    max_tokens=max_toks,
                    temperature=0.7,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield f"data: {json.dumps({'delta': delta.content})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# List available models
# ──────────────────────────────────────────────
@app.route("/models", methods=["POST"])
def list_models():
    try:
        body    = request.get_json()
        api_key = body.get("api_key", "")
        if not api_key:
            return jsonify({"error": "API key required"}), 400

        client = get_client(api_key)
        models = client.models.list()
        gpt_models = sorted(
            [m.id for m in models.data if "gpt" in m.id],
            reverse=True,
        )
        return jsonify({"models": gpt_models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Validate API key
# ──────────────────────────────────────────────
@app.route("/validate", methods=["POST"])
def validate_key():
    try:
        body    = request.get_json()
        api_key = body.get("api_key", "")
        if not api_key:
            return jsonify({"valid": False, "error": "No key provided"}), 400

        client = get_client(api_key)
        # Cheap call to verify key works
        client.models.list()
        return jsonify({"valid": True})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 401


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Ayie Backend running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
