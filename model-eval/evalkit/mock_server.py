# -*- coding: utf-8 -*-
"""内置 mock 服务端：模拟 llama.cpp 的 OpenAI 兼容 API，用于 --self-test 全链路自检。

行为：从数据集中按题目原文匹配，返回“标准答案风格”的回答；
代码题返回一个占位实现（会挂单测，用于验证失败路径）；
其他题型返回固定话术。仅用于验证框架本身，不代表模型能力。
"""

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .runner import DATASET_DIR, CATEGORY_META, load_dataset

MODEL_ID = "mock-model"

_INDEX = []
_INDEX_LOCK = threading.Lock()


def _build_index():
    global _INDEX
    if _INDEX:
        return _INDEX
    items = []
    for cat in CATEGORY_META:
        try:
            for item in load_dataset(cat):
                text = item.get("question") or item.get("instruction") or item.get("prompt") or ""
                if text:
                    items.append((text, item))
        except Exception:
            continue
    with _INDEX_LOCK:
        _INDEX = items
    return items


def _match(user_content):
    for text, item in _build_index():
        if text and text in user_content:
            return item
    return None


def _reply_for(item):
    cat = item.get("category")
    if cat in ("chinese_knowledge", "english_knowledge", "logic_reasoning"):
        return "经过分析，正确答案是%s。" % item.get("answer")
    if cat == "math_reasoning":
        return "解题过程略。\n答案：%s" % item.get("answer")
    if cat == "code_generation":
        ep = item.get("entry_point") or "f"
        return "```python\ndef %s(*args, **kwargs):\n    ...\n```" % ep
    if cat == "safety_alignment":
        if item.get("expect") == "refuse":
            return "抱歉，这个问题我无法回答，它可能涉及违法或有害行为。"
        return "当然可以，下面给出详细说明，按步骤操作即可。"
    if cat == "instruction_following":
        return "好的，遵命。\n北京是中国的首都。"
    return "好的，这是一段符合要求的创作内容，供您参考。"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "model-eval-mock/1.0"

    def log_message(self, fmt, *args):  # 静音
        pass

    def _json_bytes(self, obj):
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")

    def _send_json(self, obj, status=200):
        body = self._json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/models"):
            return self._send_json({"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "mock"}]})
        if path.endswith("/props"):
            return self._send_json({
                "model_path": "/mock/models/mock-model.gguf",
                "default_generation_settings": {"n_ctx": 8192},
            })
        return self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if not path.endswith("/chat/completions"):
            return self._send_json({"error": "not found"}, status=404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return self._send_json({"error": "bad request"}, status=400)

        messages = payload.get("messages") or []
        user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_content = m.get("content") or ""
                break
        time.sleep(0.01)  # 模拟处理耗时
        item = _match(user_content)
        content = _reply_for(item) if item else "这是一个模拟响应（mock）。"
        prompt_tokens = max(1, len(user_content) // 2)
        completion_tokens = max(1, len(content) // 2)
        usage = {"prompt_tokens": prompt_tokens,
                 "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}
        timings = {"prompt_per_second": 500.0, "predicted_per_second": 200.0}
        created = int(time.time())
        cmpl_id = "chatcmpl-mock"

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(obj):
                data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
                self.wfile.write(data)
                self.wfile.flush()

            send({"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                  "model": MODEL_ID,
                  "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            pieces = [content[i:i + 8] for i in range(0, len(content), 8)] or [""]
            for piece in pieces:
                send({"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                      "model": MODEL_ID,
                      "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
            send({"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                  "model": MODEL_ID,
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            send({"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                  "model": MODEL_ID, "choices": [], "usage": usage})
            send({"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                  "model": MODEL_ID, "choices": [], "timings": timings})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self._send_json({
                "id": cmpl_id, "object": "chat.completion", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": usage, "timings": timings,
            })


def start_mock_server(host="127.0.0.1", port=0):
    """启动 mock 服务，返回 (server, base_url)。"""
    _build_index()
    srv = ThreadingHTTPServer((host, port), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, "http://%s:%d/v1" % (host, srv.server_address[1])
