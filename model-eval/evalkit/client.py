# -*- coding: utf-8 -*-
"""llama.cpp / OpenAI 兼容客户端：流式请求、TTFT 与吞吐指标采集、错误处理。"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "model-eval/1.0"

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


def _is_loopback(host):
    if not host:
        return False
    h = host.strip("[]").lower()
    return h in _LOOPBACK_HOSTS or h.startswith("127.")


def _opener_for(url):
    """回环地址（localhost/127.x）直连，避免被本机代理拦截；其余地址遵循系统代理设置。"""
    if _is_loopback(urllib.parse.urlparse(url).hostname):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


class ChatResult(dict):
    """单次对话结果（dict 便于直接序列化）。"""

    @property
    def ok(self):
        return self.get("error") is None


def _headers(cfg):
    h = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    key = (cfg.get("server") or {}).get("api_key") or ""
    if key:
        h["Authorization"] = "Bearer " + key
    return h


def _endpoint(cfg):
    base = cfg["server"]["base_url"].rstrip("/")
    return base + "/chat/completions"


def _root(cfg):
    base = cfg["server"]["base_url"].rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _build_payload(cfg, messages, stream):
    payload = {
        "model": cfg["model"].get("name") or "default",
        "messages": messages,
        "temperature": cfg["generation"].get("temperature", 0.0),
        "top_p": cfg["generation"].get("top_p", 1.0),
        "max_tokens": cfg["generation"].get("max_tokens", 2048),
        "stream": stream,
    }
    stop = cfg["generation"].get("stop") or []
    if stop:
        payload["stop"] = stop
    ctk = cfg["model"].get("chat_template_kwargs") or {}
    if ctk:
        payload["chat_template_kwargs"] = ctk
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _finalize(content, ttft, total, usage, timings, finish_reason):
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")
    tps = None
    if timings and timings.get("predicted_per_second"):
        tps = float(timings["predicted_per_second"])
    elif completion_tokens and total:
        gen_time = max(total - (ttft or 0.0), 1e-6)
        tps = completion_tokens / gen_time
    return ChatResult(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=round(ttft * 1000, 1) if ttft is not None else None,
        total_ms=round(total * 1000, 1),
        tokens_per_sec=round(tps, 2) if tps else None,
        finish_reason=finish_reason,
        error=None,
    )


def _chat_stream(cfg, messages, timeout):
    """流式请求：返回 (result, http_error)。"""
    url = _endpoint(cfg)
    body = json.dumps(_build_payload(cfg, messages, True)).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=_headers(cfg), method="POST")
    opener = _opener_for(url)
    t0 = time.perf_counter()
    ttft, chunks, usage, timings, finish = None, [], None, None, None
    try:
        with opener.open(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("timings"):
                    timings = obj["timings"]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks.append(delta["content"])
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        return None, "HTTP %s %s" % (e.code, detail)
    except Exception as e:  # URLError / timeout / socket
        return None, "%s: %s" % (type(e).__name__, e)
    total = time.perf_counter() - t0
    return _finalize("".join(chunks), ttft, total, usage, timings, finish), None


def _chat_block(cfg, messages, timeout):
    """非流式请求（流式失败时的回退）。"""
    url = _endpoint(cfg)
    body = json.dumps(_build_payload(cfg, messages, False)).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=_headers(cfg), method="POST")
    opener = _opener_for(url)
    t0 = time.perf_counter()
    try:
        with opener.open(req, timeout=timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        return None, "HTTP %s %s" % (e.code, detail)
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)
    total = time.perf_counter() - t0
    choice = (obj.get("choices") or [{}])[0]
    content = ((choice.get("message") or {}).get("content")) or ""
    ttft = None
    return _finalize(content, ttft, total, obj.get("usage"), obj.get("timings"),
                     choice.get("finish_reason")), None


def chat(cfg, messages):
    """发送一次对话请求，始终返回 ChatResult（失败时 error 字段非空）。"""
    timeout = float(cfg["server"].get("timeout") or 600)
    result, err = _chat_stream(cfg, messages, timeout)
    if err is not None:
        # 流式失败：去掉 stream_options 再试一次，再退回非流式
        if "stream_options" in json.dumps(_build_payload(cfg, messages, True)):
            cfg_copy = json.loads(json.dumps(cfg))
            cfg_copy["generation"] = dict(cfg_copy["generation"])
            result, err2 = _chat_stream(cfg_copy, messages, timeout)
            if err2 is None:
                return result
        result, err = _chat_block(cfg, messages, timeout)
    if err is not None:
        return ChatResult(content="", prompt_tokens=None, completion_tokens=None,
                          ttft_ms=None, total_ms=None, tokens_per_sec=None,
                          finish_reason=None, error=err)
    return result


def fetch_model_info(cfg):
    """尽力获取服务端模型信息（/v1/models 与 llama.cpp /props）。"""
    info = {"reported_name": None, "model_path": None, "server": cfg["server"]["base_url"]}
    try:
        url = cfg["server"]["base_url"].rstrip("/") + "/models"
        req = urllib.request.Request(url, headers=_headers(cfg))
        with _opener_for(url).open(req, timeout=10) as resp:
            obj = json.loads(resp.read().decode("utf-8", "replace"))
        data = obj.get("data") or []
        if data and data[0].get("id"):
            info["reported_name"] = data[0]["id"]
    except Exception:
        pass
    try:
        url = _root(cfg) + "/props"
        req = urllib.request.Request(url, headers=_headers(cfg))
        with _opener_for(url).open(req, timeout=10) as resp:
            obj = json.loads(resp.read().decode("utf-8", "replace"))
        if obj.get("model_path"):
            info["model_path"] = obj["model_path"]
    except Exception:
        pass
    return info
