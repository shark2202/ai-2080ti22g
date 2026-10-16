# -*- coding: utf-8 -*-
"""连接检查：跑测评前先确认端点可达、模型可答、速度量级合理。

不依赖模型服务即可运行（用内置 mock 服务自测本模块）。
"""

import time

from . import client


def _fmt(v, unit="", width=None):
    if v is None:
        return "未知"
    s = ("%g%s" % (v, unit)) if width is None else ("%*g%s" % (width, v, unit))
    return s


def check_endpoint(cfg, probe=True):
    """检查端点。返回 (ok: bool, lines: list[str])。"""
    base = cfg["server"]["base_url"]
    timeout = float(cfg["server"].get("timeout") or 600)
    lines = ["端点: %s" % base]

    # 1) /v1/models —— 判断服务是否在监听
    t0 = time.perf_counter()
    info = client.fetch_model_info(cfg)
    probe_ms = (time.perf_counter() - t0) * 1000

    if info.get("reported_name"):
        lines.append("服务: 在线（/v1/models 响应 %.0f ms）" % probe_ms)
        lines.append("服务端报告的模型名: %s" % info["reported_name"])
        if info.get("model_path"):
            lines.append("服务端模型路径: %s" % info["model_path"])
    else:
        lines.append("服务: 未连通（/v1/models 无响应或非 OpenAI 兼容）")
        lines.append("  排查顺序：① llama-server 是否在跑（ps aux | grep llama-server）"
                     "② --host 是否为 0.0.0.0（默认只监听本机，远程访问会被拒）"
                     "③ 端口与防火墙 ④ 远程端点是否被系统代理拦截")
        return False, lines

    cfg_name = (cfg.get("model") or {}).get("name")
    if cfg_name and info.get("reported_name") and cfg_name not in str(info["reported_name"]):
        lines.append("注意: 配置中的 model.name=%s 与服务端不一致（仅影响报告标注，不影响请求）"
                     % cfg_name)

    if not probe:
        lines.append("（--no-probe：已跳过实际推理测试）")
        return True, lines

    # 2) 真实推理一次 —— 确认可生成并估算速度
    t0 = time.perf_counter()
    res = client.chat(cfg, [{"role": "user", "content": "用一句话回答：一加一等于几？"}])
    wall = time.perf_counter() - t0

    if res.get("error"):
        lines.append("推理: 失败 —— %s" % res["error"])
        lines.append("  常见原因：timeout=%gs 对 27B 偏小（建议 ≥600），"
                     "或 max_tokens 过大导致生成超时" % timeout)
        return False, lines

    snippet = " ".join((res.get("content") or "").split())[:60]
    lines.append("推理: 成功，耗时 %.1f s" % wall)
    lines.append("  回答片段: %s" % (snippet or "(空)"))
    lines.append("  首 token 延迟 TTFT: %s" % _fmt(res.get("ttft_ms"), " ms"))
    lines.append("  生成速度: %s" % _fmt(res.get("tokens_per_sec"), " tok/s"))
    lines.append("  token 用量: prompt %s / completion %s"
                 % (_fmt(res.get("prompt_tokens")), _fmt(res.get("completion_tokens"))))

    if res.get("tokens_per_sec") is not None and res["tokens_per_sec"] < 3:
        lines.append("提示: 速度低于 3 tok/s，全量 180 题可能需要数小时；"
                     "建议减小 --limit 先试跑，或确认模型已全量卸载到 GPU（-ngl 99）")
    if not (res.get("content") or "").strip():
        lines.append("警告: 模型返回空内容。若启用了思考模式，可能是 max_tokens "
                     "被推理过程耗尽；可调大 max_tokens 或改用 enable_thinking=false")
    lines.append("")
    lines.append("可以开始测评: python3 run_eval.py --config <本配置> --limit 3")
    return True, lines