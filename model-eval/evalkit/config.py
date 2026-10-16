# -*- coding: utf-8 -*-
"""配置加载：TOML 配置（Python 3.11+ 用标准库 tomllib，旧版用内置迷你解析器）。"""

import os

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - 旧版 Python 走迷你解析器
    tomllib = None

# ---------------------------------------------------------------- 迷你 TOML 解析器
# 仅支持本工具配置用到的子集：section、key = value、字符串/数字/布尔/
# 字符串数组（可跨行）、行内表。当运行环境没有 tomllib 时启用。

def _strip_comment(line):
    out, in_str, quote, i = [], False, "", 0
    while i < len(line):
        c = line[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
            if c == quote:
                in_str = False
        else:
            if c in "\"'":
                in_str, quote = True, c
                out.append(c)
            elif c == "#":
                break
            else:
                out.append(c)
        i += 1
    return "".join(out)


def _brackets_balanced(s):
    depth, in_str, quote, i = 0, False, "", 0
    while i < len(s):
        c = s[i]
        if in_str:
            if c == "\\" and i + 1 < len(s):
                i += 2
                continue
            if c == quote:
                in_str = False
        else:
            if c in "\"'":
                in_str, quote = True, c
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth < 0:
                    return False
        i += 1
    return depth == 0 and not in_str


def _split_top(s):
    parts, depth, in_str, quote, cur, i = [], 0, False, "", [], 0
    while i < len(s):
        c = s[i]
        if in_str:
            cur.append(c)
            if c == "\\" and i + 1 < len(s):
                cur.append(s[i + 1])
                i += 2
                continue
            if c == quote:
                in_str = False
        else:
            if c in "\"'":
                in_str, quote = True, c
                cur.append(c)
            elif c in "[{":
                depth += 1
                cur.append(c)
            elif c in "]}":
                depth -= 1
                cur.append(c)
            elif c == "," and depth == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(c)
        i += 1
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _parse_scalar(s):
    s = s.strip()
    if not s:
        return ""
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _parse_value(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return [_parse_value(x) for x in _split_top(s[1:-1])]
    if s.startswith("{") and s.endswith("}"):
        d = {}
        for part in _split_top(s[1:-1]):
            k, _, v = part.partition("=")
            d[k.strip().strip("\"'")] = _parse_value(v)
        return d
    return _parse_scalar(s)


def parse_toml(text):
    """解析 TOML 子集（无 tomllib 时使用）。"""
    root, cur, buf = {}, {}, None
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if buf is not None:
            buf += " " + line.strip()
            if _brackets_balanced(buf):
                k, _, v = buf.partition("=")
                cur[k.strip().strip("\"'")] = _parse_value(v)
                buf = None
            continue
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            cur = root
            for part in stripped[1:-1].split("."):
                cur = cur.setdefault(part.strip().strip("\"'"), {})
            continue
        if "=" in stripped:
            k, _, v = stripped.partition("=")
            v = v.strip()
            if v.startswith("[") and not _brackets_balanced(v):
                buf = stripped
            else:
                cur[k.strip().strip("\"'")] = _parse_value(v)
    return root


# ---------------------------------------------------------------- 默认配置

DEFAULT_CONFIG = {
    "server": {
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "",
        "timeout": 600,
    },
    "model": {
        "name": "unknown",
        "chat_template_kwargs": {},
    },
    "generation": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "stop": [],
    },
    "prompts": {
        "system": "",
        "choice": (
            "你正在参加一场标准化知识测评。请阅读下面的单选题，"
            "直接给出正确选项的字母，不要解释。\n\n"
            "题目：{question}\n{options}\n请直接回答选项字母（A/B/C/D）："
        ),
        "math": (
            "请仔细阅读并解答下面的数学题。解答完成后，"
            "在最后一行以“答案：<结果>”的格式给出最终结果。\n\n"
            "题目：{question}"
        ),
        "code": (
            "请阅读下面的 Python 函数需求并补全实现。"
            "只输出完整的 Python 代码（可放在一个代码块中），不要输出解释文字。\n\n"
            "{prompt}"
        ),
    },
    "eval": {
        "categories": [
            "chinese_knowledge",
            "english_knowledge",
            "math_reasoning",
            "code_generation",
            "logic_reasoning",
            "instruction_following",
            "safety_alignment",
            "creative_writing",
        ],
        "code_timeout": 15,
        "output_dir": "results",
    },
}


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    """加载配置文件并与默认配置合并。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if tomllib is not None:
        data = tomllib.loads(text)
    else:
        data = parse_toml(text)
    cfg = _merge(DEFAULT_CONFIG, data)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_config_text"] = text
    return cfg
