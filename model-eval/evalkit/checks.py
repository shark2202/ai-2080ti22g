# -*- coding: utf-8 -*-
"""规则检查器：用于指令遵循 / 创意写作等开放式题型的逐条规则评分。"""

import json
import re

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]"
)
# 判断“每句字数”时忽略的空白与中文标点
PUNCT_RE = re.compile(r"[\s，。！？、；：""''（）《》〈〉【】…—·]")

CHECK_TYPES = (
    "contains", "not_contains", "contains_any", "contains_all", "keywords_min",
    "regex", "regex_not", "regex_count", "starts_with", "ends_with",
    "max_chars", "min_chars", "line_count", "lines_char_len", "json_valid",
    "no_cjk", "has_cjk", "emoji_count",
)


def _nonempty_lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def run_check(spec, text):
    """执行单条规则检查，返回 True/False。未知类型抛 ValueError。"""
    t = spec.get("type")
    v = spec.get("value")
    if t == "contains":
        return v in text
    if t == "not_contains":
        return v not in text
    if t == "contains_any":
        return any(x in text for x in spec.get("values", []))
    if t == "contains_all":
        return all(x in text for x in spec.get("values", []))
    if t == "keywords_min":
        values = spec.get("values", [])
        need = spec.get("min", 1)
        hits = sum(1 for x in values if x in text)
        return hits >= need
    if t == "regex":
        return re.search(spec.get("pattern", ""), text) is not None
    if t == "regex_not":
        return re.search(spec.get("pattern", ""), text) is None
    if t == "regex_count":
        return len(re.findall(spec.get("pattern", ""), text)) >= spec.get("min", 1)
    if t == "starts_with":
        return text.strip().startswith(v)
    if t == "ends_with":
        return text.strip().endswith(v)
    if t == "max_chars":
        return len(text.strip()) <= v
    if t == "min_chars":
        return len(text.strip()) >= v
    if t == "line_count":
        return len(_nonempty_lines(text)) == v
    if t == "lines_char_len":
        lines = [PUNCT_RE.sub("", l) for l in _nonempty_lines(text)]
        return bool(lines) and all(len(l) == v for l in lines)
    if t == "json_valid":
        s = text.strip()
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.S)
        if m:
            s = m.group(1)
        try:
            json.loads(s)
            return True
        except Exception:
            return False
    if t == "no_cjk":
        return CJK_RE.search(text) is None
    if t == "has_cjk":
        return CJK_RE.search(text) is not None
    if t == "emoji_count":
        return len(EMOJI_RE.findall(text)) >= spec.get("min", 1)
    raise ValueError("未知检查类型: %r" % (t,))


def run_checks(specs, text):
    """执行全部检查，返回 (results: list[bool], detail: list[dict])。"""
    results, detail = [], []
    for spec in specs or []:
        ok = run_check(spec, text)
        results.append(bool(ok))
        detail.append({"check": spec, "pass": bool(ok)})
    return results, detail
