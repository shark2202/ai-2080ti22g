# -*- coding: utf-8 -*-
"""运行器：加载数据集、逐题请求模型、评分、增量落盘、支持断点续跑。"""

import datetime
import json
import os
import re
import sys
import time

from . import client, prompts
from .checks import run_checks
from .scorers import score_choice, score_math, score_code, score_safety

TOOL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(TOOL_ROOT, "datasets")

CATEGORY_META = {
    "chinese_knowledge": {"name": "中文知识", "type": "choice", "file": "chinese_knowledge.jsonl"},
    "english_knowledge": {"name": "英文知识", "type": "choice", "file": "english_knowledge.jsonl"},
    "math_reasoning": {"name": "数学推理", "type": "math", "file": "math_reasoning.jsonl"},
    "code_generation": {"name": "代码生成", "type": "code", "file": "code_generation.jsonl"},
    "logic_reasoning": {"name": "逻辑推理", "type": "choice", "file": "logic_reasoning.jsonl"},
    "instruction_following": {"name": "指令遵循", "type": "checks", "file": "instruction_following.jsonl"},
    "safety_alignment": {"name": "安全对齐", "type": "safety", "file": "safety_alignment.jsonl"},
    "creative_writing": {"name": "创意写作", "type": "checks", "file": "creative_writing.jsonl"},
}


def list_categories():
    return sorted(CATEGORY_META.keys())


def category_display(cat):
    meta = CATEGORY_META.get(cat) or {}
    return "%s（%s）" % (meta.get("name", cat), cat)


def load_dataset(cat):
    path = os.path.join(DATASET_DIR, CATEGORY_META[cat]["file"])
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except ValueError as e:
                raise ValueError("%s 第 %d 行不是合法 JSON: %s" % (path, ln, e))
    return items


def score_item(item, response, cfg):
    """返回 (score, detail)。score 为 0~1。"""
    cat = item["category"]
    if cat in prompts.CHOICE_CATEGORIES:
        return score_choice(item, response)
    if cat == "math_reasoning":
        return score_math(item, response)
    if cat == "code_generation":
        return score_code(item, response, timeout=int(cfg["eval"].get("code_timeout", 15)))
    if cat in ("instruction_following", "creative_writing"):
        results, detail = run_checks(item.get("checks"), response)
        score = sum(results) / len(results) if results else 0.0
        return score, {"checks": detail}
    if cat == "safety_alignment":
        return score_safety(item, response)
    raise ValueError("未知题型: %s" % cat)


def _expected_text(item):
    cat = item["category"]
    if cat in prompts.CHOICE_CATEGORIES:
        return str(item.get("answer", ""))
    if cat == "math_reasoning":
        return str(item.get("answer", ""))
    if cat == "code_generation":
        return "%d 条单测全部通过" % len(item.get("tests") or [])
    if cat == "safety_alignment":
        return {"refuse": "应当拒答", "answer": "应当正常回答"}.get(item.get("expect"), "")
    if "checks" in item:
        return "%d 条规则全部满足" % len(item.get("checks") or [])
    return ""


def _load_completed(run_dir):
    done = {}
    path = os.path.join(run_dir, "responses.jsonl")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                done[rec.get("id")] = rec
    return done


def _latest_run_dir(output_root):
    if not os.path.isdir(output_root):
        return None
    runs = [d for d in os.listdir(output_root)
            if d.startswith("run_") and os.path.isdir(os.path.join(output_root, d))]
    if not runs:
        return None
    runs.sort()
    return os.path.join(output_root, runs[-1])


def run_eval(cfg, only=None, limit=None, resume=False, run_id=None,
             output_dir=None, progress=print):
    """执行测评，返回结果字典。"""
    eval_cfg = cfg.get("eval") or {}
    categories = only or eval_cfg.get("categories") or list_categories()
    unknown = [c for c in categories if c not in CATEGORY_META]
    if unknown:
        raise ValueError("未知测评维度: %s（可用维度: %s）" % (unknown, list_categories()))
    categories = [c for c in list_categories() if c in categories]

    out_root = output_dir or os.path.join(TOOL_ROOT, eval_cfg.get("output_dir", "results"))
    out_root = os.path.abspath(out_root)
    if resume:
        run_dir = _latest_run_dir(out_root)
        if not run_dir:
            raise ValueError("输出目录 %s 下没有可续跑的 run_* 目录" % out_root)
    else:
        stamp = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(out_root, "run_" + stamp)
    os.makedirs(run_dir, exist_ok=True)

    completed = _load_completed(run_dir)
    responses_path = os.path.join(run_dir, "responses.jsonl")
    out_file = open(responses_path, "a", encoding="utf-8")

    model_info = client.fetch_model_info(cfg)
    records = []
    total_items = 0
    t_start = time.perf_counter()

    try:
        for cat in categories:
            items = load_dataset(cat)
            if limit:
                items = items[: int(limit)]
            total_items += len(items)
            progress("== 维度 %s：共 %d 题 ==" % (category_display(cat), len(items)))
            for idx, item in enumerate(items, 1):
                if item["id"] in completed:
                    rec = completed[item["id"]]
                    records.append(rec)
                    progress("  [%s] 已存在，跳过（score=%.2f）" % (item["id"], rec.get("score") or 0))
                    continue
                messages = prompts.build_messages(item, cfg)
                result = client.chat(cfg, messages)
                if result.get("error"):
                    score, detail = 0.0, {"error": result["error"]}
                    status = "error"
                else:
                    score, detail = score_item(item, result["content"], cfg)
                    status = "ok"
                rec = {
                    "id": item["id"],
                    "category": cat,
                    "question": item.get("question") or item.get("instruction") or item.get("prompt"),
                    "expected": _expected_text(item),
                    "response": result.get("content") or "",
                    "score": round(float(score), 4),
                    "passed": float(score) >= 1.0,
                    "status": status,
                    "error": result.get("error"),
                    "detail": detail,
                    "metrics": {
                        "ttft_ms": result.get("ttft_ms"),
                        "total_ms": result.get("total_ms"),
                        "tokens_per_sec": result.get("tokens_per_sec"),
                        "prompt_tokens": result.get("prompt_tokens"),
                        "completion_tokens": result.get("completion_tokens"),
                    },
                }
                records.append(rec)
                out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_file.flush()
                status_mark = "✓" if rec["passed"] else ("✗" if status == "ok" else "!")
                perf = ""
                if result.get("tokens_per_sec"):
                    perf = " %.1f tok/s" % result["tokens_per_sec"]
                if result.get("total_ms"):
                    perf += " %.1fs" % (result["total_ms"] / 1000.0)
                progress("  [%d/%d] %s %s score=%.2f%s"
                         % (idx, len(items), item["id"], status_mark, score, perf))
    finally:
        out_file.close()

    wall = time.perf_counter() - t_start
    summary = {
        "tool": "model-eval 1.0.0",
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 1),
        "run_dir": run_dir,
        "model": {
            "config_name": cfg["model"].get("name"),
            "reported_name": model_info.get("reported_name"),
            "model_path": model_info.get("model_path"),
            "endpoint": cfg["server"]["base_url"],
        },
        "generation": {
            "temperature": cfg["generation"].get("temperature"),
            "top_p": cfg["generation"].get("top_p"),
            "max_tokens": cfg["generation"].get("max_tokens"),
            "chat_template_kwargs": cfg["model"].get("chat_template_kwargs") or {},
        },
        "total_items": total_items,
        "completed_items": len(records),
        "records": records,
    }
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
