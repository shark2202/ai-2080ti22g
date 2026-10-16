# -*- coding: utf-8 -*-
"""数据集校验：字段完整性、ID 唯一性、答案合法性、代码题参考答案可通过全部单测。"""

import json
import os

from .checks import CHECK_TYPES
from .runner import CATEGORY_META, DATASET_DIR, list_categories
from .scorers import run_code_tests, to_float

CHOICE_CATS = ("chinese_knowledge", "english_knowledge", "logic_reasoning")


def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise ValueError("%s 第 %d 行 JSON 解析失败: %s" % (path, ln, e))
    return rows


def validate_datasets(verbose=True):
    """校验全部数据集，返回 (是否全部通过, 报告行列表)。"""
    problems = []
    report = []
    seen_ids = {}

    for cat in list_categories():
        meta = CATEGORY_META[cat]
        path = os.path.join(DATASET_DIR, meta["file"])
        if not os.path.exists(path):
            problems.append("%s: 文件缺失" % meta["file"])
            continue
        rows = _load_jsonl(path)
        report.append("%s: %d 题" % (meta["file"], len(rows)))
        if not rows:
            problems.append("%s: 文件为空" % meta["file"])

        for i, item in enumerate(rows, 1):
            iid = item.get("id")
            where = "%s#%d(%s)" % (meta["file"], i, iid)
            if not iid:
                problems.append("%s: 缺少 id" % where)
            elif iid in seen_ids:
                problems.append("%s: id 与 %s 重复" % (where, seen_ids[iid]))
            else:
                seen_ids[iid] = where
            if item.get("category") != cat:
                problems.append("%s: category 字段(%r)与文件不符" % (where, item.get("category")))

            if cat in CHOICE_CATS:
                opts = item.get("options") or {}
                if sorted(opts.keys()) != ["A", "B", "C", "D"]:
                    problems.append("%s: 选项必须是 A/B/C/D 四项，实际 %s" % (where, sorted(opts.keys())))
                ans = str(item.get("answer", "")).upper()
                if ans not in opts:
                    problems.append("%s: 答案 %r 不在选项中" % (where, item.get("answer")))
                if not item.get("question"):
                    problems.append("%s: 缺少 question" % where)

            elif cat == "math_reasoning":
                if to_float(item.get("answer")) is None:
                    problems.append("%s: 答案 %r 无法解析为数值" % (where, item.get("answer")))
                if not item.get("question"):
                    problems.append("%s: 缺少 question" % where)

            elif cat == "code_generation":
                if not item.get("prompt"):
                    problems.append("%s: 缺少 prompt" % where)
                if not item.get("entry_point"):
                    problems.append("%s: 缺少 entry_point" % where)
                elif ("def %s" % item["entry_point"]) not in item.get("prompt", ""):
                    problems.append("%s: prompt 中未包含函数定义 %s" % (where, item["entry_point"]))
                tests = item.get("tests") or []
                if not tests:
                    problems.append("%s: 缺少 tests" % where)

            elif cat == "safety_alignment":
                if item.get("expect") not in ("refuse", "answer"):
                    problems.append("%s: expect 必须是 refuse/answer" % where)
                if item.get("expect") == "answer" and not (item.get("keywords") or []):
                    problems.append("%s: expect=answer 时应提供 keywords" % where)

            elif cat in ("instruction_following", "creative_writing"):
                checks = item.get("checks") or []
                if not checks:
                    problems.append("%s: 缺少 checks" % where)
                for c in checks:
                    if c.get("type") not in CHECK_TYPES:
                        problems.append("%s: 未知检查类型 %r" % (where, c.get("type")))
                if not (item.get("instruction") or item.get("prompt")):
                    problems.append("%s: 缺少 instruction/prompt" % where)

    # 代码题参考答案必须通过全部单测
    sol_path = os.path.join(DATASET_DIR, "code_generation.solutions.jsonl")
    if os.path.exists(sol_path):
        solutions = {r["id"]: r for r in _load_jsonl(sol_path)}
        items = {r["id"]: r for r in _load_jsonl(
            os.path.join(DATASET_DIR, CATEGORY_META["code_generation"]["file"]))}
        for iid, item in items.items():
            sol = solutions.get(iid)
            if not sol:
                problems.append("code_generation: %s 缺少参考答案" % iid)
                continue
            frac, passed, total, err = run_code_tests(sol.get("code", ""),
                                                      item.get("tests") or [], timeout=10)
            if frac < 1.0:
                problems.append("code_generation: %s 参考答案未通过全部单测（%d/%d）%s"
                                % (iid, passed, total, err or ""))
        report.append("code_generation.solutions.jsonl: %d 条参考答案" % len(solutions))
    else:
        problems.append("code_generation.solutions.jsonl: 文件缺失")

    ok = not problems
    if verbose:
        for line in report:
            print("  [数据集] " + line)
        if problems:
            print("")
            for p in problems:
                print("  [问题] " + p)
        else:
            print("  [数据集] 全部校验通过 ✓")
    return ok, problems
