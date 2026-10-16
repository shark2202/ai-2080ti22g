# -*- coding: utf-8 -*-
"""一次性修复脚本：打乱选择题正确选项的位置。

背景：首版题库在人工撰写时把正确答案习惯性放在了 B 位
（中文知识 28/40=B、英文知识 26/40=B、逻辑推理 13/20=B），
导致"闭眼恒答 B"即可获得 65%~70% 的分数，测评失去区分度。

做法：对每道选择题，用题号派生的确定性种子重新排列options 顺序，
保证同一题在任何机器上重跑得到同一结果（可复现），
并使正确选项在 A/B/C/D 上尽量均匀分布。

用法：python3 scripts/balance_answer_positions.py [--dry-run]
"""

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "datasets")
TARGETS = [
    "chinese_knowledge.jsonl",
    "english_knowledge.jsonl",
    "logic_reasoning.jsonl",
]
LETTERS = ["A", "B", "C", "D"]


def balance(path, dry_run=False):
    name = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(ln) for ln in f if ln.strip()]

    before = {}
    for i in items:
        before[i["answer"]] = before.get(i["answer"], 0) + 1

    changed = 0
    # 严格轮转分配目标字母：A,B,C,D,A,B,... 保证分布极差 ≤1
    # （随机分配会残留偏斜，轮转才是真均匀）
    order = sorted(items, key=lambda x: x["id"])
    for idx, item in enumerate(order):
        old = item["answer"]
        options = item["options"]
        if set(options) != set(LETTERS):
            raise SystemExit("%s: %s 的选项键不是 A-D：%s" % (name, item["id"], sorted(options)))
        target = LETTERS[idx % len(LETTERS)]
        # 重排选项，使正确内容落到 target 位；用题号派生种子保证可复现
        correct_value = options[old]
        others = [options[L] for L in LETTERS if L != old]
        rng = random.Random("balance-v1:" + item["id"])
        rng.shuffle(others)
        new_options = {}
        it = iter(others)
        for L in LETTERS:
            new_options[L] = correct_value if L == target else next(it)
        item["options"] = new_options
        item["answer"] = target
        if target != old:
            changed += 1

    after = {}
    for i in items:
        after[i["answer"]] = after.get(i["answer"], 0) + 1

    spread_before = max(before.values()) - min(before.values())
    spread_after = max(after.values()) - min(after.values())
    print("%s:" % name)
    print("  修复前分布: %s  (极差 %d)" % (dict(sorted(before.items())), spread_before))
    print("  修复后分布: %s  (极差 %d)" % (dict(sorted(after.items())), spread_after))
    print("  重排了答案位置的题数: %d / %d" % (changed, len(items)))

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("  已写回")
    return spread_before, spread_after


def main():
    ap = argparse.ArgumentParser(description="打乱选择题正确选项位置，消除位置偏置")
    ap.add_argument("--dry-run", action="store_true", help="只报告将发生的变化，不写文件")
    args = ap.parse_args()

    total_before = total_after = 0
    for fn in TARGETS:
        path = os.path.join(DATASETS, fn)
        if not os.path.exists(path):
            print("跳过（文件不存在）: %s" % fn)
            continue
        sb, sa = balance(path, dry_run=args.dry_run)
        total_before += sb
        total_after += sa
        print("")

    print("合计极差: %d -> %d" % (total_before, total_after))
    return 0


if __name__ == "__main__":
    sys.exit(main())