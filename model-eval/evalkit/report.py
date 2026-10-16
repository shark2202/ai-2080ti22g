# -*- coding: utf-8 -*-
"""报告生成：按维度聚合得分与性能指标，输出 Markdown 报告与控制台摘要。"""

import hashlib
import json
import os

from .runner import CATEGORY_META, category_display


def _percentile(values, p):
    if not values:
        return None
    vs = sorted(values)
    k = max(0, min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1)))))
    return vs[k]


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(v, digits=1):
    return "-" if v is None else ("%.*f" % (digits, v))


def aggregate(summary):
    """按维度聚合得分与性能指标。"""
    by_cat = {}
    for rec in summary["records"]:
        by_cat.setdefault(rec["category"], []).append(rec)

    cats = []
    for cat in CATEGORY_META:
        recs = by_cat.get(cat) or []
        if not recs:
            continue
        n = len(recs)
        scores = [r["score"] for r in recs]
        passed = sum(1 for r in recs if r["passed"])
        errors = sum(1 for r in recs if r["status"] != "ok")
        ttfts = [r["metrics"]["ttft_ms"] for r in recs if r["metrics"].get("ttft_ms") is not None]
        tps = [r["metrics"]["tokens_per_sec"] for r in recs if r["metrics"].get("tokens_per_sec")]
        totals = [r["metrics"]["total_ms"] for r in recs if r["metrics"].get("total_ms")]
        cats.append({
            "category": cat,
            "display": category_display(cat),
            "count": n,
            "mean_score": sum(scores) / n,
            "passed": passed,
            "errors": errors,
            "ttft_mean_ms": _mean(ttfts),
            "ttft_p50_ms": _percentile(ttfts, 50),
            "ttft_p95_ms": _percentile(ttfts, 95),
            "tps_mean": _mean(tps),
            "tps_p50": _percentile(tps, 50),
            "total_mean_s": (_mean(totals) or 0) / 1000.0,
        })

    overall = _mean([c["mean_score"] for c in cats])
    all_ttft = [r["metrics"]["ttft_ms"] for r in summary["records"]
                if r["metrics"].get("ttft_ms") is not None]
    all_tps = [r["metrics"]["tokens_per_sec"] for r in summary["records"]
               if r["metrics"].get("tokens_per_sec")]
    completion_tokens = sum(r["metrics"].get("completion_tokens") or 0
                            for r in summary["records"])
    perf = {
        "ttft_mean_ms": _mean(all_ttft),
        "ttft_p50_ms": _percentile(all_ttft, 50),
        "ttft_p95_ms": _percentile(all_ttft, 95),
        "tps_mean": _mean(all_tps),
        "tps_p50": _percentile(all_tps, 50),
        "tps_p95": _percentile(all_tps, 95),
        "completion_tokens_total": completion_tokens,
    }
    return {"categories": cats, "overall": overall, "perf": perf}


def dataset_hashes():
    """计算数据集文件哈希，保证报告可复现、可追溯。"""
    from .runner import DATASET_DIR
    hashes = {}
    for fn in sorted(os.listdir(DATASET_DIR)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(DATASET_DIR, fn), "rb") as f:
                hashes[fn] = hashlib.sha256(f.read()).hexdigest()[:16]
    return hashes


def _truncate(s, n=120):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def render_markdown(summary, agg):
    m = summary["model"]
    lines = []
    lines.append("# 模型能力标准化测评报告")
    lines.append("")
    lines.append("> ⚠️ **本报告来自自建回归测评套件（model-eval），不是行业标准基准。**")
    lines.append("> 分数仅在「同一题库（见第四节数据集哈希）相同」的前提下可横向对比；")
    lines.append("> **不可**与 MMLU / C-Eval / GSM8K / HumanEval 等公开榜单或官方 model card 分数直接比较。")
    lines.append("> 需要对外可比的标准分数请用 EleutherAI lm-evaluation-harness，详见 `docs/标准与对比说明.md`。")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("| --- | --- |")
    lines.append("| 模型（配置名） | %s |" % m.get("config_name"))
    lines.append("| 模型（服务端上报） | %s |" % (m.get("reported_name") or "-"))
    lines.append("| 模型文件 | %s |" % (m.get("model_path") or "-"))
    lines.append("| 服务端点 | %s |" % m.get("endpoint"))
    lines.append("| 测评时间 | %s |" % summary.get("started_at"))
    lines.append("| 题目总数 | %d |" % summary.get("total_items"))
    lines.append("| 请求失败 | %d |" % sum(c["errors"] for c in agg["categories"]))
    lines.append("| 总耗时 | %.1f 秒 |" % summary.get("wall_seconds", 0))
    lines.append("| **综合得分** | **%.1f / 100** |" % ((agg["overall"] or 0) * 100))
    lines.append("| 生成参数 | temperature=%s, top_p=%s, max_tokens=%s |"
                 % (summary["generation"]["temperature"], summary["generation"]["top_p"],
                    summary["generation"]["max_tokens"]))
    lines.append("")
    lines.append("## 一、能力维度得分")
    lines.append("")
    lines.append("| 维度 | 得分 | 题数 | 完全通过 | 通过率 | 请求失败 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for c in agg["categories"]:
        lines.append("| %s | %.1f | %d | %d | %.1f%% | %d |"
                     % (c["display"], c["mean_score"] * 100, c["count"], c["passed"],
                        100.0 * c["passed"] / c["count"], c["errors"]))
    lines.append("")
    p = agg["perf"]
    lines.append("## 二、性能指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append("| 首 token 延迟 TTFT（平均 / P50 / P95） | %s / %s / %s ms |"
                 % (_fmt(p["ttft_mean_ms"]), _fmt(p["ttft_p50_ms"]), _fmt(p["ttft_p95_ms"])))
    lines.append("| 生成速度（平均 / P50 / P95） | %s / %s / %s tok/s |"
                 % (_fmt(p["tps_mean"], 2), _fmt(p["tps_p50"], 2), _fmt(p["tps_p95"], 2)))
    lines.append("| 生成 token 总量 | %d |" % p["completion_tokens_total"])
    lines.append("| 单题平均耗时 | %s 秒 |" % _fmt(summary.get("wall_seconds", 0) /
                                                  max(1, summary.get("completed_items", 1))))
    lines.append("")
    lines.append("## 三、错题与异常明细")
    for c in agg["categories"]:
        recs = [r for r in summary["records"]
                if r["category"] == c["category"] and not r["passed"]]
        lines.append("")
        lines.append("### %s（%.1f 分，错 %d 题）"
                     % (c["display"], c["mean_score"] * 100, len(recs)))
        if not recs:
            lines.append("")
            lines.append("全部通过。")
            continue
        lines.append("")
        for r in recs[:20]:
            if r["status"] != "ok":
                lines.append("- `%s` 请求失败：%s" % (r["id"], _truncate(r.get("error"), 100)))
                continue
            lines.append("- `%s` 期望「%s」，模型回答：「%s」（score=%.2f）"
                         % (r["id"], _truncate(r["expected"], 40),
                            _truncate(r["response"], 80), r["score"]))
        if len(recs) > 20:
            lines.append("- ……其余 %d 题见 responses.jsonl" % (len(recs) - 20))
    lines.append("")
    lines.append("## 四、数据集版本（SHA256 前16位）")
    lines.append("")
    lines.append("| 文件 | 哈希 |")
    lines.append("| --- | --- |")
    for fn, h in dataset_hashes().items():
        lines.append("| %s | `%s` |" % (fn, h))
    lines.append("")
    lines.append("---")
    lines.append("评分方法：客观题按标准答案自动比对；代码题在隔离子进程中运行单测；"
                 "指令/创作题按规则检查逐条计分；安全题按拒答识别与合规回答识别计分。"
                 "综合得分为各维度得分的等权平均。")
    lines.append("")
    lines.append("定位声明：model-eval 是自建的回归测评套件，题库为本工具自撰（题型风格参照公开基准范式，"
                 "但与任何公开基准无对应关系）。本报告用于同一题库下的版本/量化/参数 A/B 回归，"
                 "不构成行业标准测评结论。")
    lines.append("")
    return "\n".join(lines)


def write_report(summary, agg):
    run_dir = summary["run_dir"]
    md = render_markdown(summary, agg)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(run_dir, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    return os.path.join(run_dir, "report.md")


def print_console(summary, agg):
    print("")
    print("=" * 64)
    print("测评完成：%s" % summary["model"].get("reported_name") or summary["model"]["config_name"])
    print("=" * 64)
    print("%-28s %8s %6s %8s %10s" % ("维度", "得分", "题数", "通过率", "tok/s"))
    for c in agg["categories"]:
        print("%-28s %7.1f %6d %7.1f%% %10s"
              % (c["display"], c["mean_score"] * 100, c["count"],
                 100.0 * c["passed"] / c["count"], _fmt(c["tps_mean"], 1)))
    print("-" * 64)
    p = agg["perf"]
    print("综合得分: %.1f / 100    TTFT均值: %s ms    生成速度均值: %s tok/s"
          % ((agg["overall"] or 0) * 100, _fmt(p["ttft_mean_ms"]), _fmt(p["tps_mean"], 1)))
    print("报告目录: %s" % summary["run_dir"])
