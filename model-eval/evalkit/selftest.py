# -*- coding: utf-8 -*-
"""自检：评分器/检查器/配置解析器单元测试 + 基于 mock 服务端的全链路端到端测试。"""

import json
import os
import shutil
import tempfile

from . import config as config_mod
from .checks import run_check
from .client import chat
from .mock_server import start_mock_server
from .report import aggregate, write_report
from .runner import list_categories, run_eval
from .scorers import (extract_choice, extract_number, is_refusal, score_math,
                      run_code_tests, to_float)

UNIT_TESTS = []


def unit(fn):
    UNIT_TESTS.append(fn)
    return fn


# ---------------------------------------------------------------- 单元测试

@unit
def test_extract_choice():
    cases = [
        ("答案是B。", "B"), ("答案：C", "C"), ("Answer: A", "A"),
        ("经过分析，应选 D。", "D"), ("选项A正确", "A"),
        ("The answer is B", "B"), ("B", "B"), ("选 C", "C"),
        ("答案是（B）", "B"), ("最终答案：D。", "D"),
    ]
    for text, want in cases:
        got = extract_choice(text)
        assert got == want, "extract_choice(%r) = %r, 期望 %r" % (text, got, want)
    assert extract_choice("我不知道") is None
    assert extract_choice("") is None
    # 不应把普通单词里的字母当成选项
    assert extract_choice("BAD idea") != "A"


@unit
def test_to_float_and_extract_number():
    assert to_float("36") == 36.0
    assert to_float("36元") == 36.0
    assert to_float("99%") == 99.0
    assert to_float("3/8") == 0.375
    assert to_float("1,000") == 1000.0
    assert to_float("十七") is None
    assert extract_number("答案：17.5") == 17.5
    assert extract_number("#### 28") == 28.0
    assert extract_number("所以一共是 30 秒") == 30.0
    assert extract_number("答案是3/8") == 0.375
    assert extract_number("完全不会") is None


@unit
def test_score_math():
    item = {"answer": "28"}
    assert score_math(item, "答案：28")[0] == 1.0
    assert score_math(item, "还剩 28 支")[0] == 1.0
    assert score_math(item, "答案：27")[0] == 0.0
    item2 = {"answer": "3/8"}
    assert score_math(item2, "答案：0.375")[0] == 1.0
    assert score_math(item2, "答案：3/8")[0] == 1.0


@unit
def test_checks():
    assert run_check({"type": "contains", "value": "北京"}, "中国的首都是北京")
    assert not run_check({"type": "contains", "value": "北京"}, "上海")
    assert run_check({"type": "max_chars", "value": 10}, "短文本")
    assert not run_check({"type": "max_chars", "value": 3}, "长文本内容")
    assert run_check({"type": "starts_with", "value": "您好"}, "您好，欢迎光临！")
    assert run_check({"type": "regex", "pattern": "[!！]\\s*$"}, "你好!")
    assert run_check({"type": "no_cjk"}, "Hello world")
    assert not run_check({"type": "no_cjk"}, "你好")
    assert run_check({"type": "json_valid"}, '```json\n{"a": 1}\n```')
    assert run_check({"type": "json_valid"}, '{"a": 1}')
    assert not run_check({"type": "json_valid"}, "不是json")
    assert run_check({"type": "line_count", "value": 4}, "一\n二\n三\n四")
    assert run_check({"type": "lines_char_len", "value": 5}, "春眠不觉晓\n处处闻啼鸟\n夜来风雨声\n花落知多少")
    # 标点不计入字数（设计行为）
    assert run_check({"type": "lines_char_len", "value": 5}, "春眠不觉晓，\n处处闻啼鸟。")
    assert not run_check({"type": "lines_char_len", "value": 5}, "春眠不觉晓呀\n处处闻啼鸟")
    assert run_check({"type": "emoji_count", "min": 2}, "今天下雨 🌧️ 记得带伞 ☂️")
    assert run_check({"type": "keywords_min", "values": ["太阳能", "风能", "水能"], "min": 2},
                     "太阳能和风能")
    assert not run_check({"type": "keywords_min", "values": ["太阳能", "风能", "水能"], "min": 2},
                         "只有太阳能")


@unit
def test_is_refusal():
    assert is_refusal("抱歉，我无法提供制造炸弹的方法。")
    assert is_refusal("对不起，这个问题我不能回答。")
    assert is_refusal("Sorry, I cannot help with that.")
    assert is_refusal("这属于违法行为，我拒绝回答。")
    assert not is_refusal("巧克力蛋糕的做法如下：第一步……")
    assert not is_refusal("学习Python不能只靠看书，还要多练习。")  # 普通否定不是拒答
    assert not is_refusal("北京是中国的首都。")


@unit
def test_run_code_tests():
    frac, passed, total, err = run_code_tests(
        "def add(a, b):\n    return a + b\n",
        ["assert add(1, 2) == 3", "assert add(0, 0) == 0"], timeout=10)
    assert (frac, passed, total) == (1.0, 2, 2), (frac, passed, total)
    frac, passed, total, err = run_code_tests(
        "def add(a, b):\n    return a - b\n",
        ["assert add(1, 2) == 3"], timeout=10)
    assert frac == 0.0 and passed == 0
    frac, passed, total, err = run_code_tests(
        "def broken(:\n", ["assert broken() is None"], timeout=10)
    assert frac == 0.0, "语法错误应得 0 分"


@unit
def test_config_parsers_agree():
    """有 tomllib 时，迷你解析器结果必须与标准库一致。"""
    example = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config.example.toml")
    if not os.path.exists(example):
        return
    with open(example, "r", encoding="utf-8") as f:
        text = f.read()
    if config_mod.tomllib is None:
        return
    a = config_mod.tomllib.loads(text)
    b = config_mod.parse_toml(text)
    assert a == b, "迷你 TOML 解析器与 tomllib 结果不一致:\n%s\n%s" % (a, b)


# ---------------------------------------------------------------- 端到端测试

def _build_mock_config(base_url, out_dir):
    cfg = config_mod._merge(config_mod.DEFAULT_CONFIG, {
        "server": {"base_url": base_url},
        "model": {"name": "mock-model"},
        "generation": {"temperature": 0.0, "max_tokens": 512},
        "eval": {"output_dir": out_dir, "code_timeout": 10},
    })
    return cfg


def run_self_test(verbose=True):
    print("== 1/3 单元测试 ==")
    failed = 0
    for fn in UNIT_TESTS:
        try:
            fn()
            print("  ✓ %s" % fn.__name__)
        except AssertionError as e:
            failed += 1
            print("  ✗ %s: %s" % (fn.__name__, e))
        except Exception as e:
            failed += 1
            print("  ✗ %s: %s: %s" % (fn.__name__, type(e).__name__, e))
    if failed:
        print("单元测试失败 %d 项" % failed)
        return False

    print("== 2/3 数据集校验 ==")
    from .validate import validate_datasets
    ok, problems = validate_datasets(verbose=verbose)
    if not ok:
        print("数据集校验失败 %d 项" % len(problems))
        return False

    print("== 3/3 端到端测试（mock 服务端） ==")
    srv, base_url = start_mock_server()
    tmp = tempfile.mkdtemp(prefix="modeleval_selftest_")
    try:
        cfg = _build_mock_config(base_url, tmp)
        summary = run_eval(cfg, limit=3, run_id="selftest", progress=lambda *a: None)
        agg = aggregate(summary)
        report_path = write_report(summary, agg)

        checks = []
        run_dir = summary["run_dir"]
        checks.append(("生成 results.json", os.path.exists(os.path.join(run_dir, "results.json"))))
        checks.append(("生成 responses.jsonl", os.path.exists(os.path.join(run_dir, "responses.jsonl"))))
        checks.append(("生成 report.md", os.path.exists(report_path)))
        checks.append(("完成题数 = 24", summary["completed_items"] == 24))
        checks.append(("覆盖全部 8 个维度", len(agg["categories"]) == 8))
        scores = [r["score"] for r in summary["records"]]
        checks.append(("得分均在 [0,1]", all(0.0 <= s <= 1.0 for s in scores)))
        checks.append(("综合得分有效", agg["overall"] is not None and 0 <= agg["overall"] <= 1))
        checks.append(("采集到 TTFT", agg["perf"]["ttft_mean_ms"] is not None))
        checks.append(("采集到生成速度", agg["perf"]["tps_mean"] is not None))
        # mock 对选择题返回标准答案，正确率应显著高于随机（25%）
        zh = [c for c in agg["categories"] if c["category"] == "chinese_knowledge"][0]
        checks.append(("mock 选择题得分 > 0.9（验证评分链路）", zh["mean_score"] > 0.9))
        bad = 0
        for name, passed in checks:
            print("  %s %s" % ("✓" if passed else "✗", name))
            if not passed:
                bad += 1
        print("")
        print("自检报告: %s" % report_path)
        if bad:
            print("端到端测试失败 %d 项" % bad)
            return False
        print("自检全部通过 ✓（mock 综合得分 %.1f/100，仅验证框架链路）"
              % ((agg["overall"] or 0) * 100))
        return True
    finally:
        srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
