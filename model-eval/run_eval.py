#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model-eval 命令行入口。

用法：
    python3 run_eval.py --config config.toml            # 全量测评
    python3 run_eval.py --config config.toml --only math_reasoning,code_generation
    python3 run_eval.py --config config.toml --limit 5  # 每个维度只跑前 5 题
    python3 run_eval.py --config config.toml --resume   # 续跑未完成的 run
    python3 run_eval.py --check-endpoint               # 端点连通性 + 试跑一次推理
    python3 run_eval.py --self-test                     # 框架自检（不依赖模型）
    python3 run_eval.py --validate-datasets             # 数据集完整性校验
    python3 run_eval.py --list-categories               # 列出全部测评维度
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evalkit import config as config_mod          # noqa: E402
from evalkit.report import aggregate, print_console, write_report  # noqa: E402
from evalkit.runner import list_categories, run_eval  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="面向 llama.cpp / OpenAI 兼容端点的标准化模型能力测评（可复用、独立）")
    parser.add_argument("--config", "-c", default="config.toml", help="配置文件路径（默认 ./config.toml）")
    parser.add_argument("--only", help="只运行指定维度，逗号分隔")
    parser.add_argument("--limit", type=int, help="每个维度只跑前 N 题")
    parser.add_argument("--resume", action="store_true", help="续跑输出目录中最近一次未完成的测评")
    parser.add_argument("--run-id", help="自定义运行目录名（run_<id>）")
    parser.add_argument("--output-dir", help="输出目录（默认 results/）")
    parser.add_argument("--self-test", action="store_true", help="框架自检：评分器单测 + mock 服务端全链路测试")
    parser.add_argument("--validate-datasets", action="store_true", help="校验数据集完整性（含代码题参考答案执行验证）")
    parser.add_argument("--check-endpoint", action="store_true",
                        help="检查端点连通性并试跑一次推理（跑测评前建议先执行）")
    parser.add_argument("--no-probe", action="store_true",
                        help="配合 --check-endpoint：只探活不发推理请求")
    parser.add_argument("--list-categories", action="store_true", help="列出全部测评维度")
    args = parser.parse_args(argv)

    if args.list_categories:
        for c in list_categories():
            print(c)
        return 0

    if args.self_test:
        from evalkit.selftest import run_self_test
        return 0 if run_self_test() else 1

    if args.validate_datasets:
        from evalkit.validate import validate_datasets
        ok, _ = validate_datasets()
        return 0 if ok else 1

    if not os.path.exists(args.config):
        print("配置文件不存在: %s\n请复制 config.example.toml 为 config.toml 后修改。" % args.config)
        return 2

    cfg = config_mod.load_config(args.config)

    if args.check_endpoint:
        from evalkit.preflight import check_endpoint
        ok, lines = check_endpoint(cfg, probe=not args.no_probe)
        for ln in lines:
            print(ln)
        return 0 if ok else 1

    only = [c.strip() for c in args.only.split(",") if c.strip()] if args.only else None

    print("模型能力标准化测评")
    print("  端点: %s" % cfg["server"]["base_url"])
    print("  模型: %s" % cfg["model"]["name"])
    print("  维度: %s" % (", ".join(only) if only else ", ".join(cfg["eval"]["categories"])))
    print("")

    summary = run_eval(cfg, only=only, limit=args.limit, resume=args.resume,
                       run_id=args.run_id, output_dir=args.output_dir)
    agg = aggregate(summary)
    write_report(summary, agg)
    print_console(summary, agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
