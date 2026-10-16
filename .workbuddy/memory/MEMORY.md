# 项目记忆 — ai-2080ti22g（qwen3.8-27b 本地部署与测评）

## 项目目标
在 2080Ti 22GB 设备上用 llama.cpp 部署 qwen3.8-27b，并对其进行**标准化、可复用、独立**的能力测评。

## 核心资产：model-eval 测评工具（/Users/mac/ai-2080ti22g/model-eval/）
- 入口：`python3 run_eval.py --config config.toml`（全量）/ `--only <维度>` / `--limit N` / `--resume` / `--self-test` / `--validate-datasets` / `--list-categories`
- 题库在 `datasets/`（180 题，8 维度），改题后必须跑 `--validate-datasets`（代码题新增须同步补参考答案再校验）
- 结果在 `results/run_<时间戳>/`：report.md（主报告）、results.json、responses.jsonl、aggregate.json
- 被测端点只需 OpenAI 兼容接口；local/127.x 自动绕过代理；Python 3.8+ 零依赖（<3.11 用内置迷你 TOML 解析器）
- Qwen3 思考模式：默认配置关闭（enable_thinking=false，见 config.toml）；测思考能力时打开并调大 max_tokens

## 环境事实
- 本机（macOS）无 llama.cpp 服务在跑；模型部署在 2080Ti 设备（Linux/NVIDIA 盒），测评机通过 base_url 远程指向
- 沙箱设有 http_proxy=127.0.0.1:52380，任何 localhost HTTP 请求都可能被代理拦截（工具已内置回环 bypass）
- shell grep 是 toybox，`'A\|B'` 静默失效——否定性结论必须用内置 Grep（ripgrep）交叉验证

## 重要定位：model-eval 不是行业标准基准（2026-10-16 核实）
- 本工具 180 题**自撰**，与 MMLU / C-Eval / GSM8K / HumanEval **无对应关系**；分数**不可**与公开榜单或官方 model card 比较，只能同题库 A/B（报告含数据集 SHA256 可核对）。
- 行业事实标准 = 公开基准（MMLU-Pro、C-Eval/CMMLU、GSM8K、HumanEval/EvalPlus、IFEval、LiveCodeBench…）+ **EleutherAI lm-evaluation-harness**。该 harness 官方有 `gguf` 后端专测 llama-server：`lm_eval --model gguf --model_args base_url=http://IP:8080 --tasks mmlu,ceval-valid,gsm8k,humaneval`；限制：需 llama.cpp ≥2024-12（logprobs.content 格式），不支持 perplexity 类任务。
- 已交付：`docs/标准与对比说明.md`、`scripts/run_lm_eval_baseline.sh`（一键跑标准基准），且 report.py 已在每份报告顶部与底部强制打上"非行业标准"声明——不可删，这是防止误用的护栏。
- Qwen3.8-27B 事实（2026-08-14 发布，Apache 2.0，稠密 27B + 视觉编码器，原生上下文 262,144，可切思考模式）：社区 GGUF Q4_K_M≈16.5GB / Q5_K_M≈19.8GB / Q6_K≈22.0GB / Q8_0≈29GB，另约 0.9GB 视觉投影器 → 22GB 卡推荐 Q4_K_M + 8K 上下文。官方 model card 分数（Terminal-Bench 2.1 73.0、LiveCodeBench v6 90.3、IFBench 79.5、SWE-bench Pro 61.7、CoWorkBench 70.7）均为 agentic 基准，需 agent harness，本工具不可比。

## 硬件/量化参考（2080Ti 22GB）
- 27B：Q4_K_M ≈16.5GB（推荐，余量给 KV cache 与视觉投影器）、Q5_K_M ≈19.8GB（上下文需控制到 4K-8K）；Q6_K ≈22GB 基本无余量
- 预期速度：Q4 单流约 15-30 tok/s；全量 180 题约 30-60 分钟，可 --resume 中断续跑
