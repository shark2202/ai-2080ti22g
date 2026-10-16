# model-eval — 本地大模型标准化能力测评工具

针对 **llama.cpp 部署的 qwen3.8-27b（2080Ti 22GB 设备）** 的标准化能力测评工具。
设计目标：**标准化**（固定题库、固定提示词、温度 0、可复现）、**可复用**（换模型/换端点只改配置）、**独立**（只依赖 OpenAI 兼容 HTTP 接口，不绑定 llama.cpp；零第三方 Python 依赖，Python 3.8+ 即可运行）。

## 〇、定位与可比性（先读这一节）

- **本工具不是行业标准基准。** 它是**自建的回归测评套件**：180 道题由本工具自撰，题型风格参照公开基准范式，但与 MMLU / C-Eval / GSM8K / HumanEval 等**无任何对应关系**。
- **能比**：同一题库（报告内含数据集 SHA256 哈希可核对）下的 A/B —— 换量化、换上下文、思考模式开关、llama.cpp 版本、模型换版等。
- **不能比**：任何公开榜单或官方 model card 分数。本工具的"中文知识 82 分"**不是** MMLU 82 分。
- **要对外可比的标准分数**：用行业事实标准 **EleutherAI lm-evaluation-harness**，本目录提供一键脚本：

```bash
bash scripts/run_lm_eval_baseline.sh http://<2080Ti设备IP>:8080
PROFILE=full bash scripts/run_lm_eval_baseline.sh http://<IP>:8080   # 更全面但更慢
```

完整说明（事实标准是什么、官方 model card 分数对照、污染与后端差异等）见 **`docs/标准与对比说明.md`**；
安装步骤、任务档位、后端参数格式、结果解读见**`docs/标准基准测评指南.md`**。

> 本机（macOS + 沙箱环境）实测**装不上** lm_eval（沙箱拒绝创建带连字符的目录），
> 请在 2080Ti 上建 venv 安装并运行，详见标准基准测评指南。

## 一、测评维度与评分方法（共 180 题）

| 维度 | 文件 | 题数 | 题型 | 评分方式 |
| --- | --- | --- | --- | --- |
| 中文知识 | `datasets/chinese_knowledge.jsonl` | 40 | 单选 A-D | 抽取答案字母与标准答案比对 |
| 英文知识 | `datasets/english_knowledge.jsonl` | 40 | 单选 A-D | 同上（MMLU 风格） |
| 数学推理 | `datasets/math_reasoning.jsonl` | 20 | 解答题 | 抽取最终数值（支持分数/百分数/千分位），容差 1e-4 |
| 代码生成 | `datasets/code_generation.jsonl` | 20 | 函数补全 | 隔离子进程运行断言单测，按通过率计分 |
| 逻辑推理 | `datasets/logic_reasoning.jsonl` | 20 | 单选 A-D | 同知识题（三段论/数列/真假话） |
| 指令遵循 | `datasets/instruction_following.jsonl` | 16 | 约束指令 | 逐条规则检查（字数/格式/JSON/emoji 等），按通过比例计分 |
| 安全对齐 | `datasets/safety_alignment.jsonl` | 16 | 拒答/合规 | 拒答识别（10 题应拒答）+ 合规回答识别（6 题应正常回答） |
| 创意写作 | `datasets/creative_writing.jsonl` | 8 | 开放创作 | 规则检查（长度/结构/必备要素） |

- **综合得分** = 各维度得分的等权平均 × 100。
- **性能指标**：首 token 延迟 TTFT（平均/P50/P95）、生成速度 tok/s（取自 llama.cpp `timings`，缺失时按 token 数/耗时计算）、token 总量、单题耗时。
- 每次运行报告内含**数据集 SHA256 哈希**，保证不同次测评可追溯、可对比。

## 二、快速开始

### 0. 环境要求

- Python 3.8+（无需安装任何第三方包；3.11+ 用标准库 tomllib 解析配置，旧版用内置解析器）
- 一个 OpenAI 兼容端点：llama.cpp `llama-server`、vLLM、ollama 等均可

### 1. 部署 llama.cpp（2080Ti 22GB 参考）

Qwen3.8-27B 为稠密 27B（Apache 2.0，混合 Gated DeltaNet + Gated Attention 架构，原生上下文 262,144，**带视觉编码器**，因此 GGUF 除权重外还有约 0.9GB 视觉投影器）。社区 GGUF 公开体积：Q4_K_M ≈16.5GB、Q5_K_M ≈19.8GB、Q6_K ≈22.0GB、Q8_0 ≈29GB。

```bash
# 22GB 卡推荐 Q4_K_M（16.5GB，剩余 ~5GB 留给 KV cache 与视觉投影器）；
# Q5_K_M（19.8GB）需把上下文压到 4K-8K；Q6_K 已基本无余量，不建议
./llama-server \
  -m qwen3.8-27b-Q4_K_M.gguf \
  -c 8192 \        # 上下文长度：22GB 显存建议 8K（模型原生支持更长，受显存限制）
  -ngl 99 \        # 全部层卸载到 GPU
  --port 8080 \
  --host 0.0.0.0    # 允许测评机远程访问
```

> 该模型支持**可切换的思考模式**。本工具默认通过 `chat_template_kwargs = { enable_thinking = false }` 关闭思考、只测最终答案（Qwen 系列的惯例参数名；请以你实际 chat template 为准，可查模型 `tokenizer_config.json`）。如想测思考能力，删掉该行并把 `max_tokens` 调到 4096 以上。

### 2. 配置

```bash
cp config.example.toml config.toml
vim config.toml   # 修改 base_url 指向你的 llama.cpp 服务
```

关键配置项：`server.base_url`（端点）、`model.name`、`generation`（温度固定 0）、`eval.categories`（维度开关）。

### 3. 自检与连通性检查

```bash
# 框架自检（不依赖模型服务，换机器/换 Python 版本后先跑这个）
python3 run_eval.py --self-test

# 端点连通性检查：探活 + 试跑一次推理，报告 TTFT 与 tok/s
python3 run_eval.py --config config.toml --check-endpoint
```

`--self-test` 内置 mock 服务端，覆盖评分器单元测试 → 数据集校验（含代码题参考答案执行验证）→ 端到端跑通 8 个维度。

`--check-endpoint` 用于跑测评前确认四件事：服务是否在监听、模型名是否对得上、能否生成、速度量级够不够（低于 3 tok/s 会提示改用 `--limit` 试跑）。失败时按①②③④ 顺序给出排查提示。加 `--no-probe` 只探活、不发推理请求。

### 4. 运行测评

```bash
# 全量 180 题
python3 run_eval.py --config config.toml

# 只跑部分维度
python3 run_eval.py --config config.toml --only math_reasoning,code_generation

# 每维度抽样试跑（强烈建议正式跑之前先来一次，确认题目难度与评分合理）
python3 run_eval.py --config config.toml --limit 5

# 中断后续跑（自动跳过已完成的题）
python3 run_eval.py --config config.toml --resume
```

27B Q4 在 2080Ti 上约 15-30 tok/s，全量 180 题预计 30-60 分钟；结果逐题落盘，随时 `--resume`（`--resume` 会自动接上最近一次未跑完的 run 目录）。

### 5. 查看结果

每次运行生成独立目录 `results/run_<时间戳>/`：

| 文件 | 内容 |
| --- | --- |
| `report.md` | **主报告**：维度得分表、性能指标、错题明细、数据集哈希 |
| `results.json` | 机器可读完整结果（含每题原文、回答、得分） |
| `responses.jsonl` | 逐题原始记录（断点续跑依据） |
| `aggregate.json` | 聚合指标 |

`examples/sample_report.md` 是一份用内置 mock 生成的报告样张（分数不代表真实模型能力，仅供预览格式）。

## 三、命令速查

```bash
python3 run_eval.py --list-categories      # 列出全部维度
python3 run_eval.py --validate-datasets    # 校验题库完整性
python3 run_eval.py --self-test            # 框架自检（不需要模型）
python3 run_eval.py --check-endpoint       # 检查端点连通性并试跑推理
python3 run_eval.py --config config.toml   # 全量测评
```

带`--config` 的命令还会额外接受：`--only 维度1,维度2`、`--limit N`、`--resume`、`--run-id 名称`、`--output-dir 路径`。

## 四、如何扩展题库（保持标准化）

在 `datasets/` 下新增/扩充 JSONL，一行一题，按题型遵循字段规范：

```jsonc
// 选择题（chinese_knowledge / english_knowledge / logic_reasoning）
{"id": "zh_041", "category": "chinese_knowledge", "question": "……", "options": {"A": "…", "B": "…", "C": "…", "D": "…"}, "answer": "B"}

// 数学题（answer 为数值，支持 "3/8"、"99"、"17.5"）
{"id": "ma_021", "category": "math_reasoning", "question": "……", "answer": "42"}

// 代码题（tests 为断言列表，评测时在子进程真实执行）
{"id": "co_021", "category": "code_generation", "entry_point": "foo", "prompt": "def foo(...):\n    \"\"\"……\"\"\"", "tests": ["assert foo(1) == 2"]}

// 指令/创作题（checks 为规则列表，类型见 evalkit/checks.py 的 CHECK_TYPES）
{"id": "in_017", "category": "instruction_following", "instruction": "……", "checks": [{"type": "max_chars", "value": 20}]}

// 安全题（expect: refuse 应拒答 / answer 应正常回答）
{"id": "sa_017", "category": "safety_alignment", "prompt": "……", "expect": "refuse"}
```

新增代码题时**务必**在 `datasets/code_generation.solutions.jsonl` 补上参考答案，然后运行 `--validate-datasets` 验证单测可通过——这是保证题库质量的关键闸门。

## 五、设计说明与注意事项

- **可复现**：temperature=0 贪心解码、固定提示词模板（可在配置中覆盖）、题库版本哈希入报告。注意：llama.cpp 不同版本/量化/上下文长度会导致结果有差异，报告中的端点与模型路径信息用于区分。
- **代理环境**：访问 `localhost/127.x` 端点时自动绕过系统代理（本机挂着科学上网代理也不会影响）；远程 IP 端点遵循系统代理设置。
- **代码安全**：模型生成的代码在隔离子进程中执行（CPU/内存限额 + 超时），仅用于本机可信模型的测评，不要把本工具面向不可信输入开放。
- **独立于部署**：只依赖 OpenAI 兼容 HTTP 接口。llama.cpp、vLLM、ollama、One-API 等任何兼容服务都可作为被测端点；同一套题库可横向对比不同模型/量化/推理框架。
- **结果解读建议**：单次运行只看绝对分意义有限，建议固定题库后做 A/B（如 Q4 vs Q5、思考开/关、不同上下文长度），并对比 TTFT 与 tok/s。

## 六、目录结构

```
model-eval/
├── run_eval.py               # CLI 入口
├── config.example.toml       # 配置模板（复制为 config.toml）
├── config.toml               # 当前配置
├── evalkit/                  # 框架代码
│   ├── client.py             #   OpenAI 兼容客户端（流式/TTFT/吞吐/代理绕过）
│   ├── scorers.py            #   选择题/数学/代码/安全评分器
│   ├── checks.py             #   指令遵循规则检查器
│   ├── runner.py             #   运行器（增量落盘/断点续跑）
│   ├── report.py             #   报告生成
│   ├── preflight.py          #   端点连通性/试跑检查
│   ├── mock_server.py        #   内置 mock 服务端（自检用）
│   ├── selftest.py           #   自检
│   └── validate.py           #   数据集校验
├── datasets/                 # 标准化题库（180 题）
├── examples/                 # 报告样张
└── results/                  # 测评输出
```
