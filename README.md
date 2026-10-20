# ai-2080ti22g

面向 **单张 RTX 2080 Ti 22GB（SM75/Turing）** 的本地大模型工程：**能力测评 + 运行时选型调研**。

## 目录

| 目录 | 内容 |
|---|---|
| `model-eval/` | 标准化能力测评工具（自建 180 题回归套件 + lm-evaluation-harness 标准基准流程）；只依赖 OpenAI 兼容端点，换模型/端点只改 `config.toml` |
| `docs/runtime-research.zh-CN.md` | **运行时调研汇总**：设备与网络现实、5 个运行时候选对照、模型/量化权重地板、四轴"突破资源上限"、本机实测数据、三车道与三档结论、未验证清单 |
| `.workbuddy/memory/` | 逐日工作记忆（MEMORY.md 为索引） |

## 结论速查（详见 `docs/runtime-research.zh-CN.md`）

**三车道**（端口不冲突，显存互斥）：

| 车道 | 方案 | 端口 | 速度 |
|---|---|---|---|
| 27B 交互 | llama.cpp / KVMem（13 GB Q2 mix + KVMem 256K logical） | 8080 | **45–48 tok/s**（实测） |
| 并发 / 长文档 | vLLM 9B FP8（单卡 262K 上限；2B/4B 换更大 KV） | 8100 | **聚合 75–259 tok/s**（实测） |
| GPU 被占用时兜底 | CPU 引擎 `qwen3.8-27b-in-c` | 8190 | 1.5–2.5 tok/s（推算） |

**跑更大模型**：装得进显存 → llama.cpp（**35B-A3B 全显存，推算 100–300 tok/s**）；装得进"显存+内存之和" → **KTransformers**（待评估）；都装不下 → 磁盘流式（`deepseek-v4-in-c --gpu`，已验 sm_75 可编译，~0.5 tok/s）。

## 数字可信度约定

本文档所有数字标注三类：**【实测】**（本机真权重实测）、**【同架构】**（dummy 权重实测，架构/内核/显存足迹真实、逐 token 速率仅参考）、**【推算】**（未实测，不得当结论引用）。
跨机器对比不成立；换路线即换口径。

## 相关仓库

- `vLLM-2080Ti-Definitive/`：SM75 定向 vLLM fork（单卡 profile、验证脚本、10 节经验教训文档）
- `qwen3.8-2080ti-256k/`：llama.cpp/KVMem 单卡 27B 部署（当前主力）
- `qwen3.8-27b-in-c/`、`kimi-k3-in-c/`、`deepseek-v4-in-c/`：纯 C / 流式引擎