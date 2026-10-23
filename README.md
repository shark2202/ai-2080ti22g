# ai-2080ti22g

面向 **单张 RTX 2080 Ti 22GB（SM75/Turing）** 的本地大模型工程：**能力测评 + 运行时选型调研**。

## 目录

| 目录 | 内容 |
|---|---|
| `model-eval/` | 标准化能力测评工具（自建 180 题回归套件 + lm-evaluation-harness 标准基准流程）；只依赖 OpenAI 兼容端点，换模型/端点只改 `config.toml` |
| `docs/runtime-research.zh-CN.md` | **运行时调研汇总**：设备与网络现实、5 个运行时候选对照、模型/量化权重地板、四轴"突破资源上限"、本机实测数据、三车道与三档结论、未验证清单 |
| `docs/三运行时对比结论.md` | **单页选型入口**：llama.cpp / NInfer / vLLM 三方总表、四条结论、选型建议、三个必须记住的限制、数据来源标注 |
| `.workbuddy/memory/` | 逐日工作记忆（MEMORY.md 为索引） |

## 结论速查（详见 `docs/runtime-research.zh-CN.md`）

**三车道**（端口不冲突，显存互斥）：

| 车道 | 方案 | 端口 | 速度 |
|---|---|---|---|
| 27B 交互 | llama.cpp / KVMem（13 GB Q2 mix + KVMem 256K logical） | 8080 | **45–48 tok/s**（实测） |
| 27B 交互（agent / 代码负载） | **NInfer for Turing**（18.2 GB 专有 `.ninfer`，MTP d3） | 8082 | **48.5–53.0 tok/s** 单流；4 路并发聚合 **94.9**、8 路 **185.7**（实测，近线性扩展） |
| 并发 / 长文档 | vLLM 9B FP8（单卡 262K 上限；2B/4B 换更大 KV） | 8100 | **聚合 75–259 tok/s**（实测） |
| GPU 被占用时兜底 | CPU 引擎 `qwen3.8-27b-in-c` | 8190 | 1.5–2.5 tok/s（推算） |

⚠️ **2026-10-20 修订**：早期版本把"并发"划归 vLLM 专属，**该判断已过时** —— NInfer 27B 四路并发聚合 94.9 tok/s
已超过 vLLM 9B FP8 的 74.8/85.3。vLLM 相对 NInfer 只剩**上下文长度**与**模型可换性**。
完整三方对比见 [`docs/三运行时对比结论.md`](docs/三运行时对比结论.md)。

**跑更大模型**：装得进显存 → llama.cpp（**35B-A3B 全显存，推算 100–300 tok/s**）；装得进"显存+内存之和" → **KTransformers**（待评估）；都装不下 → 磁盘流式（`deepseek-v4-in-c --gpu`，已验 sm_75 可编译，~0.5 tok/s）。

## 数字可信度约定

本文档所有数字标注三类：**【实测】**（本机真权重实测）、**【同架构】**（dummy 权重实测，架构/内核/显存足迹真实、逐 token 速率仅参考）、**【推算】**（未实测，不得当结论引用）。
跨机器对比不成立；换路线即换口径。

## 相关仓库

- `vLLM-2080Ti-Definitive/`：SM75 定向 vLLM fork（单卡 profile、验证脚本、10 节经验教训文档）
- `qwen3.8-2080ti-256k/`：llama.cpp/KVMem 单卡 27B 部署（当前主力）
- `qwen3.8-27b-in-c/`、`kimi-k3-in-c/`、`deepseek-v4-in-c/`：纯 C / 流式引擎