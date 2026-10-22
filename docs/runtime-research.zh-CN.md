# 2080 Ti 22GB 运行时调研汇总

面向 **单张 RTX 2080 Ti 22GB（SM75/Turing）** 的推理运行时选型调研，汇总 2026-10-20/21 期间的实测与核对结果。

## 〇、定位与数字可信度（先读这一节）

本文是**运行时选型与容量调研**，不是模型能力测评（能力测评见 `model-eval/`）。所有数字分三类，逐项标注：

| 标注 | 含义 |
|---|---|
| **【实测】** | 本机真实测量（真权重或同架构 dummy 权重，注明） |
| **【同架构】** | dummy 权重实测：架构/内核/KV 记账/显存足迹真实，逐 token 速率仅作参考 |
| **【推算】** | 由带宽/参数量/参考机折算，**未实测**，不得当结论引用 |

跨机器对比不成立：本文数字只属于"本机 + 本路线 + 本口径"。

## 一、设备与网络现实（决定全部时间成本）

| 项 | 值 |
|---|---|
| GPU | 1× RTX 2080 Ti 22,528 MiB（驱动 580.178.04），SM75，无第二张卡 |
| CPU / 内存 | 2× Xeon E5-2673 v4（56 线程，单 NUMA 节点）/ 31 GB RAM + 2 GB swap |
| `CommitLimit − Committed_AS` | **仅 ~2.6 GiB**（ComfyUI 5.4GB、ToDesk 1.3GB、code-server ×5 ~3.5GB 占了 commit） |
| OS / 工具链 | Ubuntu 22.04.5、kernel 6.8、glibc 2.35；**只有 gcc/g++ 11 与 gcc 12（无 `g++-12`）**；系统 `nvcc` 是 11.5，可用 CUDA 在 `/usr/local/cuda-13.4` |
| 磁盘 | 916 GB 盘 / 可用 704 GB（NVMe） |

**网络（代理是最大变量）**：

| 路径 | 结果 |
|---|---|
| Clash 代理 `192.168.8.184:7890` | ~300 KB/s，且对 github / hf-mirror / pypi.org / jsdelivr / download.pytorch.org **TLS 断**（`SSL routines::unexpected eof`） |
| hf-mirror **直连** | 单流 1.09 MB/s，多连接可持续 ~2 MB/s、突发 9.5 MB/s ✅ |
| aliyun PyPI / pytorch-wheels 直连 | ✅ |
| ghfast.top / gh-proxy.com **直连** | git ✅（可作 `FLASHQLA_REPO` / `BUILD_GIT_MIRROR_PREFIX`） |
| ModelScope 直连 | ✅（多模态/GGUF 权重主源） |

**准则**：能直连不走代理；大文件多连接 + 短尝试 + 速度下限（<4KB/s 持续 30s 即弃）。

## 二、运行时候选对照

| 运行时 | 后端 | 27B 可用性 | 速度 | 结论 |
|---|---|---|---|---|
| **llama.cpp / KVMem**（`qwen3.8-2080ti-256k`） | CUDA（SM75 专项补丁 `turing-mmvq-mmq-routing.diff` 等） | ✅ 13.0 GB Q2 mix + 1.14 GB DFlash2 | **【实测】45–48 tok/s 日常、64.7 短上下文、prefill 361–380 tok/s、256K logical（KV 外置检索）** | **单卡 27B 的唯一可用答案** |
| **vLLM-2080Ti-Definitive**（fork） | CUDA（SM75 内核回退 + FlashQLA legacy GDN） | ❌ 权重地板 18.60 GB → KV ≈ 0 | 9B FP8：**【实测】prefill ~3,020 tok/s、decode 47.2、1–4 路并发 47→75（normal）/85（fast）tok/s、0 preemption** | **并发/长文档车道**；单卡 9B 可到 262K 上下文 |
| **NInfer for Turing**（`mr-september/ninfer-2080ti-22g`） | 自研 C++/CUDA 引擎（原为 RTX 5090/Blackwell，移植到 sm_75） | ⚠️ 需专有 `.ninfer` v3 权重（18.2 GB，无 GGUF reader） | **【实测】无投机 26.6–26.8 tok/s；MTP d3 按内容 33.2–53.0 tok/s**（prefill 313–348 tok/s）；**4 路并发聚合 94.9 / 8 路 185.7 tok/s（近线性扩展）**；上下文 126K(带 MTP)–152K；**本机已构建并跑通** | **第三车道**：单请求速度与 llama.cpp 相当、**并发扩展近线性**、上下文 126–152K（高于官方 97–125K），代价是 4.4 GiB 主机内存 + 18.2 GB 专有权重 |
| `qwen3.8-27b-in-c`（自研，纯 C） | 纯 CPU | ✅ IQ1_M 6.73 GB / Q4_K_M 16.46 GB | **【推算】1.5–2.5 tok/s**（参考机 2.52/1.88） | CPU 兜底车道；本机 `make strict` 通过 |
| `deepseek-v4-in-c`（第三方） | CPU + 可选 `--gpu`（dense trunk 上位） | 284B Flash ≈160 GB 磁盘 | **【推算】~0.5–0.6 tok/s**（作者实测 1.62–1.74 s/token） | 唯一"装得下的超 RAM 模型"；**已实测可为 sm_75 编译** |
| `kimi-k3-in-c`（第三方） | 纯 CPU（0 GPU） | 2.78T，检查点 **1.56 TB** | 5.6–26.5 s/token | ❌ **本机装不下**（缺约 1 TB 磁盘） |

## 三、模型与量化选项（vLLM 视角的"权重地板"）

| 目标 | 最小可用体积 | 单卡可行性 |
|---|---:|---|
| Qwen3.8-27B（FP8 / NVFP4 / INT4 / AWQ / GPTQ / bnb） | **18.60 GB**（RedHatAI INT4） | ❌ 权重后 KV 余量 ≈ 0；dense 卸载后 decode 掉到个位数 |
| Qwen3.8-27B GGUF（K/IQ 全系，含 5.71 GB IQ1_S） | — | ❌ vLLM 的 `vllm-gguf-plugin` **支持 K/IQ 类型但无 `qwen3_5`/hybrid/GDN 实现** → 架构不支持 |
| Qwen3.8-9B FP8（`compressed-tensors`） | 11.91 GB | ✅ 已验证（262K 上限、7.53× 并发 @32K） |
| **Qwen3.8-2B / 4B Distill（safetensors）** | 4.57 / 9.34 GB | ✅ **甜点区**：KV 余量 ~13 / ~8.6 GiB → **【推算】256K × 3 / × 2 并发** |
| Qwen3.6-35B-A3B（MoE，3B 激活） | 16.8–22.2 GB（IQ3_S→Q4_K_M） | ✅（llama.cpp）**全放显存 → 【推算】100–300 tok/s** |
| Qwen3.8-Flash-Next 131B-A6B | ~32–42 GB（IQ2 档） | ⚠️ 需专家卸载（显存+内存 ≈ 48 GiB）**【推算】5–20 tok/s** |
| DeepSeek-V4-Flash 284B | ~160 GB 磁盘 | ⚠️ 流式（~0.5 tok/s）或 KTransformers |

**GGUF 索引**（`local-ai-zone.github.io/gguf_models.json`，38,814 条）：对 vLLM **不可用**（架构阻塞，非量化类型）；
对 llama.cpp 有价值：Qwen3.8-27B 有 732 条，最小 IQ1_S 5.71 GB / IQ2_XXS 8.37 GB。

## 四、"突破单一资源上限"的四轴

| 轴 | 机制 | 代价 | 本机 |
|---|---|---|---|
| ① KV 层 | KV 主体/老块放内存或 SSD | 检索延迟 | ✅ 已在用（KVMem）；vLLM 侧 **【实测】前缀复用 TTFT 12.53s → 0.244s（51×）** |
| ② **权重层·MoE 专家卸载** | 专家放内存，attention+KV 在显存 | 小（只算激活专家） | ✅ **最优**：llama.cpp `-cmoe`/`--n-cpu-moe`/`-ot`（本机二进制已核对支持） |
| ③ 计算层 | CPU 参与算子计算 | 中等 | ⏳ KTransformers（2026-05 起支持 DeepSeek-V4-Flash） |
| ④ 权重层·磁盘流式 | 权重每 token 从磁盘过一遍 | 极大（秒级/token） | ⚠️ 仅"完全装不下"时（deepseek-v4-in-c / colibri） |

上游作者选型表（照抄）：装得进显存/内存 → **llama.cpp**；装得进**显存+内存之和** → **KTransformers**；
都装不下且要逐 bit 一致 → `deepseek-v4-in-c`；都装不下且要快 → **colibri**。

## 五、本机实测数据

### 5.1 vLLM（fork）单卡能力矩阵【同架构 dummy 权重】
| KV 精度 | MAX_MODEL_LEN | GPU KV tokens | 并发余量 | 显存 |
|---|---:|---:|---:|---:|
| FP8 | 32K | 246,622 | 7.53× | 18,136 MiB |
| FP8 | 131K | 501,275 | 3.82× | ~19.8 GiB（+offload） |
| FP8 | 229K | 292,864 | 1.28× | 18,136 MiB |
| FP8 | 262K | 294,012 | 1.12× | 18,136 MiB |
| TQ4NC | 262K | 444,079 | 1.69× | 17,154 MiB |
| TQ4NC + 三层 offload | 262K | 915,547（含 CPU 层） | 3.49× | 20,202 MiB |

### 5.2 vLLM 真权重（9B FP8，SHA256 已校验）
| 项 | 值 |
|---|---|
| 加载 / 质量闸门 | 4.10 s / **4/4**（Paris、2、Water、8K 长上下文取回 Paris） |
| 4K/128 参考口径 | **prefill ~3,020 tok/s、decode 47.2 tok/s、TTFT 1.36 s** |
| 并发 1/2/4（normal） | 47.29 / 40.08 / **74.80** tok/s 聚合，0 preemption |
| 并发 1/2/4（fast） | 47.28 / 42.53 / **85.25** tok/s 聚合，0 preemption |
| 关键内核 | Marlin weight-only FP8（SM75 回退）、FlashQLA legacy SM75 GDN prefill、Triton GDN decode、mamba align |

### 5.3 与显存共存（ComfyUI 占 4.56 GiB 时）
| GPU_UTIL | 预算 | KV 余量 | FP8 上下文上限 |
|---|---:|---:|---:|
| 0.90（独占） | 19.33 GiB | 4.61 GiB | ~302K |
| 0.78（共存上限） | 16.75 GiB | 2.03 GiB | ~133K |
| 0.70 | 15.04 GiB | 0.32 GiB | ~21K |

vLLM 在显存不足时**拒绝启动**（`free memory < util×total`）——这是保护行为，不是故障。


### 5.4 NInfer for Turing（本机实测 · 2026-10-22）

| 项 | 值 |
|---|---|
| 权重 | `qwen3_8_27b.ninfer` 20,437,521,664 B，sha256 `81f924d4…375da`（脚本 + 独立复算双验） |
| 启动 | 15.2 s（权重 9.1 s @1.83 GiB/s）→ `engine ready` |
| 显存 | 权重 15.4 GiB + 序列 2.36 GiB（int8 KV payload 2.19 GiB）+ runtime 2.45 GiB = **计划 17.9 GiB**；**实测峰值 18726 MiB** |
| 主机内存 | **host context pinned 3.15 GiB + host 权重 1.26 GiB**（embedding 表放 pinned host，省 1.26 GiB 显存） |
| 无投机基线 | **26.6–26.8 tok/s**（与官方 bench-short 的 27.03 ± 0.11 一致），prefill 313–348 tok/s |
| 端到端 | `ninfer-serve` :8082 已接进 pi（`~/.pi/agent/models.json` 的 `ninfer-local` provider），中文问答正常 |

#### 5.4.1 官方标准测量协议的本机复现（2026-10-22）

用 `docs/turing-port.md` 的 standard measurements（`ninfer_bench`，语料 `bench/fixtures/bench_corpus.ids`），
对照官方在 **Windows 11 / CUDA 12.9 / MSVC 14.44 / 驱动 610.88** 上 2026-10-19 发布的数据：

| Run | 指标 | 官方 | **本机（Linux / CUDA 13.4 / gcc 11.4）** | 差 |
|---|---|---:|---:|---:|
| bench-short | pp512 | 615.99 ± 0.70 | **587.78 ± 0.72** | −4.6% |
| bench-short | tg256（MTP off） | 27.03 ± 0.11 | **26.81 ± 0.18** | −0.8% |
| bench-mtp | pp512 | 594.56 ± 4.31 | **578.76 ± 1.04** | −2.7% |
| bench-mtp | tg256 | 29.58 ± 0.05 | **28.64 ± 0.16** | −3.2% |
| bench-mtp | **接受率** | 0.377 | **0.3771** | **0.0%（四位小数一致）** |
| bench-long | pp8192 | 568.63 ± 7.59 | **563.94 ± 1.67** | −0.8% |
| bench-long | pp65536 | 457.51 ± 7.26 | **435.24 ± 0.56** | −4.9% |
| bench-long | tg32 | 25.91 ± 0.01 | **25.62 ± 0.00** | −1.1% |

真文本场景（`examples/cli/messages`，官方：code_python 47.5 vs 26.3、story_en_mystery 28.8 vs 26.0）：

| 场景 | 官方 MTP / none | **本机 MTP / none** | 本机接受率 |
|---|---|---|---:|
| scenario_code_python | 47.5 / 26.3 | **48.5 / 27.1** | **83.9%** |
| scenario_code_typescript | — | 46.2 / 26.7 | 81.1% |
| scenario_story_en_mystery | 28.8 / 26.0 | **29.3 / 26.9** | 38.5% |
| scenario_story_zh_dialogue | — | 27.0 / 26.6 | **33.2%** |

→ **速度与投机行为跨平台复现良好（0.8–4.9%），接受率精确到四位小数一致。**

#### 5.4.1b 并发与上下文（本机实测）

**并发**（`ninfer-serve`，同一 prompt，逐路取服务端上报的 decode tok/s，聚合 = 各路之和；
工具 `ninfer-2080ti-22g/bench/conc_probe.py`）：

| 客户端数 | 4 通道 @64K：每路 / 聚合 | 8 通道 @32K：每路 / 聚合 |
|---:|---|---|
| 1 | 41.8 / **41.8** | 41.7 / 41.7 |
| 2 | 29.3, 28.4 / **57.7** | — |
| 4 | 22.5–24.6 / **94.9** | 22.5–24.7 / 95.0 |
| 8 | 22.1–24.5 / **185.7** | 12.3–13.7 / **103.3** |

→ **并发扩展接近线性**（1→4 路 2.27×，1→8 路 4.44×）：显存带宽被多路共享，权重每 token 只读一次。
这是 NInfer 与另外两个运行时最不一样的地方 —— 投机解码是**单流**优化（社区结论：优势到 `--parallel 4` 就消失），
而 NInfer 的**多路调度本身就能扩展**。
⚠️ 但 **8 通道 @32K 反而比 4 通道 @64K 差**（103.3 vs 185.7）：通道多 → 每路分到的 context 更少，
KV 页压力更大。**本机甜点是 4 通道 @64K**；8 通道在 64K 下直接起不来（device budget 不够）。

**上下文上限**（本机，实测边界）：

| | 本机 | 官方（Windows，桌面占 2.3 GB 显存） |
|---|---:|---:|
| 带 MTP | **129,024**（126K，20.0 GiB） | 约 97–99K |
| 不带 MTP | **155,648**（152K，19.8 GiB） | 110–125K |

→ 本机上限**高于**官方，差值正好对应"别人桌面占着 2.3 GB 显存、我们只占 0.28 GB"。
官方文档也写了"它随桌面和其他程序占多少显存而变"。

#### 5.4.2 ⚠️ 唯一没有复现的：perplexity

协议 B（`--quick --kv-dtype int8`，同一权重、同一语料、同一命令）：

| 域 | 官方（2026-10-19） | **本机** | 差 |
|---|---:|---:|---:|
| chinese_reference | 6.767452 | **5.005013** | −26.0% |
| english_long_form | 7.178341 | **6.891122** | −4.0% |
| english_reference | 6.454913 | **6.195144** | −4.0% |
| ninfer_code | 1.673458 | **1.652818** | −1.2% |
| **overall** | **4.796799** | **4.343197** | **−9.5%** |

已排除的原因：**权重不是漂移的**（HF 上该 artifact 自 2026-09-15 未更新，sha256 与我们下载的一致，
官方测量在 10-19 之后）；**语料不是漂移的**（`eval/corpora/perplexity-1m/data` 最后修改 2026-08-29）；
scored-token 数差 56（261,167 vs 261,223）官方文档自己说明"是运行时结果"。
→ 剩下的解释是**两个构建的浮点数值不等价**（Windows/MSVC/CUDA 12.9 vs Linux/gcc/CUDA 13.4；
Turing 无 bf16 张量核，bf16 运算的累加顺序随编译器与内核而变）。
**实践含义**：NInfer 的**性能结论可以跨平台引用**，但**输出逐 token 不会与官方一致**；
在 Turing 上评估模型质量时，别把 PPL 与官方数字对比。

**投机收益由输出熵决定（跨引擎同现象）** —— 同一权重、同一台机、只改输出类型：

| 输出类型 | 无投机 | MTP d3 | 接受率 | 接受长度 | 加速 |
|---|---:|---:|---:|---:|---:|
| 结构化 JSON | 26.6 | **53.0** | **97.2%** | 3.92 | 1.99× |
| 代码（英文） | 26.8 | 42.3 | 70.7% | 3.11 | 1.58× |
| 中文散文 | 26.6 | 33.2 | 48.8% | 2.45 | 1.25× |
| 中文散文（d1） | — | 33.0 | **69.9%** | 1.70 | 1.24× |

散文用 d1 时**接受率高得多（69.9% vs 48.8%）但速度持平** —— 与官方文档"散文 depth 1–2 更好、代码 depth 3 最好"一致。
⚠️ 这与 llama.cpp 侧**独立测到的同一现象**（DFlash2 在中文创作散文上接受率 0.12、加速为负；在结构化任务上 0.75–0.92）互相印证：
**投机解码的收益由输出熵决定，不由引擎决定。** 结论：agent / 代码 / 工具调用负载值得开投机；自由创作别开。

### 5.5 三个 27B 运行时的本机对照

| | llama.cpp / KVMem | NInfer | vLLM-Definitive |
|---|---|---|---|
| 27B 可用 | ✅ 13.0 GB Q2 mix | ✅ 但要 18.2 GB 专有权重 | ❌ 地板 18.60 GB / 架构阻塞 |
| 无投机 decode | 22.5 tok/s（本会话题面口径） | **26.7 tok/s** | — |
| 最好成绩 | **65.7 tok/s**（+DFlash2 结构化） | **53.0 tok/s**（+MTP 结构化 JSON） | 9B FP8：47.2 单 / 85.3 聚合 |
| **并发聚合** | 单流优化（优势到 `--parallel 4` 就消失） | **4 路 94.9 / 8 路 185.7 tok/s（近线性扩展）** | 9B FP8：4 路 74.8(normal)/85.3(fast) |
| 显存占用 | 15806 MiB | 18726 MiB 峰值 | 18136 MiB（9B FP8） |
| 上下文 | **256K logical**（KV 外置） | 64K（官方 97–99K 带 MTP） | 9B 可 262K |
| 主机内存 | 几乎不占 | **4.4 GiB pinned** | 少量 |
| 依赖成本 | 已就绪 | **13 条依赖坑**（见 `ninfer-2080ti-22g/scripts/setup_local_deps.sh`） | 已就绪 |
| 权重格式 | GGUF（通用） | **`.ninfer` 专有**（无 GGUF reader） | safetensors 量化 |

**选型结论（在本机）**：
1. **要 27B + 长上下文 + 通用权重** → llama.cpp/KVMem（256K logical，权重最灵活）
2. **要 27B + 最高单请求速度 + 愿意锁死权重格式与上下文长度** → NInfer（53.0 tok/s 结构化负载；代价：18.2 GB 专有权重 + 4.4 GiB pinned host + 64K ctx）
3. **要并发 / 长文档 / 9B** → vLLM-Definitive（27B 在这条路上是死的：GGUF 架构阻塞 + safetensors 权重地板 18.60 GB）
4. 三者**不能同时驻留**（本机 22.5 GB 卡）：llama.cpp 13.9 + NInfer 18.3 > 22.5

## 六、结论速查

**三车道**（端口不冲突，显存互斥）：
1. 27B 交互 → **llama.cpp/KVMem**（:8080，45–48 tok/s，256K logical）｜**也可用 NInfer**（:8082，53.0 tok/s 结构化负载，126–152K ctx，并发 4 路 94.9 / 8 路 185.7 tok/s）
2. 并发/长文档 → **vLLM 9B FP8**（:8100，聚合 75–259 tok/s；或改 2B/4B 换更大 KV）
   ⚠️ 2026-10-22 更新：**27B 的并发不再是 vLLM 独占** —— NInfer 27B 在 4 路并发下聚合 **94.9 tok/s**，
   已超过 vLLM 9B FP8 的 74.8(normal)/85.3(fast)。vLLM 相对 NInfer 的优势只剩**上下文长度**
   （9B 可到 262K vs NInfer 27B 152K）与**模型可换性**（vLLM 能跑 2B/4B/9B，NInfer 只认 .ninfer 权重）。
3. GPU 被占用时的兜底 → **CPU 引擎**（`qwen3.8-27b-in-c`，:8190）

**"跑更大模型"三档**：
1. 35B-A3B（≤21 GiB）→ llama.cpp 全显存 → **【推算】100–300 tok/s**
2. DeepSeek-V4-Flash 284B（>48 GiB）→ **KTransformers**（待评估）
3. 同上且要纯 C → `deepseek-v4-in-c --gpu`（sm_75 已验证可编）/ colibri → ~0.5 tok/s

## 七、复现入口

| 内容 | 位置 |
|---|---|
| vLLM 构建/启动/验证脚本与单卡 profile | `vLLM-2080Ti-Definitive/tools/single-card/`、`profiles/local/1x2080Ti/qwen9b/w8a16/` |
| vLLM 详尽经验教训（10 节、367 行） | `vLLM-2080Ti-Definitive/docs/single-card-2080ti-lessons.zh-CN.md` |
| llama.cpp/KVMem 基线、配置、基准数据 | `qwen3.8-2080ti-256k/`（`config.sh`、`docs/BENCHMARKS.md`、`benchmarks/summary.json`） |
| 纯 C 引擎 | `qwen3.8-27b-in-c/`、`kimi-k3-in-c/`、`deepseek-v4-in-c/` |

## 八、未验证清单（不得当结论）

1. 真权重的 **131K/229K/262K 长上下文** 与 **分层 offload**（仅 dummy 验证过；真权重被 ComfyUI 显存占用挡住）
2. `MODE=fast` vs `normal` 的 **cudagraph capture sizes A/B**（dummy 暗示 ~2× 并发差距，真权重待测）
3. **长跑服务上的 prefill 递减**（长上下文/混合流量后出现，全新服务不复现；已排除降频与 prefix cache）
4. **Qwen3.8-2B/4B** 在 vLLM 的单卡实测（甜点区结论目前是推算）
5. **35B-A3B** 的 100–300 tok/s（推算；需下 17.3 GB 实测）
6. `deepseek-v4-in-c` 的 sm_75 **运行时**正确性/性能（只验证过可编译）
7. **KTransformers** 在本机（SM75 + 48 GiB 组合）的可行性与速度