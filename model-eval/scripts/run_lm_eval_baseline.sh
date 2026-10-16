#!/usr/bin/env bash
# ============================================================================
# 标准基准测评：EleutherAI lm-evaluation-harness（行业事实标准）
#
# 与 model-eval 自建题库的分工：
#   model-eval→ 日常回归监控（零依赖、分钟级、自建 180 题，不可对外比较）
#   本脚本      → 对外可比的标准分数（MMLU / C-Eval / GSM8K / HumanEval / IFEval）
#
# 用法：
#   bash scripts/run_lm_eval_baseline.sh <端点URL> [模型别名]
# 例：
#   bash scripts/run_lm_eval_baseline.sh http://100.69.133.37:8080
#   bash scripts/run_lm_eval_baseline.sh http://127.0.0.1:8080qwen3.8-local
#
# 环境变量：
#   TASKS      覆盖默认任务列表（逗号分隔）
#   PROFILE    任务档位：core(默认) | full | quick
#   OUT_DIR    结果目录，默认 ./lm_eval_results
#   LM_EVAL_BIN 指定 lm_eval 可执行文件路径（用独立 venv 时需要）
# ============================================================================
set -uo pipefail

BASE_URL="${1:-http://127.0.0.1:8080}"
MODEL_ALIAS="${2:-}"
OUT_DIR="${OUT_DIR:-./lm_eval_results}"
PROFILE="${PROFILE:-core}"
TIMEOUT="${TIMEOUT:-600}"

# ---------- 任务档位 ----------
# core：国内外最常被引用的四项，耗时可控，适合日常发版前跑
# full：加上指令遵循与高难推理，耗时显著增加
# quick：只跑最小的两个，适合验证链路是否通
case "$PROFILE" in
  quick)
    DEFAULT_TASKS="gsm8k,mmlu"
    ;;
  core)
    DEFAULT_TASKS="mmlu,ceval-valid,gsm8k,humaneval"
    ;;
  full)
    DEFAULT_TASKS="mmlu,mmlu_pro,ceval-valid,gsm8k,humaneval,ifeval,arc_challenge"
    ;;
  *)
    echo "未知 PROFILE=${PROFILE}（可选 quick / core / full）" >&2
    exit 2
    ;;
esac
TASKS="${TASKS:-$DEFAULT_TASKS}"

# ---------- 定位 lm_eval ----------
LM_EVAL_BIN="${LM_EVAL_BIN:-}"
if [ -z "$LM_EVAL_BIN" ]; then
  if command -v lm_eval >/dev/null 2>&1; then
    LM_EVAL_BIN="lm_eval"
  else
    echo "=========================================================="
    echo " 未找到 lm_eval 命令"
    echo ""
    echo " 请先在独立虚拟环境中安装（强烈建议用独立 venv，勿污染系统 Python）："
    echo ""
    echo "   python3 -m venv ~/.venvs/lmeval"
    echo "   ~/.venvs/lmeval/bin/pip install -U pip"
    echo "   ~/.venvs/lmeval/bin/pip install lm_eval"
    echo ""
    echo " 装好后可用环境变量指定路径："
    echo "   LM_EVAL_BIN=~/.venvs/lmeval/bin/lm_eval \\"
    echo "     bash scripts/run_lm_eval_baseline.sh $BASE_URL"
    echo ""
    echo " 注意：lm_eval 依赖较重（torch/datasets/evaluate），本机若为"
    echo "       macOS 建议 Python 3.10~3.12；也可直接在 2080Ti 上安装，"
    echo "       通过 --base_url 指向该端点远程测评。"
    echo "=========================================================="
    exit 1
  fi
fi

echo "=========================================================="
echo " lm-evaluation-harness 标准基准（PROFILE=${PROFILE}）"
echo "=========================================================="
echo " 端点   : ${BASE_URL}"
echo " 模型   : ${MODEL_ALIAS:-（由服务端决定）}"
echo " 任务   : ${TASKS}"
echo " 输出   : ${OUT_DIR}"
echo "=========================================================="
echo ""

# ---------- 预检：端点必须可达 ----------
echo "[1/3] 预检端点连通性..."
HEALTH_URL="${BASE_URL%/}/v1/models"
if command -v curl >/dev/null 2>&1; then
  if ! curl -s --max-time 10 "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "  警告: ${HEALTH_URL} 无响应，测评大概率全部失败。" >&2
    echo "  请先确认 llama-server 在跑且 --host 0.0.0.0。" >&2
    read -r -p "  仍要继续？(y/N) " ans
    case "$ans" in
      [yY]|[yY][eE][sS]) ;;
      *) echo "已取消。"; exit 1;;
    esac
  else
    echo "  端点可达 ✓"
  fi
else
  echo "  未找到 curl，跳过预检"
fi
echo ""

# ---------- 自动取模型名 ----------
# local-completions / local-chat-completions 这两个后端必须显式指定 model
if [ -z "$MODEL_ALIAS" ] && command -v curl >/dev/null 2>&1; then
  MODEL_ALIAS="$(curl -s --max-time 10 "${HEALTH_URL}" \
    | tr ',' '\n' | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
fi
echo "[2/3] 模型标识: ${MODEL_ALIAS:-（未取到，若后端需要模型名请手动传入第二个参数）}"
echo ""

mkdir -p "$OUT_DIR"

# ---------- 运行 ----------
echo "[3/3] 开始跑标准基准，耗时取决于模型速度与任务规模..."
echo ""

# 后端选择（参数格式已按 lm-evaluation-harness 官方文档核实）：
#   gguf
#     --model_args base_url=http://host:8080
#     注意 base_url 不带 /v1；后端自行拼接 /v1/completions。
#     走 /v1/completions 而非 chat/completions，支持 generate_until + loglikelihood。
#     llama.cpp 需 ≥ 2024-12（依赖 logprobs.content 格式，旧 token_logprobs 不支持）。
#     并发度默认从 /props 的 total_slots 自动探测，可用 parallel=N 覆盖。
#     不支持 loglikelihood_rolling，故 wikitext 等 perplexity 任务不可用。
#
#   local-completions
#     --model_args model=<别名>,base_url=http://host:8080/v1/completions
#     注意 base_url 必须是到 /v1/completions 的完整路径（与 gguf 不同）。
#     作为 gguf 不可用时的替代，同样支持 loglikelihood 类任务。
BACKEND="${BACKEND:-gguf}"

case "$BACKEND" in
  gguf)
    MODEL_ARGS="base_url=${BASE_URL%/}"
    if [ -n "$MODEL_ALIAS" ]; then
      # router 模式（多模型）下必须指定别名
      MODEL_ARGS="${MODEL_ARGS},model=${MODEL_ALIAS}"
    fi
    ;;
  local-completions)
    if [ -z "$MODEL_ALIAS" ]; then
      echo "local-completions 后端需要模型名，请作为第二个参数传入。" >&2
      exit 2
    fi
    MODEL_ARGS="model=${MODEL_ALIAS},base_url=${BASE_URL%/}/v1/completions"
    ;;
  *)
    echo "未知 BACKEND=${BACKEND}（可选 gguf / local-completions）" >&2
    exit 2
    ;;
esac

LOG="${OUT_DIR}/lm_eval_${PROFILE}.log"
RESULTS="${OUT_DIR}/${PROFILE}"

"$LM_EVAL_BIN" --model "$BACKEND" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size 1 \
  --output_path "$RESULTS" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

# ---------- generate_until 类任务（如 ifeval）需要 chat 后端 ----------
# 官方明确：local-chat-completions 只支持 generate_until（无 logprobs），
# 而 ifeval 属于该类，不能用 gguf 后端跑，否则会报 output type 不匹配。
CHAT_TASKS=""
for t in ifeval; do
  case ",${TASKS}," in
    *",${t},"*) CHAT_TASKS="${CHAT_TASKS}${CHAT_TASKS:+,}${t}";;
  esac
done

if [ -n "$CHAT_TASKS" ]; then
  echo ""
  echo ">>> 补跑 chat 类任务（${CHAT_TASKS}）：这些任务需 generate_until，"
  echo "    按官方说明应使用 local-chat-completions后端。"
  if [ -z "$MODEL_ALIAS" ]; then
    echo "    未取到模型名，跳过。如需跑请手动传入模型别名。"
  else
    "$LM_EVAL_BIN" --model local-chat-completions \
      --model_args "model=${MODEL_ALIAS},base_url=${BASE_URL%/}/v1/chat/completions" \
      --tasks "$CHAT_TASKS" \
      --batch_size 1 \
      --apply_chat_template \
      --output_path "${OUT_DIR}/${PROFILE}_chat" 2>&1 \
      | tee "${OUT_DIR}/lm_eval_${PROFILE}_chat.log" || \
      echo "    chat 类任务运行失败（可能该模型 chat template 不支持 apply_chat_template），不影响上面的结果"
  fi
fi

echo ""
if [ "$RC" -eq 0 ]; then
  echo "=========================================================="
  echo " 完成。结果：${RESULTS}"
  echo " 日志  ：${LOG}"
  echo "=========================================================="
else
  echo " lm_eval 退出码 ${RC}，未全部成功。" >&2
  echo " 常见原因："
  echo "   ① llama.cpp 版本过旧 —— gguf 后端需 2024-12 及更新（logprobs.content 格式）"
  echo "   ② 端点被代理拦截 —— 100.x 等非回环地址需确认 http_proxy 未劫持"
  echo "   ③ 任务名拼写错误 —— 用 '$LM_EVAL_BIN --tasks list' 查看可用任务"
  echo "   ④ 数据集下载失败 —— 首次运行需联网拉取 HuggingFace 数据集"
  echo " 详见 ${LOG}"
fi

# ---------- 汇总 ----------
cat > "${OUT_DIR}/README.txt" << EOF
lm-evaluation-harness 标准基准结果
==================================
运行时间 : $(date "+%Y-%m-%d %H:%M:%S")
端点: ${BASE_URL}
模型   : ${MODEL_ALIAS:-（服务端决定）}
后端   : ${BACKEND}
档位   : ${PROFILE}
任务   : ${TASKS}

结果目录: ${RESULTS}
日志    : ${LOG}

解读提醒
--------
1. 这些是公开基准分数，可以与官方 model card、论文、排行榜比较。
2. 但仍受以下因素影响，读数时须一并考虑：
   - 量化精度：本次被测为 Q2/K 量化的第三方微调权重，
     分数天然低于官方 FP16/BF16 原版，不可直接等同。
   - 数据污染：公开基准的题目可能已进入模型训练语料，分数会偏高。
   - 推理后端：llama.cpp 与 vLLM 的采样实现不同，同一权重结果会有差异。
   - prompt模板：chat template 不同，指令类任务（ifeval）分数波动较大。
3. 本清单仅覆盖 loglikelihood 类任务。gguf 后端不支持
   perplexity 类任务（如 wikitext），需要时应换用其他后端或
   换 HuggingFace 直接加载权重。
EOF

echo ""
echo "汇总信息已写入 ${OUT_DIR}/README.txt"
exit $RC