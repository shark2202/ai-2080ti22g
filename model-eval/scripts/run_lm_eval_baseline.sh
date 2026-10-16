#!/usr/bin/env bash
# ============================================================================
# 标准基准对照脚本：用行业事实标准 EleutherAI lm-evaluation-harness
# 对同一个 llama.cpp 端点跑公开基准，产出可对外比较的分数。
#
# 与 model-eval 的分工：
#   model-eval            → 日常回归监控（自建题库，零依赖，秒级起）
#   本脚本 / lm_eval      → 对外可比的标准基准分数（MMLU / C-Eval / GSM8K 等）
#
# 前置：pip install lm_eval    （需要 Python 3.10+）
# 用法：bash scripts/run_lm_eval_baseline.sh http://<2080Ti设备IP>:8080 [模型别名]
# ============================================================================
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8080}"
MODEL_ALIAS="${2:-}"
TIMEOUT="${TIMEOUT:-600}"

if ! command -v lm_eval >/dev/null 2>&1; then
  echo "未找到 lm_eval。请先安装：pip install lm_eval" >&2
  exit 1
fi

MODEL_ARGS="base_url=${BASE_URL}"
if [ -n "$MODEL_ALIAS" ]; then
  MODEL_ARGS="${MODEL_ARGS},model=${MODEL_ALIAS}"
fi

# 公开基准任务：中文/英文知识、数学、代码、指令遵循
# 注意：mmlu / ceval-valid / humaneval 属 loglikelihood 型任务，建议走 gguf 后端
TASKS_LOGLIK="mmlu,ceval-valid,gsm8k,humaneval"

OUT_DIR="${OUT_DIR:-./lm_eval_results}"
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo " lm-evaluation-harness 标准基准"
echo " 端点   : ${BASE_URL}"
echo " 模型   : ${MODEL_ALIAS:-（服务端默认）}"
echo " 任务   : ${TASKS_LOGLIK}"
echo " 注意   : gguf 后端不支持 perplexity 类任务（如 wikitext）"
echo "          llama.cpp 需 2024-12 及更新版本（logprobs.content 格式）"
echo "=========================================================="

# A. llama.cpp 专用后端（llama-server 的 OpenAI 兼容接口）
if [ -n "$MODEL_ALIAS" ]; then
  lm_eval --model gguf \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS_LOGLIK}" \
    --batch_size 1 \
    --output_path "${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/lm_eval_gguf.log"
else
  lm_eval --model gguf \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS_LOGLIK}" \
    --batch_size 1 \
    --output_path "${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/lm_eval_gguf.log"
fi

# B. 通用 OpenAI 兼容 chat 后端（可选，用于 chat 类任务如 ifeval）
#    需要模型名，可从 llama-server 的 /v1/models 获取：
#    curl -s ${BASE_URL}/v1/models
CHAT_MODEL="${CHAT_MODEL:-$(curl -s --max-time 5 "${BASE_URL}/v1/models" \
  | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)}"

if [ -n "$CHAT_MODEL" ]; then
  echo ""
  echo ">>> 运行 chat 类标准基准（model=${CHAT_MODEL}）"
  lm_eval --model local-chat-completions \
    --model_args "model=${CHAT_MODEL},base_url=${BASE_URL}/v1" \
    --tasks ifeval \
    --batch_size 1 \
    --output_path "${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/lm_eval_chat.log" || \
    echo "ifeval 运行失败（可能该任务需要特定模板），不影响上面的结果"
else
  echo "未能从 ${BASE_URL}/v1/models 获取模型名，跳过 chat 类基准"
fi

echo ""
echo "结果已保存到 ${OUT_DIR}/"
echo "解读提醒：公开基准分数同样受数据污染、量化精度与推理后端影响；"
echo "         与官方 model card 分数不同属正常现象，详见 docs/标准与对比说明.md"
