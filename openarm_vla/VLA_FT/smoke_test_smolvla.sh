#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SmolVLA 微调 smoke test
#
# 目的：在最少代价下证明整条 VLA fine-tune 链路能端到端跑通——
#   (1) 用 collect_dataset.py 采 2 episode 作为 smoke dataset (若尚不存在)
#   (2) MODE=full STEPS=20 BATCH_SIZE=2 跑一次，verify
#   (3) MODE=lora STEPS=20 BATCH_SIZE=2 跑一次，verify
# 任何一步失败整个脚本立刻退出。
#
# 可调的 env vars（默认值够 smoke 用）：
#   SMOKE_DATASET_ROOT   /.../Datasets/_smoke_test          smoke shards 的家
#   SMOKE_REPO_ID        openarm/lift_cube_smoke            smoke 用的 repo_id
#   OUTPUT_ROOT          /.../logs/smolvla_smoke            两次训练的输出父目录
#   STEPS                20
#   BATCH_SIZE           2
#   SAVE_FREQ            20    (== STEPS → 必跑一次最终 save)
#   LOG_FREQ             5
#   SKIP_DATA_COLLECT    0     (=1 则即便没 smoke shard 也不重新采)
#   SKIP_FULL            0
#   SKIP_LORA            0
#   FORCE_FRESH_OUTPUT   0     (=1 时重跑前删 OUTPUT_ROOT 下的两个子目录)
# -----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"  # /home/lcw/workspace/openarm

SMOKE_DATASET_ROOT="${SMOKE_DATASET_ROOT:-${REPO_ROOT}/openarm_vla/Datasets/_smoke_test}"
SMOKE_REPO_ID="${SMOKE_REPO_ID:-openarm/lift_cube_smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/smolvla_smoke}"
STEPS="${STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAVE_FREQ="${SAVE_FREQ:-${STEPS}}"
LOG_FREQ="${LOG_FREQ:-5}"
SKIP_DATA_COLLECT="${SKIP_DATA_COLLECT:-0}"
SKIP_FULL="${SKIP_FULL:-0}"
SKIP_LORA="${SKIP_LORA:-0}"
FORCE_FRESH_OUTPUT="${FORCE_FRESH_OUTPUT:-0}"

CONDA_BASE="${CONDA_BASE:-/home/lcw/workspace/miniforge3}"
ISAAC_ENV_NAME="${ISAAC_ENV_NAME:-env_isaaclab}"
SMOLVLA_ENV_NAME="${SMOLVLA_ENV_NAME:-smolvla}"

# 默认走 offline，避免内网代理 403；需要在线拉权重时手动 HF_HUB_OFFLINE=0 再跑。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DISABLED=true

FT_SCRIPT="${HERE}/finetune_smolvla.sh"
VERIFY_PY="${HERE}/verify_finetune_output.py"
COLLECT_PY="${REPO_ROOT}/openarm_vla/Data_Collector/collect_dataset.py"

say() { echo "[smoke_test_smolvla] $*"; }

[[ -x "${FT_SCRIPT}" ]] || { echo "finetune_smolvla.sh 不可执行: ${FT_SCRIPT}" >&2; exit 1; }
[[ -f "${VERIFY_PY}" ]] || { echo "verify_finetune_output.py 不存在: ${VERIFY_PY}" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Step 1: 准备 smoke dataset (2 个成功 episode)
# -----------------------------------------------------------------------------
existing_shard=""
if [[ -d "${SMOKE_DATASET_ROOT}" ]]; then
  # 找第一个含 meta/info.json 的子目录。shard root = info.json 再往上两级。
  while IFS= read -r -d '' cand; do
    existing_shard="$(dirname "$(dirname "${cand}")")"
    break
  done < <(find "${SMOKE_DATASET_ROOT}" -maxdepth 3 -name "info.json" -path "*/meta/info.json" -print0 2>/dev/null)
fi

if [[ -n "${existing_shard}" ]]; then
  say "复用已有 smoke shard: ${existing_shard}"
  SMOKE_SHARD="${existing_shard}"
elif [[ "${SKIP_DATA_COLLECT}" == "1" ]]; then
  echo "[smoke_test_smolvla] SKIP_DATA_COLLECT=1 且 ${SMOKE_DATASET_ROOT} 下没有任何 shard，放弃" >&2
  exit 1
else
  say "没找到已有 smoke shard，调用 collect_dataset.py 采 2 个 episode..."
  mkdir -p "${SMOKE_DATASET_ROOT}"

  # collect_dataset.py 需要 env_isaaclab (3.11 Isaac Lab)。
  ISAAC_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
  [[ -f "${ISAAC_SH}" ]] || { echo "conda 找不到: ${ISAAC_SH}" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "${ISAAC_SH}"
  conda activate "${ISAAC_ENV_NAME}"

  (
    cd "${REPO_ROOT}"
    OPENARM_COLLECT_NUM_SUCCESS=2 \
    OPENARM_COLLECT_DATASET_ROOT="${SMOKE_DATASET_ROOT}" \
    OPENARM_COLLECT_DATASET_REPO_ID="${SMOKE_REPO_ID}" \
    python "${COLLECT_PY}"
  )

  conda deactivate

  # 再去找一遍。
  while IFS= read -r -d '' cand; do
    SMOKE_SHARD="$(dirname "$(dirname "${cand}")")"
    break
  done < <(find "${SMOKE_DATASET_ROOT}" -maxdepth 3 -name "info.json" -path "*/meta/info.json" -print0 2>/dev/null)
  [[ -n "${SMOKE_SHARD:-}" ]] || { echo "[smoke_test_smolvla] 采集失败：${SMOKE_DATASET_ROOT} 下依然找不到 shard" >&2; exit 1; }
  say "新采 smoke shard: ${SMOKE_SHARD}"
fi

# -----------------------------------------------------------------------------
# Step 2/3: 依次跑 full / lora
# -----------------------------------------------------------------------------
run_one() {
  local mode="$1"
  local out_dir="${OUTPUT_ROOT}/${mode}"

  if [[ -d "${out_dir}" ]]; then
    if [[ "${FORCE_FRESH_OUTPUT}" == "1" ]]; then
      say "FORCE_FRESH_OUTPUT=1，删掉旧 ${out_dir}"
      rm -rf "${out_dir}" "${out_dir}.log"
    else
      echo "[smoke_test_smolvla] ${out_dir} 已存在，请加 FORCE_FRESH_OUTPUT=1 清掉再跑" >&2
      exit 1
    fi
  fi
  mkdir -p "$(dirname "${out_dir}")"

  say "===== MODE=${mode} start ====="
  say "  OUTPUT_DIR=${out_dir}"
  say "  DATASET_ROOT=${SMOKE_SHARD}"
  say "  STEPS=${STEPS} BATCH_SIZE=${BATCH_SIZE} SAVE_FREQ=${SAVE_FREQ} LOG_FREQ=${LOG_FREQ}"

  MODE="${mode}" \
  DATASET_ROOT="${SMOKE_SHARD}" \
  DATASET_REPO_ID="${SMOKE_REPO_ID}" \
  OUTPUT_DIR="${out_dir}" \
  STEPS="${STEPS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SAVE_FREQ="${SAVE_FREQ}" \
  LOG_FREQ="${LOG_FREQ}" \
  EVAL_FREQ=0 \
  NUM_WORKERS=0 \
  LOG_FILE="${out_dir}.log" \
    bash "${FT_SCRIPT}"

  say "===== MODE=${mode} finetune done，跑 verify ====="

  # verify 里要用 lerobot / torch / peft，用 smolvla env。
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${SMOLVLA_ENV_NAME}"

  python "${VERIFY_PY}" "${out_dir}" --log-file "${out_dir}.log" --min-steps 1

  conda deactivate
}

if [[ "${SKIP_FULL}" == "1" ]]; then
  say "SKIP_FULL=1，跳过 full fine-tune"
else
  run_one full
fi

if [[ "${SKIP_LORA}" == "1" ]]; then
  say "SKIP_LORA=1，跳过 lora fine-tune"
else
  run_one lora
fi

echo ""
echo "[PASS] SmolVLA smoke test (full + lora) OK"
echo "       - smoke shard: ${SMOKE_SHARD}"
echo "       - full out:    ${OUTPUT_ROOT}/full"
echo "       - lora out:    ${OUTPUT_ROOT}/lora"
