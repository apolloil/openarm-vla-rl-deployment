#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SmolVLA 微调启动脚本（Full / LoRA 两种模式同一入口）
#
# 读取配置的方式是环境变量（便于 smoke_test 与真实微调复用）：
#   MODE              full | lora                 必填：默认 full
#   DATASET_ROOT      某个 collect_dataset.py 产出的 shard 根目录；包含
#                     meta/, data/, videos/ 三个子目录               必填
#   DATASET_REPO_ID   LeRobot dataset 的 repo_id（meta/info.json 里那个）
#                     默认 openarm/lift_cube_expert_v0
#   OUTPUT_DIR        训练产物目录；必须不存在（lerobot 拒绝覆盖）     必填
#   STEPS             训练步数                               默认 20000
#   BATCH_SIZE        批大小                                 默认 32
#   LR                optimizer 学习率（overrides 策略 preset）
#                     full 默认 1e-4，lora 默认 1e-3
#   SAVE_FREQ         checkpoint 频率（步）                   默认 5000
#   LOG_FREQ          日志频率（步）                          默认 50
#   EVAL_FREQ         留 0 就不跑在线 eval（smoke 一定 0）    默认 0
#   SEED              随机种子                               默认 1000
#   NUM_WORKERS       dataloader 进程数                      默认 2
#   PEFT_R            LoRA rank（仅 MODE=lora 生效）          默认 64
#   MIXED_PRECISION   传给 accelerate：no | fp16 | bf16       默认 bf16
#   LOG_FILE          日志落盘路径；默认 $OUTPUT_DIR/train.log
#
# SmolVLA base (lerobot/smolvla_base) 的输入要求：
#   observation.state                (STATE，会被 pad 到 32 维)
#   observation.images.camera1/2/3   (VISUAL，3x256x256)
# 我们数据集里是：
#   observation.state.ee_pose        (4D，x/y/z/finger_width)
#   observation.state.goal           (3D，ignored — 目标已经画在图里)
#   observation.images.scene         (720x1280x3)
# 所以 rename_map 把 ee_pose -> state，scene -> camera1；
# 缺的 camera2 / camera3 通过 --policy.empty_cameras=2 用 0 占位；
# observation.state.goal 不映射 => 不在 policy.input_features 里 => 被静默忽略。
# -----------------------------------------------------------------------------
set -euo pipefail

# -----------------------------------------------------------------------------
# 0. 激活 smolvla conda env + 依赖自检
# -----------------------------------------------------------------------------
CONDA_BASE="${CONDA_BASE:-/home/lcw/workspace/miniforge3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-smolvla}"
LEROBOT_REPO_PATH="${LEROBOT_REPO_PATH:-/home/lcw/workspace/openarm/lerobot}"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "[finetune_smolvla] conda 找不到：${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  echo "[finetune_smolvla] 重设 CONDA_BASE 环境变量再跑" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

if ! python - <<'PY'
import sys
try:
    import lerobot  # noqa: F401
    import peft     # noqa: F401
except Exception as e:
    print(f"[finetune_smolvla] import failed: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
import shutil
if shutil.which("lerobot-train") is None:
    print("[finetune_smolvla] lerobot-train CLI 不在 PATH，检查 pip install 是否完整", file=sys.stderr)
    sys.exit(1)
PY
then
  echo ""
  echo "[finetune_smolvla] 依赖自检失败。请在 ${CONDA_ENV_NAME} env 里装好 lerobot + peft：" >&2
  echo "    conda activate ${CONDA_ENV_NAME}" >&2
  echo "    pip install -e \"${LEROBOT_REPO_PATH}[peft]\"" >&2
  echo "如果内网没 pip 能访问的源，peft (纯 Python) 可以直接从别的 env 复制过来：" >&2
  echo "    cp -r <other_env>/lib/python3.*/site-packages/peft* \\" >&2
  echo "          ${CONDA_BASE}/envs/${CONDA_ENV_NAME}/lib/python3.12/site-packages/" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# 1. 读配置 + 必填校验
# -----------------------------------------------------------------------------
MODE="${MODE:-full}"
case "${MODE}" in
  full|lora) ;;
  *) echo "[finetune_smolvla] MODE 必须是 full 或 lora，收到: '${MODE}'" >&2; exit 2 ;;
esac

: "${DATASET_ROOT:?必须指定 DATASET_ROOT (某个 collect_dataset.py 产出的 shard)}"
: "${OUTPUT_DIR:?必须指定 OUTPUT_DIR (训练输出目录，必须不存在)}"

DATASET_REPO_ID="${DATASET_REPO_ID:-openarm/lift_cube_expert_v0}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-0}"
SEED="${SEED:-1000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PEFT_R="${PEFT_R:-64}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

if [[ "${MODE}" == "full" ]]; then
  LR="${LR:-1e-4}"
else
  # PEFT 通常用更高的 lr，因为只更新少量 adapter 参数。
  LR="${LR:-1e-3}"
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[finetune_smolvla] DATASET_ROOT 不存在: ${DATASET_ROOT}" >&2
  exit 3
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "[finetune_smolvla] ${DATASET_ROOT} 看起来不是 LeRobot shard（缺 meta/info.json）" >&2
  exit 3
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
  echo "[finetune_smolvla] OUTPUT_DIR 已存在: ${OUTPUT_DIR}" >&2
  echo "                   lerobot-train 不会覆盖既有目录，请换一个或删掉它。" >&2
  exit 3
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"
mkdir -p "$(dirname "${LOG_FILE}")"

# -----------------------------------------------------------------------------
# 2. rename_map 构造
#    注意：lerobot 的 draccus parser 读 JSON 字典，传 CLI 的时候整个值要是一个
#    字符串；由 bash 的单引号传过去。
# -----------------------------------------------------------------------------
RENAME_MAP='{"observation.images.scene":"observation.images.camera1","observation.state.ee_pose":"observation.state"}'

# -----------------------------------------------------------------------------
# 3. 构造 lerobot-train 参数
# -----------------------------------------------------------------------------
CMD=(
  lerobot-train
  --policy.path=lerobot/smolvla_base
  --policy.device=cuda
  --policy.empty_cameras=2
  "--policy.optimizer_lr=${LR}"
  "--policy.push_to_hub=false"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--output_dir=${OUTPUT_DIR}"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--save_freq=${SAVE_FREQ}"
  "--log_freq=${LOG_FREQ}"
  "--eval_freq=${EVAL_FREQ}"
  "--seed=${SEED}"
  "--num_workers=${NUM_WORKERS}"
  "--wandb.enable=false"
  "--rename_map=${RENAME_MAP}"
)

if [[ "${MODE}" == "lora" ]]; then
  CMD+=(
    --peft.method_type=LORA
    "--peft.r=${PEFT_R}"
  )
fi

# -----------------------------------------------------------------------------
# 4. 打印 + 执行
# -----------------------------------------------------------------------------
echo "==================================================================="
echo "[finetune_smolvla] MODE=${MODE}"
echo "[finetune_smolvla] DATASET_ROOT=${DATASET_ROOT}"
echo "[finetune_smolvla] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[finetune_smolvla] LOG_FILE=${LOG_FILE}"
echo "[finetune_smolvla] STEPS=${STEPS}  BATCH_SIZE=${BATCH_SIZE}  LR=${LR}"
echo "[finetune_smolvla] SAVE_FREQ=${SAVE_FREQ}  LOG_FREQ=${LOG_FREQ}  EVAL_FREQ=${EVAL_FREQ}"
echo "[finetune_smolvla] MIXED_PRECISION=${MIXED_PRECISION}"
[[ "${MODE}" == "lora" ]] && echo "[finetune_smolvla] PEFT_R=${PEFT_R}"
echo "[finetune_smolvla] CMD:"
printf '  %q \\\n' "${CMD[@]}"
echo "==================================================================="

# 关掉在线功能 + 让 accelerate 走 bf16。
# HF_HUB_OFFLINE=1：默认用已缓存的模型，避免内网代理封 huggingface.co。
# 确实需要拉新权重时用 HF_HUB_OFFLINE=0 再跑。
export WANDB_DISABLED=true
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export ACCELERATE_MIXED_PRECISION="${MIXED_PRECISION}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"

# 同时写 stdout 和日志文件，方便后续 verify_finetune_output.py 从 log 里拉 loss。
set -o pipefail
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
