#!/usr/bin/env bash
# Sequentially run play_lift.py for scene presets 0–9 (one process at a time; saves VRAM).
# Videos go to: <openarm repo>/Multi-Scene-Video/
#
#   CHECKPOINT_PATH=logs/rsl_rl/openarm_lift/<run>/model_3999.pt \
#     bash openarm_vla/Data_Collector/run_play_multi_scene_videos.sh
#
# Requires: conda env ``env_isaaclab``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/Multi-Scene-Video"
PLAY_PY="${SCRIPT_DIR}/play_lift.py"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${PLAY_PY}" ]]; then
  echo "ERROR: missing ${PLAY_PY}" >&2
  exit 1
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: set CHECKPOINT_PATH=/path/to/model.pt before running this script" >&2
  exit 1
fi

# shellcheck disable=SC1091
if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/workspace/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/workspace/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/mambaforge/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/mambaforge/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found (tried ~/miniforge3, ~/workspace/miniforge3, …)" >&2
  exit 1
fi

conda activate env_isaaclab

cd "${REPO_ROOT}"

echo "[INFO] Repo:    ${REPO_ROOT}"
echo "[INFO] Videos:  ${OUT_DIR}"
echo "[INFO] Checkpoint: ${CHECKPOINT_PATH}"
echo "[INFO] Serial presets 0..9 (strictly one Python/Isaac process at a time)."

for i in $(seq 0 9); do
  echo ""
  echo "========================================"
  echo "[INFO] Starting preset ${i} / 9"
  echo "========================================"
  python "${PLAY_PY}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --video_output_dir "${OUT_DIR}" \
    --scene_preset_id "${i}"
  echo "[INFO] Finished preset ${i}"
done

echo ""
echo "[INFO] All presets done. MP4s in: ${OUT_DIR}"
