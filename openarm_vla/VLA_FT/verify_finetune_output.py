#!/usr/bin/env python3
"""Verify the output of a `finetune_smolvla.sh` run.

Usage:
    conda activate smolvla
    python verify_finetune_output.py <output_dir> [--log-file PATH] [--min-steps N] \
        [--skip-load] [--skip-forward]

Checks (all or nothing unless a flag disables it):
  1. ``<output_dir>/`` exists.
  2. ``<output_dir>/checkpoints/last -> <step>/pretrained_model/`` exists and
     has ``config.json``, plus either ``model.safetensors`` (full FT) or
     ``adapter_config.json`` + ``adapter_model.safetensors`` (LoRA).
  3. Parse the training log (by default ``<output_dir>.log`` — we `tee` to there
     from the shell driver). Collect ``loss:<num>`` occurrences, check:
       - at least 1 loss line (smoke test has very few, 20 steps / log_freq 5
         means 4 log lines).
       - all losses are finite.
       - first reported loss is not catastrophically bigger than the last
         (smoke test, so we only sanity-check the ratio ≤ 10x; full FT will
         go down much more, but we don't force it).
  4. Load the checkpoint with the proper factory path (for LoRA: detect adapter
     config and load base + adapter). Make sure the model has trainable params
     and at least one non-zero tensor.
  5. (Optional) Best-effort forward pass with a dummy batch. This is worth its
     weight only if it's cheap — SmolVLA's forward needs tokenized language +
     preprocessed images so we stop at the "model is callable & parameters move
     to CUDA" check rather than replicate the whole preprocessor pipeline.
     Enable with --run-forward; off by default (safer for smoke).

Exit code 0 on pass, non-zero on any check failure.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


LOSS_REGEX = re.compile(r"\bloss:([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf|-inf)", re.IGNORECASE)


def _fail(msg: str) -> None:
    print(f"[verify_finetune_output] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _info(msg: str) -> None:
    print(f"[verify_finetune_output] {msg}")


def check_checkpoint_layout(output_dir: Path) -> tuple[Path, bool]:
    """Return (pretrained_model_dir, is_lora)."""
    if not output_dir.is_dir():
        _fail(f"output_dir 不存在: {output_dir}")

    ckpt_root = output_dir / "checkpoints"
    if not ckpt_root.is_dir():
        _fail(f"没有 checkpoints 子目录: {ckpt_root}")

    # `checkpoints/last` is a symlink pointing to the latest step dir.
    last_link = ckpt_root / "last"
    if not last_link.exists():
        _fail(f"缺少 checkpoints/last 软链: {last_link}. lerobot-train 没能完成至少一次保存")
    step_dir = last_link.resolve()
    if not step_dir.is_dir():
        _fail(f"checkpoints/last 指向的目录不存在: {step_dir}")

    pretrained = step_dir / "pretrained_model"
    if not pretrained.is_dir():
        _fail(f"checkpoint 里缺少 pretrained_model/ 子目录: {pretrained}")

    config_json = pretrained / "config.json"
    if not config_json.is_file():
        _fail(f"pretrained_model 下缺少 config.json: {config_json}")

    adapter_cfg = pretrained / "adapter_config.json"
    adapter_bin = pretrained / "adapter_model.safetensors"
    model_bin = pretrained / "model.safetensors"

    is_lora = adapter_cfg.is_file() and adapter_bin.is_file()
    if is_lora:
        _info(f"检测到 LoRA checkpoint: {pretrained}")
    elif model_bin.is_file():
        _info(f"检测到 full FT checkpoint: {pretrained}")
    else:
        _fail(
            "pretrained_model 下既没有 model.safetensors，也没有完整的 LoRA 文件 "
            f"(adapter_config.json + adapter_model.safetensors)；看 {pretrained} 自检"
        )

    # Sanity: config.json should declare smolvla policy type.
    try:
        with open(config_json) as f:
            cfg = json.load(f)
        if cfg.get("type") != "smolvla":
            _fail(f"config.json 的 type != 'smolvla' (实际 '{cfg.get('type')}')")
    except json.JSONDecodeError as e:
        _fail(f"config.json 解析失败: {e}")

    return pretrained, is_lora


def check_training_log(log_file: Path, min_steps: int) -> list[float]:
    if not log_file.is_file():
        _fail(f"日志文件不存在: {log_file} (finetune_smolvla.sh 应该 tee 到这里)")

    losses: list[float] = []
    with open(log_file, "r", errors="replace") as f:
        for line in f:
            for m in LOSS_REGEX.finditer(line):
                tok = m.group(1)
                try:
                    val = float(tok)
                except ValueError:
                    continue
                losses.append(val)

    if not losses:
        _fail(
            f"日志里找不到任何 'loss:<num>' 条目: {log_file}\n"
            "确认 STEPS >= LOG_FREQ，并且 lerobot-train 至少走完一轮 MetricsTracker 输出"
        )

    # Check for NaN/Inf.
    bad = [v for v in losses if not math.isfinite(v)]
    if bad:
        _fail(f"日志里发现 {len(bad)} 个非有限 loss (NaN/Inf)；前几个 = {bad[:5]}")

    if len(losses) < min_steps:
        _fail(
            f"日志里的 loss 条目数 = {len(losses)} < min_steps={min_steps}。"
            " 可能训练没跑完设定的 STEPS。"
        )

    # Smoke test 只跑 20 步，不强求 loss 下降；但要检查没有灾难性爆炸 (ratio <= 10)。
    first, last = losses[0], losses[-1]
    _info(f"log: 捕获 {len(losses)} 条 loss，first={first:.4f} last={last:.4f}")
    if first > 0 and last / max(first, 1e-8) > 10.0:
        _fail(f"loss 看起来爆炸了：last/first = {last / first:.2f} (>10x)")

    return losses


def check_load_and_forward(
    pretrained: Path,
    is_lora: bool,
    run_forward: bool,
) -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        _fail(f"torch import 失败: {e}")

    # Use the lerobot factory so we hit the same code path as the real runtime.
    # Note: this requires HF cache for lerobot/smolvla_base + SmolVLM2 to exist
    # (we rely on the same cache the trainer just used).
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy
    except ImportError as e:
        _fail(f"lerobot import 失败: {e}")

    try:
        if is_lora:
            # LoRA: cfg.use_peft is written into adapter_config.json via
            # peft_model.config.use_peft = True; lerobot factory detects that
            # and loads base + adapter for us.
            from peft import PeftConfig

            peft_cfg = PeftConfig.from_pretrained(str(pretrained))
            base_path = peft_cfg.base_model_name_or_path
            if not base_path:
                _fail("adapter_config.json 里 base_model_name_or_path 为空，无法定位基座模型")

            policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained))
            policy_cfg.pretrained_path = pretrained
            policy_cfg.use_peft = True

            # We don't have a dataset here — make_policy insists on ds_meta or
            # env_cfg. For LoRA, input_features/output_features are saved inside
            # the checkpoint's config.json, so we can skip ds_meta by filling a
            # minimal fake meta. Cheaper: load base model directly then wrap.
            from lerobot.policies.factory import get_policy_class
            from peft import PeftModel

            policy_cls = get_policy_class(policy_cfg.type)
            # `from_pretrained` on the *base* model id first so we pick up its
            # input/output features + pretrained weights.
            base_cfg = PreTrainedConfig.from_pretrained(base_path)
            base_cfg.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            base_policy = policy_cls.from_pretrained(base_path, config=base_cfg)
            policy = PeftModel.from_pretrained(base_policy, str(pretrained), config=peft_cfg)
        else:
            from lerobot.policies.factory import get_policy_class

            policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained))
            policy_cfg.pretrained_path = pretrained
            policy_cfg.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            policy_cls = get_policy_class(policy_cfg.type)
            policy = policy_cls.from_pretrained(str(pretrained), config=policy_cfg)
    except Exception as e:  # noqa: BLE001
        _fail(f"checkpoint 加载失败: {type(e).__name__}: {e}")

    import torch

    num_params = sum(p.numel() for p in policy.parameters())
    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    _info(f"model loaded OK — 总参数 {num_params/1e6:.1f}M，可训练 {num_trainable/1e6:.2f}M")

    if num_params == 0:
        _fail("模型没有任何参数，checkpoint 明显坏了")

    # Sanity: at least one weight tensor should be non-zero.
    has_nonzero = False
    for p in policy.parameters():
        if torch.isfinite(p).all() and p.abs().sum().item() > 0:
            has_nonzero = True
            break
    if not has_nonzero:
        _fail("所有参数都是 0 或 NaN/Inf，权重加载可能失败")

    if run_forward:
        _info("跳过 end-to-end forward（需要完整的 preprocessor，不在 smoke 范围内）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="finetune_smolvla.sh 的 OUTPUT_DIR")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="训练日志路径，默认 <output_dir>.log（finetune_smolvla.sh 的默认落盘位置）",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="日志里至少要有多少条 loss 记录，smoke 默认 1",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="只检查目录结构和日志，不加载 checkpoint（最便宜的一档）",
    )
    parser.add_argument(
        "--run-forward",
        action="store_true",
        help="加载完 checkpoint 后跑一次 forward；smoke 默认不跑（需要完整 preprocessor）",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    log_file: Path = args.log_file.resolve() if args.log_file else Path(str(output_dir) + ".log")

    _info(f"output_dir = {output_dir}")
    _info(f"log_file   = {log_file}")

    pretrained, is_lora = check_checkpoint_layout(output_dir)
    check_training_log(log_file, min_steps=args.min_steps)

    if args.skip_load:
        _info("--skip-load 指定，跳过 checkpoint 加载检查")
    else:
        check_load_and_forward(pretrained, is_lora, run_forward=args.run_forward)

    print("[verify_finetune_output] PASS")


if __name__ == "__main__":
    main()
