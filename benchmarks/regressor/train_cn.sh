#!/usr/bin/env bash
# Phase B — fine-tune the MF-ControlNet on (exact-label conditioning -> image) pairs.
#
# Flags justified:
#   --controlnet_model_name_or_path xinsir/controlnet-tile-sdxl-1.0   warm start: tile already
#       conditions on a continuous-tone reference, so we only have to move it from "copy the
#       reference" to "realise these statistics". Converges far faster than from-UNet init.
#   --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix  SDXL's own VAE is
#       numerically unstable in low precision.
#   --mixed_precision bf16   3090 is Ampere -> bf16 available, and it is safer than fp16 for
#       TRAINING (no loss-scale blowups).
#   --gradient_checkpointing --use_8bit_adam --set_grads_to_none   the three flags that take
#       SDXL-ControlNet training from ~38GB down into a 24GB card.
#   batch 1 x accum 4 = effective batch 4.
#   NOTE: xformers is deliberately NOT used — torch 2.5 SDPA is the default and is enough.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONIOENCODING=utf-8
R=benchmarks/regressor
STEPS="${1:-3000}"          # first run: 3000 (sanity). full run: 10000
RES="${2:-1024}"            # drop to 768 if OOM

accelerate launch --mixed_precision bf16 $R/train_scripts/train_controlnet_sdxl.py \
  --pretrained_model_name_or_path "SG161222/RealVisXL_V4.0" \
  --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
  --controlnet_model_name_or_path "xinsir/controlnet-tile-sdxl-1.0" \
  --output_dir "$R/cn_model" \
  --train_data_dir "$R/cn_dataset" \
  --image_column image --conditioning_image_column conditioning_image --caption_column text \
  --resolution "$RES" \
  --learning_rate 1e-5 \
  --max_train_steps "$STEPS" \
  --train_batch_size 1 --gradient_accumulation_steps 4 \
  --gradient_checkpointing --use_8bit_adam --set_grads_to_none \
  --mixed_precision bf16 \
  --checkpointing_steps 500 --validation_steps 500 \
  --validation_image "$R/val_cond_uniform.png" "$R/val_cond_gradient.png" \
  --validation_prompt "dense forest canopy, natural texture, photorealistic, fine detail" \
                      "rocky mountain terrain, natural texture, photorealistic, fine detail" \
  --seed 0 \
  --report_to tensorboard
