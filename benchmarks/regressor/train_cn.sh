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
# MEASURED on this 3090 (24GB), and both limits are real:
#  * SPEED: 1024 res + grad-checkpointing = ~12 s per MICRO-step, so accum 4 => 48 s/step
#    => 40 h for 3000 steps. Too slow.
#  * MEMORY: at 768 the process alone held 23.9 GB (baseline with it killed: 0.35 GB) and
#    spilled to host RAM under Windows WDDM -> GPU pinned at 100% with step progress
#    FROZEN for 7 min. Resident cost is UNet(bf16) + ControlNet + fp32 grads + 8-bit Adam
#    state, before activations; the memory-saving flags are not enough at 768.
# So: 512 res (activations ~0.44x of 768), accum 2. ~6 s/step => 2000 steps ~ 3.5 h.
# CAVEAT: trained at 512 but sampled at 1024. The conditioning maps are smooth/low-frequency
# so the control pathway should transfer, but VERIFY at the gates; a higher-res continuation
# needs >24GB or latent+embedding precaching.
STEPS="${1:-2000}"
RES="${2:-512}"
ACCUM="${3:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation

accelerate launch --num_processes 1 --mixed_precision bf16 \
  $R/train_scripts/train_controlnet_sdxl.py \
  --pretrained_model_name_or_path "SG161222/RealVisXL_V4.0" \
  --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
  --controlnet_model_name_or_path "xinsir/controlnet-tile-sdxl-1.0" \
  --output_dir "$R/cn_model" \
  --train_data_dir "$R/cn_dataset" \
  --image_column image --conditioning_image_column conditioning_image --caption_column text \
  --resolution "$RES" \
  --learning_rate 1e-5 \
  --max_train_steps "$STEPS" \
  --train_batch_size 1 --gradient_accumulation_steps "$ACCUM" \
  --gradient_checkpointing --use_8bit_adam --set_grads_to_none \
  --mixed_precision bf16 \
  --checkpointing_steps 500 --validation_steps 500 \
  --validation_image "$R/val_cond_uniform.png" "$R/val_cond_gradient.png" \
  --validation_prompt "dense forest canopy, natural texture, photorealistic, fine detail" \
                      "rocky mountain terrain, natural texture, photorealistic, fine detail" \
  --seed 0 \
  --report_to tensorboard
