#!/usr/bin/env bash
# Resume the interrupted overnight pass: finish SDXL round 2, then re-cache, retrain, validate.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONIOENCODING=utf-8
R=benchmarks/regressor
LOG=$R/NIGHT_LOG2.md
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo -e "$1" | tee -a "$LOG"; }

log "\n# FINISH PASS (resume after shutdown) — started $(stamp)\n"
# joint_model_before.pt already holds the pre-pass model (snapshotted before the run).

log "## Stage 3 (resume): SDXL diverse keep-all round 2 — $(stamp)"
python -u $R/generate_corpus_sdxl.py --prompts-module prompts,prompts_intricate,prompts_ood \
  --per-family 10 --seed-base 200000 --out $R/corpus/images/diffusion_diverse2 >> "$LOG" 2>&1 \
  && log "stage 3 OK" || log "stage 3 FAILED"

log "\n## Stage 4: re-cache VAE latents (full 3971-img corpus) — $(stamp)"
rm -f $R/corpus/latents.npy $R/corpus/latents_meta.csv
python -u $R/latent_regressor.py --cache-latents --vae-res 1024 --epochs 2 >> "$LOG" 2>&1 \
  && log "stage 4 OK" || log "stage 4 FAILED"

log "\n## Stage 5: retrain joint regressor (400 ep) — $(stamp)"
python -u $R/joint_regressor.py --epochs 400 --patience 45 >> "$LOG" 2>&1 \
  && log "stage 5 OK" || log "stage 5 FAILED"

log "\n## Stage 6a: before/after on diverse content — $(stamp)"
python -u $R/eval_joint_compare.py >> "$LOG" 2>&1 && log "stage 6a OK" || log "stage 6a FAILED"

log "\n## Stage 6b: refreshed reachable atlas — $(stamp)"
python -u $R/reachable_atlas.py --out $R/reachable_pass2 >> "$LOG" 2>&1 \
  && log "stage 6b OK" || log "stage 6b FAILED"

python - <<'PY' >> "$LOG" 2>&1
import pandas as pd, numpy as np
d = pd.read_csv("benchmarks/regressor/corpus/manifest.csv")
print("\nfinal corpus:", len(d)); print(d.groupby("source").size().to_string())
c = d[(d.c1>0.5)&(d.c1<2.5)&(d.c2>-1.5)&(d.c2<0.2)]
H,_,_ = np.histogram2d(c.c1, c.c2, bins=[np.linspace(0.6,2.3,7), np.linspace(-1.4,0.0,8)])
print("empty joint cells:", int((H==0).sum()), "/", H.size, "(was 9/42 at start)")
PY

log "\n## FINISH DONE — $(stamp)"
