#!/usr/bin/env bash
# Overnight joint-(c1,c2)-balanced corpus pass + retrain + validate.
# Each stage logs to NIGHT_LOG2.md and continues on error so we keep partial results.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONIOENCODING=utf-8
R=benchmarks/regressor
LOG=$R/NIGHT_LOG2.md
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo -e "$1" | tee -a "$LOG"; }

log "\n# Overnight joint-balanced pass — started $(stamp)\n"
# snapshot the current joint model so the eval compares pre-pass vs post-pass
cp $R/joint_model.pt $R/joint_model_before.pt 2>/dev/null && log "snapshotted before-model"

log "\n## Stage 1: scaffold joint-fill (intricate substrates, c1xc2 grid) — $(stamp)"
python -u $R/mf_scaffold.py --corpus --corpus-families prompts_intricate \
  --corpus-c1-targets "0.7,1.1,1.5,1.9" --corpus-targets "-0.2,-0.5,-0.8,-1.1" \
  --corpus-seeds 2 --corpus-strength 0.45 --res 1024 >> "$LOG" 2>&1 \
  && log "stage 1 OK" || log "stage 1 FAILED"

log "\n## Stage 2: scaffold joint-fill (OOD substrates) — $(stamp)"
python -u $R/mf_scaffold.py --corpus --corpus-families prompts_ood \
  --corpus-c1-targets "0.7,1.1,1.5,1.9" --corpus-targets "-0.2,-0.5,-0.8,-1.1" \
  --corpus-seeds 1 --corpus-strength 0.45 --res 1024 >> "$LOG" 2>&1 \
  && log "stage 2 OK" || log "stage 2 FAILED"

log "\n## Stage 3: SDXL diverse keep-all round 2 (fresh seeds) — $(stamp)"
python -u $R/generate_corpus_sdxl.py --prompts-module prompts,prompts_intricate,prompts_ood \
  --per-family 10 --seed-base 200000 --out $R/corpus/images/diffusion_diverse2 >> "$LOG" 2>&1 \
  && log "stage 3 OK" || log "stage 3 FAILED"

log "\n## Stage 4: re-cache VAE latents — $(stamp)"
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

# corpus + coverage summary
python - <<'PY' >> "$LOG" 2>&1
import pandas as pd, numpy as np
d = pd.read_csv("benchmarks/regressor/corpus/manifest.csv")
print("\nfinal corpus:", len(d)); print(d.groupby("source").size().to_string())
c = d[(d.c1>0.5)&(d.c1<2.5)&(d.c2>-1.5)&(d.c2<0.2)]
H,_,_ = np.histogram2d(c.c1, c.c2, bins=[np.linspace(0.6,2.3,7), np.linspace(-1.4,0.0,8)])
print("empty joint cells:", int((H==0).sum()), "/", H.size, "(was 9/42 at start)")
PY

log "\n## DONE — $(stamp)"
