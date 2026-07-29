"""Big pass: 120 scene + multifractal-gradient images (15 scenes x 8 gradient variants,
fixed seed per scene). Saves all + a contact sheet. ~20 min on the 3090."""
import sys
from pathlib import Path
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mf_controlnet import build_pipe, generate, field_to_control, DEFAULT_CN
from mf_creative import gradient_field

SCENES = [
    "dense green forest canopy seen from above",
    "autumn forest treetops, aerial view",
    "vibrant coral reef underwater",
    "rugged rocky mountain landscape",
    "meadow full of wildflowers",
    "aerial view of a branching river delta",
    "rippled desert sand dunes",
    "dense crowd of people seen from above",
    "tangled tropical jungle vegetation",
    "frost-covered pine forest, aerial",
    "rows of purple lavender field, aerial",
    "city street grid at night, aerial",
    "cracked dry lakebed mud flats",
    "ocean waves and sea foam, aerial",
    "rows of green vineyard, aerial",
]
# (direction, cn_scale, c2_hi) — c2_lo fixed at -0.20
VARIANTS = [
    ("h", 0.7, -0.85), ("v", 0.7, -0.85), ("d", 0.8, -0.90), ("radial", 0.8, -0.90),
    ("h", 0.9, -0.95), ("v", 0.6, -0.80), ("d", 0.9, -0.95), ("radial", 0.6, -0.80),
]
C1, C2_LO, STEPS, GEND = 1.3, -0.20, 30, 0.8


def main():
    out = HERE / "gradient_pass"; (out / "img").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {DEFAULT_CN} ...", flush=True)
    pipe = build_pipe(DEFAULT_CN, device)

    n = len(SCENES) * len(VARIANTS); k = 0
    thumb = 220
    sheet = Image.new("RGB", (len(VARIANTS) * thumb, len(SCENES) * thumb), "black")
    import time; t0 = time.time()
    for si, scene in enumerate(SCENES):
        for vi, (d, cn, c2hi) in enumerate(VARIANTS):
            k += 1
            ctrl = field_to_control(gradient_field(C1, C2_LO, c2hi, d, si))
            img = generate(pipe, scene, ctrl, cn, guidance_end=GEND, steps=STEPS,
                           seed=si, device=device)
            img.save(out / "img" / f"s{si:02d}_{d}_cn{cn:.1f}_c2{c2hi:.2f}.png")
            sheet.paste(img.resize((thumb, thumb)), (vi * thumb, si * thumb))
            if k % 8 == 0:
                el = (time.time() - t0) / 60
                print(f"  [{k}/{n}] {scene[:22]:22s} el={el:.1f}m", flush=True)
                sheet.save(out / "contact_sheet.png")          # periodic flush
    sheet.save(out / "contact_sheet.png")
    print(f"DONE {k} images in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
