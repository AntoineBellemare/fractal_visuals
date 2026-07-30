"""
prompts_creatures.py — COMPLEX / semantic content for controlled-multifractality tests.

Everything so far has been texture or scene. This set is deliberately *object-like*: creatures,
faces, organisms, and — most importantly for pareidolia — AMBIGUOUS content where a face or
creature is only suggested by the texture. That last group is the actual research target: the
percept should emerge from the statistics, not from a literal depiction.

Three bands:
  creature  — explicit organisms (control test: does guidance survive strong semantics?)
  face      — explicit faces/portraits (hardest semantic prior)
  emergent  — pareidolia bait: face/creature merely SUGGESTED by cloud, rock, bark, smoke
"""

TAIL = ", sharp focus, fine detail, natural light, photorealistic, high resolution, 8K"
NEG = ("low quality, blurry, out of focus, distorted, deformed, watermark, text, letters, "
       "logo, signature, frame, border, cartoon, illustration, render, cgi")


def _p(c):
    return [c + TAIL]


PROMPTS = {
    # ---------- explicit creatures ----------
    "deep_sea_creature": _p("a translucent bioluminescent deep-sea creature with delicate tendrils"),
    "owl": _p("an owl perched on a branch, intricate feather detail, facing the camera"),
    "insect_macro": _p("extreme macro of a beetle, iridescent chitin and fine hairs"),
    "octopus": _p("an octopus on the seafloor, textured skin and suckers"),
    "lizard": _p("a chameleon on a branch, granular patterned skin"),
    "coral_creature": _p("a strange soft-bodied organism made of living coral and polyps"),

    # ---------- explicit faces ----------
    "stone_face": _p("a weathered human face carved into rough stone, eroded surface"),
    "lion_face": _p("close-up of a lion's face, dense fur and whiskers"),
    "bark_face": _p("a face carved into old tree bark, deep grain and fissures"),
    "ice_face": _p("a human face sculpted from cracked ice, frosted surface"),

    # ---------- emergent / pareidolia bait ----------
    "cloud_face": _p("billowing storm clouds whose shapes suggest a vague human face"),
    "rock_creature": _p("an eroded rock formation that resembles a crouching creature"),
    "smoke_figure": _p("dense curling smoke that suggests a shadowy figure"),
    "root_face": _p("a tangle of tree roots and moss that suggests a hidden face"),
    "lichen_creature": _p("lichen and mineral staining on stone suggesting a creature silhouette"),
    "wave_creature": _p("churning sea foam and waves suggesting the form of a large animal"),
}

BAND_OF = {}
for _f, _b in [
    ("deep_sea_creature", "creature"), ("owl", "creature"), ("insect_macro", "creature"),
    ("octopus", "creature"), ("lizard", "creature"), ("coral_creature", "creature"),
    ("stone_face", "face"), ("lion_face", "face"), ("bark_face", "face"), ("ice_face", "face"),
    ("cloud_face", "emergent"), ("rock_creature", "emergent"), ("smoke_figure", "emergent"),
    ("root_face", "emergent"), ("lichen_creature", "emergent"), ("wave_creature", "emergent"),
]:
    BAND_OF[_f] = _b

ALL_FAMILIES = list(PROMPTS.keys())
BANDS = ["creature", "face", "emergent"]
