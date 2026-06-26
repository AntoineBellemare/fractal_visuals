"""
prompts_ood.py — out-of-distribution substrates for probing generalization.

The regressor/guidance corpus is natural textures (rock, cloud, fluid, lichen, rust,
frost, ink, smoke, dendrites...). These prompts deliberately leave that distribution —
man-made, textile, geometric, biological-different — to test whether c1/c2 control still
works on content the model never trained on. Same interface as prompts.py.
"""

HEAD = "extreme macro photograph, "
TAIL = ", sharp focus, natural light, fine detail, photorealistic, 8K"
NEG = ("low quality, blurry, distorted, watermark, text, words, letters, numbers, logo, "
       "frame, border, grid, illustration, cartoon, human, person, face, eyes, figure, "
       "hand, symbols")


def _p(*core):
    return [HEAD + c + TAIL for c in core]


PROMPTS = {
    # ---- man-made / geometric ----
    "circuit_board": _p(
        "green printed circuit board, copper traces winding between tiny components",
        "densely routed PCB macro, solder pads and conductive pathways",
        "motherboard close-up, layered copper tracks and vias",
        "circuit board with surface-mount components and branching traces",
        "electronic board macro, metallic interconnect lines on green substrate",
    ),
    "crumpled_foil": _p(
        "crumpled aluminium foil, sharp creases and bright specular highlights",
        "wrinkled metallic foil surface, faceted folds catching light",
        "scrunched silver foil macro, chaotic ridges and reflective planes",
        "creased tin foil, angular crumple pattern with glints",
        "compressed aluminium sheet, dense network of folds and dents",
    ),
    "honeycomb": _p(
        "beeswax honeycomb, hexagonal cells filled with amber honey",
        "raw honeycomb macro, waxy hexagon lattice catching warm light",
        "empty honeycomb structure, regular six-sided cells in pale wax",
        "dripping honeycomb, glistening hexagonal chambers",
        "honeycomb cross-section, layered hexagonal wax cells",
    ),
    "woven_fabric": _p(
        "coarse woven linen close-up, interlaced warp and weft threads",
        "macro of burlap weave, rough fibrous crosshatch texture",
        "tweed fabric weave, intertwined colored yarns",
        "denim twill macro, diagonal interlocked cotton threads",
        "canvas weave close-up, dense orthogonal thread grid of fibers",
    ),
    "knitted_wool": _p(
        "chunky knitted wool, rows of soft interlocking stitches",
        "cable-knit sweater macro, twisted plaited wool loops",
        "hand-knitted wool texture, fuzzy interlooped yarn",
        "wool stockinette stitches close-up, V-shaped loops in rows",
        "felted wool surface, matted tangled fibers",
    ),
    # ---- biological-different (texture, no faces) ----
    "peacock_feather": _p(
        "peacock feather barbs close-up, iridescent blue-green filaments",
        "feather macro, fine parallel barbs splitting into barbules",
        "shimmering plumage detail, interlocking iridescent feather strands",
        "exotic bird feather, structured rows of glossy barbs",
        "feather eye-spot macro, concentric bands of colored barbs",
    ),
    "snake_scales": _p(
        "snake skin scales macro, overlapping keeled diamond plates",
        "reptile scale pattern, tessellated glossy overlapping shields",
        "python skin close-up, mosaic of patterned interlocking scales",
        "lizard scale texture, fine granular beaded plates",
        "serpent scales, rows of overlapping keratin tiles",
    ),
    # ---- geological scene (aerial, OOD framing) ----
    "cracked_desert": _p(
        "aerial view of cracked desert mud flat, polygonal fissure network",
        "dry lakebed from above, curling clay plates and deep cracks",
        "parched playa, hierarchical mud-crack mosaic to the horizon",
        "drought-cracked earth aerial, interlocking dried clay polygons",
        "desert hardpan macro, branching desiccation cracks",
    ),
}

ALL_FAMILIES = list(PROMPTS.keys())
