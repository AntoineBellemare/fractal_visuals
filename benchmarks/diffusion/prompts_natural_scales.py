"""
prompts_natural_scales.py — the natural world across SCALES, for controlled-FD /
controlled-multifractality stimulus generation.

Organised micro -> macro: microscopy, microbial colonies, plants, trees/canopy,
sand/sediment, rock/mineral, water/ice, atmosphere, terrain/aerial. Each family is a
photoreal *substrate*; the multifractal statistics (c1 -> fractal dimension,
c2 -> multifractality) are imposed by guidance, not by the prompt.

Same interface as prompts.py: HEAD/TAIL/NEG/PROMPTS/ALL_FAMILIES, plus SCALE_OF{} mapping
each family to its scale band so results can be grouped by scale.
"""

HEAD = ""
TAIL = ", sharp focus, fine detail, natural light, photorealistic, high resolution, 8K"
NEG = ("low quality, blurry, out of focus, distorted, watermark, text, words, letters, "
       "numbers, logo, signature, frame, border, grid, illustration, cartoon, painting, "
       "render, cgi, human, person, face, hands, figure")


def _p(*core):
    return [c + TAIL for c in core]


PROMPTS = {
    # ---------------- MICROSCOPY ----------------
    "diatoms": _p("dark-field microscope image of diatoms, ornate silica shells, glassy radial patterning",
                  "scanning electron micrograph of diatom frustules, porous geometric shells"),
    "pollen_grains": _p("scanning electron micrograph of pollen grains, spiky textured spheres clustered together",
                        "microscope view of mixed pollen grains, sculpted exine surfaces"),
    "cell_tissue": _p("histology microscope slide of plant tissue, stained cell walls forming a packed mosaic",
                      "brightfield micrograph of leaf cross-section, ordered cell chambers"),
    "blood_smear": _p("microscope blood smear, densely packed red cells on a pale field",
                      "phase-contrast micrograph of cells in suspension, overlapping discs"),
    "plankton": _p("microscope view of marine plankton, translucent filaments and radiolarian shells",
                   "dark-field micrograph of mixed plankton, spiny drifting organisms"),
    "mineral_thin_section": _p("polarized light micrograph of a rock thin section, interlocking birefringent crystals",
                               "petrographic thin section under crossed polars, mosaic of coloured mineral grains"),
    "snow_crystal_micro": _p("microscope photograph of a single snow crystal, dendritic ice arms",
                             "macro micrograph of hoar frost crystals, fine branching ice needles"),

    # ---------------- MICROBIAL / COLONIES ----------------
    "bacterial_colony": _p("petri dish bacterial colony, spreading concentric growth with crenulated margins",
                           "agar plate with bacterial colonies, ringed growth fronts and satellite specks"),
    "biofilm": _p("microbial biofilm sheet on a wet surface, glistening layered mat with ripples",
                  "bacterial biofilm macro, slimy stratified film with wrinkled folds"),
    "mold_growth": _p("mold colony spreading on a damp surface, fuzzy fractal-edged blooms",
                      "petri dish fungal growth, feathery filament fringes and spore centres"),
    "slime_mold": _p("Physarum slime mold plasmodium, branching vein network across damp bark",
                     "yellow slime mold spreading in a reticulate transport web over leaf litter"),
    "mycelium": _p("white fungal mycelium threading through dark substrate, dense hyphal filaments",
                   "mycelial mat on decaying wood, radiating rhizomorph cords"),
    "algae_bloom": _p("green algae bloom on still water, swirling filament mats",
                      "macro of freshwater algae, tangled green threads and bubbles"),

    # ---------------- PLANTS ----------------
    "leaf_venation": _p("macro of a leaf skeleton, hierarchical venation branching into fine veinlets",
                        "backlit leaf close-up, dense reticulate vein network"),
    "moss": _p("macro of a moss cushion, dense green shoots and capsules",
               "forest floor moss carpet close-up, tiny leafy stems packed together"),
    "fern_frond": _p("macro of a fern frond, repeating pinnae and self-similar leaflets",
                     "unfurling fern fiddlehead and fronds, layered green blades"),
    "lichen": _p("crustose lichen on slate, concentric rosettes and crinkled margins",
                 "foliose lichen on bark, ruffled overlapping lobes"),
    "succulent": _p("macro of a succulent rosette, tightly packed geometric leaves",
                    "cluster of small succulents from above, repeating fleshy rosettes"),
    "bark": _p("close-up of rugged tree bark, deep fissures and flaking plates",
               "birch bark macro, peeling papery layers and dark lenticels"),
    "root_system": _p("exposed root system in soil, tangled branching rootlets",
                      "washed plant roots against dark soil, fine ramifying network"),
    "seed_head": _p("macro of a dandelion seed head, radiating filament pappus",
                    "dried umbel seed head close-up, radiating spokes and florets"),
    "flower_field": _p("meadow of wildflowers from above, dense scattered blooms",
                       "field of lavender rows seen from above, textured purple bands"),
    "grass_tussock": _p("macro of dense grass blades, overlapping thin leaves",
                        "dry tussock grass close-up, tangled straw-coloured stems"),

    # ---------------- TREES / CANOPY ----------------
    "forest_canopy": _p("dense green forest canopy seen from directly above, packed crowns",
                        "tropical rainforest canopy aerial, layered treetops"),
    "autumn_canopy": _p("autumn forest treetops from above, mixed orange and gold crowns",
                        "aerial view of deciduous forest in fall colour"),
    "winter_branches": _p("bare winter tree branches against a pale sky, recursive branching",
                          "canopy of leafless trees from below, dense twig network"),
    "pine_forest": _p("frost-covered pine forest seen from above, dark conifer spires in snow",
                      "aerial of dense evergreen forest, packed conifer crowns"),
    "forest_floor": _p("forest floor with fallen leaves and twigs, littered organic debris",
                       "woodland ground close-up, moss, leaf litter and roots"),
    "mangrove": _p("mangrove roots in shallow water from above, tangled prop-root maze",
                   "aerial of mangrove forest edge, braided channels and vegetation"),

    # ---------------- SAND / SEDIMENT ----------------
    "sand_dunes": _p("rippled desert sand dunes from above, sinuous crests and shadows",
                     "aerial of a dune field, repeating wind-formed ridges"),
    "sand_grains": _p("extreme macro of sand grains, individual translucent mineral particles",
                      "close-up of coarse beach sand, mixed grain sizes and shell fragments"),
    "sand_ripples": _p("wet sand ripples on a beach at low tide, fine parallel ridges",
                       "underwater sand ripples seen from above, rhythmic sediment waves"),
    "mud_cracks": _p("cracked dry lakebed mud flats, polygonal desiccation plates",
                     "parched clay surface, hierarchical curling mud cracks"),
    "sediment_layers": _p("stratified sediment cliff face, fine banded layers",
                          "varved sediment core close-up, alternating light and dark laminae"),
    "gravel_bed": _p("river gravel bed seen from above, packed rounded pebbles",
                     "scree slope of broken angular rock fragments"),

    # ---------------- ROCK / MINERAL ----------------
    "granite": _p("polished granite surface, crystalline speckled grains",
                  "rough granite boulder face, quartz and feldspar crystals"),
    "marble": _p("polished marble slab, branching grey veins in white stone",
                 "Carrara marble surface, soft flowing veining"),
    "basalt": _p("columnar basalt formation from above, packed polygonal columns",
                 "weathered basalt surface, vesicular dark volcanic rock"),
    "agate": _p("polished agate slice, concentric banded chalcedony",
                "fortification agate cross-section, sharp nested bands"),
    "weathered_stone": _p("weathered limestone wall, pitted eroded surface with lichen stains",
                          "sea-weathered rock face, honeycomb salt weathering pockets"),
    "cliff_face": _p("rugged rocky cliff face, fractured ledges and talus",
                     "eroded canyon wall, layered rock with deep gullies"),
    "geode": _p("geode interior lined with quartz crystals, glittering druzy surface",
                "amethyst geode cross-section, packed crystal points"),

    # ---------------- WATER / ICE ----------------
    "sea_foam": _p("sea foam on wet sand from above, lacy dissolving bubbles",
                   "ocean waves and foam seen from above, churned white streaks"),
    "water_ripples": _p("sunlit water surface ripples, refracted caustic patterns",
                        "rain drops on a pond surface, interfering circular ripples"),
    "frost_pattern": _p("window frost ferns, dendritic ice crystals on cold glass",
                        "hoar frost on a leaf, feathery ice growth"),
    "ice_sheet": _p("cracked sea ice from above, fractured floes and leads",
                    "glacier surface with crevasses, blue ice fissures"),
    "snow_surface": _p("wind-sculpted snow surface, sastrugi ridges and drifts",
                       "fresh snow crust close-up, granular sparkling texture"),
    "bubbles": _p("dense soap bubble foam macro, packed polyhedral cells",
                  "bubbles frozen in clear ice, trapped spheres at many sizes"),

    # ---------------- ATMOSPHERE ----------------
    "cirrus": _p("wispy cirrus clouds in a pale sky, fibrous ice streaks",
                 "high cirrus filaments at altitude, feathered strands"),
    "cumulus": _p("towering cumulus clouds, billowing cauliflower masses",
                  "field of fair-weather cumulus from above, scattered puffs"),
    "mammatus": _p("mammatus cloud undersides, hanging pouched lobes at dusk",
                   "turbulent storm cloud underbelly, bulbous sagging cells"),
    "smoke_plume": _p("turbulent smoke plume, billowing eddies and thinning wisps",
                      "backlit incense smoke, laminar threads breaking into vortices"),
    "fog_bank": _p("fog rolling over a forest seen from above, soft layered mist",
                   "morning mist over water, diffuse drifting veils"),

    # ---------------- TERRAIN / AERIAL ----------------
    "river_delta": _p("aerial view of a branching river delta, distributary channels in sediment",
                      "satellite-like view of a braided river, interwoven channels"),
    "mountain_range": _p("aerial of a rugged mountain range, ridges and valleys in relief",
                         "snow-dusted mountain ridges from above, dendritic drainage"),
    "canyon": _p("aerial of a deep eroded canyon system, branching tributary gorges",
                 "badlands terrain from above, densely dissected rills"),
    "coastline": _p("aerial of a rocky coastline, indented bays and headlands",
                    "barrier island coast from above, tidal channels and sandbars"),
    "salt_flat": _p("aerial of a salt flat, polygonal evaporite crust",
                    "salt pan surface with crystalline ridges, geometric cells"),
    "tundra_polygons": _p("aerial of arctic patterned ground, ice-wedge polygons",
                          "permafrost tundra from above, mosaic of thaw ponds"),
    "volcanic_terrain": _p("aerial of a lava field, ropey and blocky flow textures",
                           "volcanic crater slopes from above, radiating erosion gullies"),
}

SCALE_OF = {}
for _fams, _scale in [
    (["diatoms", "pollen_grains", "cell_tissue", "blood_smear", "plankton",
      "mineral_thin_section", "snow_crystal_micro"], "microscopy"),
    (["bacterial_colony", "biofilm", "mold_growth", "slime_mold", "mycelium",
      "algae_bloom"], "microbial"),
    (["leaf_venation", "moss", "fern_frond", "lichen", "succulent", "bark",
      "root_system", "seed_head", "flower_field", "grass_tussock"], "plant"),
    (["forest_canopy", "autumn_canopy", "winter_branches", "pine_forest",
      "forest_floor", "mangrove"], "tree"),
    (["sand_dunes", "sand_grains", "sand_ripples", "mud_cracks", "sediment_layers",
      "gravel_bed"], "sand"),
    (["granite", "marble", "basalt", "agate", "weathered_stone", "cliff_face",
      "geode"], "rock"),
    (["sea_foam", "water_ripples", "frost_pattern", "ice_sheet", "snow_surface",
      "bubbles"], "water_ice"),
    (["cirrus", "cumulus", "mammatus", "smoke_plume", "fog_bank"], "atmosphere"),
    (["river_delta", "mountain_range", "canyon", "coastline", "salt_flat",
      "tundra_polygons", "volcanic_terrain"], "terrain"),
]:
    for _f in _fams:
        SCALE_OF[_f] = _scale

ALL_FAMILIES = list(PROMPTS.keys())
SCALES = ["microscopy", "microbial", "plant", "tree", "sand", "rock", "water_ice",
          "atmosphere", "terrain"]
