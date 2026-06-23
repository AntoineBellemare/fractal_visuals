"""
prompts_intricate.py — intricate, strongly-multifractal pareidolia substrates.

Companion to prompts.py. Where prompts.py covers the 28 family-paired natural textures,
this set targets the HIGH-c2 regime: branching, dendritic, turbulent, reticulated and
weathered structures that (a) carry strong nested intermittency (the c2 we want to
control) and (b) are classic pareidolia substrates — the eye supplies faces/figures, so
the prompts describe the STRUCTURE and explicitly forbid actual faces/figures.

Same interface as prompts.py (HEAD/TAIL/NEG/_p/PROMPTS/ALL_FAMILIES), so generate_corpus_sdxl
and guided_sample can load it via --prompts-module prompts_intricate. These families are
diffusion-only (no procedural counterpart); analyze.py's family pairing does not apply.
"""

# Refined: give a CLEAN photoreal substrate and let guidance/scaffold set the
# multifractality. The old "nested/hyperdetailed/labyrinthine" tail homogenised detail
# (raised the c2 floor) and triggered guidance artifacts, so it's gone.
HEAD = "extreme macro photograph, "
TAIL = ", sharp focus, natural light, fine detail, photorealistic, 8K"

# Forbid literal faces/figures/text: pareidolia must come from the viewer, not the render.
# (text in NEG also suppresses the text-grid decode artifact seen under strong guidance.)
NEG = ("low quality, blurry, distorted, watermark, text, words, letters, numbers, logo, "
       "signature, frame, border, grid, illustration, cartoon, painting, sketch, render, "
       "cgi, human, person, face, eyes, figure, animal, hand, symbols")


def _p(*core):
    return [HEAD + c + TAIL for c in core]


PROMPTS = {
    # ---------- BIOLOGICAL / ORGANIC (branching, reticulated) ----------
    "lichen_crust": _p(
        "crustose lichen colony spreading across slate, concentric rosettes and tendrils",
        "map lichen mosaic on granite, interlocking polygonal patches with crinkled edges",
        "foliose lichen tangle, ruffled lobes and rhizines, blue-green and ochre",
        "dense lichen encrustation on bark, branching filaments and pustules",
        "leprose lichen powder spreading in fractal margins over weathered stone",
    ),
    "slime_mold": _p(
        "Physarum slime mold plasmodium, branching vein network across damp wood",
        "yellow slime mold tendrils forming an optimized transport web on leaf litter",
        "slime mold migrating front, pulsating reticulate channels and lobes",
        "myxomycete network, anastomosing veins thinning into delicate capillaries",
        "slime mold fruiting bodies clustered, tangled stalks and webbed strands",
    ),
    "mycelium_net": _p(
        "white fungal mycelium network threading through dark substrate, dense filaments",
        "mycelial mat colonizing wood, radiating hyphae and branching nodes",
        "cobweb mycelium spreading in fractal fans across compost",
        "rhizomorph cords of fungus, root-like branching black strands on bark",
        "fungal hyphae web under microscope, interwoven threads and junctions",
    ),
    "coral_brain": _p(
        "brain coral surface, convoluted meandering ridges and grooves, labyrinthine",
        "maze coral close-up, sinuous interlocking valleys, calcite texture",
        "brain coral polyp ridges, nested folds tightening into a maze",
        "fossil coral cross-section, convoluted septa and chambers",
        "Platygyra coral meanders, cerebral folds in pale limestone",
    ),
    "capillary_network": _p(
        "leaf venation skeleton, hierarchical branching veins from midrib to capillaries",
        "dried leaf vein network backlit, fractal reticulation of fine veinlets",
        "root system mat, tangled hairlike rootlets branching through soil",
        "river-delta capillary pattern etched in dried silt, dendritic channels",
        "blood-vessel-like mineral network in translucent agate, fine ramifying threads",
    ),

    # ---------- MINERAL / DENDRITIC ----------
    "manganese_dendrite": _p(
        "manganese dendrites on limestone, black fern-like fractal mineral growth",
        "pyrolusite dendrites branching across pale sandstone, delicate tree-like fronds",
        "moss agate dendritic inclusions, green-black mineral ferns in translucent stone",
        "dendritic crystallization on a bedding plane, self-similar branching filaments",
        "iron-oxide dendrites spreading in fractal arborescence over chalk",
    ),
    "frost_fern": _p(
        "window frost ferns, ice crystal dendrites branching across cold glass",
        "feathery hoar frost, fractal ice fronds radiating from nucleation points",
        "frost flowers on a frozen pane, interlocking crystalline ferns",
        "rime ice crystals, dense dendritic frost lattice catching light",
        "fern frost macro, nested ice branches splitting into ever finer tips",
    ),
    "salt_efflorescence": _p(
        "salt efflorescence bloom on brick, crystalline white fractal crust",
        "mineral efflorescence spreading over concrete, feathery salt whiskers",
        "evaporite crust on dried mud, polygonal salt ridges and blooms",
        "gypsum efflorescence on cave wall, branching crystalline tendrils",
        "saline crust crystallizing in dendritic fans across stone",
    ),

    # ---------- DECAY / WEATHERING ----------
    "peeling_paint": _p(
        "cracked peeling paint layers on rusted metal, curling flakes and fissures",
        "alligatored varnish craquelure, nested cracking in aged enamel",
        "weathered ship hull paint, blistered peeling scabs over corrosion",
        "multilayer paint delaminating, intricate flake mosaic and lifted edges",
        "crazed lacquer surface, fine fractal crack network over faded color",
    ),
    "copper_patina": _p(
        "oxidized copper verdigris patina, mottled blue-green corrosion blooms",
        "weathered bronze surface, dappled patina with crystalline corrosion fronts",
        "aged copper roof close-up, turquoise oxidation spreading in fractal patches",
        "malachite-like patina creeping over metal, nested tarnish gradients",
        "corroded brass with verdigris dendrites and pitting, intricate mottling",
    ),
    "rust_corrosion": _p(
        "blooming rust on steel, layered orange corrosion scabs and flaking laminae",
        "oxidized iron plate, pitted rust craters with concentric tide rings",
        "weathered corten steel, mottled rust gradients and crystalline flaking",
        "rust bleeding across riveted metal, branching stain tendrils",
        "deep corrosion pitting, nested rust nodules and exfoliating layers",
    ),
    "mold_colony": _p(
        "mold colony spreading on damp plaster, fuzzy fractal-edged blooms",
        "black mold growth fronts, dendritic spore tendrils on a wall",
        "petri dish fungal colony, ringed growth margins and feathery edges",
        "mildew creeping over fabric, mottled colonies with diffuse fractal borders",
        "spreading mold mat, overlapping circular colonies and filament fringes",
    ),
    "cracked_glaze": _p(
        "crazed ceramic glaze, fine craquelure network over celadon",
        "dried mud cracks, polygonal plates curling at intricate fissured edges",
        "crackle-glaze porcelain, nested random crack mosaic catching ink",
        "parched lakebed crust, hierarchical desiccation cracks in clay",
        "old oil-painting craquelure, branching cracks through varnish",
    ),

    # ---------- FLUID / TURBULENT ----------
    "ferrofluid": _p(
        "ferrofluid spike labyrinth under a magnet, self-organizing black peaks",
        "magnetic fluid maze, interlocking spiked ridges and valleys",
        "ferrofluid instability fronts, branching spikes splitting recursively",
        "oily ferrofluid surface, hedgehog spike lattice with mirror sheen",
        "ferrofluid in a field, labyrinthine ridge pattern of nested spikes",
    ),
    "ink_bloom": _p(
        "black ink blooming in water, turbulent tendrils and recursive curls",
        "ink diffusing into milk, fractal plumes branching into fine filaments",
        "dye dispersing in water, mushrooming vortices and thread-like wisps",
        "ink drop turbulence, nested swirls unraveling into capillary threads",
        "smoke-like ink cloud in water, billowing self-similar eddies",
    ),
    "marbled_ink": _p(
        "suminagashi marbled ink, concentric rings dragged into feathered swirls",
        "ebru paper marbling, combed nested loops and fine tendrils of pigment",
        "oil-and-ink marble, turbulent veining and folded color strata",
        "marbled endpaper pattern, intricate stone-and-vein swirls",
        "fluid marbling, raked spirals breaking into delicate filaments",
    ),
    "oil_iridescence": _p(
        "iridescent oil film on a wet road, swirling interference colors and eddies",
        "thin oil slick on water, marbled rainbow sheen with turbulent boundaries",
        "petrol film iridescence, nested color contours folding into vortices",
        "soap-film interference macro, shifting iridescent bands and cells",
        "oil-on-water spectral swirls, fractal mixing fronts",
    ),

    # ---------- ATMOSPHERIC ----------
    "mammatus_cloud": _p(
        "mammatus cloud undersides, pouched lobes hanging in nested bulges",
        "turbulent storm cloud pouches, self-similar sagging cells at dusk",
        "mammatus formation backlit, rounded lobes tiling the sky",
        "chaotic cloud underbelly, bulbous nested pouches and shadowed folds",
        "asperitas cloud waves, churning rippled undulations overhead",
    ),
    "smoke_eddies": _p(
        "turbulent smoke plume, billowing eddies curling into fine filaments",
        "incense smoke in still air, laminar threads breaking into vortices",
        "dense smoke wall, self-similar rolls and wisps tearing apart",
        "rising smoke turbulence, nested mushrooming curls dissolving to haze",
        "backlit smoke, intricate vortex streets and thinning tendrils",
    ),
}

ALL_FAMILIES = list(PROMPTS.keys())
