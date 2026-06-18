"""
Prompts for the diffusion natural-texture sweep.

32 families * 5 prompt variations. Each family gets prompts that target the
distinctive visual character of its procedural counterpart, while staying in
high-detail close-up natural-texture territory (no people, no objects).
"""

NEG = ("low quality, blurry, distorted, watermark, text, logo, signature, "
       "frame, border, drawing, illustration, cartoon, painting, sketch, "
       "human, person, face, hand, object, geometric")

# Anchor that boosts photorealistic detail and discourages stylization.
HEAD = "extreme macro photograph, "
TAIL = ", professional photography, sharp focus, 8K, high detail, photorealistic"


def _p(*core):
    return [HEAD + c + TAIL for c in core]


PROMPTS = {
    # ---------- ROCK ----------
    "marble": _p(
        "polished white marble slab with delicate gray veins flowing across the surface",
        "Carrara marble countertop closeup, soft branching veins, natural daylight",
        "antique Italian marble texture with intertwined dark veins, museum lighting",
        "translucent white marble fragment with subtle gold veining, studio light",
        "honed marble surface with mineral streaks, even diffuse light, museum-grade",
    ),
    "agate": _p(
        "polished agate slice with concentric colored bands, vivid layered mineral, bright studio light",
        "blue lace agate slab cross-section showing wavy parallel bands, gem photography",
        "Brazilian agate geode interior, layered banded chalcedony, jewel-tone palette",
        "fortification agate with sharp angular bands, mineral specimen lighting",
        "translucent agate slice glowing against backlight, fine banding, gem-quality",
    ),
    "weathered": _p(
        "weathered limestone wall surface with patchy lichen stains and mineral deposits, overcast daylight",
        "old sandstone block with rain-eroded blotches, neutral gray stains, ancient ruin texture",
        "weathered concrete surface with rust streaks and dark patina, urban decay",
        "lichen-covered boulder face, mottled gray-green stains, natural forest light",
        "saltwater-weathered stone with white mineral deposits, sea cliff texture",
    ),
    "granite": _p(
        "polished gray granite surface with sharp crystalline grains, bright studio light",
        "black and pink granite countertop closeup, mica flecks visible, kitchen lighting",
        "natural rough granite boulder face, individual quartz crystals catching sunlight",
        "salt-and-pepper granite slab, even crystal distribution, museum lighting",
        "pink granite surface showing feldspar crystals and dark biotite specks, macro detail",
    ),
    "vesicular": _p(
        "vesicular basalt surface with countless small gas bubbles, dark volcanic rock, raking light",
        "porous pumice stone closeup, irregular vesicles, dry studio light",
        "scoria volcanic rock surface, deep dark cavities, dramatic shadow",
        "lava rock with frozen bubble cavities, rough porous surface, natural daylight",
        "weathered vesicular basalt with mineral-stained pore interiors, overcast light",
    ),
    "turbulent": _p(
        "turbulently flowing molten rock that has solidified mid-swirl, ropy pahoehoe lava surface",
        "obsidian flow surface with chaotic swirl patterns, glossy black volcanic glass",
        "fluid lava flow stone with intertwined ribbon textures, low raking sunlight",
        "frozen flow-stone surface with curling sworls and chaotic eddies, natural light",
        "viscous magma surface texture, twisted ribbons of dark rock, dramatic side light",
    ),
    "schist": _p(
        "mica schist rock surface with shimmering parallel foliation planes, natural daylight",
        "garnet schist closeup, fine wavy foliations with dark mineral grains, museum lighting",
        "weathered schist boulder face, layered laminations catching light, mountain trail",
        "graphite-rich schist surface, dark wavy foliations, raking studio light",
        "silvery muscovite schist showing reflective layered sheets, geologist's specimen photo",
    ),
    "serpentinite": _p(
        "polished serpentinite rock slab, mottled deep green with darker veins, museum specimen light",
        "raw serpentinite outcrop face, mottled green-black surface with thin white veinlets, natural daylight",
        "weathered serpentinite boulder, splotchy green and rust patina, forest light",
        "antigorite serpentinite closeup, mottled olive-green with chrysotile streaks, geology specimen",
        "polished California serpentinite showing dark vein network across green matrix, studio light",
    ),
    "flow_banded": _p(
        "flow-banded rhyolite surface with parallel colored bands, smooth banded lava, raking light",
        "banded iron formation closeup, alternating rust-red and gray laminations, geological specimen",
        "flow-banded obsidian with thin parallel stripes, glossy volcanic glass, low light",
        "rhyolite flow bands closeup, ribbon-like alternating textures, natural sunlight",
        "banded fluorite rock face, wide parallel mineral bands, museum specimen lighting",
    ),
    "convoluted": _p(
        "convoluted metamorphic rock surface with chaotic folded layers, dramatic light",
        "ptygmatic folded gneiss closeup, tight chaotic folds within layered host rock",
        "highly contorted migmatite surface, dark folded layers within lighter matrix, natural light",
        "intensely folded slate with chaotic crinkle pattern, raking studio light",
        "convoluted folded marble surface, swirling bands of contrasting colors, museum lighting",
    ),
    "gneiss": _p(
        "high-grade gneiss rock face with alternating light and dark mineral bands, natural daylight",
        "polished gneiss slab closeup, classic granite-like banding, museum lighting",
        "weathered gneiss outcrop with parallel laminations, mountain trail photography",
        "augen gneiss surface showing lens-shaped feldspar crystals in banded matrix, geologist's photo",
        "biotite-rich gneiss showing dark mineral bands alternating with quartz-feldspar layers, studio light",
    ),

    # ---------- CLOUD ----------
    "cascade_lognormal": _p(
        "puffy cumulus clouds filling the sky, sunlit billowing tops, blue background",
        "isolated cumulus cloud against deep blue sky, sharp edges, midday sunlight",
        "cumulus humilis cloud field photographed from above, dense puffy tops, aerial view",
        "towering cumulus clouds with strong shadow contrast, dramatic late afternoon light",
        "small cumulus puffs at low altitude, evenly distributed across blue sky, soft sunlight",
    ),
    "stratified": _p(
        "horizontal stratus cloud layers stacked across the sky, parallel cloud bands, dawn light",
        "altostratus cloud sheet seen from below, even gray layer with subtle texture, overcast",
        "stratocumulus cloud field with parallel rolls, low altitude, late afternoon",
        "layered cloud strata seen from an airplane window, parallel bands at sunset",
        "uniform stratus deck filling the sky, soft horizontal banding, diffuse daylight",
    ),
    "billow": _p(
        "Kelvin-Helmholtz billow clouds rolling across the sky, wave-like cloud crests, dawn light",
        "rotor cloud waves over mountain ridge, parallel curling billows, dramatic side light",
        "lenticular cloud waves stacked over a peak, smooth wave crests, golden hour",
        "asperitas cloud formation seen from below, undulating wave pattern across the sky, dramatic light",
        "wave cloud sheet with regular rippled crests, evening sky, low warm sun",
    ),
    "cirrus": _p(
        "wispy cirrus clouds streaking across blue sky, fine ice crystal filaments, high altitude",
        "cirrus fibratus closeup with fine wispy strands, deep blue background, midday sun",
        "cirrus uncinus mares-tails sweeping across the sky, delicate hooked strands, soft sunlight",
        "high-altitude cirrus stripes against a clear blue sky, parallel ice crystal trails, daylight",
        "thin cirrostratus veil with subtle striations across the sky, soft diffuse sunlight",
    ),
    "cloud_mrw": _p(
        "turbulent cumulus cloud with intricate multi-scale puffs, sharp sunlit edges, intense afternoon light",
        "massive convective cloud face showing complex turbulence, dramatic side light",
        "turbulent thunderhead cloud closeup, complex bubbling cauliflower structure, dramatic light",
        "highly textured cumulonimbus cloud showing nested turbulent puffs, raking sunlight",
        "intricate cloud face with multi-scale convective detail, sharp contrast, late afternoon",
    ),
    "cloud_multifractional": _p(
        "mixed-texture cloud showing both smooth and turbulent regions in the same frame, dramatic sky",
        "complex sky with regions of soft haze and regions of structured puffs, golden hour",
        "varied cloud texture across the frame, smooth wisps blending into puffy detail, soft daylight",
        "transitional cloud zone where stratus blends into cumulus, mixed character, overcast",
        "complex multi-character cloud composition, smooth and rough regions side by side, daylight",
    ),
    "warped_fbm": _p(
        "soft smooth gradient cloud sheet across the sky, gentle undulations, overcast daylight",
        "uniform overcast cloud cover with very subtle smooth variations, diffuse soft light",
        "gentle cloud gradient against soft blue sky, no sharp features, even daylight",
        "smooth cloud canvas with very low-frequency undulations, overcast sky",
        "uniform stratus layer with extremely soft tonal variations, diffuse daylight",
    ),
    "ridged": _p(
        "ridged altocumulus cloud bands stretching across the sky, parallel rolls, sunrise light",
        "regular cloud street pattern, parallel cloud rolls along wind axis, daylight",
        "lines of cumulus clouds aligned in parallel cloud streets, aerial view from above",
        "wave-like ridged altocumulus, alternating cloud and clear-sky bands, late afternoon",
        "horizontal cloud ridges with sharp parallel structure, dramatic dawn light",
    ),

    # ---------- FLUID ----------
    "curl_weave": _p(
        "intricate turbulent water surface with interwoven curl patterns, top-down view",
        "swirling oil-on-water surface with woven curls of color, macro photography",
        "marbled paper texture with intertwined curling streaks of pigment, top-down studio light",
        "ferrofluid surface with woven curl spikes, dramatic side light, lab photography",
        "magnetic fluid showing woven curl patterns under uniform field, top-down macro",
    ),
    "choppy": _p(
        "choppy ocean surface with countless small wavelets catching sunlight, aerial close view",
        "wind-rippled lake surface with sharp small waves, dawn light",
        "rough sea surface texture with chaotic small whitecaps, daytime aerial photo",
        "choppy harbor water with overlapping small waves, raking afternoon sun",
        "wind-blown river surface with chaotic ripples and tiny crests, top-down macro",
    ),
    "eddies": _p(
        "turbulent river flow with visible eddies and vortices, top-down aerial view, daylight",
        "vortex street downstream of a bridge pier, visible swirls on water surface, aerial",
        "tidal channel with chaotic eddies and counter-currents, top-down photography, soft daylight",
        "stream flowing around a rock with visible vortex shedding, top-down macro",
        "turbulent wake behind a moving boat, swirling eddies on water surface, aerial",
    ),
    "vorticity": _p(
        "smoke or dye injected into a turbulent flow showing vortex structures, lab photography",
        "ink in water revealing vortex tube structures, sharp detail, lab macro",
        "dye-visualized turbulent flow showing rotating vortex cores, fluid dynamics laboratory",
        "Schlieren image of turbulent vortex structures in fluid, scientific photography",
        "stretched vortex structures in laminar-turbulent transition flow, lab macro",
    ),
    "dye_diffusion": _p(
        "single drop of red ink diffusing into still water, tendrils spreading outward, top-down macro",
        "blue dye spreading through clear water, fine tendrils branching out, lab photography",
        "two colors of dye merging in water with sharp diffusion fronts, top-down macro",
        "ink drop slowly dissolving into milk, hair-thin diffusion fronts, dramatic light",
        "fluorescein dye spreading through water, glowing tendrils branching outward, lab light",
    ),
    "rheoscopic": _p(
        "rheoscopic fluid in a sealed jar showing streaks of suspended platelets in flow, side light",
        "Kalliroscope fluid revealing flow lines via reflective platelets, lab photography",
        "rheoscopic fluid with bright streaky highlights tracing the velocity field, side raking light",
        "mica-platelet suspension showing flow visualization streaks under rotation, lab macro",
        "rheoscopic fluid in laminar shear flow, parallel highlight streaks, scientific photography",
    ),

    # ---------- BARK ----------
    "burled_oak": _p(
        "burled oak bark surface with dense overlapping knots and rough grain, close-up forest photography",
        "old oak tree trunk with intricate burl patterns and dense knot clusters, natural daylight",
        "burled walnut wood surface closeup, dense knot clusters in flowing grain, gallery light",
        "tree burl growth on an old oak, intricate knot field on bark surface, forest light",
        "burled oak veneer with dense knot clusters and contorted grain, woodworking studio photography",
    ),
    "riven_oak": _p(
        "riven oak board surface with prominent knots and split grain, natural studio light",
        "rough-sawn oak plank with large dark knots and radial cracks from each knot, raking light",
        "split oak log surface showing huge prominent knots and grain cracks radiating outward",
        "weathered oak fence rail with deep grain cracks emanating from old knots, raking sunlight",
        "old oak floorboard showing wide grain bending around large dark knots, gallery photography",
    ),

    # ---------- SMOKE ----------
    "chimney_plume": _p(
        "narrow plume of dark smoke rising from a chimney against a dim sky, dramatic backlight",
        "thin smoke column rising straight up from a small fire, dusk sky, atmospheric photography",
        "rising column of campfire smoke, narrow plume, evening forest backlight",
        "industrial smokestack plume rising vertically, narrow column on overcast sky",
        "single rising smoke plume from a chimney, narrow opaque column, dawn light",
    ),
    "billow_smoke": _p(
        "thick billowing smoke filling the frame, dramatic backlighting, volumetric haze",
        "broad smoke cloud rising from a fire, dense billowing mass, late afternoon backlight",
        "huge billowing smoke cloud rolling upward, dense opaque smoke, dramatic backlight",
        "dense black smoke billows from a burning building, broad rising cloud, raking sun",
        "wildfire smoke plume filling the sky, broad dense billows, golden hour backlight",
    ),

    # ---------- FIRE ----------
    "firestorm": _p(
        "raging bonfire flames filling the frame, chaotic swirling fire tongues, dramatic night light",
        "intense house fire with chaotic vortices of flame, night photography",
        "swirling wildfire flames seen up close, twisted fire tongues, dramatic night light",
        "industrial flare stack at night, chaotic flame vortices roaring upward",
        "controlled burn with raging flame vortices and turbulent fire structure, dramatic night",
    ),
    "lava_pool": _p(
        "molten lava lake surface seen from above, glowing orange-red cells separated by dark cooled crust",
        "active lava pool with cellular brightness pattern, dark crust between bright pools, volcanic crater",
        "Hawaiian basalt lava lake closeup, glowing molten cells with dark cooled boundaries, dusk light",
        "Kilauea-like lava pond surface, cellular flow with bright cores and dark rims, aerial macro",
        "viscous lava lake showing tessellated cells of glow and crust, volcanic photography",
    ),
    "volcanic_fissure": _p(
        "volcanic fissure with meandering vertical cracks pouring glowing fire upward, night photography",
        "Icelandic fissure eruption with narrow vertical fire vents through dark lava field, dramatic night",
        "volcanic rift with multiple thin vertical lava fountains in the dark, night light",
        "small fissure vent with a single tall vertical lava spray, night photography, dramatic",
        "active rift eruption with multiple meandering fire cracks rising from dark rock, night",
    ),
}

ALL_FAMILIES = list(PROMPTS.keys())

if __name__ == "__main__":
    n_total = sum(len(v) for v in PROMPTS.values())
    print(f"{len(PROMPTS)} families, {n_total} prompts total "
          f"({n_total // len(PROMPTS)} per family)")
    for fam, pp in PROMPTS.items():
        print(f"  {fam:24s} {len(pp)} prompts")
