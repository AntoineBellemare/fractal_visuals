"""
mfractal: generate and quantify multifractal / multifractional 2D fields
for pareidolia stimulus research.

Two flagships (spectral, full-brightness):
    from mfractal import multifractal_cloud, multifractional, punch
Three process-based organic generators:
    from mfractal import dla, stir_and_fold, branching_network
Natural-surface textures for pareidolia (rock / marble / agate / cracked stone):
    from mfractal import rock_texture, marble, agate, weathered, granite, cellular_stone
"""
from . import generators, quantify, descriptors, stimuli, flagships, organic, textures, fluids, dataset, clouds, bark, fire, metal, modulations, levy, wavelet_cascade, attractors, interference, cloudscape as _cloudscape_mod
from .flagships import (multifractal_cloud, multifractional, punch,
                        complexity, sample_battery)
from .organic import dla, stir_and_fold, branching_network
from .textures import (rock_texture, marble, agate, weathered, granite,
                       cellular_stone, colorize, list_palettes, PALETTES,
                       colorize_natural, NATURAL_PALETTES,
                       colorize_for_family, FAMILY_PALETTES,
                       strata, gneiss, concentric, breccia, porphyry,
                       dendritic, vesicular, stylolite, mylonite, stromatolite,
                       boudinage, liesegang, convoluted, turbulent, flow_banded, schist,
                       serpentinite, veined, complexity_range,
                       flow_banded, schist, chert, serpentinite, veined,
                       ROCK_FAMILIES, COMPLEX_FAMILIES)
from .fluids import (eddies, vorticity, plume, curl_weave, dye_diffusion,
                     rheoscopic, FLUID_FAMILIES, CX_FLUIDS)
from .cloudscape import cloudscape, SKY_PALETTES
from .clouds import (cascade_lognormal, billow, cirrus, ridged,
                     warped_fbm, cloud_multifractional, billow_smoke,
                     CLOUD_FAMILIES)
from .bark import burled_oak, riven_oak, BARK_FAMILIES
from .fire import firestorm, lava_pool, volcanic_fissure, FIRE_FAMILIES
from .metal import rust_bloom, rust_streaks, rust_pitted, METAL_FAMILIES
from .modulations import symmetrize
from .levy import universal_cascade, LEVY_FAMILIES
from .wavelet_cascade import prescribed_cascade, WAVELET_FAMILIES
from .attractors import de_jong_attractor, ATTRACTOR_FAMILIES
from .interference import wave_interference, INTERFERENCE_FAMILIES
from .dataset import build_dataset, validate_dataset, generate
from .quantify import wavelet_leaders_2d
from .quantify import (mfdfa_2d, moments_2d, legendre, spectrum_summary,
                       box_count_support, local_fd_map, heterogeneity)
from .descriptors import (directional_hurst, anisotropy_index, subband_anisotropy,
                          autocorr_radial, coherence_length, integral_length,
                          local_c2_map, heterogeneity_full, feature_vector)

__all__ = [
    "generators", "quantify", "stimuli", "flagships", "organic", "textures", "fluids",
    "bark", "burled_oak", "riven_oak", "BARK_FAMILIES",
    "fire", "firestorm", "lava_pool", "volcanic_fissure", "FIRE_FAMILIES",
    "metal", "rust_bloom", "rust_streaks", "rust_pitted", "METAL_FAMILIES",
    "modulations", "symmetrize",
    "levy", "universal_cascade", "LEVY_FAMILIES",
    "wavelet_cascade", "prescribed_cascade", "WAVELET_FAMILIES",
    "attractors", "de_jong_attractor", "ATTRACTOR_FAMILIES",
    "interference", "wave_interference", "INTERFERENCE_FAMILIES",
    "eddies", "vorticity", "plume", "curl_weave", "dye_diffusion", "rheoscopic", "FLUID_FAMILIES", "CX_FLUIDS",
    "dataset", "build_dataset", "validate_dataset", "generate", "wavelet_leaders_2d",
    "clouds", "cloudscape", "SKY_PALETTES", "CLOUD_FAMILIES", "cascade_lognormal", "billow", "cirrus",
    "ridged", "warped_fbm", "cloud_multifractional", "billow_smoke",
    "multifractal_cloud", "multifractional", "punch", "complexity", "sample_battery",
    "dla", "stir_and_fold", "branching_network",
    "rock_texture", "marble", "agate", "weathered", "granite", "cellular_stone",
    "colorize", "list_palettes", "PALETTES", "colorize_natural", "NATURAL_PALETTES",
    "colorize_for_family", "FAMILY_PALETTES",
    "strata", "gneiss", "concentric", "breccia", "porphyry", "dendritic", "vesicular",
    "stylolite", "mylonite", "stromatolite", "boudinage", "liesegang",
    "convoluted", "turbulent", "flow_banded", "schist", "serpentinite", "veined",
    "complexity_range",
    "flow_banded", "schist", "chert", "serpentinite", "veined",
    "ROCK_FAMILIES", "COMPLEX_FAMILIES",
    "mfdfa_2d", "moments_2d", "legendre", "spectrum_summary",
    "box_count_support", "local_fd_map", "heterogeneity",
    "descriptors",
    "directional_hurst", "anisotropy_index", "subband_anisotropy",
    "autocorr_radial", "coherence_length", "integral_length",
    "local_c2_map", "heterogeneity_full", "feature_vector",
]
