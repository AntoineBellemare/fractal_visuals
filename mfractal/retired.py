"""
retired.py — the single source of truth for generator families that no longer exist.

WHY THIS EXISTS. Families get pruned (see commit af17c50, which dropped four generators that
were pixel-identical to a family already in the catalogue). The knowledge of *what* was dropped
and *what replaced it* was recorded in `benchmarks/make_figures.py` — a figures script — so
nothing else in the repo could consult it. The result: `datasets/build_pareidolia.py` still lists
three retired families, and `mf.generate` fails on them with a bare

    AttributeError: module 'mfractal.textures' has no attribute 'choppy'

thrown from deep inside a generation loop, hours in. 17 of the 106 stimuli in
`datasets/pareidolia/` were built from those names.

The fix is not to resurrect the duplicates — the prune was correct, they were riffs on a family
that is still here. It is to make retirement a *first-class, queryable* fact of the package, so
that a stale family list fails immediately with a message that says what to use instead.

Two kinds of retirement, and they are not the same thing:
  * ALIASED   — the family was a near-duplicate of one that remains. There IS a replacement, and
                asking for the old name can be transparently redirected.
  * WITHDRAWN — the family behaved badly (estimator artifacts, inverted or non-monotone response
                to the complexity knob). There is NO replacement and a redirect would be a lie.
"""
from __future__ import annotations

# name -> (replacement or None, reason)
RETIRED: dict[str, tuple[str | None, str]] = {
    # --- ALIASED: pixel-identical to a family that is still in the catalogue (af17c50) ---
    "stratified":     ("cascade_lognormal", "pixel-identical at matched seed (r > 0.95)"),
    "choppy":         ("cascade_lognormal", "pixel-identical at matched seed (r 0.986)"),
    "cloud_mrw":      ("prescribed_cascade", "pixel-identical at matched seed (r 0.964)"),
    "levy_marble":    ("universal_cascade",  "pixel-identical at matched seed (r 0.995)"),
    # --- WITHDRAWN: no replacement; the generator's behaviour was the problem ---
    "plume":          (None, "estimator artifact on sparse bright structure (c2 ~ -8)"),
    "cellular_stone": (None, "inverted rho (more cells = more uniform fragmentation)"),
    "concentric":     (None, "inverted rho (more rings = more uniform spacing)"),
    "breccia":        (None, "inverted rho (more fragments = more uniform tiling)"),
    "veined":         (None, "high variance per seed, weak monotonicity"),
}


class RetiredFamilyError(KeyError):
    """Raised when a retired family is requested and no redirect was permitted."""


def is_retired(family: str) -> bool:
    return family in RETIRED


def replacement_for(family: str) -> str | None:
    """The canonical family to use instead, or None if the family was withdrawn outright."""
    return RETIRED.get(family, (None, ""))[0]


def resolve_family(family: str, *, allow_retired: bool = False) -> str:
    """Canonical name for `family`.

    Live families pass through untouched. A retired ALIASED family either redirects to its
    replacement (allow_retired=True) or raises. A retired WITHDRAWN family always raises, because
    silently substituting a different texture would corrupt whatever is being measured.
    """
    if family not in RETIRED:
        return family
    repl, why = RETIRED[family]
    if repl is None:
        raise RetiredFamilyError(
            f"'{family}' was withdrawn and has no replacement ({why}). "
            f"Remove it from the family list."
        )
    if allow_retired:
        return repl
    raise RetiredFamilyError(
        f"'{family}' was retired ({why}); use '{repl}' instead, or pass allow_retired=True "
        f"to redirect automatically."
    )


def validate_families(families, *, allow_retired: bool = False) -> list[str]:
    """Check a whole family list UP FRONT and resolve it.

    Call this when a script starts rather than discovering a stale name several hundred
    generations into a run. Reports every bad name at once instead of one per crash.
    """
    out, problems = [], []
    for f in families:
        try:
            out.append(resolve_family(f, allow_retired=allow_retired))
        except RetiredFamilyError as e:
            problems.append(str(e))
    if problems:
        raise RetiredFamilyError(
            f"{len(problems)} retired famil{'y' if len(problems) == 1 else 'ies'} in this list:\n  "
            + "\n  ".join(problems))
    return out


# Backwards compatibility: benchmarks/make_figures.py used to own this mapping.
DROPPED = {k: (f"{v[1]}; use '{v[0]}'" if v[0] else v[1]) for k, v in RETIRED.items()}
