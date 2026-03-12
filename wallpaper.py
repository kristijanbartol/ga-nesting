"""
Wallpaper group policies for texture alignment evaluation.

Each policy encodes the orientation compatibility rules for one wallpaper
group.  The evaluator calls `policy.seam_compatible(rho_i, rho_j)` to decide
whether two adjacent patches can be seam-aligned given their grain rotations,
instead of hardcoding stripe-specific parity checks.

Adding a new group: subclass WallpaperPolicy, add an entry to POLICIES.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class WallpaperPolicy(ABC):
    """Orientation compatibility rules for a wallpaper group."""

    @abstractmethod
    def seam_compatible(self, rho_i: int, rho_j: int) -> bool:
        """
        Return True if grain rotations rho_i and rho_j (each in 0..3,
        i.e. multiples of 90°) can produce a valid seam alignment for
        this wallpaper group.  When False the evaluator adds a hard
        penalty and skips the phase-mismatch computation.
        """
        ...


class StripesPolicy(WallpaperPolicy):
    """
    PM — horizontal (or vertical) stripes.

    A 90° rotation turns horizontal stripes into vertical ones, making
    any seam between a 0°/180° patch and a 90°/270° patch impossible to
    align.  Only same-parity orientations are compatible.
    """
    def seam_compatible(self, rho_i: int, rho_j: int) -> bool:
        return (rho_i % 2) == (rho_j % 2)


class GridPolicy(WallpaperPolicy):
    """
    PMM — grid / check / plaid patterns.

    The pattern has 4-fold rotational symmetry: rotating by any multiple
    of 90° produces an identical-looking texture.  All orientation pairs
    are therefore compatible at a seam.
    """
    def seam_compatible(self, rho_i: int, rho_j: int) -> bool:
        return True


POLICIES: dict[str, WallpaperPolicy] = {
    "stripes": StripesPolicy(),
    "grid":    GridPolicy(),
}


def get_policy(wallpaper_group: str) -> WallpaperPolicy:
    if wallpaper_group not in POLICIES:
        raise ValueError(
            f"Unknown wallpaper group: {wallpaper_group!r}. "
            f"Valid options: {list(POLICIES)}"
        )
    return POLICIES[wallpaper_group]
