"""
Wallpaper group policies for texture alignment evaluation.

Each policy encodes the orientation compatibility rules and lattice geometry
for one wallpaper group.  The evaluator calls `policy.seam_compatible()` to
decide whether two adjacent patches can be seam-aligned, and
`policy.lattice_directions()` to build the TextureLattice with the correct
axis orientations.

Adding a new group: subclass WallpaperPolicy, add an entry to POLICIES.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class WallpaperPolicy(ABC):
    """Orientation compatibility rules and lattice geometry for a wallpaper group."""

    @abstractmethod
    def seam_compatible(self, rho_i: int, rho_j: int) -> bool:
        """
        Return True if grain rotations rho_i and rho_j (each in 0..3,
        i.e. multiples of 90°) can produce a valid seam alignment for
        this wallpaper group.  When False the evaluator adds a hard
        penalty and skips the phase-mismatch computation.
        """
        ...

    def lattice_directions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (u_dir, v_dir) — the two axis directions of the texture
        lattice as unit vectors.  The default is the standard orthogonal
        horizontal/vertical basis.  Override for rotated or oblique lattices.
        """
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])


class StripesPolicy(WallpaperPolicy):
    """
    PM — horizontal stripes.

    A 90° rotation turns horizontal stripes into vertical ones, making
    any seam between a 0°/180° patch and a 90°/270° patch impossible to
    align.  Only same-parity orientations are compatible.
    """
    def seam_compatible(self, rho_i: int, rho_j: int) -> bool:
        return (rho_i % 2) == (rho_j % 2)


class DiagonalStripesPolicy(StripesPolicy):
    """
    PM — diagonal stripes at 45°.

    Same orientation compatibility as horizontal stripes (rotating 90°
    mirrors the diagonal, so same-parity rule holds).  The only difference
    is the lattice is rotated 45°: u runs along [1,1]/√2 and v along
    [-1,1]/√2.
    """
    def lattice_directions(self) -> tuple[np.ndarray, np.ndarray]:
        s = 1.0 / np.sqrt(2.0)
        return np.array([s, s]), np.array([-s, s])


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
    "stripes":          StripesPolicy(),
    "diagonal_stripes": DiagonalStripesPolicy(),
    "grid":             GridPolicy(),
    "p4":               GridPolicy(),   # 4-fold rotation, no mirrors — same seam math as grid
    "p4m":              GridPolicy(),   # 4-fold + mirrors (polka dots) — same seam math as grid
}


def get_policy(wallpaper_group: str) -> WallpaperPolicy:
    if wallpaper_group not in POLICIES:
        raise ValueError(
            f"Unknown wallpaper group: {wallpaper_group!r}. "
            f"Valid options: {list(POLICIES)}"
        )
    return POLICIES[wallpaper_group]
