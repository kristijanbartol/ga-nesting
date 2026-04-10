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

    def glide_transforms(self) -> list:
        """
        Return a list of callables (phi: (N,2) → phi': (N,2)) representing
        glide-reflection symmetries in fractional phase coordinates.
        The default (no glide symmetries) returns an empty list.
        Override for PG, PMG, PGG and related groups.
        """
        return []

    def phase_axes(self) -> tuple[bool, bool]:
        """
        Return (use_u, use_v) — which lattice axes affect visual alignment.

        For stripe-based patterns the texture depends only on the V lattice
        direction, so U-phase mismatch is irrelevant and should be excluded
        from the metric and the Stage2 solver.  Grid/rotational patterns
        depend on both axes.
        """
        return (True, True)


class StripesPolicy(WallpaperPolicy):
    """
    PM — horizontal stripes.

    A 90° rotation turns horizontal stripes into vertical ones, making
    any seam between a 0°/180° patch and a 90°/270° patch impossible to
    align.  Only same-parity orientations are compatible.
    """
    def seam_compatible(self, rho_i: int, rho_j: int) -> bool:
        return (rho_i % 2) == (rho_j % 2)

    def phase_axes(self) -> tuple[bool, bool]:
        return (False, True)  # stripes depend only on V


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


class HerringbonePolicy(StripesPolicy):
    """
    PG — herringbone (glide reflection only, no mirror axes).

    Seam compatibility is the same as stripes (same-parity rho), but adjacent
    patches may also align via the glide reflection: reflect u and shift v by
    half a period.  The evaluator uses glide_transforms() to check this.
    """
    def glide_transforms(self):
        return [lambda p: np.stack([1.0 - p[:, 0], (p[:, 1] + 0.5) % 1.0], axis=1)]


class ChevronPolicy(StripesPolicy):
    """
    PMG — chevron (one mirror axis + one glide reflection axis).

    Like stripes for orientation compatibility.  Two extra symmetries:
    a pure mirror (flip v) and a glide (shift u by half + flip v).
    """
    def glide_transforms(self):
        return [
            lambda p: np.stack([p[:, 0], 1.0 - p[:, 1]], axis=1),                            # mirror v
            lambda p: np.stack([(p[:, 0] + 0.5) % 1.0, 1.0 - p[:, 1]], axis=1),             # glide: shift u + mirror v
        ]


class BrickPolicy(StripesPolicy):
    """
    PGG — brick bond (two perpendicular glide reflection axes, no mirror axes).

    Like stripes for orientation compatibility.  Two glide symmetries:
    flip u + shift v by half, and shift u by half + flip v.
    """
    def glide_transforms(self):
        return [
            lambda p: np.stack([1.0 - p[:, 0], (p[:, 1] + 0.5) % 1.0], axis=1),             # glide 1: flip u + shift v
            lambda p: np.stack([(p[:, 0] + 0.5) % 1.0, 1.0 - p[:, 1]], axis=1),             # glide 2: shift u + flip v
        ]


POLICIES: dict[str, WallpaperPolicy] = {
    "stripes":          StripesPolicy(),
    "diagonal_stripes": DiagonalStripesPolicy(),
    "grid":             GridPolicy(),
    "p4":               GridPolicy(),        # 4-fold rotation, no mirrors — same seam math as grid
    "p4m":              GridPolicy(),        # 4-fold + mirrors (polka dots) — same seam math as grid
    "pg":               HerringbonePolicy(), # glide reflection only — herringbone
    "pmg":              ChevronPolicy(),     # mirror + glide — chevron
    "pgg":              BrickPolicy(),       # two glide axes — brick bond
}


def get_policy(wallpaper_group: str) -> WallpaperPolicy:
    if wallpaper_group not in POLICIES:
        raise ValueError(
            f"Unknown wallpaper group: {wallpaper_group!r}. "
            f"Valid options: {list(POLICIES)}"
        )
    return POLICIES[wallpaper_group]
