import math
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from .item import NestingItem

# ==============================================================================
# 1. THE FABRIC ROLL (State)
# ==============================================================================

@dataclass
class FabricState:
    width: float
    # item, centroid_x, centroid_y, placed_poly
    placed_items: List[Tuple[any, float, float, Polygon]] = field(default_factory=list)
    _geometries: List[Polygon] = field(default_factory=list)
    _tree: Optional[STRtree] = None

    def place(self, item, x: float, y: float):
        """Places item such that its LOCAL ORIGIN (centroid) is at (x, y)."""
        placed_poly = item.place_at(x, y)
        self.placed_items.append((item, x, y, placed_poly))
        self._geometries.append(placed_poly)
        self._tree = STRtree(self._geometries)

    def is_overlapping(self, candidate_poly: Polygon) -> bool:
        minx, miny, maxx, maxy = candidate_poly.bounds
        if minx < -1e-5 or maxx > self.width + 1e-5:
            return True
        if miny < -1e-5:
            return True
        if not self._geometries:
            return False
        indices = self._tree.query(candidate_poly)
        for idx in indices:
            if candidate_poly.intersects(self._geometries[idx]):
                return True
        return False

    @property
    def total_height(self) -> float:
        return max(p.bounds[3] for p in self._geometries) if self._geometries else 0.0


# ==============================================================================
# 2. THE ENGINE
# ==============================================================================

class NestingEngine:
    def __init__(self, fabric_width: float, texture_spec):
        self.width = fabric_width
        self.texture = texture_spec

    def nest(
        self,
        items: List,
        permutation: Optional[List[int]] = None,
        rotations: Optional[List[int]] = None,
        heuristic: Optional[object] = None,
    ) -> FabricState:
        fabric = FabricState(self.width)

        # --- Ordering (pi) ---
        if permutation is None:
            sorted_items = sorted(items, key=lambda x: x.area, reverse=True)
        else:
            sorted_items = []
            seen = set()
            for idx in permutation:
                idx = int(idx)
                if 0 <= idx < len(items) and idx not in seen:
                    sorted_items.append(items[idx])
                    seen.add(idx)
            for i, it in enumerate(items):
                if i not in seen:
                    sorted_items.append(it)

        # --- Per-item grain rotations (rho) ---
        if rotations is not None:
            for i, it in enumerate(items):
                if i < len(rotations):
                    it.set_rotation(float(int(rotations[i]) % 4 * 90))

        h = int(heuristic or 0)

        for item in sorted_items:
            placed = False

            for (cx, cy) in self._skyline_candidates(fabric, item, h):
                if not fabric.is_overlapping(item.place_at(cx, cy)):
                    fabric.place(item, cx, cy)
                    placed = True
                    break

            if not placed:
                # Fallback: push above the current skyline, snapped to lattice.
                tx = self.texture.period_x
                ty = self.texture.period_y
                ox, oy = getattr(item, "phase_offset", (0.0, 0.0))
                i_minx, i_miny, _, _ = item.shape.bounds

                safe_cy = oy + math.ceil(
                    (fabric.total_height - i_miny - oy) / ty
                ) * ty
                safe_cx = ox + math.ceil((-i_minx - ox) / tx) * tx
                fabric.place(item, safe_cx, safe_cy)

        return fabric

    def _skyline_candidates(
        self, fabric: FabricState, item: NestingItem, h: int = 0
    ) -> List[Tuple[float, float]]:
        """
        Skyline Bottom-Left with full texture-lattice sweep.

        For every valid lattice x-column (one per texture period across the
        fabric width), compute the minimum lattice y that clears:
          - the fabric floor  (cy + item_miny >= 0)
          - every placed item whose AABB overlaps in x  (AABB skyline)
        then snap cy up to the nearest lattice y-position.

        This produces O(fabric_width / period_x) candidates — ~30 for typical
        settings — each grounded in the actual skyline, so the first valid
        candidate is usually the tightest available placement.

        h controls candidate ordering (tie-breaking):
          0 – (top_edge, cx)  : globally lowest placement, then leftmost  [default]
          1 – (cx, top_edge)  : leftmost column first, then lowest within it
          2 – top_edge only   : purely greedy on minimising total height
        """
        tx = self.texture.period_x
        ty = self.texture.period_y
        ox, oy = getattr(item, "phase_offset", (0.0, 0.0))

        i_minx, i_miny, i_maxx, i_maxy = item.shape.bounds

        # ── Valid centroid x-range ──────────────────────────────────────────
        # cx + i_minx >= 0   →   cx >= -i_minx
        # cx + i_maxx <= W   →   cx <= W - i_maxx
        cx_lo = -i_minx
        cx_hi = self.width - i_maxx

        k_lo = math.ceil((cx_lo - ox) / tx)
        k_hi = math.floor((cx_hi - ox) / tx)

        if k_hi < k_lo:
            # Item wider than the fabric — single best-effort column.
            k_lo = k_hi = round((cx_lo - ox) / tx)

        # ── One candidate per x-column ──────────────────────────────────────
        candidates: List[Tuple[float, float]] = []

        for k in range(k_lo, k_hi + 1):
            cx = ox + k * tx

            # Fabric-floor constraint: cy + i_miny >= 0
            cy_min = -i_miny

            # AABB skyline: raise cy_min to clear every placed item that
            # overlaps in x with this item at column cx.
            for _, _, _, placed_poly in fabric.placed_items:
                p_minx, _p_miny, p_maxx, p_maxy = placed_poly.bounds
                x_overlap = (
                    cx + i_maxx > p_minx + 1e-6 and
                    cx + i_minx < p_maxx - 1e-6
                )
                if x_overlap:
                    # New item bottom (cy + i_miny) must reach or exceed p_maxy.
                    cy_min = max(cy_min, p_maxy - i_miny)

            # Snap cy_min up to the nearest lattice y-position.
            k_y = math.ceil((cy_min - oy) / ty)
            cy = oy + k_y * ty

            candidates.append((cx, cy))

        # ── Sort by chosen tie-breaking criterion ───────────────────────────
        mode = h % 3
        if mode == 0:
            candidates.sort(key=lambda p: (p[1] + i_maxy, p[0]))
        elif mode == 1:
            candidates.sort(key=lambda p: (p[0], p[1] + i_maxy))
        else:
            candidates.sort(key=lambda p: p[1] + i_maxy)

        return candidates
