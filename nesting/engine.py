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
    # item, x, y (centroid), poly
    placed_items: List[Tuple[any, float, float, Polygon]] = field(default_factory=list)
    _geometries: List[Polygon] = field(default_factory=list)
    _tree: Optional[STRtree] = None

    def place(self, item, x: float, y: float):
        """Places item such that its LOCAL ORIGIN is at (x, y)."""
        placed_poly = item.place_at(x, y)
        self.placed_items.append((item, x, y, placed_poly))
        self._geometries.append(placed_poly)
        self._tree = STRtree(self._geometries)

    def is_overlapping(self, candidate_poly: Polygon) -> bool:
        minx, miny, maxx, maxy = candidate_poly.bounds
        # Strict Fabric Boundary Checks
        if minx < -1e-5 or maxx > self.width + 1e-5: return True
        if miny < -1e-5: return True
        
        if not self._geometries: return False
        indices = self._tree.query(candidate_poly)
        for idx in indices:
            if candidate_poly.intersects(self._geometries[idx]):
                return True
        return False

    @property
    def total_height(self) -> float:
        return max([p.bounds[3] for p in self._geometries]) if self._geometries else 0.0

# ==============================================================================
# 2. THE STRATEGIST (Heuristic)
# ==============================================================================

class BottomLeftHeuristic:
    """
    Standard 'Bottom-Left' Strategy.
    Proposes positions by trying to nudge the new item against the corners 
    of existing items.
    """
    def propose_positions(self, 
                          item_poly: Polygon, 
                          fabric: FabricState) -> List[Tuple[float, float]]:
        """
        Returns a list of (x, y) candidates.
        """
        candidates = []
        
        # 1. Always propose (0,0) - The bottom-left corner of the roll
        candidates.append((0.0, 0.0))
        
        if not fabric.placed_items:
            return candidates
            
        # 2. Generate "Corner Points"
        # For every placed item P, and every corner of P,
        # we try to place the new item C such that one of C's corners touches P's corner.
        # This is a simplified "NFP-like" approach.
        
        item_bounds = item_poly.bounds # (minx, miny, maxx, maxy)
        item_w = item_bounds[2] - item_bounds[0]
        item_h = item_bounds[3] - item_bounds[1]
        
        # Optimization: Only look at the bounding box corners of placed items
        # A full vertex-to-vertex check is O(V^2), this is O(N).
        for _, px, py, p_poly in fabric.placed_items:
            p_minx, p_miny, p_maxx, p_maxy = p_poly.bounds
            
            # Candidate: To the Right of P
            candidates.append((p_maxx, p_miny))
            # Candidate: On Top of P
            candidates.append((p_minx, p_maxy))
            # Candidate: Top-Right corner logic
            candidates.append((p_maxx, p_miny - item_h)) # Align tops?
            
        # Sort candidates by Y (primary) and X (secondary) to prefer Bottom-Left
        # This sort order is what makes it "Bottom-Left"
        candidates.sort(key=lambda pos: (pos[1], pos[0]))
        
        return candidates

# ==============================================================================
# 3. THE CONTROLLER (Engine)
# ==============================================================================

class NestingEngine:
    def __init__(self, fabric_width: float, texture_spec):
        self.width = fabric_width
        self.texture = texture_spec # e.g., TextureSpec(period_x=10, period_y=50)

    def nest(self, items: List) -> FabricState:
        fabric = FabricState(self.width)
        # Sort Largest Area First
        sorted_items = sorted(items, key=lambda x: x.area, reverse=True)

        for item in sorted_items:
            placed = False
            
            # 1. Get current search candidates (proposing Bottom-Left of AABB)
            # We use a simple Skyline/Bottom-Left hybrid
            candidates = self._get_candidates(fabric)
            
            for (cx, cy) in candidates:
                # 2. COORDINATE CONVERSION
                # cx, cy is the target for the item's MIN_X, MIN_Y.
                # We need to find where the CENTROID (0,0) should go.
                # offset = (0,0) - (min_x, min_y)
                off_x, off_y = item.bottom_left_offset
                target_centroid_x = cx - off_x
                target_centroid_y = cy - off_y

                # 3. TEXTURE SNAP
                # Snap an anchor point to the lattice defined in TextureSpec.
                # If seam_anchor_local is provided (from exported seamlines), we snap that.
                # Otherwise, we fall back to snapping the item's local origin.
                tx = self.texture.period_x
                ty = self.texture.period_y
                
                if getattr(item, 'seam_anchor_local', None) is not None:
                    ax_local, ay_local = float(item.seam_anchor_local[0]), float(item.seam_anchor_local[1])
                    anchor_x = target_centroid_x + ax_local
                    anchor_y = target_centroid_y + ay_local

                    snapped_anchor_x = round(anchor_x / tx) * tx
                    snapped_anchor_y = round(anchor_y / ty) * ty

                    snapped_x = snapped_anchor_x - ax_local
                    snapped_y = snapped_anchor_y - ay_local
                else:
                    # NEW: phase offset shifts the snapping lattice for this item
                    ox, oy = getattr(item, "phase_offset", (0.0, 0.0))

                    snapped_x = round((target_centroid_x - ox) / tx) * tx + ox
                    snapped_y = round((target_centroid_y - oy) / ty) * ty + oy
                
                # 4. VALIDATION
                test_poly = item.place_at(snapped_x, snapped_y)
                if not fabric.is_overlapping(test_poly):
                    fabric.place(item, snapped_x, snapped_y)
                    placed = True
                    break
            
            if not placed:
                # Fallback: Just push it above the current skyline
                off_x, off_y = item.bottom_left_offset
                safe_y = fabric.total_height - off_y + 10.0
                fabric.place(item, -off_x, safe_y)

        return fabric

    def _get_candidates(self, fabric) -> List[Tuple[float, float]]:
        """Proposes Bottom-Left corner locations."""
        pts = [(0.0, 0.0)]
        for _, _, _, poly in fabric.placed_items:
            minx, miny, maxx, maxy = poly.bounds
            pts.append((maxx, miny)) # To the right
            pts.append((minx, maxy)) # On top
        return sorted(pts, key=lambda p: (p[1], p[0]))
