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
    
    # List of placed items: (ItemObject, Position_X, Position_Y, RotatedPolygon)
    placed_items: List[Tuple[NestingItem, float, float, Polygon]] = field(default_factory=list)
    
    # Spatial Index for fast intersection checks
    _tree: Optional[STRtree] = None
    _geometries: List[Polygon] = field(default_factory=list)
    
    def place(self, item: NestingItem, x: float, y: float):
        """Commit an item to the fabric."""
        # 1. Create the final placed polygon in global coords
        placed_poly = item.place_at(x, y)
        
        # 2. Store
        self.placed_items.append((item, x, y, placed_poly))
        self._geometries.append(placed_poly)
        
        # 3. Rebuild Tree (Naive approach; incremental is harder in Python)
        # For <50 items, rebuilding is instantaneous.
        self._tree = STRtree(self._geometries)

    def is_overlapping(self, candidate_poly: Polygon) -> bool:
        """
        Checks if candidate_poly intersects ANY placed item.
        Returns True if collision detected.
        """
        if not self._geometries:
            return False
            
        # 1. Check Fabric Width Limits
        minx, miny, maxx, maxy = candidate_poly.bounds
        if minx < 0 or maxx > self.width:
            return True
        if miny < 0: # Fabric starts at Y=0
            return True
            
        # 2. Query R-Tree for potential overlaps (Bounding Box check)
        # query() returns indices of geometries that *might* intersect
        candidate_indices = self._tree.query(candidate_poly)
        
        # 3. Precise Polygon Intersection check
        for idx in candidate_indices:
            existing_poly = self._geometries[idx]
            if candidate_poly.intersects(existing_poly):
                return True
                
        return False

    @property
    def total_height(self) -> float:
        """The total length of fabric used (Max Y)."""
        if not self._geometries:
            return 0.0
        # Check bounds of all items
        max_y = 0.0
        for _, _, _, poly in self.placed_items:
            max_y = max(max_y, poly.bounds[3])
        return max_y

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
        self.texture = texture_spec

    def nest(self, items: List[NestingItem]) -> FabricState:
        
        fabric = FabricState(self.width)
        heuristic = BottomLeftHeuristic()
        
        # Define Texture Periods (Grid Spacing)
        # Default to small value (1mm) if no texture is provided to simulate "continuous" placement
        if self.texture:
            tx = self.texture.period_x
            ty = self.texture.period_y
        else:
            tx, ty = 1.0, 1.0 

        # Sort items by Area (Largest First)
        sorted_items = sorted(items, key=lambda x: x.area, reverse=True)
        
        for item in sorted_items:
            # 1. Rotate (Fixed to 0 for now)
            item.set_rotation(0) 
            
            # 2. Get Candidates (Proposed for Bottom-Left Corner)
            raw_candidates = heuristic.propose_positions(item.shape, fabric)
            
            best_pos = None # Stores (centroid_x, centroid_y)
            
            # Pre-calculate offset to save time
            minx_offset, miny_offset = item.bottom_left_offset
            
            # 3. Evaluate Candidates
            for (corner_x, corner_y) in raw_candidates:
                
                # A. Convert "Corner Proposal" -> "Centroid Proposal"
                # corner_x is where the Heuristic wants the bottom-left to be.
                # centroid_x is where the item center must be to achieve that.
                # Formula: Center = Corner - Offset (because Offset = Corner - Center)
                # Wait, Offset = minx (negative number).
                # So Corner = Center + minx  =>  Center = Corner - minx
                centroid_x = corner_x - minx_offset
                centroid_y = corner_y - miny_offset
                
                # B. Snap CENTROID to Texture Grid
                # We align the "anchor" (center) to the lattice
                snapped_cx = round(centroid_x / tx) * tx
                snapped_cy = round(centroid_y / ty) * ty
                
                # C. Generate Test Polygon at this snapped centroid location
                test_poly = item.place_at(snapped_cx, snapped_cy)
                
                # D. Check Bounds (Critical!)
                # Now that we snapped, did we accidentally push the bottom below zero?
                poly_minx, poly_miny, poly_maxx, poly_maxy = test_poly.bounds
                
                if poly_minx < 0 or poly_maxx > self.width:
                    continue # Out of width bounds
                if poly_miny < 0:
                    continue # Out of height bounds (Below Zero check)

                # E. Check Collision
                if not fabric.is_overlapping(test_poly):
                    # Found a valid spot!
                    best_pos = (snapped_cx, snapped_cy)
                    break 
            
            # 4. Commit
            if best_pos:
                fabric.place(item, best_pos[0], best_pos[1])
            else:
                # Fallback: Place high above everything
                # Align safe Y to grid as well
                current_max_y = fabric.total_height
                safe_y = round((current_max_y - miny_offset + 10.0) / ty) * ty
                
                # We also need a safe X (start at 0 + offset)
                safe_x = round((0.0 - minx_offset) / tx) * tx
                
                fabric.place(item, safe_x, safe_y)
                
        return fabric
