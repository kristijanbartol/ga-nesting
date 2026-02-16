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
        
        # Sort items by Area (Largest First) - standard packing heuristic
        # In the full EA, this order comes from the 'pi' gene.
        sorted_items = sorted(items, key=lambda x: x.area, reverse=True)
        
        for item in sorted_items:
            # 1. Rotate (Fixed to 0 for now, EA would set this)
            item.set_rotation(0) 
            
            # 2. Get Candidates (Raw positions)
            # The heuristic proposes based on geometry only
            raw_candidates = heuristic.propose_positions(item.shape, fabric)
            
            best_pos = None
            
            # 3. Evaluate Candidates
            for (x, y) in raw_candidates:
                
                # --- TEXTURE SNAPPING LOGIC ---
                # This is the Core Novelty.
                # If we have a texture, we can't just place at (x,y).
                # We must shift (x,y) so the item's anchor aligns with the lattice.
                
                # Assume Texture Period = (Tx, Ty)
                # We want: (x + anchor_x) % Tx == 0
                # So: x_snapped = round_up_to_multiple(x + anchor_x, Tx) - anchor_x
                
                # Simplified for this test: Snap to nearest 10mm grid
                # In real code, use self.texture.period_x
                tx, ty = 10.0, 10.0 
                
                snapped_x = round(x / tx) * tx
                snapped_y = round(y / ty) * ty
                
                # Create test polygon at snapped position
                test_poly = item.place_at(snapped_x, snapped_y)
                
                # Check Collision
                if not fabric.is_overlapping(test_poly):
                    best_pos = (snapped_x, snapped_y)
                    break # Found the best spot (first valid in sorted list)
            
            # 4. Commit or Fail
            if best_pos:
                fabric.place(item, best_pos[0], best_pos[1])
                print(f"   Placed {item.name} at {best_pos}")
            else:
                # Fallback: If no candidate works (rare with infinite height),
                # Place it way above the highest item.
                # (Simple 'Skyline' fallback)
                safe_y = fabric.total_height + 1.0
                fabric.place(item, 0.0, safe_y)
                print(f"   Placed {item.name} at fallback (0, {safe_y})")
                
        return fabric
