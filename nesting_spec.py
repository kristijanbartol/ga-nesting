from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

# ==============================================================================
# 1. NESTING INPUTS (The "Item")
# ==============================================================================

@dataclass
class NestingItem:
    """
    Represents a discrete item to be packed.
    This can be a single patch (e.g., Sleeve) or a locked constellation.
    """
    item_id: int
    
    # The polygon boundary centered at (0,0).
    # Shape: (N, 2). Clockwise or CCW doesn't matter, but must be consistent.
    polygon: np.ndarray 
    
    # The "Grain Line" vector relative to the polygon coordinate system.
    # Usually [1, 0] (X-axis) for standard parameterization.
    grain_vector: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    
    # Metadata for debugging
    name: str = "patch"
    
    # NEW: Texture Phase Anchor
    # A specific point on the polygon (e.g., center, or specific seam vertex)
    # that MUST align with the texture lattice if the item is "Relaxed".
    # Default: (0,0)
    texture_anchor_point: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    def get_rotated_polygon(self, rotation_degrees: float) -> np.ndarray:
        """Returns cached or computed rotated vertices."""
        pass
    
    def get_aabb(self, rotation_degrees: float) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, max_x, max_y)."""
        pass

# ==============================================================================
# 2. THE FABRIC CONTEXT
# ==============================================================================

@dataclass
class FabricState:
    """
    Mutable state representing the current roll.
    """
    width: float
    length_used: float = 0.0
    
    # List of placed polygons for collision detection
    # Stored as (Polygon, Position_X, Position_Y)
    placed_items: List[Tuple[np.ndarray, float, float]] = field(default_factory=list)
    
    # Spatial Index (e.g., R-Tree or Grid) for fast collision checks
    # We will define this abstractly for now.
    spatial_index: Any = None 

# ==============================================================================
# 3. HEURISTIC STRATEGY (The Logic)
# ==============================================================================

class NestingHeuristic(ABC):
    """
    Abstract Base Class for placement strategies.
    Examples: Bottom-Left, Skyline, Max-Rectangle.
    """
    
    @abstractmethod
    def propose_positions(self, 
                          item: NestingItem, 
                          fabric: FabricState, 
                          rotation: float) -> List[Tuple[float, float]]:
        """
        Generates a list of candidate (x, y) positions for the item.
        These candidates are usually 'corner points' of the current layout.
        
        Note: The Heuristic does NOT check for validity/collision. 
        It only proposes "reasonable" spots to try.
        """
        pass

# ==============================================================================
# 4. THE NESTER (The Controller)
# ==============================================================================

@dataclass
class LayoutSolution:
    """The final result of packing."""
    # Map: ItemID -> (x, y, rotation)
    placements: Dict[int, Tuple[float, float, float]]
    
    total_length: float
    efficiency: float  # Area_Parts / Area_Marker
    
    # If valid=False, the nester failed to fit everything (constraints broken)
    valid: bool 

class NestingEngine(ABC):
    
    @abstractmethod
    def nest(self, 
             items: List[NestingItem], 
             permutation: List[int],     # Order from Genotype.pi
             rotations: List[int],       # Rotations from Genotype.rho
             heuristic: NestingHeuristic,
             texture_spec: Any) -> LayoutSolution:
        """
        Executes the packing.
        
        The Loop:
          1. Pick next item from 'permutation'.
          2. Apply 'rotation' (Global + Relative).
          3. Ask 'heuristic' for candidate positions.
          4. IF Texture == Relaxed:
               Filter candidates -> Snap to Texture Lattice.
          5. Check Collision for each candidate.
          6. Place at first valid spot.
          7. Update FabricState.
        """
        pass
