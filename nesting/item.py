import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class NestingItem:
    """
    A distinct piece of fabric to be nested.
    Wraps a shapely Polygon and handles its local transformations.
    """
    item_id: int
    name: str
    
    # The raw vertices (N, 2) centered at (0,0) locally.
    original_vertices: np.ndarray 
    
    # Patch index as used by export_seamlines (e.g., 1, 2, ...)
    patch_idx: Optional[int] = None

    # Optional anchor point (in LOCAL coordinates) used for texture alignment.
    # When present, the nesting engine will snap this point to the texture lattice.
    seam_anchor_local: Optional[Tuple[float, float]] = None
    
    # The active Shapely geometry (after rotation)
    # initialized in __post_init__
    shape: Polygon = field(init=False)
    
    # We track the current rotation state to avoid re-computing
    current_rotation: float = 0.0

    def __post_init__(self):
        """Convert raw numpy vertices to Shapely Polygon."""
        # Ensure the polygon is valid and simple
        poly = Polygon(self.original_vertices)
        if not poly.is_valid:
            poly = poly.buffer(0) # Attempt auto-fix for self-intersections
        self.shape = poly

    def set_rotation(self, angle_degrees: float):
        """
        Applies a rotation to the original shape.
        Resets the item to (0,0) but rotated.
        """
        if abs(angle_degrees - self.current_rotation) < 1e-5:
            return

        # Always rotate from the ORIGINAL to avoid accumulation errors
        base_shape = Polygon(self.original_vertices)
        self.shape = rotate(base_shape, angle_degrees, origin=(0, 0))
        self.current_rotation = angle_degrees

    def place_at(self, x: float, y: float) -> Polygon:
        """
        Returns a NEW Polygon object translated to (x, y).
        Does NOT modify the stored internal state (immutability for safety).
        """
        return translate(self.shape, xoff=x, yoff=y)

    @property
    def area(self) -> float:
        return self.shape.area

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns (minx, miny, maxx, maxy)."""
        return self.shape.bounds
    
    @property
    def bottom_left_offset(self) -> Tuple[float, float]:
        """
        Returns the vector from the Centroid (0,0) to the Bottom-Left corner (minx, miny).
        Used to correct placement so the piece sits strictly inside the positive quadrant.
        """
        minx, miny, _, _ = self.shape.bounds
        # shape is centered at (0,0), so minx/miny are usually negative
        return (minx, miny)
