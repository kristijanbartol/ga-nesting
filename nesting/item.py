import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from dataclasses import dataclass, field
from typing import Tuple


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
