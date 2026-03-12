from typing import List, Tuple, Dict, Optional, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

# ==============================================================================
# 1. STATIC DEFINITIONS (The "Library")
# ==============================================================================

@dataclass(frozen=True)
class LandmarkDefinition:
    """
    Definition of a semantic landmark region on the mesh surface.
    The region is defined as a topological quad projected onto the manifold.
    """
    name: str
    
    # The 4 corner vertices defining the region of interest (ROI).
    # Order: [Bottom-Left, Bottom-Right, Top-Right, Top-Left]
    # Corresponding to (0,0), (1,0), (1,1), (0,1) in the delta space.
    boundary_corners: Tuple[int, int, int, int]

class SeamPathType(Enum):
    GEODESIC = "geodesic"     # Standard shortest path (e.g., Side Seam)
    DUAL = "dual"             # Split path (Front + Back), e.g., Armhole or Neck Opening

@dataclass(frozen=True)
class SeamDefinition:
    """
    Static definition of a topological cut connecting two landmarks.
    """
    name: str
    start_landmark: str
    end_landmark: str
    path_type: SeamPathType = SeamPathType.GEODESIC     # Hints the geometry engine on how to cut between these points

    # NEW: Store specific geometric parameters here.
    # e.g. {"front_z": 0.05, "back_z": -0.1}
    geometry_hints: Dict[str, float] = field(default_factory=dict)

    # Static importance weight for texture alignment (0.0 = excluded).
    # DUAL (boundary) seams default to 0.0; GEODESIC interior seams set explicitly.
    importance: float = 0.0

# ==============================================================================
# 2. PROBLEM CONTEXT (Immutable)
# ==============================================================================

@dataclass(frozen=True)
class TextureSpec:
    name: str
    period_x: float
    period_y: float
    wallpaper_group: str = "stripes"

    def get_valid_symmetries(self) -> List[float]:
        """
        Returns allowed relative rotation angles (degrees) for this group.
        Stripes (PM)  -> [0, 180]
        Grid   (PMM)  -> [0, 90, 180, 270]
        """
        from wallpaper import get_policy, StripesPolicy, GridPolicy
        policy = get_policy(self.wallpaper_group)
        if isinstance(policy, GridPolicy):
            return [0.0, 90.0, 180.0, 270.0]
        return [0.0, 180.0]  # StripesPolicy and future 2-fold groups

    def wallpaper_policy(self):
        """Return the WallpaperPolicy for this texture's group."""
        from wallpaper import get_policy
        return get_policy(self.wallpaper_group)

@dataclass(frozen=True)
class ProblemInstance:
    """
    The immutable context for a SPECIFIC optimization run.
    Contains strictly the subset of data required for the selected seams.
    """
    # Metadata
    mesh_path: str
    fabric_width: float
    texture: TextureSpec

    # Geometry Data (Loaded Once)
    # Required for the LandmarkMapper to perform interpolation
    mesh_vertices: np.ndarray # Shape (N, 3)
    mesh_faces: np.ndarray    # Shape (F, 3)
    
    # Active Logic
    active_landmarks: Tuple[LandmarkDefinition, ...]
    active_seam_topology: Tuple[Tuple[int, int], ...] # Indices into active_landmarks

    seam_names: Tuple[str, ...]

    # Store the full objects, not just types, so we can access 'geometry_hints'
    active_seam_definitions: Tuple[SeamDefinition, ...]

    # Genotype.alpha[i] corresponds to active_seam_types[i]
    active_seam_types: Tuple[SeamPathType, ...]

    # Per-seam static importance weights (all seams, incl. DUAL with 0.0).
    seam_importances: Tuple[float, ...] = ()

    @property
    def num_landmarks(self) -> int:
        return len(self.active_landmarks)

    @property
    def num_seams(self) -> int:
        return len(self.active_seam_topology)

# ==============================================================================
# 3. GENOTYPE (Decision Variables)
# ==============================================================================

@dataclass
class Genotype:
    """
    The fixed-length vector representation optimized by the EA.
    """
    # 1. Geometry Parameters
    # Dimensionality: 2 * num_landmarks
    # Structure: [u_0, v_0, u_1, v_1, ..., u_m, v_m]
    # Range: [0.0, 1.0]
    # Maps to a coordinate within the Quad Patch of the corresponding landmark.
    delta: np.ndarray 

    # 2. Fabrication Orientation
    # Dimensionality: num_charts (derived from num_seams topology)
    # Discrete [0, 1, 2, 3] for EACH chart.
    # Controls Global Rotation (if Root) or Relative Symmetry (if Child).
    rho: np.ndarray    

    # 3. Alignment Logic
    # Dimensionality: num_seams
    # Binary [0, 1] for EACH active seam.
    # 0 = Relaxed (Independent), 1 = Locked (Constellation).
    alpha: np.ndarray  

    # 4. Nesting Order
    # Dimensionality: num_charts
    # Permutation of chart indices.
    pi: np.ndarray     

    # 5. Heuristic Selector
    # Integer ID for the packing strategy.
    heuristic_id: int

# ==============================================================================
# 4. PIPELINE INTERFACES
# ==============================================================================

@dataclass
class FlatPattern:
    """Intermediate representation: A flattened 2D polygon."""
    chart_id: int
    polygon: np.ndarray  # (V, 2) vertices centered at (0,0)
    
    # Map: SeamIndex (from ProblemInstance) -> (M, 2) boundary line vertices
    seam_boundaries: Dict[int, np.ndarray] 

@dataclass
class NestingItem:
    """
    Atomic unit for the nesting engine.
    Can be a single piece or a 'Constellation' of locked pieces.
    """
    root_chart_id: int
    
    # Combined polygon of the root + all locked children (transformed)
    # Used for collision detection
    contour: np.ndarray 
    
    # The actual pieces inside this item (for final coordinate reporting)
    # List of (ChartID, RelativeTransformMatrix)
    sub_charts: List[Tuple[int, np.ndarray]] 
    
    # Global rotation logic derived from Root's rho
    global_rotation_deg: float 

@dataclass
class LayoutResult:
    """Final output of the evaluation pipeline."""
    # Map: ChartID -> (x, y, rotation_deg) on the fabric
    placements: Dict[int, Tuple[float, float, float]]
    
    marker_length: float
    total_distortion: float
    texture_mismatch_score: float

class PhenotypeMapper(ABC):
    """
    Abstract Base Class for the decoder pipeline.
    """
    @abstractmethod
    def decode_geometry(self, 
                        instance: ProblemInstance, 
                        delta: np.ndarray) -> List[FlatPattern]:
        """
        Stage 1:
        1. Map delta (u,v) -> 3D Vertex IDs (Bilinear Interp + Snap).
        2. Compute Geodesic Cuts between active landmarks.
        3. Parameterize (LSCM) patches to 2D.
        """
        pass

    @abstractmethod
    def build_constellations(self, 
                             instance: ProblemInstance,
                             patterns: List[FlatPattern], 
                             alpha: np.ndarray, 
                             rho: np.ndarray) -> List[NestingItem]:
        """
        Stage 2:
        1. Build connectivity graph from alpha.
        2. Merge locked pieces using TextureSpec logic.
        3. Assign global/relative rotations using rho.
        """
        pass

    @abstractmethod
    def nest_layout(self, 
                    instance: ProblemInstance,
                    items: List[NestingItem], 
                    pi: np.ndarray, 
                    heuristic_id: int) -> LayoutResult:
        """
        Stage 3:
        1. Sort items by pi.
        2. Place using Heuristic.
        3. Snap Relaxed items to Texture Lattice.
        """
        pass
