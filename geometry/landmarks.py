from spec import LandmarkDefinition


CORE_LANDMARKS = {
    "Upper": {
        "Neck": LandmarkDefinition(
            name="Neck",
            boundary_corners=(4301, 5279, 4199, 4762),
        ),
        "Shoulder": LandmarkDefinition(
            name="Shoulder",
            boundary_corners=(5274, 6446, 4122, 4723)
        ),
        "Armpit": LandmarkDefinition(
            name="Armpit",
            boundary_corners=(4755, 4751, 5230, 4163)
        ),
        "Hip": LandmarkDefinition(
            name="Hip",
            boundary_corners=(4144, 5261, 5250, 4286)
        ),
    },
    "Lower": {
        "Hip": LandmarkDefinition(
            name="Hip",
            boundary_corners=(4144, 5261, 5250, 4286)
        ),
        "Crotch": LandmarkDefinition(
            name="Crotch",
            boundary_corners=(3149, 3149, 1364, 1364)
        )
    }
}


LONG_LANDMARKS = {
    "Upper": {
        "Sleeve_Up": LandmarkDefinition(
            name="Sleeve_Up",
            boundary_corners=(6335, 6335, 5400, 5400)
        ),
        "Sleeve_Down": LandmarkDefinition(
            name="Sleeve_Down",
            boundary_corners=(5391, 5391, 5398, 5398)
        )
    },
    "Lower": {
        "Ankle_Outer": LandmarkDefinition(
            name="Pant_End",
            boundary_corners=(6582, 6582, 6722, 6722)
        ),
        "Ankle_Inner": LandmarkDefinition(
            name="Pant_End",
            boundary_corners=(6593, 6593, 6595, 6595)
        ),
    }
}

SHORT_LANDMARKS = {
    "Upper": {
        "Sleeve_Up": LandmarkDefinition(
            name="Sleeve_Up",
            boundary_corners=(5176, 5176, 5184, 5184)
        ),
        "Sleeve_Down": LandmarkDefinition(
            name="Sleeve_Down",
            boundary_corners=(5137, 5137, 5218, 5218)
        )
    },
    "Lower": {
        "Ankle_Outer": LandmarkDefinition(
            name="Pant_End",
            boundary_corners=(4521, 4521, 4518, 4518)
        ),
        "Ankle_Inner": LandmarkDefinition(
            name="Pant_End",
            boundary_corners=(4657, 4657, 4526, 4526)
        ),
    }
}


ONESIE_LANDMARKS = {
    "Neck": LandmarkDefinition(
        name="Neck",
        boundary_corners=(4301, 5279, 4199, 4762),
    ),
    "Neck_Front": LandmarkDefinition(
        name="Neck_Front",
        boundary_corners=(3169, 3169, 3169, 3169),
    ),
    "Shoulder": LandmarkDefinition(
        name="Shoulder",
        boundary_corners=(5274, 6446, 4122, 4723)
    ),
    "Armpit": LandmarkDefinition(
        name="Armpit",
        boundary_corners=(4755, 4751, 5230, 4163)
    ),
    "Hip": LandmarkDefinition(
        name="Hip",
        boundary_corners=(4984, 4984, 4921, 4921)
    ),
    "Hip_Front": LandmarkDefinition(
        name="Hip_Front",
        boundary_corners=(3160, 3160, 3160, 3160)
    ),
    "Hip_Back": LandmarkDefinition(
        name="Hip_Back",
        boundary_corners=(1784, 1784, 1784, 1784)
    ),
    "Crotch": LandmarkDefinition(
        name="Crotch",
        boundary_corners=(3149, 3149, 1364, 1364)
    ),
    "Ankle_Outer": LandmarkDefinition(
        name="Pant_End",
        boundary_corners=(6582, 6582, 6722, 6722)
    ),
    "Ankle_Inner": LandmarkDefinition(
        name="Pant_End",
        boundary_corners=(6593, 6593, 6595, 6595)
    ),
}

ONESIE_LONG_LANDMARKS = {
    "Sleeve_Up": LandmarkDefinition(
        name="Sleeve_Up",
        boundary_corners=(6335, 6335, 5400, 5400)
    ),
    "Sleeve_Down": LandmarkDefinition(
        name="Sleeve_Down",
        boundary_corners=(5391, 5391, 5398, 5398)
    ),
}


# ---------------------------------------------------------------------------
# Midline landmarks — not mirrored (x≈0 so L/R would be the same vertex).
# These are added directly to the landmark lib without going through
# generate_symmetric_landmarks.  They are marked is_derived=True so
# LandmarkMapper computes their vertex IDs automatically from a reference
# landmark at the same height using find_midline_vidx.
# ---------------------------------------------------------------------------

LOWER_MIDLINE_LANDMARKS = {
    "Hip_Front": LandmarkDefinition(
        name="Hip_Front",
        boundary_corners=(0, 0, 0, 0),   # unused — vertex derived at runtime
        is_derived=True,
    ),
    "Hip_Back": LandmarkDefinition(
        name="Hip_Back",
        boundary_corners=(0, 0, 0, 0),   # unused — vertex derived at runtime
        is_derived=True,
    ),
}

# Specifies how each derived midline landmark is computed:
#   (derived_name, reference_landmark_name, is_front)
# reference_landmark_name must be a sampled (non-derived) landmark whose
# vertex ID is available after pass 1 of map_genotype_to_vertices.
LOWER_DERIVED_MIDLINE: tuple = (
    ("Hip_Front", "Hip_L", True),
    ("Hip_Back",  "Hip_L", False),
)

# Same structure for the upper (shirt/sleeveless) garment — derived from its own Hip_L.
UPPER_MIDLINE_LANDMARKS = {
    "Hip_Front": LandmarkDefinition(
        name="Hip_Front",
        boundary_corners=(0, 0, 0, 0),   # unused — vertex derived at runtime
        is_derived=True,
    ),
    "Hip_Back": LandmarkDefinition(
        name="Hip_Back",
        boundary_corners=(0, 0, 0, 0),   # unused — vertex derived at runtime
        is_derived=True,
    ),
}

UPPER_DERIVED_MIDLINE: tuple = (
    ("Hip_Front", "Hip_L", True),
    ("Hip_Back",  "Hip_L", False),
)

SHIRTLESS_SEAMS = [
    "Side_L",
    "Side_R",
    "Neck_Opening",
    "Shoulder_R",
    "Shoulder_L",
    "Armhole_R",
    "Armhole_L",
    "Waist_Hem_Front_R",
    "Waist_Hem_Front_L",
    "Waist_Hem_Back_R",
    "Waist_Hem_Back_L",
]
SHIRT_SEAMS = [
    "Side_L",
    "Side_R",
    "Neck_Opening",
    "Shoulder_R",
    "Shoulder_L",
    "Armhole_R",
    "Armhole_L",
    "Waist_Hem_Front_R",
    "Waist_Hem_Front_L",
    "Waist_Hem_Back_R",
    "Waist_Hem_Back_L",
    "Sleeve_Upper_L",
    "Sleeve_Upper_R",
    "Sleeve_Edge_L",
    "Sleeve_Edge_R",
    "Sleeve_Lower_L",
    "Sleeve_Lower_R",
]
PANT_SEAMS = [
    "Side_R",
    "Pant_End_R",
    "Inner_Seam_R",
    "Rise_Front",
    "Rise_Back",
    "Inner_Seam_L",
    "Pant_End_L",
    "Side_L",
    "Waist_Hem_Front_R",
    "Waist_Hem_Front_L",
    "Waist_Hem_Back_R",
    "Waist_Hem_Back_L",
]
ONESIE_SEAMS = [
    "Neck_Opening",
    "Shoulder_R",
    "Armhole_R",
    "Side_Upper_R",
    "Side_Lower_R",
    "Pant_End_R",
    "Inner_Seam_R",
    "Rise_Front",
    "Front_Zipper",
    "Rise_Back",
    "Inner_Seam_L",
    "Pant_End_L",
    "Side_Lower_L",
    "Side_Upper_L",
    "Armhole_L",
    "Shoulder_L",
]
ONESIE_WITH_SLEEVES_SEAMS = ONESIE_SEAMS + [
    "Sleeve_Upper_R",
    "Sleeve_Edge_R",
    "Sleeve_Lower_R",
    "Sleeve_Upper_L",
    "Sleeve_Edge_L",
    "Sleeve_Lower_L",
]

ACTIVE_SEAMS = PANT_SEAMS.copy()
SHOULDER_KPT_IDX = 5335
HIP_KPT_IDX = 4418   # side hip vertex used as Y anchor for front/back midline detection
