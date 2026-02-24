from spec import LandmarkDefinition


# TODOs:
#  - move seamline definitions (`topologies.py`) to geometry/
#  - test all garment types (sleeveless shirt, long shirt, short shirt, pants, shorts, dress, onesie)
#  - add onesie definition(s)
#  - add more than one seamline definition per garment type (for example, split in the middle / jacket style) + cut in the middle back
#  - add complicated onesie definition (teaser?)

#  - adjust flattening hyperparameters as part of the GA (more parameters!!)

#  - add more texture pattern types (start with stripes and grid)
#  - visualize in the simulation in 3D
#  - include fit loss component (read from the resulting scales and evaluate using existing utils)
# 
#  - adjust hyperparameters to compare the results

# OPTIONAL / EXPERIMENTS:
#  - produce for multiple sizes, e.g., S, M, L, XL (two modes: each size has individual parameters vs. common parameters)
#  - more complex pattern symmetries


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
        "Waist": LandmarkDefinition(
            name="Waist",
            boundary_corners=(6524, 6557, 4984, 4921)
        ),
        #"Waist_Mid": LandmarkDefinition(
        #    name="Waist_Mid",
        #    boundary_corners=(6524, 6557, 4984, 4921)
        #)
    },
    "Lower": {
        "Waist": LandmarkDefinition(
            name="Waist",
            boundary_corners=(6524, 6557, 4984, 4921)
        ),
        "Waist_Mid": LandmarkDefinition(
            name="Waist_Mid",
            boundary_corners=(6524, 6557, 4984, 4921)
        ),
        "Hip": LandmarkDefinition(
            name="Hip",
            boundary_corners=(6524, 6557, 4984, 4921)
        ),
        "Crotch": LandmarkDefinition(
            name="Crotch",
            boundary_corners=(6524, 6557, 4984, 4921)
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
        "Leg_Outer": LandmarkDefinition(
            name="Pant_End",
            boundary_corners=(6582, 6582, 6722, 6722)
        ),
        "Leg_Inner": LandmarkDefinition(
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


SHIRTLESS_SEAMS = [
    "Side_L", 
    "Side_R", 
    "Neck_Opening", 
    "Shoulder_R", 
    "Shoulder_L", 
    "Armhole_R", 
    "Armhole_L", 
    "Waist_Hem"
]
SHIRT_SEAMS = [
    "Side_L", 
    "Side_R", 
    "Neck_Opening", 
    "Shoulder_R", 
    "Shoulder_L",
    "Armhole_R", 
    "Armhole_L", 
    "Waist_Hem", 
    "Sleeve_Upper_L", 
    "Sleeve_Upper_R", 
    "Sleeve_Edge_L", 
    "Sleeve_Edge_R",
    "Sleeve_Lower_L",
    "Sleeve_Lower_R"
]
PANT_SEAMS = [
    "Side_L",
    "Side_R",
    "Pant_End_L",
    "Pant_End_R",
    "Crotch_Front",
    "Crotch_Back",
    "Inner_Seam_L",
    "Inner_Seam_R",
    "Waistband_Front_L",
    "Waistband_Front_R",
    "Waistband_Back_L",
    "Waistband_Back_R"
]
ONESIE_SEAMS = []

ACTIVE_SEAMS = SHIRT_SEAMS.copy()
SHOULDER_KPT_IDX = 5335
