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
        "Waist": LandmarkDefinition(
            name="Waist",
            boundary_corners=(6524, 6524, 6385, 6385)
        ),
        #"Waist_Mid": LandmarkDefinition(
        #    name="Waist_Mid",
        #    boundary_corners=(6524, 6557, 4984, 4921)
        #)
    },
    "Lower": {
        "Hip": LandmarkDefinition(
            name="Hip",
            boundary_corners=(4984, 4984, 4921, 4921)
        ),
        "Waist_Front": LandmarkDefinition(
            name="Waist_Front",
            boundary_corners=(3160, 3160, 3160, 3160)
        ),
        "Waist_Back": LandmarkDefinition(
            name="Waist_Front",
            boundary_corners=(1783, 1783, 1783, 1783)
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
    "Side_R",
    "Pant_End_R",
    "Inner_Seam_R",
    "Rise_Front",
    "Rise_Back",
    "Inner_Seam_L",
    "Pant_End_L",
    "Side_L",
    "Waist_Hem",
]
ONESIE_SEAMS = []

ACTIVE_SEAMS = PANT_SEAMS.copy()
SHOULDER_KPT_IDX = 5335
