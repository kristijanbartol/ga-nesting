from spec import SeamDefinition, SeamPathType


def build_shirt_topology(landmark_lib):
    """
    Defines the standard shirt seams based on generated L/R landmarks.
    """
    seams = {}
    
    # 1. Neck Opening (Left Neck <-> Right Neck)
    # Usually a hole, so we use DUAL to get front/back collar lines
    seams["Neck_Opening"] = SeamDefinition(
        "Neck_Opening", "Neck_L", "Neck_R", 
        path_type=SeamPathType.DUAL 
    )
    
    # 2. Right Shoulder (Neck R <-> Shoulder R)
    seams["Shoulder_R"] = SeamDefinition(
        "Shoulder_R", "Neck_R", "Shoulder_R", 
        path_type=SeamPathType.GEODESIC
    )
    
    # 3. Right Armhole (Shoulder R <-> Armpit R)
    # SPECIAL CASE: This is a single logical item, but geometry 
    # will cut two paths (Front/Back)
    seams["Armhole_R"] = SeamDefinition(
        "Armhole_R", "Shoulder_R", "Armpit_R", 
        path_type=SeamPathType.DUAL
    )
    
    # 4. Right Side Seam (Armpit R <-> Waist R)
    seams["Side_R"] = SeamDefinition(
        "Side_R", "Armpit_R", "Waist_R", 
        path_type=SeamPathType.GEODESIC
    )
    
    # 5. Waist/Hem (Waist R <-> Waist L)
    # Assuming this loops around the back/front (Dual) or just a bottom cut?
    # Usually the bottom of a shirt is an opening.
    seams["Waist_Hem"] = SeamDefinition(
        "Waist_Hem", "Waist_R", "Waist_L", 
        path_type=SeamPathType.DUAL
    )
    
    # 6. Left Side Seam (Waist L <-> Armpit L)
    seams["Side_L"] = SeamDefinition(
        "Side_L", "Waist_L", "Armpit_L", 
        path_type=SeamPathType.GEODESIC
    )
    
    # 7. Left Armhole (Armpit L <-> Shoulder L)
    seams["Armhole_L"] = SeamDefinition(
        "Armhole_L", "Armpit_L", "Shoulder_L", 
        path_type=SeamPathType.DUAL # Special Case
    )
    
    # 8. Left Shoulder (Shoulder L <-> Neck L)
    seams["Shoulder_L"] = SeamDefinition(
        "Shoulder_L", "Shoulder_L", "Neck_L", 
        path_type=SeamPathType.GEODESIC
    )
    
    return seams
