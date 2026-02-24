from spec import SeamDefinition, SeamPathType


def build_sleeveless_shirt_topology(landmark_lib):
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


def build_shirt_topology(landmark_lib):
    """
    Defines the standard shirt seams (with sleeves) based on generated L/R landmarks.
    """
    seams = {}
    
    # 1. Neck Opening (Left Neck <-> Right Neck)
    seams["Neck_Opening"] = SeamDefinition(
        "Neck_Opening", "Neck_L", "Neck_R", 
        path_type=SeamPathType.DUAL 
    )
    
    # 2. Right Shoulder (Neck R <-> Shoulder R)
    seams["Shoulder_R"] = SeamDefinition(
        "Shoulder_R", "Neck_R", "Shoulder_R", 
        path_type=SeamPathType.GEODESIC
    )
    
    # 3. Right Upper Sleeve Seam (Shoulder R <-> Elbow R)
    seams["Sleeve_Upper_R"] = SeamDefinition(
        "Sleeve_Upper_R", "Sleeve_Up_R", "Shoulder_R",
        path_type=SeamPathType.GEODESIC
    )

    # 4. Right Sleeve Edge (Elbow R <-> Wrist R)
    seams["Sleeve_Edge_R"] = SeamDefinition(
        "Sleeve_Edge_R", "Sleeve_Up_R", "Sleeve_Down_R",
        path_type=SeamPathType.DUAL
    )

    # 5. Right Lower Sleeve Seam (Wrist R <-> Armpit R)
    seams["Sleeve_Lower_R"] = SeamDefinition(
        "Sleeve_Lower_R", "Sleeve_Down_R", "Armpit_R",
        path_type=SeamPathType.GEODESIC
    )

    # 6. Right Armhole (Shoulder R <-> Armpit R)
    seams["Armhole_R"] = SeamDefinition(
        "Armhole_R", "Shoulder_R", "Armpit_R", 
        path_type=SeamPathType.DUAL
    )
    
    # 7. Right Side Seam (Armpit R <-> Waist R)
    seams["Side_R"] = SeamDefinition(
        "Side_R", "Armpit_R", "Waist_R", 
        path_type=SeamPathType.GEODESIC
    )
    
    # 8. Waist/Hem (Waist R <-> Waist L)
    seams["Waist_Hem"] = SeamDefinition(
        "Waist_Hem", "Waist_R", "Waist_L", 
        path_type=SeamPathType.DUAL
    )
    
    # 9. Left Side Seam (Waist L <-> Armpit L)
    seams["Side_L"] = SeamDefinition(
        "Side_L", "Waist_L", "Armpit_L", 
        path_type=SeamPathType.GEODESIC
    )
    
    # 10. Left Armhole (Armpit L <-> Shoulder L)
    seams["Armhole_L"] = SeamDefinition(
        "Armhole_L", "Armpit_L", "Shoulder_L", 
        path_type=SeamPathType.DUAL
    )

    # 11. Left Upper Sleeve Seam (Shoulder L <-> Elbow L)
    seams["Sleeve_Upper_L"] = SeamDefinition(
        "Sleeve_Upper_L", "Sleeve_Up_L", "Shoulder_L",
        path_type=SeamPathType.GEODESIC
    )

    # 12. Left Sleeve Edge (Elbow L <-> Wrist L)
    seams["Sleeve_Edge_L"] = SeamDefinition(
        "Sleeve_Edge_L", "Sleeve_Up_L", "Sleeve_Down_L",
        path_type=SeamPathType.DUAL
    )

    # 13. Left Lower Sleeve Seam (Wrist L <-> Armpit L)
    seams["Sleeve_Lower_L"] = SeamDefinition(
        "Sleeve_Lower_L", "Sleeve_Down_L", "Armpit_L",
        path_type=SeamPathType.GEODESIC
    )

    # 14. Left Shoulder (Shoulder L <-> Neck L)
    seams["Shoulder_L"] = SeamDefinition(
        "Shoulder_L", "Shoulder_L", "Neck_L", 
        path_type=SeamPathType.GEODESIC
    )
    
    return seams


def build_pant_topology(landmark_lib):
    """
    Defines the standard pant seams based on generated L/R landmarks.
    """
    seams = {}

    # 1. Right Side Seam (Waist R <-> Ankle R)
    seams["Side_R"] = SeamDefinition(
        "Side_R", "Waist_R", "Ankle_R",
        path_type=SeamPathType.GEODESIC
    )

    # 2. Right Pant End / Hem (Ankle R opening)
    seams["Pant_End_R"] = SeamDefinition(
        "Pant_End_R", "Ankle_R", "Ankle_R_Inner",
        path_type=SeamPathType.DUAL
    )

    # 3. Right Inner Seam (Ankle R Inner <-> Crotch R)
    seams["Inner_Seam_R"] = SeamDefinition(
        "Inner_Seam_R", "Ankle_R_Inner", "Crotch_R",
        path_type=SeamPathType.GEODESIC
    )

    # 4. In-Between Seam Front (Crotch R <-> Crotch L, front side)
    # Shared between left and right - only defined once
    seams["Inbetween_Front"] = SeamDefinition(
        "Inbetween_Front", "Crotch", "Waist_Mid_Front",
        path_type=SeamPathType.GEODESIC
    )

    # 5. In-Between Seam Back (Crotch R <-> Crotch L, back side)
    # Shared between left and right - only defined once
    seams["Inbetween_Back"] = SeamDefinition(
        "Inbetween_Back", "Crotch", "Waist_Mid_Back",
        path_type=SeamPathType.GEODESIC
    )

    # 6. Waist Right Front (Crotch R <-> Waist R, front)
    seams["Waist_R_Front"] = SeamDefinition(
        "Waist_R_Front", "Waist_Mid_Front", "Waist_R",
        path_type=SeamPathType.GEODESIC
    )

    # 7. Waist Right Back (Crotch R <-> Waist R, back)
    seams["Waist_R_Back"] = SeamDefinition(
        "Waist_R_Back", "Waist_Mid_Back", "Waist_R",
        path_type=SeamPathType.GEODESIC
    )

    # --- Symmetric Left Side ---

    # 8. Waist Left Front (Waist_Mid_Front <-> Waist L, front)
    seams["Waist_L_Front"] = SeamDefinition(
        "Waist_L_Front", "Waist_Mid_Front", "Waist_L",
        path_type=SeamPathType.GEODESIC
    )

    # 9. Waist Left Back (Waist_Mid_Back <-> Waist L, back)
    seams["Waist_L_Back"] = SeamDefinition(
        "Waist_L_Back", "Waist_Mid_Back", "Waist_L",
        path_type=SeamPathType.GEODESIC
    )

    # 10. Left Inner Seam (Crotch <-> Ankle L Inner)
    seams["Inner_Seam_L"] = SeamDefinition(
        "Inner_Seam_L", "Crotch", "Ankle_L_Inner",
        path_type=SeamPathType.GEODESIC
    )

    # 11. Left Pant End / Hem (Ankle L Inner <-> Ankle L opening)
    seams["Pant_End_L"] = SeamDefinition(
        "Pant_End_L", "Ankle_L_Inner", "Ankle_L",
        path_type=SeamPathType.DUAL
    )

    # 12. Left Side Seam (Ankle L <-> Waist L)
    seams["Side_L"] = SeamDefinition(
        "Side_L", "Ankle_L", "Waist_L",
        path_type=SeamPathType.GEODESIC
    )

    return seams


def build_test_topology(landmark_lib):
    seams = {}
    
    # 1. Neck Opening (Left Neck <-> Right Neck)
    # Usually a hole, so we use DUAL to get front/back collar lines
    seams["s1"] = SeamDefinition(
        "s1", "1", "2", 
        path_type=SeamPathType.GEODESIC 
    )
    
    # 2. Right Shoulder (Neck R <-> Shoulder R)
    seams["s2"] = SeamDefinition(
        "s2", "2", "3", 
        path_type=SeamPathType.GEODESIC
    )
    
    # 3. Right Armhole (Shoulder R <-> Armpit R)
    # SPECIAL CASE: This is a single logical item, but geometry 
    # will cut two paths (Front/Back)
    seams["s3"] = SeamDefinition(
        "s3", "3", "4", 
        path_type=SeamPathType.GEODESIC
    )
    
    # 4. Right Side Seam (Armpit R <-> Waist R)
    seams["s4"] = SeamDefinition(
        "s4", "4", "1", 
        path_type=SeamPathType.GEODESIC
    )
    
    return seams
