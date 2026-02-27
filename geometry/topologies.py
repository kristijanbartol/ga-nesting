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
    Defines pant seams that produce 4 patches (front/back × left/right leg).

    Landmark names must match what generate_symmetric_landmarks produces, i.e.
    each source key becomes <key>_L (original) and <key>_R (mirror).

    Required source keys in LONG_LANDMARKS["Lower"]:
      "Ankle_Outer"   -> Ankle_Outer_L, Ankle_Outer_R  (outer ankle / hem corner)
      "Ankle_Inner"   -> Ankle_Inner_L, Ankle_Inner_R  (inner ankle / inseam corner)
    Required source keys in CORE_LANDMARKS["Lower"]:
      "Hip"           -> Hip_L, Hip_R           (side-seam top / waistband corners)
      "Crotch"        -> Crotch_L, Crotch_R     (inseam + rise bottom)
      "Waist_Front"   -> Waist_Front_L/R        (front-rise top, front waist centre)
      "Waist_Back"    -> Waist_Back_L/R         (back-rise top, back waist centre)
    """
    seams = {}

    # 1. Right outer side seam  (Hip_R -> Ankle_Outer_R)
    seams["Side_R"] = SeamDefinition(
        "Side_R", "Hip_R", "Ankle_Outer_R",
        path_type=SeamPathType.GEODESIC
    )

    # 2. Right leg hem opening  (DUAL: front + back hem paths)
    seams["Pant_End_R"] = SeamDefinition(
        "Pant_End_R", "Ankle_Outer_R", "Ankle_Inner_R",
        path_type=SeamPathType.DUAL
    )

    # 3. Right inseam  (Ankle_Inner_R -> Crotch_R)
    seams["Inner_Seam_R"] = SeamDefinition(
        "Inner_Seam_R", "Ankle_Inner_R", "Crotch_R",
        path_type=SeamPathType.GEODESIC
    )

    # 4. Front rise  (Crotch_R -> Waist_Front_L, front body surface)
    seams["Rise_Front"] = SeamDefinition(
        "Rise_Front", "Crotch_R", "Waist_Front_L",
        path_type=SeamPathType.GEODESIC
    )

    # 5. Back rise  (Crotch_L -> Waist_Back_L, back body surface)
    seams["Rise_Back"] = SeamDefinition(
        "Rise_Back", "Crotch_L", "Waist_Back_L",
        path_type=SeamPathType.GEODESIC
    )

    # 6. Left inseam  (Crotch_L -> Ankle_Inner_L)
    seams["Inner_Seam_L"] = SeamDefinition(
        "Inner_Seam_L", "Crotch_L", "Ankle_Inner_L",
        path_type=SeamPathType.GEODESIC
    )

    # 7. Left leg hem opening  (DUAL: front + back hem paths)
    seams["Pant_End_L"] = SeamDefinition(
        "Pant_End_L", "Ankle_Inner_L", "Ankle_Outer_L",
        path_type=SeamPathType.DUAL
    )

    # 8. Left outer side seam  (Ankle_Outer_L -> Hip_L)
    seams["Side_L"] = SeamDefinition(
        "Side_L", "Ankle_Outer_L", "Hip_L",
        path_type=SeamPathType.GEODESIC
    )

    # 9. Waistband opening  (DUAL: front + back waist hem paths)
    seams["Waist_Hem"] = SeamDefinition(
        "Waist_Hem", "Hip_R", "Hip_L",
        path_type=SeamPathType.DUAL
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
