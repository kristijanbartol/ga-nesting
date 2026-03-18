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
        path_type=SeamPathType.DUAL,
        importance=0.0  # boundary opening — no neighboring patch
    )

    # 2. Right Shoulder (Neck R <-> Shoulder R)
    seams["Shoulder_R"] = SeamDefinition(
        "Shoulder_R", "Neck_R", "Shoulder_R",
        path_type=SeamPathType.GEODESIC,
        importance=0.0  # shoulder seam — visible
    )

    # 3. Right Armhole (Shoulder R <-> Armpit R)
    # SPECIAL CASE: This is a single logical item, but geometry
    # will cut two paths (Front/Back)
    seams["Armhole_R"] = SeamDefinition(
        "Armhole_R", "Shoulder_R", "Armpit_R",
        path_type=SeamPathType.DUAL,
        importance=0.0  # boundary opening — no neighboring patch
    )

    # 4. Right Side Seam (Armpit R <-> Hip R)
    seams["Side_R"] = SeamDefinition(
        "Side_R", "Armpit_R", "Hip_R",
        path_type=SeamPathType.GEODESIC,
        importance=1.0  # side seam — visible
    )

    # 5. Hem opening: front and back paths each split at the midline landmark
    #    Hip_R -> Hip_Front -> Hip_L  (front hem)
    #    Hip_R -> Hip_Back  -> Hip_L  (back hem)
    seams["Waist_Hem_Front_R"] = SeamDefinition(
        "Waist_Hem_Front_R", "Hip_R", "Hip_Front",
        path_type=SeamPathType.GEODESIC,
        importance=0.0  # boundary opening — no neighboring patch
    )
    seams["Waist_Hem_Front_L"] = SeamDefinition(
        "Waist_Hem_Front_L", "Hip_Front", "Hip_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )
    seams["Waist_Hem_Back_R"] = SeamDefinition(
        "Waist_Hem_Back_R", "Hip_R", "Hip_Back",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )
    seams["Waist_Hem_Back_L"] = SeamDefinition(
        "Waist_Hem_Back_L", "Hip_Back", "Hip_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )

    # 6. Left Side Seam (Hip L <-> Armpit L)
    seams["Side_L"] = SeamDefinition(
        "Side_L", "Hip_L", "Armpit_L",
        path_type=SeamPathType.GEODESIC,
        importance=1.0  # side seam — visible
    )

    # 7. Left Armhole (Armpit L <-> Shoulder L)
    seams["Armhole_L"] = SeamDefinition(
        "Armhole_L", "Armpit_L", "Shoulder_L",
        path_type=SeamPathType.DUAL,  # Special Case
        importance=0.0  # boundary opening — no neighboring patch
    )

    # 8. Left Shoulder (Shoulder L <-> Neck L)
    seams["Shoulder_L"] = SeamDefinition(
        "Shoulder_L", "Shoulder_L", "Neck_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.0  # shoulder seam — visible
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
        path_type=SeamPathType.DUAL,
        importance=0.0  # boundary opening — no neighboring patch
    )

    # 2. Right Shoulder (Neck R <-> Shoulder R)
    seams["Shoulder_R"] = SeamDefinition(
        "Shoulder_R", "Neck_R", "Shoulder_R",
        path_type=SeamPathType.GEODESIC,
        importance=1.0  # shoulder seam — visible
    )

    # 3. Right Upper Sleeve Seam (Shoulder R <-> Elbow R)
    seams["Sleeve_Upper_R"] = SeamDefinition(
        "Sleeve_Upper_R", "Sleeve_Up_R", "Shoulder_R",
        path_type=SeamPathType.GEODESIC,
        importance=0.7  # upper sleeve seam — visible
    )

    # 4. Right Sleeve Edge (Elbow R <-> Wrist R)
    seams["Sleeve_Edge_R"] = SeamDefinition(
        "Sleeve_Edge_R", "Sleeve_Up_R", "Sleeve_Down_R",
        path_type=SeamPathType.DUAL,
        importance=0.0  # boundary opening (cuff hem) — no neighboring patch
    )

    # 5. Right Lower Sleeve Seam (Wrist R <-> Armpit R)
    seams["Sleeve_Lower_R"] = SeamDefinition(
        "Sleeve_Lower_R", "Sleeve_Down_R", "Armpit_R",
        path_type=SeamPathType.GEODESIC,
        importance=0.6  # underarm sleeve seam — less visible
    )

    # 6. Right Armhole (Shoulder R <-> Armpit R)
    seams["Armhole_R"] = SeamDefinition(
        "Armhole_R", "Shoulder_R", "Armpit_R",
        path_type=SeamPathType.DUAL,
        importance=0.0  # boundary opening — no neighboring patch
    )

    # 7. Right Side Seam (Armpit R <-> Hip R)
    seams["Side_R"] = SeamDefinition(
        "Side_R", "Armpit_R", "Hip_R",
        path_type=SeamPathType.GEODESIC,
        importance=1.0  # side seam — visible
    )

    # 8. Hem opening: front and back paths each split at the midline landmark
    #    Hip_R -> Hip_Front -> Hip_L  (front hem)
    #    Hip_R -> Hip_Back  -> Hip_L  (back hem)
    seams["Waist_Hem_Front_R"] = SeamDefinition(
        "Waist_Hem_Front_R", "Hip_R", "Hip_Front",
        path_type=SeamPathType.GEODESIC,
        importance=0.0  # boundary opening — no neighboring patch
    )
    seams["Waist_Hem_Front_L"] = SeamDefinition(
        "Waist_Hem_Front_L", "Hip_Front", "Hip_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )
    seams["Waist_Hem_Back_R"] = SeamDefinition(
        "Waist_Hem_Back_R", "Hip_R", "Hip_Back",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )
    seams["Waist_Hem_Back_L"] = SeamDefinition(
        "Waist_Hem_Back_L", "Hip_Back", "Hip_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )

    # 9. Left Side Seam (Hip L <-> Armpit L)
    seams["Side_L"] = SeamDefinition(
        "Side_L", "Hip_L", "Armpit_L",
        path_type=SeamPathType.GEODESIC,
        importance=1.0  # side seam — visible
    )

    # 10. Left Armhole (Armpit L <-> Shoulder L)
    seams["Armhole_L"] = SeamDefinition(
        "Armhole_L", "Armpit_L", "Shoulder_L",
        path_type=SeamPathType.DUAL,
        importance=0.0  # boundary opening — no neighboring patch
    )

    # 11. Left Upper Sleeve Seam (Shoulder L <-> Elbow L)
    seams["Sleeve_Upper_L"] = SeamDefinition(
        "Sleeve_Upper_L", "Sleeve_Up_L", "Shoulder_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.7  # upper sleeve seam — visible
    )

    # 12. Left Sleeve Edge (Elbow L <-> Wrist L)
    seams["Sleeve_Edge_L"] = SeamDefinition(
        "Sleeve_Edge_L", "Sleeve_Up_L", "Sleeve_Down_L",
        path_type=SeamPathType.DUAL,
        importance=0.0  # boundary opening (cuff hem) — no neighboring patch
    )

    # 13. Left Lower Sleeve Seam (Wrist L <-> Armpit L)
    seams["Sleeve_Lower_L"] = SeamDefinition(
        "Sleeve_Lower_L", "Sleeve_Down_L", "Armpit_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.6  # underarm sleeve seam — less visible
    )

    # 14. Left Shoulder (Shoulder L <-> Neck L)
    seams["Shoulder_L"] = SeamDefinition(
        "Shoulder_L", "Shoulder_L", "Neck_L",
        path_type=SeamPathType.GEODESIC,
        importance=1.0  # shoulder seam — visible
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
      "Hip_Front"     -> Hip_Front               (front-rise top, front hip centre — midline, no L/R split)
      "Hip_Back"      -> Hip_Back                (back-rise top, back hip centre — midline, no L/R split)
      "Crotch"        -> Crotch_L, Crotch_R     (inseam + rise bottom)
    """
    seams = {}

    # 1. Right outer side seam  (Hip_R -> Ankle_Outer_R)
    seams["Side_R"] = SeamDefinition(
        "Side_R", "Hip_R", "Ankle_Outer_R",
        path_type=SeamPathType.GEODESIC,
        importance=1.0  # outer side — most visible
    )

    # 2. Right leg hem opening  (DUAL: front + back hem paths)
    seams["Pant_End_R"] = SeamDefinition(
        "Pant_End_R", "Ankle_Outer_R", "Ankle_Inner_R",
        path_type=SeamPathType.DUAL,
        importance=0.0  # boundary opening — no neighboring patch
    )

    # 3. Right inseam  (Ankle_Inner_R -> Crotch_R)
    seams["Inner_Seam_R"] = SeamDefinition(
        "Inner_Seam_R", "Ankle_Inner_R", "Crotch_R",
        path_type=SeamPathType.GEODESIC,
        importance=0.3  # inner leg — hidden when worn
    )

    # 4. Front rise  (Crotch_R -> Hip_Front, front body surface)
    seams["Rise_Front"] = SeamDefinition(
        "Rise_Front", "Crotch_R", "Hip_Front",
        path_type=SeamPathType.GEODESIC,
        importance=1.0  # front crotch rise — visible
    )

    # 5. Back rise  (Crotch_R -> Hip_Back, back body surface)
    # NOTE: uses Crotch_R (same as Rise_Front and Inner_Seam_R) so that all four
    # crotch seams share a single mesh vertex.  Crotch_L and Crotch_R are generated
    # by x-flipping Crotch and doing a KDTree snap; because the crotch vertices are
    # not exactly on the x=0 midline they resolve to different mesh vertices, leaving
    # an uncut "bridge" that merges two patches.  Using the same landmark for all
    # four seams guarantees they meet at one point and produces the correct 4 patches.
    seams["Rise_Back"] = SeamDefinition(
        "Rise_Back", "Crotch_R", "Hip_Back",
        path_type=SeamPathType.GEODESIC,
        importance=0.8  # back crotch rise — visible but less than front
    )

    # 6. Left inseam  (Crotch_R -> Ankle_Inner_L)
    # See note on Rise_Back above — Crotch_R used for the same reason.
    seams["Inner_Seam_L"] = SeamDefinition(
        "Inner_Seam_L", "Crotch_R", "Ankle_Inner_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.3  # inner leg — hidden when worn
    )

    # 7. Left leg hem opening  (DUAL: front + back hem paths)
    seams["Pant_End_L"] = SeamDefinition(
        "Pant_End_L", "Ankle_Inner_L", "Ankle_Outer_L",
        path_type=SeamPathType.DUAL,
        importance=0.0  # boundary opening — no neighboring patch
    )

    # 8. Left outer side seam  (Ankle_Outer_L -> Hip_L)
    seams["Side_L"] = SeamDefinition(
        "Side_L", "Ankle_Outer_L", "Hip_L",
        path_type=SeamPathType.GEODESIC,
        importance=1.0  # outer side — most visible
    )

    # 9. Waistband opening: front and back paths each split at the midline landmark
    #    Hip_R -> Hip_Front -> Hip_L  (front waistband)
    #    Hip_R -> Hip_Back  -> Hip_L  (back waistband)
    seams["Waist_Hem_Front_R"] = SeamDefinition(
        "Waist_Hem_Front_R", "Hip_R", "Hip_Front",
        path_type=SeamPathType.GEODESIC,
        importance=0.0  # boundary opening — no neighboring patch
    )
    seams["Waist_Hem_Front_L"] = SeamDefinition(
        "Waist_Hem_Front_L", "Hip_Front", "Hip_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )
    seams["Waist_Hem_Back_R"] = SeamDefinition(
        "Waist_Hem_Back_R", "Hip_R", "Hip_Back",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )
    seams["Waist_Hem_Back_L"] = SeamDefinition(
        "Waist_Hem_Back_L", "Hip_Back", "Hip_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )

    return seams


def build_onesie_topology(landmark_lib):
    """
    Defines seams for a sleeveless onesie: shirt body (no waist hem) connected
    to pants (no waistband opening) via continuous side seams through the hip.
    A front zipper seam runs from Hip_Front up to Neck_Front.

    Produces 4 body patches: front-right, back-right, front-left, back-left
    (each spanning torso + leg), plus the usual crotch topology.
    """
    seams = {}

    # 1. Neck opening (DUAL: front + back collar lines)
    seams["Neck_Opening"] = SeamDefinition(
        "Neck_Opening", "Neck_L", "Neck_R",
        path_type=SeamPathType.DUAL,
        importance=0.0
    )

    # 2. Right shoulder
    seams["Shoulder_R"] = SeamDefinition(
        "Shoulder_R", "Neck_R", "Shoulder_R",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )

    # 3. Right armhole (DUAL: front + back armhole paths)
    seams["Armhole_R"] = SeamDefinition(
        "Armhole_R", "Shoulder_R", "Armpit_R",
        path_type=SeamPathType.DUAL,
        importance=0.0
    )

    # 4. Right upper side seam (armpit → hip, replaces shirt Side_R + removes Waist_Hem)
    seams["Side_Upper_R"] = SeamDefinition(
        "Side_Upper_R", "Armpit_R", "Hip_R",
        path_type=SeamPathType.GEODESIC,
        importance=1.0
    )

    # 5. Right lower side seam (hip → ankle outer)
    seams["Side_Lower_R"] = SeamDefinition(
        "Side_Lower_R", "Hip_R", "Ankle_Outer_R",
        path_type=SeamPathType.GEODESIC,
        importance=1.0
    )

    # 6. Right leg hem opening
    seams["Pant_End_R"] = SeamDefinition(
        "Pant_End_R", "Ankle_Outer_R", "Ankle_Inner_R",
        path_type=SeamPathType.DUAL,
        importance=0.0
    )

    # 7. Right inseam (ankle inner → crotch)
    seams["Inner_Seam_R"] = SeamDefinition(
        "Inner_Seam_R", "Ankle_Inner_R", "Crotch_R",
        path_type=SeamPathType.GEODESIC,
        importance=0.3
    )

    # 8. Front rise (crotch → front hip center)
    seams["Rise_Front"] = SeamDefinition(
        "Rise_Front", "Crotch_R", "Hip_Front",
        path_type=SeamPathType.GEODESIC,
        importance=1.0
    )

    # 9. Front zipper (front hip center → front neck center)
    # Together with Rise_Front this forms the continuous front opening:
    # Neck_Front → Hip_Front → Crotch
    seams["Front_Zipper"] = SeamDefinition(
        "Front_Zipper", "Hip_Front", "Neck_Front",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )

    # 10. Back rise (crotch → back hip center)
    # Crotch_R shared with Rise_Front and inseams — see note in build_pant_topology.
    seams["Rise_Back"] = SeamDefinition(
        "Rise_Back", "Crotch_R", "Hip_Back",
        path_type=SeamPathType.GEODESIC,
        importance=0.8
    )

    # 11. Left inseam (crotch → ankle inner left)
    seams["Inner_Seam_L"] = SeamDefinition(
        "Inner_Seam_L", "Crotch_R", "Ankle_Inner_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.3
    )

    # 12. Left leg hem opening
    seams["Pant_End_L"] = SeamDefinition(
        "Pant_End_L", "Ankle_Inner_L", "Ankle_Outer_L",
        path_type=SeamPathType.DUAL,
        importance=0.0
    )

    # 13. Left lower side seam (ankle outer → hip)
    seams["Side_Lower_L"] = SeamDefinition(
        "Side_Lower_L", "Ankle_Outer_L", "Hip_L",
        path_type=SeamPathType.GEODESIC,
        importance=1.0
    )

    # 14. Left upper side seam (hip → armpit)
    seams["Side_Upper_L"] = SeamDefinition(
        "Side_Upper_L", "Hip_L", "Armpit_L",
        path_type=SeamPathType.GEODESIC,
        importance=1.0
    )

    # 15. Left armhole (DUAL: front + back armhole paths)
    seams["Armhole_L"] = SeamDefinition(
        "Armhole_L", "Armpit_L", "Shoulder_L",
        path_type=SeamPathType.DUAL,
        importance=0.0
    )

    # 16. Left shoulder
    seams["Shoulder_L"] = SeamDefinition(
        "Shoulder_L", "Shoulder_L", "Neck_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.0
    )

    return seams


def build_onesie_with_sleeves_topology(landmark_lib):
    """
    Onesie with long sleeves. Identical to build_onesie_topology plus the
    six sleeve seams from build_shirt_topology.
    """
    seams = build_onesie_topology(landmark_lib)

    # Right sleeve
    seams["Sleeve_Upper_R"] = SeamDefinition(
        "Sleeve_Upper_R", "Sleeve_Up_R", "Shoulder_R",
        path_type=SeamPathType.GEODESIC,
        importance=0.7
    )
    seams["Sleeve_Edge_R"] = SeamDefinition(
        "Sleeve_Edge_R", "Sleeve_Up_R", "Sleeve_Down_R",
        path_type=SeamPathType.DUAL,
        importance=0.0
    )
    seams["Sleeve_Lower_R"] = SeamDefinition(
        "Sleeve_Lower_R", "Sleeve_Down_R", "Armpit_R",
        path_type=SeamPathType.GEODESIC,
        importance=0.6
    )

    # Left sleeve
    seams["Sleeve_Upper_L"] = SeamDefinition(
        "Sleeve_Upper_L", "Sleeve_Up_L", "Shoulder_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.7
    )
    seams["Sleeve_Edge_L"] = SeamDefinition(
        "Sleeve_Edge_L", "Sleeve_Up_L", "Sleeve_Down_L",
        path_type=SeamPathType.DUAL,
        importance=0.0
    )
    seams["Sleeve_Lower_L"] = SeamDefinition(
        "Sleeve_Lower_L", "Sleeve_Down_L", "Armpit_L",
        path_type=SeamPathType.GEODESIC,
        importance=0.6
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
