from typing import Dict


def analyze_contour(features: Dict[str, float]) -> Dict[str, bool]:
    """
    Analyze extracted shape features to classify the object.

    Returns:
    - Dictionary with high-level attributes.
    """
    attributes = {
        "is_man_made": False,
        "fracture_detected": False,
        "long_object": False,
        "round_object": False,
        "compact_object": False,
        "long_skeleton": False,
        "rigid_object": False,
    }

    # Heuristic for man-made object:
    if (
        (
            features.get("solidity", 0) > 0.85 and
            features.get("num_corners", 0) in [3, 4, 6, 8, 10, 12]
        ) or (
            features.get("eccentricity", 0) > 0.95 and
            features.get("skeleton_length", 0) > 300
        )
    ):
        attributes["is_man_made"] = True

    if (
        features.get("num_defects", 0) > 5 or
        features.get("num_corners", 0) > 10 or
        features.get("skeleton_length", 0) > 200 or
        features.get("eccentricity", 0) > 0.95
    ):
        attributes["fracture_detected"] = True

    if (
        features.get("eccentricity", 0) > 0.95 and
        (features.get("aspect_ratio", 0) > 3 or features.get("skeleton_length", 0) > 300) and
        features.get("circularity", 0) < 0.4
    ):
        attributes["long_object"] = True

    if (
        features.get("circularity", 0) > 0.65 and
        features.get("eccentricity", 0) < 0.6 and
        features.get("solidity", 0) > 0.9
    ):
        attributes["round_object"] = True

    # Compact convex object (box, brick)
    if (
        features.get("solidity", 0) > 0.95 and
        features.get("extent", 0) > 0.8 and
        features.get("num_corners", 0) in [4, 6]
    ):
        attributes["compact_object"] = True

    # Long skeleton
    if (
        features.get("skeleton_length", 0) > 300 and
        (features.get("area", 0) / max(features.get("skeleton_length", 1), 1)) < 3
    ):
        attributes["long_skeleton"] = True

    attributes["rigid_object"] = (
        (features.get("solidity") > 0.85 and features.get("extent") > 0.85 and features.get("num_defects") <= 2) or
        (features.get("eccentricity", 0) > 0.98 and attributes['long_object'] and features.get("skeleton_length", 0) > 300)
    )

    return attributes