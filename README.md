# 🧠 ContourIQ

Smart Shape Analysis from Segmented Images

ContourIQ is a modular, AI-assisted pipeline that analyzes object contours from segmented images to extract meaningful insights — including object type classification (e.g. man-made vs. natural) and structural features like fractures, shape complexity, and geometry. Built for flexibility, precision, and scalability.


## Explanation

 Breakdown of each of the key contour features used in ContourIQ and their scientific meaning and why they matter for object classification:
🔹 1. Area

    Definition: Total number of pixels inside the contour.

    Why it matters: Gives a sense of object size. Very small areas may be noise; large ones are likely full objects.

🔹 2. Perimeter

    Definition: Total length of the contour's outline.

    Why it matters: Used with area to derive shape descriptors like circularity. Irregular or jagged shapes have longer perimeters.

🔹 3. Circularity

    Formula: (4⋅π⋅Area)/Perimeter2(4⋅π⋅Area)/Perimeter2

    Range: [0, 1], where 1 = perfect circle

    Why it matters: Helps distinguish round, smooth shapes from angular or elongated ones. Useful for detecting wheels, balls, etc.

🔹 4. Aspect Ratio

    Formula: width / height of bounding rectangle

    Why it matters: High values (e.g., >2 or <0.5) often indicate elongated objects like sticks, poles, or rods.

🔹 5. Extent

    Formula: Area/(Bounding Box Area)Area/(Bounding Box Area)

    Why it matters: Measures how much of the bounding box the object fills. Low extent → sparse or oddly shaped object.

🔹 6. Solidity

    Formula: Area/Convex Hull AreaArea/Convex Hull Area

    Why it matters: Measures shape regularity. Perfectly convex = 1. Low solidity = jagged or hollow shapes → good fracture indicator.

🔹 7. Eccentricity

    Formula: 1−(b/a)21−(b/a)2

    ​ where a = major axis, b = minor axis

    Range: 0 (circle) to 1 (line)

    Why it matters: High eccentricity means very elongated → useful to catch pipes, bars, beams, etc.

🔹 8. Skeleton Length

    Definition: Number of foreground pixels in the skeletonized shape.

    Why it matters: Good proxy for object complexity and length. Long skeletons = wires, ropes, etc.

🔹 9. Hu Moments (1–7)

    Definition: Set of 7 numbers derived from image moments, invariant to scale, rotation, and translation.

    Why it matters: Powerful for general shape comparison, especially in model-based matching.

🔹 10. Convexity Defects

    Definition: Deep indentations from the convex hull.

    Why it matters: More defects → more "fractured" or irregular. Great for detecting broken objects or natural debris.

🔹 11. Number of Corners

    Definition: Count of points in the polygonal approximation of the contour.

    Why it matters: Regular shapes (like boxes) have few corners. A high number often signals natural or deformed objects.

🔹 12. Fourier Descriptor (First Harmonic Magnitude)

    Definition: Strength of the first frequency component in the shape's Fourier transform.

    Why it matters: Low values → smooth, regular shapes. High values → irregular outlines.