# 🧠 ContourIQ

Smart Shape Analysis from Segmented Images

ContourIQ is a modular, AI-assisted pipeline that analyzes object contours from segmented images to extract meaningful insights — including object type classification (e.g. man-made vs. natural) and structural features like fractures, shape complexity, and geometry. Built for flexibility, precision, and scalability.


# ContourIQ: Key Contour Features Explained

ContourIQ uses a variety of scientifically grounded shape features to classify and understand object contours extracted from segmented images. Here's a breakdown of the key features and why each matters in real-world object classification.

---

## 🔹 1. Area
- **Definition:** Total number of pixels inside the contour.
- **Why it matters:** Provides object size. Very small areas might be noise; large ones are typically full objects.

---

## 🔹 2. Perimeter
- **Definition:** Total length of the contour's boundary.
- **Why it matters:** Used alongside area to assess shape smoothness. Jagged or complex shapes have longer perimeters.

---

## 🔹 3. Circularity
- **Formula:** $\frac{4 \cdot \pi \cdot \text{Area}}{\text{Perimeter}^2}$
- **Range:** [0, 1], with 1 representing a perfect circle.
- **Why it matters:** Distinguishes round objects (e.g., wheels, balls) from angular or irregular ones.

---

## 🔹 4. Aspect Ratio
- **Formula:** $\frac{\text{Width}}{\text{Height}}$ of the bounding box.
- **Why it matters:** High ratios indicate elongated objects like sticks or rods.

---

## 🔹 5. Extent
- **Formula:** $\frac{\text{Area}}{\text{Bounding Box Area}}$
- **Why it matters:** Indicates how fully the object occupies its bounding box. Low extent often points to sparse or odd shapes.

---

## 🔹 6. Solidity
- **Formula:** $\frac{\text{Area}}{\text{Convex Hull Area}}$
- **Why it matters:** Measures convexity. Lower values suggest more indentations or hollow regions—useful for detecting fractures.

---

## 🔹 7. Eccentricity
- **Formula:** $\sqrt{1 - (b/a)^2}$ where a = major axis and b = minor axis
- **Range:** 0 (circle) to 1 (line)
- **Why it matters:** High eccentricity values indicate very elongated objects—like pipes or beams.

---

## 🔹 8. Skeleton Length
- **Definition:** Total number of pixels in the skeletonized version of the object.
- **Why it matters:** A proxy for shape complexity or length. Long skeletons are characteristic of wires, cords, or branches.

---

## 🔹 9. Hu Moments (1–7)
- **Definition:** A set of seven invariant shape descriptors.
- **Why it matters:** Remain stable under scaling, translation, and rotation—ideal for comparing and matching complex shapes.

---

## 🔹 10. Convexity Defects
- **Definition:** Points where the contour deviates inward from its convex hull.
- **Why it matters:** More defects suggest irregular or broken surfaces—important for fracture detection.

---

## 🔹 11. Number of Corners
- **Definition:** Number of points in a simplified polygonal contour.
- **Why it matters:** Helps distinguish regular (e.g., square) from irregular or organic shapes.

---

## 🔹 12. Fourier Descriptor (First Harmonic Magnitude)
- **Definition:** Strength of the first frequency component in the Fourier transform of the contour.
- **Why it matters:** A low magnitude typically indicates a regular, smooth outline. High values imply jagged or noisy shapes.

---

These features form the foundation of ContourIQ's shape intelligence pipeline, powering object classification from raw segmentation data with precision and interpretability.

---

# 🧠 Contour-Based Attribute Analysis in ContourIQ

Beyond raw geometric features, ContourIQ intelligently derives higher-level **shape-based attributes** using heuristic rules. These attributes help interpret the **functional identity** or **structural nature** of each object.

## 🔍 Defined Attributes & Logic

### ✅ `is_man_made`
Detects regularity and simplicity in shape — typical of manufactured objects.
- `solidity > 0.85`
- `num_corners` in [3, 4, 6, 8]  
  **OR**
- `eccentricity > 0.95` and `skeleton_length > 300`

---

### ✅ `fracture_detected`
Flags objects with jagged or broken outlines.
- `num_defects > 5`  
- `num_corners > 10`  
- `solidity < 0.85`

---

### ✅ `long_object`
Catches elongated shapes like pipes or rods.
- `eccentricity > 0.95`
- `aspect_ratio > 3` or `skeleton_length > 300`
- `circularity < 0.4`

---

### ✅ `round_object`
Detects circular rigid bodies like wheels, balls.
- `circularity > 0.85`
- `eccentricity < 0.6`
- `solidity > 0.9`

---

### ✅ `compact_object`
Represents box-like, filled-in shapes.
- `solidity > 0.95`
- `extent > 0.8`
- `num_corners` in [4, 6]

---

### ✅ `long_skeleton`
Indicates fine, thread-like structures.
- `skeleton_length > 300`
- `area / skeleton_length < 3`

---

### ✅ `rigid_object`
Flags shape-consistent, solid, manufactured forms.
- `solidity > 0.8`, `extent > 0.5`, and `num_defects < 6`  
  **OR**
- `eccentricity > 0.98`, `long_object = True`, `skeleton_length > 300`


# 📋 Attribute Analysis Rationale

| Attribute         | Condition(s) | Rationale |
|--------------------|--------------|-----------|
| **is_man_made** | solidity > 0.85 and num_corners in [3,4,6,8] OR eccentricity > 0.95 and skeleton_length > 300 | Man-made objects are typically geometrically regular or long and uniform in structure like pipes or rods. |
| **fracture_detected** | num_defects > 5 OR num_corners > 10 OR solidity < 0.85 | High irregularity or jaggedness indicates potential cracks or deformities typical of fractured objects. |
| **long_object** | eccentricity > 0.95 AND (aspect_ratio > 3 OR skeleton_length > 300) AND circularity < 0.4 | Long objects are thin, stretched, and not round, e.g., wires, sticks, pipes. |
| **round_object** | circularity > 0.85 AND eccentricity < 0.6 AND solidity > 0.9 | Round, filled objects like wheels or balls are highly regular with few concavities. |
| **compact_object** | solidity > 0.95 AND extent > 0.8 AND num_corners in [4,6] | Compact and dense shapes like boxes or bricks fill space efficiently with defined edges. |
| **long_skeleton** | skeleton_length > 300 AND area / skeleton_length < 3 | Very long skeletons with little area indicate threadlike structures such as cables or roots. |
| **rigid_object** | (solidity > 0.8 AND extent > 0.5 AND num_defects < 6) OR (eccentricity > 0.98 AND long_object AND skeleton_length > 300) | Rigid objects are structurally coherent and resist deformation — either compact or uniform like pipes. |