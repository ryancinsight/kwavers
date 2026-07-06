// Type-safe 3D vector operations for computational physics.
//
// # Design Rationale
//
// This module provides a single source of truth (SSOT) for 3D vector arithmetic
// used throughout the codebase. Previously, `dot`, `sub`, `add`, `scale`,
// `cross`, and `norm` were duplicated across 12+ files (BEM solver, FEM assembly,
// geometry processing, GPU kernels, etc.), leading to:
//
// - Inconsistent implementations (e.g., different normalization strategies)
// - Maintenance burden when fixing bugs or adding optimizations
// - No SIMD acceleration for vector operations
//
// # Theorem (Vector Space Axioms)
//
// All operations satisfy the vector space axioms over ℝ³:
// - **Closure**: ∀a,b ∈ ℝ³: a+b ∈ ℝ³, a·b ∈ ℝ
// - **Commutativity**: a+b = b+a
// - **Associativity**: (a+b)+c = a+(b+c)
// - **Distributivity**: α(a+b) = αa + αb
// - **Identity**: a+0 = a, 1·a = a
//
// # Performance
//
// All functions are `#[inline(always)]` to enable auto-vectorization by LLVM.
// For batch operations, use the SIMD-accelerated versions in `simd_safe`.
//
// # Memory Layout
//
// Vectors use `[f64; 3]` (SoA-friendly) rather than a struct to ensure:
// - Cache-friendly contiguous storage in `Vec<[f64; 3]>`
// - Direct compatibility with geometry and backend buffer newtypes
// - Direct use in FFI/GPU buffer layouts

#[cfg(test)]
mod tests;

/// Compute the dot product of two 3D vectors.
///
/// ```text
/// a · b = a₀b₀ + a₁b₁ + a₂b₂
/// ```
///
/// # Complexity: O(1), 3 FLOPs (2 mul + 1 add)
#[inline(always)]
#[must_use]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the cross product of two 3D vectors.
///
/// ```text
/// a × b = [a₁b₂ - a₂b₁, a₂b₀ - a₀b₂, a₀b₁ - a₁b₀]
/// ```
///
/// # Theorem (Cross Product Properties)
/// - **Anticommutativity**: a × b = -(b × a)
/// - **Orthogonality**: (a × b) · a = 0 and (a × b) · b = 0
/// - **Magnitude**: |a × b| = |a|·|b|·sin(θ) = area of parallelogram
///
/// # Complexity: O(1), 9 FLOPs (6 mul + 3 sub)
#[inline(always)]
#[must_use]
pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Subtract two 3D vectors: a - b.
///
/// # Complexity: O(1), 3 FLOPs
#[inline(always)]
#[must_use]
pub fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Add two 3D vectors: a + b.
///
/// # Complexity: O(1), 3 FLOPs
#[inline(always)]
#[must_use]
pub fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scale a 3D vector by a scalar: s · a.
///
/// # Complexity: O(1), 3 FLOPs
#[inline(always)]
#[must_use]
pub fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Compute the squared Euclidean norm of a 3D vector.
///
/// ```text
/// |a|² = a₀² + a₁² + a₂²
/// ```
///
/// # Theorem (Squared Norm)
/// Using squared norm avoids the expensive `sqrt` call when only comparing
/// magnitudes (e.g., distance comparisons). This is a standard optimization
/// in computational geometry.
///
/// # Complexity: O(1), 5 FLOPs (3 mul + 2 add)
#[inline(always)]
#[must_use]
pub fn norm_sq3(a: [f64; 3]) -> f64 {
    dot3(a, a)
}

/// Compute the Euclidean norm (magnitude) of a 3D vector.
///
/// ```text
/// |a| = sqrt(a₀² + a₁² + a₂²)
/// ```
///
/// # Complexity: O(1), 5 FLOPs + 1 sqrt
#[inline(always)]
#[must_use]
pub fn norm3(a: [f64; 3]) -> f64 {
    norm_sq3(a).sqrt()
}

/// Compute the distance between two 3D points.
///
/// ```text
/// d(a, b) = |a - b|
/// ```
///
/// # Complexity: O(1), 8 FLOPs + 1 sqrt
#[inline(always)]
#[must_use]
pub fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm3(sub3(a, b))
}

/// Compute the squared distance between two 3D points.
///
/// Use this instead of `distance3` when only comparing distances,
/// as it avoids the expensive `sqrt` call.
///
/// # Complexity: O(1), 8 FLOPs
#[inline(always)]
#[must_use]
pub fn distance_sq3(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm_sq3(sub3(a, b))
}

/// Normalize a 3D vector to unit length.
///
/// Returns `None` if the vector has zero magnitude.
///
/// ```text
/// normalize(a) = a / |a|
/// ```
///
/// # Complexity: O(1), 7 FLOPs + 1 sqrt + 1 div
#[inline(always)]
#[must_use]
pub fn normalize3(a: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm3(a);
    if n < 1e-300 {
        None
    } else {
        let inv_n = 1.0 / n;
        Some(scale3(a, inv_n))
    }
}

/// Normalize a 3D vector, returning a fallback if zero-length.
///
/// This is useful in geometric computations where a degenerate
/// vector (e.g., from coincident points) should not cause failure.
#[inline(always)]
#[must_use]
pub fn normalize3_or(a: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    normalize3(a).unwrap_or(fallback)
}

/// Compute the angle between two 3D vectors in radians.
///
/// ```text
/// θ = arccos((a · b) / (|a| · |b|))
/// ```
///
/// Returns `None` if either vector is zero-length.
///
/// # Theorem (Angle Formula)
/// Follows from the geometric definition of the dot product:
/// a · b = |a|·|b|·cos(θ)
///
/// # Complexity: O(1), 11 FLOPs + 2 sqrt + 1 acos
#[inline(always)]
#[must_use]
pub fn angle_between3(a: [f64; 3], b: [f64; 3]) -> Option<f64> {
    let na = norm3(a);
    let nb = norm3(b);
    if na < 1e-300 || nb < 1e-300 {
        return None;
    }
    let cos_theta = (dot3(a, b) / (na * nb)).clamp(-1.0, 1.0);
    Some(cos_theta.acos())
}

/// Linearly interpolate between two 3D vectors.
///
/// ```text
/// lerp(a, b, t) = (1-t)·a + t·b
/// ```
///
/// # Properties:
/// - t = 0 → a
/// - t = 1 → b
/// - t = 0.5 → midpoint
///
/// # Complexity: O(1), 7 FLOPs
#[inline(always)]
#[must_use]
pub fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    let mt = 1.0 - t;
    [
        mt * a[0] + t * b[0],
        mt * a[1] + t * b[1],
        mt * a[2] + t * b[2],
    ]
}

/// Compute the area of a triangle given its three vertices.
///
/// Uses the cross product formula:
/// ```text
/// Area = ½ · |(p₂-p₁) × (p₃-p₁)|
/// ```
///
/// # Theorem (Triangle Area via Cross Product)
/// The magnitude of the cross product of two edge vectors equals twice
/// the triangle area. This is equivalent to Heron's formula but more
/// numerically stable for thin triangles.
///
/// # Complexity: O(1), 18 FLOPs + 1 sqrt
#[inline(always)]
#[must_use]
pub fn triangle_area3(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let e1 = sub3(p2, p1);
    let e2 = sub3(p3, p1);
    0.5 * norm3(cross3(e1, e2))
}

/// Compute the characteristic length of a triangle (maximum edge length).
///
/// This is used in BEM/FEM mesh quality assessment and quadrature
/// depth selection.
///
/// # Complexity: O(1), 9 FLOPs + 3 sqrt
#[inline(always)]
#[must_use]
pub fn triangle_characteristic_length(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let d12 = distance3(p1, p2);
    let d23 = distance3(p2, p3);
    let d31 = distance3(p3, p1);
    d12.max(d23).max(d31)
}

/// Compute the centroid of a triangle.
///
/// ```text
/// centroid = (p₁ + p₂ + p₃) / 3
/// ```
///
/// # Complexity: O(1), 6 FLOPs
#[inline(always)]
#[must_use]
pub fn triangle_centroid3(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> [f64; 3] {
    [
        (p1[0] + p2[0] + p3[0]) / 3.0,
        (p1[1] + p2[1] + p3[1]) / 3.0,
        (p1[2] + p2[2] + p3[2]) / 3.0,
    ]
}

/// Compute barycentric coordinates of point p with respect to triangle (p1, p2, p3).
///
/// Returns (u, v, w) such that p = u·p1 + v·p2 + w·p3 and u+v+w = 1.
///
/// # Theorem (Barycentric Coordinates)
/// For a point p in the plane of triangle (p1, p2, p3), the barycentric
/// coordinates are uniquely determined by solving:
/// ```text
/// p = p1 + v·(p2-p1) + w·(p3-p1)
/// u = 1 - v - w
/// ```
///
/// The point is inside the triangle iff u, v, w ∈ [0, 1].
///
/// # Returns
/// `(u, v, w)` or `(1, 0, 0)` if the triangle is degenerate.
///
/// # Complexity: O(1), ~20 FLOPs
#[inline(always)]
#[must_use]
pub fn barycentric3(p: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> (f64, f64, f64) {
    let v0 = sub3(p2, p1);
    let v1 = sub3(p3, p1);
    let v2 = sub3(p, p1);

    let dot00 = dot3(v0, v0);
    let dot01 = dot3(v0, v1);
    let dot02 = dot3(v0, v2);
    let dot11 = dot3(v1, v1);
    let dot12 = dot3(v1, v2);

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < 1e-18 {
        return (1.0, 0.0, 0.0);
    }

    let inv_denom = 1.0 / denom;
    let v = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let w = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    let u = 1.0 - v - w;

    (u, v, w)
}

/// Fused multiply-add for 3D vectors: a · s + b.
///
/// More efficient than separate scale and add operations.
///
/// # Complexity: O(1), 6 FLOPs (vs 9 for separate operations)
#[inline(always)]
#[must_use]
pub fn fma3(a: [f64; 3], s: f64, b: [f64; 3]) -> [f64; 3] {
    [a[0] * s + b[0], a[1] * s + b[1], a[2] * s + b[2]]
}

/// Negate a 3D vector: -a.
///
/// # Complexity: O(1), 3 FLOPs
#[inline(always)]
#[must_use]
pub fn neg3(a: [f64; 3]) -> [f64; 3] {
    [-a[0], -a[1], -a[2]]
}
