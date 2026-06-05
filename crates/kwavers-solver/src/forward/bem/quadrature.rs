//! Gaussian quadrature rules for triangular surface elements.
//!
//! # Algorithm: Dunavant Triangle Quadrature
//!
//! **Theorem** (Dunavant 1985): A p-th degree quadrature rule for a triangle
//! integrates any polynomial of degree ≤ p exactly:
//! ```text
//!   ∫_T f dA ≈ A_T · Σᵢ wᵢ · f(λᵢ₁, λᵢ₂, λᵢ₃)
//! ```
//! where λ = (λ₁, λ₂, λ₃) are barycentric coordinates, wᵢ are weights
//! satisfying Σwᵢ = 1, and A_T is the triangle area.
//!
//! ## 3-Point Rule (Degree 2)
//!
//! Points at midpoints of edges: λ = (1/6, 1/6, 2/3) + permutations.
//! Weights: wᵢ = 1/3. Integrates polynomials up to degree 2 exactly.
//!
//! ## 7-Point Rule (Degree 5, Radon 1948 / Dunavant 1985 Table 1)
//!
//! Integrates polynomials up to degree 5. Points include the centroid and
//! symmetric permutations of two orbits.
//!
//! Coordinates and weights (Dunavant 1985, Table I, p=5):
//! ```text
//!   Centroid:   (1/3, 1/3, 1/3), w = 9/40 = 0.225
//!   6 points from orbit (a, b, b):
//!     a = 0.059715871789770, b = 0.470142064105115, w = 0.132394152788506 (×3)
//!     a = 0.797426985353087, b = 0.101286507323456, w = 0.125939180544827 (×3)
//! ```
//!
//! Verification: Σwᵢ = 0.225 + 3×0.132394 + 3×0.125939 = 1.000 ✓
//!
//! # References
//!
//! - Dunavant, D.A. (1985) High degree efficient symmetrical Gaussian
//!   quadrature rules for the triangle. *Int. J. Numer. Methods Eng.*
//!   21(6):1129–1148. doi:10.1002/nme.1620210612
//! - Radon, J. (1948) Zur mechanischen Kubatur. *Monatshefte für Mathematik*
//!   52(4):286–300.
//!
//! # Validation
//!
//! - `test_3pt_weights_sum`: Σwᵢ = 1 (consistency condition).
//! - `test_7pt_weights_sum`: Σwᵢ = 1.
//! - `test_3pt_constant`: integrates constant = area of unit triangle.
//! - `test_7pt_degree5_polynomial`: integrates degree-5 monomial exactly.

/// A single quadrature point in barycentric coordinates with weight.
#[derive(Debug, Clone, Copy)]
pub struct QuadPoint {
    /// Barycentric coordinates (λ₁, λ₂, λ₃), sum = 1
    pub bary: [f64; 3],
    /// Quadrature weight wᵢ (sum of all weights = 1)
    pub weight: f64,
}

/// 3-point Gaussian quadrature rule for triangles (degree 2, exact for polynomials ≤ 2).
///
/// Points located at midpoints of edges.
///
/// # Reference
/// Hammer, P.C., Marlowe, O.J. & Stroud, A.H. (1956) *Math. Tables Aids Comput.*
/// 10(56):130–137.
pub const TRIANGLE_3PT: [QuadPoint; 3] = [
    QuadPoint {
        bary: [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        weight: 1.0 / 3.0,
    },
    QuadPoint {
        bary: [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        weight: 1.0 / 3.0,
    },
    QuadPoint {
        bary: [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        weight: 1.0 / 3.0,
    },
];

/// 7-point Gaussian quadrature rule for triangles (degree 5, Dunavant 1985 Table I).
///
/// Integrates polynomials up to degree 5 exactly with 7 function evaluations.
pub const TRIANGLE_7PT: [QuadPoint; 7] = [
    // Centroid
    QuadPoint {
        bary: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        weight: 0.225,
    },
    // Orbit 1: a=0.059716, b=0.470142 (×3)
    QuadPoint {
        bary: [0.059715871789770, 0.470142064105115, 0.470142064105115],
        weight: 0.132394152788506,
    },
    QuadPoint {
        bary: [0.470142064105115, 0.059715871789770, 0.470142064105115],
        weight: 0.132394152788506,
    },
    QuadPoint {
        bary: [0.470142064105115, 0.470142064105115, 0.059715871789770],
        weight: 0.132394152788506,
    },
    // Orbit 2: a=0.797427, b=0.101287 (×3)
    QuadPoint {
        bary: [0.797426985353087, 0.101286507323456, 0.101286507323456],
        weight: 0.125939180544827,
    },
    QuadPoint {
        bary: [0.101286507323456, 0.797426985353087, 0.101286507323456],
        weight: 0.125939180544827,
    },
    QuadPoint {
        bary: [0.101286507323456, 0.101286507323456, 0.797426985353087],
        weight: 0.125939180544827,
    },
];

/// Evaluate a function at a quadrature point on a triangle defined by vertices.
///
/// Maps barycentric coordinates to Cartesian coordinates:
/// ```text
///   r = λ₁·p₁ + λ₂·p₂ + λ₃·p₃
/// ```
#[inline]
#[must_use]
pub fn map_to_triangle(bary: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> [f64; 3] {
    [
        bary[2].mul_add(p3[0], bary[0].mul_add(p1[0], bary[1] * p2[0])),
        bary[2].mul_add(p3[1], bary[0].mul_add(p1[1], bary[1] * p2[1])),
        bary[2].mul_add(p3[2], bary[0].mul_add(p1[2], bary[1] * p2[2])),
    ]
}

/// Compute the area and outward unit normal of a triangle defined by 3 vertices.
///
/// # Returns
///
/// `(area, normal)` where `area = |v₁ × v₂| / 2` and `normal = (v₁ × v₂) / |v₁ × v₂|`.
#[inline]
#[must_use]
pub fn triangle_area_normal(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> (f64, [f64; 3]) {
    let v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let v2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];

    let cx = v1[1].mul_add(v2[2], -(v1[2] * v2[1]));
    let cy = v1[2].mul_add(v2[0], -(v1[0] * v2[2]));
    let cz = v1[0].mul_add(v2[1], -(v1[1] * v2[0]));
    let norm = cz.mul_add(cz, cx.mul_add(cx, cy * cy)).sqrt();

    let area = 0.5 * norm;
    let normal = if norm > 1e-15 {
        [cx / norm, cy / norm, cz / norm]
    } else {
        [0.0, 0.0, 1.0]
    };

    (area, normal)
}

/// Integrate a scalar function over a triangle using the specified quadrature rule.
///
/// ```text
///   ∫_T f dA ≈ A_T · Σᵢ wᵢ · f(rᵢ)
/// ```
///
/// # Arguments
///
/// * `f` — function to integrate, takes Cartesian coordinates
/// * `p1, p2, p3` — triangle vertices
/// * `rule` — quadrature rule (e.g., `&TRIANGLE_7PT`)
#[must_use]
pub fn integrate_triangle<F>(
    f: F,
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
    rule: &[QuadPoint],
) -> f64
where
    F: Fn([f64; 3]) -> f64,
{
    let (area, _) = triangle_area_normal(p1, p2, p3);
    let sum: f64 = rule
        .iter()
        .map(|qp| {
            let r = map_to_triangle(qp.bary, p1, p2, p3);
            qp.weight * f(r)
        })
        .sum();
    area * sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Weight sum must equal 1 for both rules (consistency condition).
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_weights_sum_to_one() {
        let sum3: f64 = TRIANGLE_3PT.iter().map(|q| q.weight).sum();
        assert!((sum3 - 1.0).abs() < 1e-12, "3pt weights sum={:.6e}", sum3);

        let sum7: f64 = TRIANGLE_7PT.iter().map(|q| q.weight).sum();
        assert!((sum7 - 1.0).abs() < 1e-12, "7pt weights sum={:.6e}", sum7);
    }

    /// Barycentric coordinates sum to 1 for each quadrature point.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_barycentric_coordinates_sum_to_one() {
        for qp in TRIANGLE_3PT.iter().chain(TRIANGLE_7PT.iter()) {
            let s = qp.bary[0] + qp.bary[1] + qp.bary[2];
            assert!((s - 1.0).abs() < 1e-12, "bary sum={:.6e}", s);
        }
    }

    /// Integrating the constant function f=1 gives the triangle area.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_integrate_constant_equals_area() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0]; // right triangle, area = 0.5

        let area_3pt = integrate_triangle(|_| 1.0, p1, p2, p3, &TRIANGLE_3PT);
        let area_7pt = integrate_triangle(|_| 1.0, p1, p2, p3, &TRIANGLE_7PT);

        assert!((area_3pt - 0.5).abs() < 1e-12, "3pt area={:.6e}", area_3pt);
        assert!((area_7pt - 0.5).abs() < 1e-12, "7pt area={:.6e}", area_7pt);
    }

    /// 3-point rule integrates degree-2 monomials exactly.
    ///
    /// For a right triangle with vertices (0,0), (1,0), (0,1):
    /// ∫∫ x² dA = 1/12.
    /// # Panics
    /// - Panics if assertion fails: `3pt ∫x² error: rel={:.3e}`.
    ///
    #[test]
    fn test_3pt_degree2_exact() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];

        let result = integrate_triangle(|r| r[0] * r[0], p1, p2, p3, &TRIANGLE_3PT);
        let exact = 1.0 / 12.0;
        let rel = (result - exact).abs() / exact;
        assert!(rel < 1e-12, "3pt ∫x² error: rel={:.3e}", rel);
    }

    /// 7-point rule integrates degree-5 monomials exactly.
    ///
    /// For the unit right triangle: ∫∫ x⁵ dA = 1/252.
    /// # Panics
    /// - Panics if assertion fails: `7pt ∫x⁵ error: rel={:.3e}`.
    ///
    #[test]
    fn test_7pt_degree5_exact() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];

        // ∫₀¹ ∫₀^{1-x} x⁵ dy dx = ∫₀¹ x⁵(1-x) dx = 1/6 - 1/7 = 1/42
        let result = integrate_triangle(|r| r[0].powi(5), p1, p2, p3, &TRIANGLE_7PT);
        let exact = 1.0 / 42.0;
        let rel = (result - exact).abs() / exact;
        assert!(rel < 1e-10, "7pt ∫x⁵ error: rel={:.3e}", rel);
    }

    /// 3-point rule is INSUFFICIENT for degree-5: test shows error > 1%.
    ///
    /// This confirms that higher-order quadrature is needed for smooth but
    /// oscillatory integrands (e.g., BEM Green's function products).
    /// # Panics
    /// - Panics if assertion fails: `3pt rule should have >0.1% error for degree-5: err={:.3e}`.
    ///
    #[test]
    fn test_3pt_insufficient_for_degree5() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];

        let result_3pt = integrate_triangle(|r| r[0].powi(5), p1, p2, p3, &TRIANGLE_3PT);
        let exact = 1.0 / 42.0;
        let err_3pt = (result_3pt - exact).abs() / exact;
        // 3-point rule cannot integrate degree-5 exactly → error must be > 0.1%
        assert!(
            err_3pt > 0.001,
            "3pt rule should have >0.1% error for degree-5: err={:.3e}",
            err_3pt
        );
    }

    /// Verify triangle_area_normal for a known equilateral triangle.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_triangle_area_normal() {
        // Equilateral triangle of side 1 in z=0 plane, area = sqrt(3)/4
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.5, 3.0_f64.sqrt() / 2.0, 0.0];
        let (area, normal) = triangle_area_normal(p1, p2, p3);
        let expected_area = 3.0_f64.sqrt() / 4.0;
        assert!((area - expected_area).abs() < 1e-12, "area={:.6e}", area);
        // Normal should be (0, 0, 1)
        assert!(
            normal[2].abs() > 0.999,
            "normal z component: {:.6e}",
            normal[2]
        );
    }
}
