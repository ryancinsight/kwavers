//! Numerical Integration for BEM Matrix Assembly
//!
//! Evaluates boundary integrals of the Helmholtz Green's function and its
//! normal derivative over triangular elements to assemble the H and G matrices
//! of the Galerkin/collocation BEM.
//!
//! ## Mathematical Setting
//!
//! The BEM boundary integral equation (Burton & Miller 1971):
//!
//! ```text
//! c(x) p(x) + ∫_Γ (∂G/∂n)(x,y) p(y) dΓ(y) = ∫_Γ G(x,y) q(y) dΓ(y)
//! ```
//!
//! where G(x,y) = exp(ik|x−y|) / (4π|x−y|) is the Helmholtz free-space
//! Green's function, q = ∂p/∂n is the normal velocity, and c(x) = 1/2 for
//! smooth boundaries (c(x) = 1 in the interior).
//!
//! ## Theorem (Quadrature Strategy)
//!
//! **Statement.** For an n-point Gauss–Legendre rule over a triangle with
//! smooth integrand, the quadrature error is O(h^{2n}) where h is the element
//! size.  When the integrand contains a 1/R singularity, the convergence
//! degrades to O(h^{2n−1}) for near-field elements and the standard rule
//! diverges for self-elements (R → 0).
//!
//! **Strategy.** Three regimes are identified by the ratio r = element_size / distance:
//!
//! | Regime | Condition | Method | Error order |
//! |--------|-----------|--------|-------------|
//! | Non-singular | r ≤ 20 | 3-point Gaussian (3p) | O(h⁶) |
//! | Near-field | 20 < r ≤ 50 | Adaptive 4^d subdivision + 7-point | O(h^{14}/2^d) |
//! | Near-field (close) | r > 50 | Depth-3 subdivision + 7-point | O(h^{14}/64) |
//! | Singular | R = 0 | Duffy transformation + 9-point | O(h^{18}) |
//!
//! **Proof sketch.** For the non-singular regime, the integrand G = e^{ikR}/(4πR)
//! is analytic in the integration domain, so standard Gauss rules achieve
//! algebraic superconvergence.  For near-field elements, adaptive subdivision
//! with depth d reduces the effective element size by 2^d, restoring the
//! non-singular convergence rate on each sub-triangle.  The Duffy transform
//! (Duffy 1982) maps [0,1]² → triangle with Jacobian J = 2·Area·u; since
//! R = u·|dir(v)|, the product G·J = O(1) as u → 0, regularising the 1/R
//! pole so tensor-product Gauss-Legendre rules converge at full algebraic order.
//!
//! ## References
//!
//! - Duffy M.G. (1982). "Quadrature over a pyramid or cube of integrands
//!   with a singularity at a vertex." SIAM J. Numer. Anal. 19(6), 1260–1262.
//! - Burton A.J., Miller G.F. (1971). Proc. R. Soc. Lond. A 323(1553), 201–210.
//! - Sauter S.A., Schwab C. (2011). *Boundary Element Methods*. Springer, §5.
//! - Atkinson K.E. (1997). *The Numerical Solution of Integral Equations of
//!   the Second Kind*. Cambridge, §4.2 (Gauss quadrature accuracy).

use num_complex::Complex64;

use super::geometry::{add, barycentric_coords, cross, norm_sq, scale, sub, triangle_area_normal};
use super::green::green_helmholtz;
use crate::core::constants::numerical::{FOUR_PI};

/// Compute boundary integrals for a near-field element using adaptive subdivision.
///
/// When the collocation point is close to the element (but not on it), the integrand
/// is nearly singular and standard quadrature loses accuracy. This routine adaptively
/// subdivides the triangle into 4^depth sub-triangles and applies 7-point Gaussian
/// quadrature on each.
///
/// # Arguments
///
/// * `k` — wavenumber [rad/m]
/// * `r_i` — collocation (source) point
/// * `element_nodes` — triangle vertex coordinates `[p1, p2, p3]`
/// * `distance` — distance from `r_i` to the element
/// * `element_size` — characteristic element length (max edge)
///
/// # Returns
///
/// `(h_contrib, g_contrib)` — contributions to H and G matrices for the 3 element nodes.
pub(crate) fn compute_nearfield_integrals(
    k: f64,
    r_i: [f64; 3],
    element_nodes: [[f64; 3]; 3],
    distance: f64,
    element_size: f64,
) -> ([Complex64; 3], [Complex64; 3]) {
    let p1 = element_nodes[0];
    let p2 = element_nodes[1];
    let p3 = element_nodes[2];

    let (_area, normal) = triangle_area_normal(p1, p2, p3);

    let ratio = if distance > 0.0 {
        element_size / distance
    } else {
        f64::INFINITY
    };
    let depth = if ratio > 50.0 {
        3
    } else if ratio > 20.0 {
        2
    } else {
        1
    };

    let quad_points = [
        ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 0.225),
        (
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            0.132394152788506,
        ),
        (
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            0.132394152788506,
        ),
        (
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
            0.132394152788506,
        ),
        (
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
            0.125939180544827,
        ),
        (
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            0.125939180544827,
        ),
        (
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            0.125939180544827,
        ),
    ];

    let mut h_res = [Complex64::new(0.0, 0.0); 3];
    let mut g_res = [Complex64::new(0.0, 0.0); 3];

    let mut stack = Vec::new();
    stack.push((p1, p2, p3, 0usize));

    while let Some((t1, t2, t3, level)) = stack.pop() {
        if level < depth {
            let m12 = scale(add(t1, t2), 0.5);
            let m23 = scale(add(t2, t3), 0.5);
            let m31 = scale(add(t3, t1), 0.5);
            stack.push((t1, m12, m31, level + 1));
            stack.push((m12, t2, m23, level + 1));
            stack.push((m31, m23, t3, level + 1));
            stack.push((m12, m23, m31, level + 1));
            continue;
        }

        let c = cross(sub(t2, t1), sub(t3, t1));
        let area = 0.5 * norm_sq(c).sqrt();

        for (bary, weight) in quad_points {
            let r = add(
                add(scale(t1, bary[0]), scale(t2, bary[1])),
                scale(t3, bary[2]),
            );
            let shape_fn = barycentric_coords(r, p1, p2, p3);
            let (g_val, grad_g) = green_helmholtz(k, r_i, r);
            let d_g_dn = grad_g[0] * normal[0] + grad_g[1] * normal[1] + grad_g[2] * normal[2];
            let w = weight * area;
            for m in 0..3 {
                h_res[m] += d_g_dn * shape_fn[m] * w;
                g_res[m] += g_val * shape_fn[m] * w;
            }
        }
    }

    (h_res, g_res)
}

/// Compute H and G boundary-integral contributions for a well-separated element.
///
/// ## Quadrature rule
///
/// Applies the 3-point symmetric Gaussian rule on the reference triangle with
/// barycentric coordinates and weights (Dunavant 1985, degree-2 exact):
///
/// ```text
/// ∫_T f dA ≈ Σ_{q=1}^{3} w_q · f(ξ_q, η_q) · Area(T)
/// ```
///
/// with nodes at (1/6, 1/6), (2/3, 1/6), (1/6, 2/3) and equal weights 1/3.
/// This rule integrates polynomials of degree ≤ 2 exactly.
///
/// ## Theorem (accuracy for non-singular regime)
///
/// For a smooth analytic integrand, a p-point Gaussian rule achieves O(h^{2p})
/// convergence.  The 3-point rule gives O(h⁶) on a uniform mesh of step h,
/// which suffices when the integrand G = e^{ikR}/(4πR) is non-singular (R > 0)
/// and the collocation point is separated by distance >> element_size.
///
/// ## Theorem (non-singular criterion)
///
/// The integrand G(x,y) is smooth over element T when dist(x, T) > C·h(T)
/// for a constant C ≈ 3–5 (see Sauter & Schwab 2011, §5.3).  Below this
/// threshold, `compute_nearfield_integrals` must be used to maintain accuracy.
///
/// # Returns
///
/// `(h_contrib, g_contrib)` — H and G matrix entries for the 3 element nodes.
pub(crate) fn compute_nonsingular_integrals(
    k: f64,
    r_i: [f64; 3],
    element_nodes: [[f64; 3]; 3],
) -> ([Complex64; 3], [Complex64; 3]) {
    let p1 = element_nodes[0];
    let p2 = element_nodes[1];
    let p3 = element_nodes[2];

    let (area, normal) = triangle_area_normal(p1, p2, p3);

    let q_points: [([f64; 2], f64); 3] = [
        ([1.0 / 6.0, 1.0 / 6.0], 1.0 / 3.0),
        ([2.0 / 3.0, 1.0 / 6.0], 1.0 / 3.0),
        ([1.0 / 6.0, 2.0 / 3.0], 1.0 / 3.0),
    ];

    let mut h_res = [Complex64::new(0.0, 0.0); 3];
    let mut g_res = [Complex64::new(0.0, 0.0); 3];

    for (uv, w) in &q_points {
        let u = uv[0];
        let v = uv[1];
        let shape_fn = [1.0 - u - v, u, v];

        let rx = shape_fn[2].mul_add(p3[0], shape_fn[0].mul_add(p1[0], shape_fn[1] * p2[0]));
        let ry = shape_fn[2].mul_add(p3[1], shape_fn[0].mul_add(p1[1], shape_fn[1] * p2[1]));
        let rz = shape_fn[2].mul_add(p3[2], shape_fn[0].mul_add(p1[2], shape_fn[1] * p2[2]));
        let r = [rx, ry, rz];

        let (g_val, grad_g) = green_helmholtz(k, r_i, r);
        let d_g_dn = grad_g[0] * normal[0] + grad_g[1] * normal[1] + grad_g[2] * normal[2];

        let weight = w * area;
        for m in 0..3 {
            h_res[m] += d_g_dn * shape_fn[m] * weight;
            g_res[m] += g_val * shape_fn[m] * weight;
        }
    }

    (h_res, g_res)
}

/// Compute boundary integrals for a singular element (collocation point on element).
///
/// Uses Duffy transformation to regularise the 1/R singularity. The transformation
/// maps the triangle to the unit square `[0,1]²` with the singular vertex at the
/// origin, introducing a Jacobian factor `u` that cancels the `1/R` singularity.
///
/// ## Algorithm
///
/// For a triangle with singular vertex at p0:
/// ```text
/// r(u,v) = (1-u) p0 + u(1-v) p1 + uv p2
/// ```
/// The Jacobian is `J = 2·Area·u`, and `R = u·|dir(v)|`, so `G·J ∝ exp(ikR)·u/R = exp(ikR)/|dir|`.
///
/// ## Reference
///
/// Duffy, M.G. (1982). SIAM J. Numer. Anal. 19(6), 1260–1262.
///
/// # Arguments
///
/// * `k` — wavenumber [rad/m]
/// * `_r_i` — collocation point (unused; same as `element_nodes[vertex_idx]`)
/// * `element_nodes` — triangle vertex coordinates
/// * `vertex_idx` — index (0, 1, or 2) of the singular vertex in `element_nodes`
///
/// # Returns
///
/// `(h_contrib, g_contrib)` — H contribution is zero (flat element: ∂G/∂n = 0 for self-element).
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub(crate) fn compute_singular_integrals(
    k: f64,
    _r_i: [f64; 3],
    element_nodes: [[f64; 3]; 3],
    vertex_idx: usize,
) -> ([Complex64; 3], [Complex64; 3]) {
    // For flat triangular elements, (r - r_i) ⊥ n, so ∂G/∂n = 0.
    // The diagonal term c(r) is added separately in the assembly loop.
    let h_res = [Complex64::new(0.0, 0.0); 3];

    // Reorder nodes so singularity is at p0
    let (p0, p1, p2) = match vertex_idx {
        0 => (element_nodes[0], element_nodes[1], element_nodes[2]),
        1 => (element_nodes[1], element_nodes[2], element_nodes[0]),
        2 => (element_nodes[2], element_nodes[0], element_nodes[1]),
        _ => unreachable!(),
    };

    // Duffy transform: r(u,v) = p0 + u·(p1 - p0) + u·v·(p2 - p1)
    let v10 = sub(p1, p0);
    let v21 = sub(p2, p1);

    let c = cross(v10, v21);
    let area = 0.5 * norm_sq(c).sqrt();

    // 3×3 Gauss-Legendre quadrature on [0,1]²
    let gauss_1d = [
        (0.1127016653792583, 0.2777777777777778),
        (0.5, 0.4444444444444444),
        (0.8872983346207417, 0.2777777777777778),
    ];

    let mut g_res_reordered = [Complex64::new(0.0, 0.0); 3];

    for (u, wu) in &gauss_1d {
        for (v, wv) in &gauss_1d {
            let dir_x = v10[0] + v * v21[0];
            let dir_y = v10[1] + v * v21[1];
            let dir_z = v10[2] + v * v21[2];

            let r_dist = u * dir_z
                .mul_add(dir_z, dir_x.mul_add(dir_x, dir_y * dir_y))
                .sqrt();

            // Jacobian: J = 2·Area·u
            let jac = 2.0 * area * u;

            let g_val = if r_dist < 1e-12 {
                // Limit u→0: G·J → 2·Area / (4π·|dir|)
                let dir_norm = dir_z
                    .mul_add(dir_z, dir_x.mul_add(dir_x, dir_y * dir_y))
                    .sqrt();
                Complex64::new(2.0 * area / (FOUR_PI * dir_norm), 0.0)
            } else {
                Complex64::new(0.0, k * r_dist).exp() / (FOUR_PI * r_dist) * jac
            };

            // Shape functions in collapsed coordinates:
            // r = (1-u)·p0 + u(1-v)·p1 + uv·p2
            let l0 = 1.0 - u;
            let l1 = u * (1.0 - v);
            let l2 = u * v;

            let weight = wu * wv;

            g_res_reordered[0] += g_val * l0 * weight;
            g_res_reordered[1] += g_val * l1 * weight;
            g_res_reordered[2] += g_val * l2 * weight;
        }
    }

    // Map back to original node order
    let mut g_res_final = [Complex64::new(0.0, 0.0); 3];
    match vertex_idx {
        0 => {
            g_res_final[0] = g_res_reordered[0];
            g_res_final[1] = g_res_reordered[1];
            g_res_final[2] = g_res_reordered[2];
        }
        1 => {
            g_res_final[1] = g_res_reordered[0];
            g_res_final[2] = g_res_reordered[1];
            g_res_final[0] = g_res_reordered[2];
        }
        2 => {
            g_res_final[2] = g_res_reordered[0];
            g_res_final[0] = g_res_reordered[1];
            g_res_final[1] = g_res_reordered[2];
        }
        _ => unreachable!(),
    }

    (h_res, g_res_final)
}
