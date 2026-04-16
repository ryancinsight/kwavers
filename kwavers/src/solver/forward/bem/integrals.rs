//! Numerical integration routines for BEM matrix assembly.
//!
//! Provides Gaussian quadrature for evaluating boundary integrals of the
//! Helmholtz Green's function over triangular elements. Three integration
//! strategies are used depending on the distance between the collocation
//! point and the element:
//!
//! - **Non-singular**: Standard 3-point Gaussian quadrature for well-separated elements.
//! - **Near-field**: Adaptive subdivision with 7-point quadrature for nearly-singular elements.
//! - **Singular**: Duffy transformation for self-elements (collocation point on element).
//!
//! ## References
//!
//! - Duffy, M.G. (1982). "Quadrature over a pyramid or cube of integrands
//!   with a singularity at a vertex." SIAM J. Numer. Anal. 19(6), 1260–1262.
//! - Sauter, S.A. & Schwab, C. (2011). *Boundary Element Methods*. Springer, §5.

use num_complex::Complex64;
use std::f64::consts::PI;

use super::geometry::{add, barycentric_coords, cross, norm_sq, scale, sub, triangle_area_normal};
use super::green::green_helmholtz;

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
            let d_g_dn =
                grad_g[0] * normal[0] + grad_g[1] * normal[1] + grad_g[2] * normal[2];
            let w = weight * area;
            for m in 0..3 {
                h_res[m] += d_g_dn * shape_fn[m] * w;
                g_res[m] += g_val * shape_fn[m] * w;
            }
        }
    }

    (h_res, g_res)
}

/// Compute boundary integrals for a non-singular (well-separated) element.
///
/// Uses 3-point Gaussian quadrature on the triangle. Suitable when the
/// collocation point is far from the element (distance >> element_size).
///
/// # Returns
///
/// `(h_contrib, g_contrib)` — contributions to H and G matrices for the 3 element nodes.
pub(crate) fn compute_nonsingular_integrals(
    k: f64,
    r_i: [f64; 3],
    element_nodes: [[f64; 3]; 3],
) -> ([Complex64; 3], [Complex64; 3]) {
    let p1 = element_nodes[0];
    let p2 = element_nodes[1];
    let p3 = element_nodes[2];

    let (area, normal) = triangle_area_normal(p1, p2, p3);

    let q_points = [
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

        let rx = shape_fn[0] * p1[0] + shape_fn[1] * p2[0] + shape_fn[2] * p3[0];
        let ry = shape_fn[0] * p1[1] + shape_fn[1] * p2[1] + shape_fn[2] * p3[1];
        let rz = shape_fn[0] * p1[2] + shape_fn[1] * p2[2] + shape_fn[2] * p3[2];
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

            let r_dist = u * (dir_x * dir_x + dir_y * dir_y + dir_z * dir_z).sqrt();

            // Jacobian: J = 2·Area·u
            let jac = 2.0 * area * u;

            let g_val = if r_dist < 1e-12 {
                // Limit u→0: G·J → 2·Area / (4π·|dir|)
                let dir_norm = (dir_x * dir_x + dir_y * dir_y + dir_z * dir_z).sqrt();
                Complex64::new(2.0 * area / (4.0 * PI * dir_norm), 0.0)
            } else {
                Complex64::new(0.0, k * r_dist).exp() / (4.0 * PI * r_dist) * jac
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
