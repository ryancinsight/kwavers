//! Post-processing utilities for BEM solutions.
//!
//! Provides vertex normal computation and incident wave field generation
//! for boundary element computations.

use num_complex::Complex64;

/// Solution of BEM system.
#[derive(Debug, Clone)]
pub struct BemSolution {
    /// Pressure on boundary nodes
    pub boundary_pressure: ndarray::Array1<Complex64>,
    /// Normal velocity on boundary nodes
    pub boundary_velocity: ndarray::Array1<Complex64>,
    /// Wavenumber used in solution
    pub wavenumber: f64,
}

/// Compute area-weighted vertex normals from a triangulated surface.
///
/// For each vertex $v_i$, the normal is the area-weighted average of all incident
/// triangle normals:
///
/// $\hat{n}_i = \frac{\sum_{t \ni v_i} A_t \hat{n}_t}{\left\|\sum_{t \ni v_i} A_t \hat{n}_t\right\|}$
///
/// where $A_t = \tfrac{1}{2}\|(\mathbf{p}_1 - \mathbf{p}_0) \times (\mathbf{p}_2 - \mathbf{p}_0)\|$
/// is the triangle area and $\hat{n}_t$ is the unit outward normal.
///
/// # Reference
///
/// Max, N.L. (1999). "Weights for computing vertex normals from facet normals."
/// *Journal of Graphics Tools*, 4(2), 1–6.
pub fn compute_vertex_normals(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> Vec<[f64; 3]> {
    let nv = vertices.len();
    let mut normals = vec![[0.0f64; 3]; nv];

    for tri in triangles {
        let p0 = vertices[tri[0]];
        let p1 = vertices[tri[1]];
        let p2 = vertices[tri[2]];

        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        // Cross product e1 × e2 (magnitude = 2 × triangle area)
        let cx = e1[1] * e2[2] - e1[2] * e2[1];
        let cy = e1[2] * e2[0] - e1[0] * e2[2];
        let cz = e1[0] * e2[1] - e1[1] * e2[0];

        for &vi in tri {
            normals[vi][0] += cx;
            normals[vi][1] += cy;
            normals[vi][2] += cz;
        }
    }

    // Normalise
    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-15 {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        }
    }

    normals
}

/// Compute incident plane-wave pressure and its normal derivative at boundary vertices.
///
/// For a plane wave $p_{\text{inc}}(\mathbf{x}) = A e^{i k \hat{d} \cdot \mathbf{x}}$:
///
/// $p_{\text{inc}}(\mathbf{x}_i) = A \exp(i k \hat{d} \cdot \mathbf{x}_i)$
///
/// $\frac{\partial p_{\text{inc}}}{\partial n}(\mathbf{x}_i) = i k (\hat{d} \cdot \hat{n}_i) \, p_{\text{inc}}(\mathbf{x}_i)$
///
/// # Arguments
///
/// * `vertices`    — Boundary surface vertex positions (m)
/// * `normals`     — Outward unit normals at each vertex (from [`compute_vertex_normals`])
/// * `direction`   — Unit propagation direction $\hat{d}$ (normalised internally)
/// * `wavenumber`  — Acoustic wavenumber $k = 2\pi f / c$
/// * `amplitude`   — Complex amplitude $A$
///
/// # Returns
///
/// `(p_inc, dp_inc_dn)` — incident pressure and normal derivative, one entry per vertex.
///
/// # Reference
///
/// Colton, D. & Kress, R. (1998). *Inverse Acoustic and Electromagnetic Scattering Theory*,
/// 2nd ed., Springer. §3.1.
pub fn plane_wave_incident(
    vertices: &[[f64; 3]],
    normals: &[[f64; 3]],
    direction: [f64; 3],
    wavenumber: f64,
    amplitude: Complex64,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let dlen = (direction[0] * direction[0]
        + direction[1] * direction[1]
        + direction[2] * direction[2])
        .sqrt()
        .max(1e-15);
    let d = [
        direction[0] / dlen,
        direction[1] / dlen,
        direction[2] / dlen,
    ];

    let ik = Complex64::new(0.0, wavenumber);

    let n = vertices.len();
    let mut p_inc = Vec::with_capacity(n);
    let mut dp_inc_dn = Vec::with_capacity(n);

    for (xi, ni) in vertices.iter().zip(normals.iter()) {
        let phase = d[0] * xi[0] + d[1] * xi[1] + d[2] * xi[2];
        let pi = amplitude * Complex64::from_polar(1.0, wavenumber * phase);
        let dn = d[0] * ni[0] + d[1] * ni[1] + d[2] * ni[2];
        p_inc.push(pi);
        dp_inc_dn.push(ik * dn * pi);
    }

    (p_inc, dp_inc_dn)
}
