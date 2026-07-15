//! Rigid-body (SE(2)) reparametrisation of a sound-speed template and its
//! analytic parameter Jacobian (MOFI; Bates et al. 2026).
//!
//! The template `c(x_orig)` is transformed into `c_φ(x_trans) = c(T_φ⁻¹ x_trans)`
//! with `T_φ` the SE(2) rigid map (rotation `θ` about the grid centre +
//! translation `δ = (δ₁, δ₂)`). Sampling on the regular grid requires bilinear
//! interpolation at the back-mapped coordinates. The transform acts in the
//! `(x, y)` plane and is applied independently to each `z`-slice (so it is a true
//! SE(2) transform for `nz = 1` and a plane-stacked SE(2) transform otherwise).
//!
//! The chained MOFI gradient is `∂f/∂φ = (∂c_φ/∂φ)ᵀ ∂f/∂c`. This module supplies
//! the analytic `∂c_φ/∂φ` (the exact derivative of the bilinear interpolant, the
//! same quantity reverse-mode autodiff would produce in the original paper) and
//! the projection `(∂c_φ/∂φ)ᵀ g`.

use leto::Array3;

/// SE(2) rigid-body transform parameters.
///
/// `theta_rad` is the rotation about the chosen centre; `delta_x_m`/`delta_y_m`
/// are the translations along the grid `x`/`y` axes in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidTransform {
    pub theta_rad: f64,
    pub delta_x_m: f64,
    pub delta_y_m: f64,
}

impl RigidTransform {
    /// The identity transform `φ = 0`.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            theta_rad: 0.0,
            delta_x_m: 0.0,
            delta_y_m: 0.0,
        }
    }
}

/// Centre of rotation in fractional index coordinates and the grid spacings.
#[derive(Debug, Clone, Copy)]
pub(super) struct PlaneGeometry {
    pub cx: f64,
    pub cy: f64,
    pub dx: f64,
    pub dy: f64,
}

impl PlaneGeometry {
    pub(super) fn centered(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        Self {
            cx: (nx - 1) as f64 / 2.0,
            cy: (ny - 1) as f64 / 2.0,
            dx,
            dy,
        }
    }
}

/// Bilinear sample of one `z`-slice plus the analytic in-plane gradient of the
/// bilinear interpolant. Returns `(value, dc/dx, dc/dy)` in physical units.
/// Out-of-domain back-mapped points return `(background, 0, 0)`.
pub(super) fn bilinear_with_gradient(
    template: &Array3<f64>,
    k: usize,
    fi: f64,
    fj: f64,
    geom: &PlaneGeometry,
    background: f64,
) -> (f64, f64, f64) {
    let [nx, ny, _] = template.shape();
    if fi < 0.0 || fj < 0.0 || fi > (nx - 1) as f64 || fj > (ny - 1) as f64 {
        return (background, 0.0, 0.0);
    }
    let i0 = fi.floor() as usize;
    let j0 = fj.floor() as usize;
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(ny - 1);
    let tx = fi - i0 as f64;
    let ty = fj - j0 as f64;

    let c00 = template[[i0, j0, k]];
    let c10 = template[[i1, j0, k]];
    let c01 = template[[i0, j1, k]];
    let c11 = template[[i1, j1, k]];

    let value = (1.0 - tx) * (1.0 - ty) * c00
        + tx * (1.0 - ty) * c10
        + (1.0 - tx) * ty * c01
        + tx * ty * c11;
    // Derivative of the bilinear interpolant w.r.t. fractional index, then /dα.
    let dc_dfi = (1.0 - ty) * (c10 - c00) + ty * (c11 - c01);
    let dc_dfj = (1.0 - tx) * (c01 - c00) + tx * (c11 - c10);
    (value, dc_dfi / geom.dx, dc_dfj / geom.dy)
}

/// Transform the template by `phi` (bilinear back-mapping). Out-of-domain points
/// take `background`.
pub(super) fn transform_template(
    template: &Array3<f64>,
    phi: &RigidTransform,
    geom: &PlaneGeometry,
    background: f64,
) -> Array3<f64> {
    let [nx, ny, nz] = template.shape();
    let (s, c) = phi.theta_rad.sin_cos();
    let mut out = Array3::from_elem([nx, ny, nz], background);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let x_trans = (i as f64 - geom.cx) * geom.dx;
                let y_trans = (j as f64 - geom.cy) * geom.dy;
                let vx = x_trans - phi.delta_x_m;
                let vy = y_trans - phi.delta_y_m;
                // x_orig = Rᵀ v.
                let xo = c * vx + s * vy;
                let yo = -s * vx + c * vy;
                let fi = geom.cx + xo / geom.dx;
                let fj = geom.cy + yo / geom.dy;
                let (val, _, _) = bilinear_with_gradient(template, k, fi, fj, geom, background);
                out[[i, j, k]] = val;
            }
        }
    }
    out
}

/// Transformed model plus the per-voxel parameter Jacobian
/// `(∂c_φ/∂θ, ∂c_φ/∂δ₁, ∂c_φ/∂δ₂)`.
///
/// With `x_orig = Rᵀ(x_trans − δ)` and `v = x_trans − δ`:
/// ```text
/// ∂x_orig/∂δ₁ = (−cosθ,  sinθ)
/// ∂x_orig/∂δ₂ = (−sinθ, −cosθ)
/// ∂x_orig/∂θ  = (−sinθ·vx + cosθ·vy, −cosθ·vx − sinθ·vy)
/// ∂c_φ/∂φⱼ    = ∇c(x_orig) · ∂x_orig/∂φⱼ
/// ```
pub(super) struct TransformWithJacobian {
    pub model: Array3<f64>,
    pub d_theta: Array3<f64>,
    pub d_delta_x: Array3<f64>,
    pub d_delta_y: Array3<f64>,
}

pub(super) fn transform_with_jacobian(
    template: &Array3<f64>,
    phi: &RigidTransform,
    geom: &PlaneGeometry,
    background: f64,
) -> TransformWithJacobian {
    let [nx, ny, nz] = template.shape();
    let (s, c) = phi.theta_rad.sin_cos();
    let mut model = Array3::from_elem([nx, ny, nz], background);
    let mut d_theta = Array3::zeros((nx, ny, nz));
    let mut d_delta_x = Array3::zeros((nx, ny, nz));
    let mut d_delta_y = Array3::zeros((nx, ny, nz));

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let x_trans = (i as f64 - geom.cx) * geom.dx;
                let y_trans = (j as f64 - geom.cy) * geom.dy;
                let vx = x_trans - phi.delta_x_m;
                let vy = y_trans - phi.delta_y_m;
                let xo = c * vx + s * vy;
                let yo = -s * vx + c * vy;
                let fi = geom.cx + xo / geom.dx;
                let fj = geom.cy + yo / geom.dy;
                let (val, dcx, dcy) = bilinear_with_gradient(template, k, fi, fj, geom, background);
                model[[i, j, k]] = val;
                // ∂x_orig/∂δ and ∂x_orig/∂θ, contracted with ∇c = (dcx, dcy).
                d_delta_x[[i, j, k]] = dcx * (-c) + dcy * s;
                d_delta_y[[i, j, k]] = dcx * (-s) + dcy * (-c);
                let dxo_dtheta = -s * vx + c * vy;
                let dyo_dtheta = -c * vx - s * vy;
                d_theta[[i, j, k]] = dcx * dxo_dtheta + dcy * dyo_dtheta;
            }
        }
    }
    TransformWithJacobian {
        model,
        d_theta,
        d_delta_x,
        d_delta_y,
    }
}

/// Project a pixel-wise model gradient `g = ∂f/∂c_φ` onto the SE(2) parameter
/// space: `(∂f/∂θ, ∂f/∂δ₁, ∂f/∂δ₂) = (⟨g, ∂c_φ/∂θ⟩, ⟨g, ∂c_φ/∂δ₁⟩, ⟨g, ∂c_φ/∂δ₂⟩)`.
pub(super) fn project_gradient(g: &Array3<f64>, jac: &TransformWithJacobian) -> [f64; 3] {
    let g_theta = (g * &jac.d_theta).iter().sum::<f64>();
    let g_dx = (g * &jac.d_delta_x).iter().sum::<f64>();
    let g_dy = (g * &jac.d_delta_y).iter().sum::<f64>();
    [g_theta, g_dx, g_dy]
}

#[cfg(test)]
mod transform_tests {
    use super::*;

    /// The analytic parameter Jacobian matches a central finite difference of the
    /// transformed template (validates `∂c_φ/∂φ` voxel-for-voxel).
    #[test]
    fn jacobian_matches_finite_difference() {
        let (nx, ny) = (24usize, 24);
        let mut template = Array3::from_elem([nx, ny, 1], 1500.0_f64);
        // Asymmetric smooth feature so all three parameters move c_φ.
        for j in 0..ny {
            for i in 0..nx {
                let r2 = (i as f64 - 9.0).powi(2) + (j as f64 - 14.0).powi(2);
                template[[i, j, 0]] += 600.0 * (-r2 / 12.0).exp();
            }
        }
        let geom = PlaneGeometry::centered(nx, ny, 1e-3, 1e-3);
        let bg = 1500.0;
        let phi = RigidTransform {
            theta_rad: 0.12,
            delta_x_m: 1.5e-3,
            delta_y_m: -0.8e-3,
        };
        let jac = transform_with_jacobian(&template, &phi, &geom, bg);

        let eps_t = 1e-5;
        let eps_d = 1e-7;
        let fd = |a: &RigidTransform, b: &RigidTransform, h: f64| {
            (&transform_template(&template, a, &geom, bg)
                - &transform_template(&template, b, &geom, bg))
                .mapv(|x| x / (2.0 * h))
        };
        let dtheta_fd = fd(
            &RigidTransform {
                theta_rad: phi.theta_rad + eps_t,
                ..phi
            },
            &RigidTransform {
                theta_rad: phi.theta_rad - eps_t,
                ..phi
            },
            eps_t,
        );
        let ddx_fd = fd(
            &RigidTransform {
                delta_x_m: phi.delta_x_m + eps_d,
                ..phi
            },
            &RigidTransform {
                delta_x_m: phi.delta_x_m - eps_d,
                ..phi
            },
            eps_d,
        );
        // Compare on the interior where the bilinear interpolant is smooth.
        let mut max_t = 0.0_f64;
        let mut max_x = 0.0_f64;
        for j in 4..ny - 4 {
            for i in 4..nx - 4 {
                max_t = max_t.max((jac.d_theta[[i, j, 0]] - dtheta_fd[[i, j, 0]]).abs());
                max_x = max_x.max((jac.d_delta_x[[i, j, 0]] - ddx_fd[[i, j, 0]]).abs());
            }
        }
        // Tolerances reflect the piecewise-bilinear gradient vs a finite step.
        assert!(max_t < 5.0, "∂c_φ/∂θ vs FD max abs diff too large: {max_t}");
        assert!(
            max_x < 1e-2,
            "∂c_φ/∂δ₁ vs FD max abs diff too large: {max_x}"
        );
    }
}
