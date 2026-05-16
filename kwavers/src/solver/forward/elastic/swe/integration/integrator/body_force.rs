use super::super::super::types::ElasticBodyForceConfig;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;

/// Evaluate a single body force configuration at grid cell `(i, j, k)` and time `t`.
///
/// Returns the force vector `[fx, fy, fz]` in N/m³.
pub(super) fn evaluate(
    grid: &Grid,
    body_force: &ElasticBodyForceConfig,
    i: usize,
    j: usize,
    k: usize,
    time: f64,
) -> KwaversResult<[f64; 3]> {
    let x = i as f64 * grid.dx;
    let y = j as f64 * grid.dy;
    let z = k as f64 * grid.dz;

    match body_force {
        ElasticBodyForceConfig::GaussianImpulse {
            center_m,
            sigma_m,
            direction,
            t0_s,
            sigma_t_s,
            impulse_n_per_m3_s,
        } => {
            if !sigma_t_s.is_finite() || *sigma_t_s <= 0.0 {
                return Ok([0.0, 0.0, 0.0]);
            }

            let dx = x - center_m[0];
            let dy = y - center_m[1];
            let dz = z - center_m[2];

            let sx = sigma_m[0];
            let sy = sigma_m[1];
            let sz = sigma_m[2];
            if !sx.is_finite()
                || !sy.is_finite()
                || !sz.is_finite()
                || sx <= 0.0
                || sy <= 0.0
                || sz <= 0.0
            {
                return Ok([0.0, 0.0, 0.0]);
            }

            let spatial_factor = (-0.5
                * (dz / sz).mul_add(dz / sz, (dx / sx).mul_add(dx / sx, (dy / sy) * (dy / sy))))
            .exp();

            let dt = time - *t0_s;
            let temporal_factor = (-(dt * dt) / (2.0 * sigma_t_s * sigma_t_s)).exp()
                / (sigma_t_s * (2.0 * std::f64::consts::PI).sqrt());

            let dir_norm = direction[2]
                .mul_add(
                    direction[2],
                    direction[0].mul_add(direction[0], direction[1] * direction[1]),
                )
                .sqrt();
            if !dir_norm.is_finite() || dir_norm < 1e-12 {
                return Ok([0.0, 0.0, 0.0]);
            }

            let scale = impulse_n_per_m3_s * spatial_factor * temporal_factor / dir_norm;
            Ok([
                scale * direction[0],
                scale * direction[1],
                scale * direction[2],
            ])
        }
    }
}
