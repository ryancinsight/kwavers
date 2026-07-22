use super::super::super::types::ElasticBodyForceConfig;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversResult, NumericalError};
use kwavers_grid::Grid;

/// Validate parameters whose failure is independent of grid position and time.
pub(super) fn validate(body_force: &ElasticBodyForceConfig) -> KwaversResult<()> {
    let ElasticBodyForceConfig::GaussianImpulse {
        center_m,
        sigma_m,
        direction,
        t0_s,
        sigma_t_s,
        impulse_n_per_m3_s,
    } = body_force;

    if center_m
        .iter()
        .chain(sigma_m)
        .chain(direction)
        .chain([t0_s, sigma_t_s, impulse_n_per_m3_s])
        .any(|value| !value.is_finite())
    {
        return Err(NumericalError::InvalidOperation(
            "Gaussian body-force parameters must be finite".to_owned(),
        )
        .into());
    }
    if sigma_m.iter().any(|&sigma| sigma <= 0.0) || *sigma_t_s <= 0.0 {
        return Err(NumericalError::InvalidOperation(
            "Gaussian body-force spatial and temporal widths must be positive".to_owned(),
        )
        .into());
    }
    let direction_norm_sq = direction[2].mul_add(
        direction[2],
        direction[0].mul_add(direction[0], direction[1] * direction[1]),
    );
    if direction_norm_sq < f64::MIN_POSITIVE {
        return Err(NumericalError::InvalidOperation(
            "Gaussian body-force direction must be nonzero".to_owned(),
        )
        .into());
    }
    Ok(())
}

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
) -> [f64; 3] {
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
            let dx = x - center_m[0];
            let dy = y - center_m[1];
            let dz = z - center_m[2];

            let sx = sigma_m[0];
            let sy = sigma_m[1];
            let sz = sigma_m[2];

            let spatial_factor = (-0.5
                * (dz / sz).mul_add(dz / sz, (dx / sx).mul_add(dx / sx, (dy / sy) * (dy / sy))))
            .exp();

            let dt = time - *t0_s;
            let temporal_factor =
                (-(dt * dt) / (2.0 * sigma_t_s * sigma_t_s)).exp() / (sigma_t_s * (TWO_PI).sqrt());

            let dir_norm = direction[2]
                .mul_add(
                    direction[2],
                    direction[0].mul_add(direction[0], direction[1] * direction[1]),
                )
                .sqrt();

            let scale = impulse_n_per_m3_s * spatial_factor * temporal_factor / dir_norm;
            [
                scale * direction[0],
                scale * direction[1],
                scale * direction[2],
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::error::KwaversError;

    fn valid_force() -> ElasticBodyForceConfig {
        ElasticBodyForceConfig::GaussianImpulse {
            center_m: [0.0; 3],
            sigma_m: [1.0; 3],
            direction: [1.0, 0.0, 0.0],
            t0_s: 0.0,
            sigma_t_s: 1.0,
            impulse_n_per_m3_s: 1.0,
        }
    }

    fn assert_invalid(force: &ElasticBodyForceConfig, expected: &str) {
        let error = validate(force).expect_err("invalid body force must be rejected");
        match error {
            KwaversError::Numerical(NumericalError::InvalidOperation(message)) => {
                assert_eq!(message, expected);
            }
            other => panic!("expected InvalidOperation, got {other}"),
        }
    }

    #[test]
    fn validation_rejects_nonpositive_width() {
        let mut force = valid_force();
        let ElasticBodyForceConfig::GaussianImpulse { sigma_m, .. } = &mut force;
        sigma_m[1] = 0.0;

        assert_invalid(
            &force,
            "Gaussian body-force spatial and temporal widths must be positive",
        );
    }

    #[test]
    fn validation_rejects_nonfinite_parameter() {
        let mut force = valid_force();
        let ElasticBodyForceConfig::GaussianImpulse { center_m, .. } = &mut force;
        center_m[0] = f64::NAN;

        assert_invalid(&force, "Gaussian body-force parameters must be finite");
    }

    #[test]
    fn validation_rejects_zero_direction() {
        let mut force = valid_force();
        let ElasticBodyForceConfig::GaussianImpulse { direction, .. } = &mut force;
        *direction = [0.0; 3];

        assert_invalid(&force, "Gaussian body-force direction must be nonzero");
    }
}
