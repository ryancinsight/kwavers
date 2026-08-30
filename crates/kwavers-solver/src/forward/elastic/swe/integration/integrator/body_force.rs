use super::super::super::types::ElasticBodyForceConfig;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversResult, NumericalError};
use kwavers_grid::Grid;

#[derive(Debug)]
struct PreparedForceParameters {
    direction: [f64; 3],
    t0_s: f64,
    sigma_t_s: f64,
    impulse_n_per_m3_s: f64,
    direction_norm: f64,
    temporal_factor: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GridSignature {
    dimensions: [usize; 3],
    spacing_bits: [u64; 3],
}

impl GridSignature {
    fn new(grid: &Grid) -> Self {
        Self {
            dimensions: [grid.nx, grid.ny, grid.nz],
            spacing_bits: [grid.dx.to_bits(), grid.dy.to_bits(), grid.dz.to_bits()],
        }
    }
}

/// Spatially prepared Gaussian forces for repeated time integration.
///
/// Gaussian spatial factors are separable and invariant across velocity-Verlet
/// steps. Storing one factor per force and coordinate axis requires
/// `O(forces × (nx + ny + nz))` memory instead of a volume-sized cache, while
/// [`Self::update_time`] evaluates each temporal Gaussian once per acceleration
/// pass rather than once per force and voxel.
#[derive(Debug)]
pub(crate) struct PreparedBodyForces {
    parameters: Box<[PreparedForceParameters]>,
    x_factors: Box<[f64]>,
    y_factors: Box<[f64]>,
    z_factors: Box<[f64]>,
    grid_signature: GridSignature,
}

impl PreparedBodyForces {
    pub(crate) fn new(grid: &Grid, body_forces: &[ElasticBodyForceConfig]) -> KwaversResult<Self> {
        for body_force in body_forces {
            validate(body_force)?;
        }

        let mut parameters = Vec::new();
        parameters
            .try_reserve_exact(body_forces.len())
            .map_err(|_| body_force_allocation_error())?;
        parameters.extend(body_forces.iter().map(|body_force| {
            let ElasticBodyForceConfig::GaussianImpulse {
                direction,
                t0_s,
                sigma_t_s,
                impulse_n_per_m3_s,
                ..
            } = body_force;
            PreparedForceParameters {
                direction: *direction,
                t0_s: *t0_s,
                sigma_t_s: *sigma_t_s,
                impulse_n_per_m3_s: *impulse_n_per_m3_s,
                direction_norm: direction_norm(direction),
                temporal_factor: 0.0,
            }
        }));

        Ok(Self {
            parameters: parameters.into_boxed_slice(),
            x_factors: prepare_axis_factors::<0>(grid.nx, grid.dx, body_forces)?,
            y_factors: prepare_axis_factors::<1>(grid.ny, grid.dy, body_forces)?,
            z_factors: prepare_axis_factors::<2>(grid.nz, grid.dz, body_forces)?,
            grid_signature: GridSignature::new(grid),
        })
    }

    pub(crate) fn validate_grid(&self, grid: &Grid) -> KwaversResult<()> {
        if self.grid_signature != GridSignature::new(grid) {
            return Err(NumericalError::InvalidOperation(
                "Prepared Gaussian body forces do not match the integrator grid".to_owned(),
            )
            .into());
        }
        Ok(())
    }

    pub(crate) fn update_time(&mut self, time: f64) {
        for parameters in &mut self.parameters {
            parameters.temporal_factor =
                temporal_factor(time, parameters.t0_s, parameters.sigma_t_s);
        }
    }

    #[inline]
    pub(crate) fn force_at(&self, i: usize, j: usize, k: usize) -> [f64; 3] {
        let [nx, ny, nz] = self.grid_signature.dimensions;
        debug_assert!(i < nx);
        debug_assert!(j < ny);
        debug_assert!(k < nz);
        let force_count = self.parameters.len();
        let x = &self.x_factors[i * force_count..(i + 1) * force_count];
        let y = &self.y_factors[j * force_count..(j + 1) * force_count];
        let z = &self.z_factors[k * force_count..(k + 1) * force_count];
        let mut force = [0.0_f64; 3];
        for (((parameters, &x_factor), &y_factor), &z_factor) in
            self.parameters.iter().zip(x).zip(y).zip(z)
        {
            let spatial_factor = (x_factor * y_factor) * z_factor;
            let scale = parameters.impulse_n_per_m3_s * spatial_factor * parameters.temporal_factor
                / parameters.direction_norm;
            force[0] += scale * parameters.direction[0];
            force[1] += scale * parameters.direction[1];
            force[2] += scale * parameters.direction[2];
        }
        force
    }
}

fn prepare_axis_factors<const AXIS: usize>(
    length: usize,
    spacing: f64,
    body_forces: &[ElasticBodyForceConfig],
) -> KwaversResult<Box<[f64]>> {
    debug_assert!(AXIS < 3);
    let factor_count = length.checked_mul(body_forces.len()).ok_or_else(|| {
        NumericalError::InvalidOperation(
            "Gaussian body-force axis profile size overflow".to_owned(),
        )
    })?;
    let mut factors = Vec::new();
    factors
        .try_reserve_exact(factor_count)
        .map_err(|_| body_force_allocation_error())?;
    for coordinate_index in 0..length {
        factors.extend(body_forces.iter().map(|body_force| {
            let ElasticBodyForceConfig::GaussianImpulse {
                center_m, sigma_m, ..
            } = body_force;
            axis_factor(coordinate_index, spacing, center_m[AXIS], sigma_m[AXIS])
        }));
    }
    debug_assert_eq!(factors.len(), factor_count);
    Ok(factors.into_boxed_slice())
}

fn body_force_allocation_error() -> kwavers_core::error::KwaversError {
    NumericalError::InvalidOperation(
        "Insufficient memory for prepared Gaussian body-force profiles".to_owned(),
    )
    .into()
}

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
    match body_force {
        ElasticBodyForceConfig::GaussianImpulse {
            direction,
            t0_s,
            sigma_t_s,
            impulse_n_per_m3_s,
            ..
        } => {
            let scale = impulse_n_per_m3_s
                * spatial_factor(grid, body_force, i, j, k)
                * temporal_factor(time, *t0_s, *sigma_t_s)
                / direction_norm(direction);
            [
                scale * direction[0],
                scale * direction[1],
                scale * direction[2],
            ]
        }
    }
}

fn spatial_factor(
    grid: &Grid,
    body_force: &ElasticBodyForceConfig,
    i: usize,
    j: usize,
    k: usize,
) -> f64 {
    let ElasticBodyForceConfig::GaussianImpulse {
        center_m, sigma_m, ..
    } = body_force;
    let x_factor = axis_factor(i, grid.dx, center_m[0], sigma_m[0]);
    let y_factor = axis_factor(j, grid.dy, center_m[1], sigma_m[1]);
    let z_factor = axis_factor(k, grid.dz, center_m[2], sigma_m[2]);
    (x_factor * y_factor) * z_factor
}

fn axis_factor(index: usize, spacing: f64, center: f64, sigma: f64) -> f64 {
    let scaled = (index as f64 * spacing - center) / sigma;
    (-0.5 * scaled * scaled).exp()
}

fn temporal_factor(time: f64, center: f64, sigma: f64) -> f64 {
    let offset = time - center;
    (-(offset * offset) / (2.0 * sigma * sigma)).exp() / (sigma * TWO_PI.sqrt())
}

fn direction_norm(direction: &[f64; 3]) -> f64 {
    direction[2]
        .mul_add(
            direction[2],
            direction[0].mul_add(direction[0], direction[1] * direction[1]),
        )
        .sqrt()
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
    fn prepared_forces_match_direct_evaluation() {
        let grid = Grid::new(3, 2, 2, 0.25, 0.5, 0.75).expect("valid test grid");
        let mut second = valid_force();
        let ElasticBodyForceConfig::GaussianImpulse {
            center_m,
            direction,
            t0_s,
            impulse_n_per_m3_s,
            ..
        } = &mut second;
        *center_m = [0.25, 0.5, 0.75];
        *direction = [0.0, 2.0, -1.0];
        *t0_s = 0.125;
        *impulse_n_per_m3_s = 3.0;
        let forces = [valid_force(), second];
        let mut prepared =
            PreparedBodyForces::new(&grid, &forces).expect("valid forces must prepare");

        for time in [-2.0, 0.0, 0.25, 2.0] {
            prepared.update_time(time);
            for index in 0..(grid.nx * grid.ny * grid.nz) {
                let i = index / (grid.ny * grid.nz);
                let j = (index / grid.nz) % grid.ny;
                let k = index % grid.nz;
                let mut expected = [0.0; 3];
                for body_force in &forces {
                    let value = evaluate(&grid, body_force, i, j, k, time);
                    for component in 0..3 {
                        expected[component] += value[component];
                    }
                }

                let actual = prepared.force_at(i, j, k);
                assert_eq!(
                    actual, expected,
                    "prepared force at [{i}, {j}, {k}], t={time}"
                );
            }
        }
    }

    #[test]
    fn prepared_forces_reject_a_different_grid() {
        let grid = Grid::new(3, 2, 2, 0.25, 0.5, 0.75).expect("valid test grid");
        let different_grid = Grid::new(3, 2, 2, 0.5, 0.5, 0.75).expect("valid different grid");
        let prepared =
            PreparedBodyForces::new(&grid, &[valid_force()]).expect("valid force must prepare");

        let error = prepared
            .validate_grid(&different_grid)
            .expect_err("grid-specific factors must reject a different grid");

        match error {
            KwaversError::Numerical(NumericalError::InvalidOperation(message)) => assert_eq!(
                message,
                "Prepared Gaussian body forces do not match the integrator grid"
            ),
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
