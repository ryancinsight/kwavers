use super::{compute_time_step, AcousticValidationCase, AcousticWorkspace};
use crate::core::error::KwaversResult;
use crate::domain::imaging::photoacoustic::{InitialPressure, PhotoacousticScenario};
use ndarray::Array3;

/// Canonical acoustic forward model for photoacoustic propagation.
///
/// # Governing Equation
/// The retained CPU propagator solves the homogeneous scalar wave equation
/// `∂²p/∂t² = c²∇²p` with an explicit second-order central-difference update.
///
/// # Numerical Algorithm
/// Spatial derivatives are approximated with a 7-point Laplacian on the
/// Cartesian grid. The time step is chosen from the scenario CFL factor using
/// [`compute_time_step`].
///
/// # Stability
/// Stability depends on the configured CFL factor and the minimum grid spacing.
#[derive(Debug, Default)]
pub struct AcousticForwardModel;

impl AcousticForwardModel {
    /// Propagate.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn propagate(
        &self,
        scenario: &PhotoacousticScenario,
        initial_pressure: &InitialPressure,
    ) -> KwaversResult<(Vec<Array3<f64>>, Vec<f64>)> {
        let grid = &scenario.grid;
        let (nx, ny, nz) = grid.dimensions();
        let speed_of_sound = scenario.config.acoustic.speed_of_sound_m_s;
        let num_time_steps = scenario.config.acoustic.num_time_steps;
        let snapshot_interval = scenario.config.acoustic.snapshot_interval.max(1);
        let dt = compute_time_step(scenario);
        let _validation_case = AcousticValidationCase {
            name: "canonical_explicit_wave_step",
            cfl_limit: scenario.config.acoustic.cfl_factor,
        };

        let mut workspace = AcousticWorkspace::new((nx, ny, nz));
        workspace.current.assign(&initial_pressure.pressure);
        workspace.previous.assign(&initial_pressure.pressure);

        let c2_dt2 = speed_of_sound.powi(2) * dt.powi(2);
        let inv_dx2 = 1.0 / grid.dx.powi(2);
        let inv_dy2 = 1.0 / grid.dy.powi(2);
        let inv_dz2 = 1.0 / grid.dz.powi(2);

        let mut pressure_fields = Vec::with_capacity((num_time_steps / snapshot_interval) + 2);
        let mut time_points = Vec::with_capacity((num_time_steps / snapshot_interval) + 2);
        pressure_fields.push(workspace.current.clone());
        time_points.push(0.0);

        for step in 1..=num_time_steps {
            for i in 0..nx {
                let im = i.saturating_sub(1);
                let ip = (i + 1).min(nx - 1);
                for j in 0..ny {
                    let jm = j.saturating_sub(1);
                    let jp = (j + 1).min(ny - 1);
                    for k in 0..nz {
                        let km = k.saturating_sub(1);
                        let kp = (k + 1).min(nz - 1);
                        let center = workspace.current[[i, j, k]];
                        let laplacian = (2.0f64.mul_add(-center, workspace.current[[i, j, kp]]) + workspace.current[[i, j, km]]).mul_add(inv_dz2, (2.0f64.mul_add(-center, workspace.current[[ip, j, k]]) + workspace.current[[im, j, k]]).mul_add(inv_dx2, (2.0f64.mul_add(-center, workspace.current[[i, jp, k]])
                                + workspace.current[[i, jm, k]]) * inv_dy2));
                        workspace.next[[i, j, k]] =
                            2.0f64.mul_add(center, -workspace.previous[[i, j, k]]) + c2_dt2 * laplacian;
                    }
                }
            }

            std::mem::swap(&mut workspace.previous, &mut workspace.current);
            std::mem::swap(&mut workspace.current, &mut workspace.next);

            if step % snapshot_interval == 0 {
                pressure_fields.push(workspace.current.clone());
                time_points.push(step as f64 * dt);
            }
        }

        Ok((pressure_fields, time_points))
    }
}
