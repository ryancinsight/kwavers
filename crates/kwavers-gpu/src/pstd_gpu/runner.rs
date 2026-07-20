//! High-level entry point for GPU-resident PSTD acoustic simulation.
//!
//! Encapsulates buffer preparation, CPML profile evaluation, sensor-mask
//! indexing, source preparation, and the [`GpuPstdSolver`] invocation. The
//! pykwavers binding layer previously held this orchestration; lifting it to
//! kwavers lets clinical adapters (e.g. the breast-UST FWI reconstruction
//! dataset) drive the GPU path directly when the `gpu` feature is enabled.
//!
//! # Constraints inherited from [`GpuPstdSolver`]
//! - Grid dimensions must be powers of two.
//! - Per-axis dimension ≤ 1,024.
//! - Single precision throughout the GPU pipeline; sensor data widened to
//!   `f64` on return.
//!
//! # Output
//! Returns `leto::Array2<f64>` of shape `(num_sensors, time_steps)` with the
//! pressure recorded at each sensor index per step.

use crate::pstd_gpu::{
    validate_gpu_pstd_dimensions, AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays,
    PstdAutoDeviceProvider, PstdOutputRequest, PstdRunInputs, PstdRunResult, PstdRunState,
    SolverParams, WgpuPstdStateProvider,
};
use kwavers_boundary::cpml::{CPMLConfig, CPMLProfiles};
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
use kwavers_source::GridSource;
use leto::{Array2 as LetoArray2, Array3 as LetoArray3};
use std::f64::consts::PI;

/// GPU PSTD acquisition settings.
#[derive(Clone, Copy, Debug)]
pub struct GpuPstdRunConfig {
    /// Number of time steps to integrate.
    pub time_steps: usize,
    /// Time step in seconds.
    pub dt: f64,
    /// Power-law absorption coefficient [dB/(MHz·cm)]; `0.0` disables
    /// fractional absorption even when the medium stores a material coefficient.
    /// When enabled, nonzero per-voxel medium coefficients take precedence and
    /// this value fills voxels without a material coefficient.
    pub alpha_coeff_db: f64,
    /// Power-law absorption exponent `y`; an enabled fractional model cannot
    /// use the singular value `1.0`.
    pub alpha_power: f64,
    /// CPML thickness in cells; `None` selects the automatic limit.
    pub pml_size: Option<usize>,
    /// Per-axis CPML thickness override; `None` uses `pml_size`.
    pub pml_size_xyz: Option<(usize, usize, usize)>,
    /// `true` places CPML inside the grid; `false` zeroes the per-cell PML
    /// damping (treat as a free interior with no absorbing layer).
    pub pml_inside: bool,
    /// Optional per-axis CPML α override.
    pub pml_alpha_xyz: Option<(f64, f64, f64)>,
}

impl Default for GpuPstdRunConfig {
    fn default() -> Self {
        Self {
            time_steps: 0,
            dt: 0.0,
            alpha_coeff_db: 0.0,
            alpha_power: 1.0,
            pml_size: None,
            pml_size_xyz: None,
            pml_inside: true,
            pml_alpha_xyz: None,
        }
    }
}

/// Drive a GPU-resident PSTD acoustic simulation and return sensor pressure
/// traces.
///
/// A nonzero medium `B/A` coefficient enables the nonlinear equation of state.
/// Fractional power-law absorption is controlled only by
/// [`GpuPstdRunConfig::alpha_coeff_db`]; zero is lossless even when the medium
/// stores a material absorption coefficient.
///
/// # Errors
/// - GPU PSTD requires power-of-2 grid dimensions with each axis ≤ 1,024.
/// - GPU device acquisition failures bubble up via the selected provider.
/// - Invalid medium, source, or sensor inputs return [`KwaversError::InvalidInput`].
pub fn run_gpu_pstd(
    grid: &Grid,
    medium: &dyn Medium,
    source: &GridSource,
    sensor_mask: &LetoArray3<bool>,
    config: GpuPstdRunConfig,
) -> KwaversResult<LetoArray2<f64>> {
    run_gpu_pstd_with_provider::<WgpuPstdStateProvider>(grid, medium, source, sensor_mask, config)
}

/// Drive a GPU-resident PSTD simulation and return the explicitly requested
/// provider outputs.
///
/// Use [`PstdOutputRequest::with_peak_pressure`] for a pointwise temporal
/// pressure envelope; a final pressure field is not an equivalent substitute.
///
/// # Errors
///
/// Returns [`KwaversError::InvalidInput`] for an invalid grid, source, medium,
/// sensor mask, configuration, or provider acquisition failure.
pub fn run_gpu_pstd_with_outputs(
    grid: &Grid,
    medium: &dyn Medium,
    source: &GridSource,
    sensor_mask: &LetoArray3<bool>,
    config: GpuPstdRunConfig,
    output_request: PstdOutputRequest,
) -> KwaversResult<PstdRunResult> {
    run_gpu_pstd_with_provider_outputs::<WgpuPstdStateProvider>(
        grid,
        medium,
        source,
        sensor_mask,
        config,
        output_request,
    )
}

/// Drive a GPU-resident PSTD acoustic simulation with an explicit GPU state
/// provider.
///
/// # Errors
/// - GPU PSTD requires power-of-2 grid dimensions with each axis ≤ 1,024.
/// - GPU device acquisition failures bubble up via the selected provider.
/// - Invalid medium, source, or sensor inputs return [`KwaversError::InvalidInput`].
pub fn run_gpu_pstd_with_provider<P>(
    grid: &Grid,
    medium: &dyn Medium,
    source: &GridSource,
    sensor_mask: &LetoArray3<bool>,
    config: GpuPstdRunConfig,
) -> KwaversResult<LetoArray2<f64>>
where
    P: PstdAutoDeviceProvider,
    P::State: PstdRunState,
{
    let time_steps = config.time_steps;
    let sensor_data_f32 = run_gpu_pstd_with_provider_outputs::<P>(
        grid,
        medium,
        source,
        sensor_mask,
        config,
        PstdOutputRequest::sensor_traces(),
    )?
    .sensor_data;
    let n_sensors = sensor_data_f32.len() / time_steps;
    LetoArray2::from_shape_vec(
        [n_sensors, time_steps],
        sensor_data_f32.into_iter().map(f64::from).collect(),
    )
    .map_err(|err| {
        KwaversError::InvalidInput(format!("GPU PSTD sensor trace shape mismatch: {err}"))
    })
}

/// Drive a GPU-resident PSTD simulation with an explicit state provider and
/// return the explicitly requested provider outputs.
///
/// # Errors
///
/// Returns [`KwaversError::InvalidInput`] for an invalid grid, source, medium,
/// sensor mask, configuration, or provider acquisition failure.
pub fn run_gpu_pstd_with_provider_outputs<P>(
    grid: &Grid,
    medium: &dyn Medium,
    source: &GridSource,
    sensor_mask: &LetoArray3<bool>,
    config: GpuPstdRunConfig,
    output_request: PstdOutputRequest,
) -> KwaversResult<PstdRunResult>
where
    P: PstdAutoDeviceProvider,
    P::State: PstdRunState,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let GpuPstdRunConfig {
        time_steps,
        dt,
        alpha_coeff_db,
        alpha_power,
        pml_size,
        pml_size_xyz,
        pml_inside,
        pml_alpha_xyz,
    } = config;

    validate_gpu_pstd_dimensions(nx, ny, nz)?;
    let total = nx * ny * nz;
    if time_steps == 0 || !dt.is_finite() || dt <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "GPU PSTD requires time_steps > 0 and finite positive dt; got steps={time_steps} dt={dt}"
        )));
    }

    let (default_thickness, max_allowed) = cpml_thickness_limits(nx, ny, nz);
    let thickness = pml_size.unwrap_or(default_thickness).min(max_allowed);

    let cpml_config = if let Some((px, py, pz)) = pml_size_xyz {
        let mut cfg = CPMLConfig::with_per_dimension_thickness(px, py, pz);
        if let Some((ax, ay, az)) = pml_alpha_xyz {
            cfg = cfg.with_alpha_xyz(ax, ay, az);
        }
        cfg
    } else {
        let mut cfg = CPMLConfig::with_thickness(thickness);
        if let Some((ax, ay, az)) = pml_alpha_xyz {
            cfg = cfg.with_alpha_xyz(ax, ay, az);
        }
        cfg
    };

    let c_ref = medium.max_sound_speed();
    let profiles = CPMLProfiles::new(&cpml_config, grid, c_ref, dt)?;

    let mut pml_sgx_3d = vec![1.0f32; total];
    let mut pml_sgy_3d = vec![1.0f32; total];
    let mut pml_sgz_3d = vec![1.0f32; total];
    let mut pml_x_3d = vec![1.0f32; total];
    let mut pml_y_3d = vec![1.0f32; total];
    let mut pml_z_3d = vec![1.0f32; total];

    if pml_inside {
        let exp_half = |sigma: &leto::Array1<f64>| -> Vec<f32> {
            sigma
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect()
        };
        let pml_sgx_1d = exp_half(&profiles.sigma_x_sgx);
        let pml_sgy_1d = exp_half(&profiles.sigma_y_sgy);
        let pml_sgz_1d = exp_half(&profiles.sigma_z_sgz);
        let pml_x_1d = exp_half(&profiles.sigma_x);
        let pml_y_1d = exp_half(&profiles.sigma_y);
        let pml_z_1d = exp_half(&profiles.sigma_z);

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let flat = ix * ny * nz + iy * nz + iz;
                    pml_sgx_3d[flat] = pml_sgx_1d[ix];
                    pml_sgy_3d[flat] = pml_sgy_1d[iy];
                    pml_sgz_3d[flat] = pml_sgz_1d[iz];
                    pml_x_3d[flat] = pml_x_1d[ix];
                    pml_y_3d[flat] = pml_y_1d[iy];
                    pml_z_3d[flat] = pml_z_1d[iz];
                }
            }
        }
    }

    let mut c0_flat = vec![c_ref as f32; total];
    let mut rho0_flat = vec![DENSITY_WATER_NOMINAL as f32; total];
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let flat = ix * ny * nz + iy * nz + iz;
                c0_flat[flat] = medium.sound_speed(ix, iy, iz) as f32;
                rho0_flat[flat] = medium.density(ix, iy, iz) as f32;
            }
        }
    }

    let sensor_indices = collect_sensor_indices(sensor_mask)?;

    let pressure_source =
        super::prepare_pstd_pressure_source(grid, source, &c0_flat, dt, time_steps)?;

    let mut vel_x_indices: Vec<u32> = Vec::new();
    let mut vel_x_signals: Vec<f32> = Vec::new();
    let mut velocity_source_correction = false;

    if let (Some(u_mask), Some(u_signal)) = (&source.u_mask, &source.u_signal) {
        let mask_flat = u_mask.as_slice_memory_order().ok_or_else(|| {
            KwaversError::InvalidInput("u_mask must be dense row-major array".into())
        })?;
        if mask_flat.len() != total {
            return Err(KwaversError::InvalidInput(format!(
                "u_mask has {} cells but grid has {total}",
                mask_flat.len()
            )));
        }
        for (i, &v) in mask_flat.iter().enumerate() {
            if v != 0.0 {
                vel_x_indices.push(i as u32);
            }
        }
        let n_vel = vel_x_indices.len();
        let n_sig_srcs = u_signal.shape()[1];
        if n_vel > 0 && n_sig_srcs != 1 && n_sig_srcs != n_vel {
            return Err(KwaversError::InvalidInput(format!(
                "u_signal has {n_sig_srcs} source rows for {n_vel} velocity-source cells; expected 1 or {n_vel}"
            )));
        }
        velocity_source_correction =
            super::source::source_mode_uses_kspace_correction(source.u_mode, "velocity")?;
        let n_sig_cols = u_signal.shape()[2].min(time_steps);
        vel_x_signals = vec![0.0f32; n_vel * time_steps];
        for src_idx in 0..n_vel {
            let sig_row = if n_sig_srcs == 1 { 0 } else { src_idx };
            for step in 0..n_sig_cols {
                vel_x_signals[src_idx * time_steps + step] = u_signal[[0, sig_row, step]] as f32;
            }
        }
    }

    let has_absorption = power_law_absorption_enabled(alpha_coeff_db, alpha_power)?;

    let bon_a_flat: Vec<f32> = (0..total)
        .map(|flat| {
            let ix = flat / (ny * nz);
            let iy = (flat % (ny * nz)) / nz;
            let iz = flat % nz;
            (medium.nonlinearity(ix, iy, iz) / 2.0) as f32
        })
        .collect();
    let has_nonlinear = has_nonlinear_coefficient(&bon_a_flat);

    let (absorb_nabla1_flat, absorb_nabla2_flat, absorb_tau_flat, absorb_eta_flat) =
        if has_absorption {
            let dk_x = TWO_PI / (nx as f64 * grid.dx);
            let dk_y = TWO_PI / (ny as f64 * grid.dy);
            let dk_z = TWO_PI / (nz as f64 * grid.dz);
            let singularity_thresh: f64 = 1e-8;
            let y = alpha_power;

            let mut n1 = vec![0.0f32; total];
            let mut n2 = vec![0.0f32; total];
            let mut tau_v = vec![0.0f32; total];
            let mut eta_v = vec![0.0f32; total];

            for flat in 0..total {
                let ix = flat / (ny * nz);
                let iy = (flat % (ny * nz)) / nz;
                let iz = flat % nz;

                let kix = if ix <= nx / 2 {
                    ix as f64
                } else {
                    (nx - ix) as f64
                } * dk_x;
                let kiy = if iy <= ny / 2 {
                    iy as f64
                } else {
                    (ny - iy) as f64
                } * dk_y;
                let kiz = if iz <= nz / 2 {
                    iz as f64
                } else {
                    (nz - iz) as f64
                } * dk_z;
                let k_mag = (kix * kix + kiy * kiy + kiz * kiz).sqrt();

                if k_mag > singularity_thresh {
                    n1[flat] = k_mag.powf(y - 2.0) as f32;
                    n2[flat] = k_mag.powf(y - 1.0) as f32;
                }

                // Match the CPU PowerLaw contract: an enabled run uses the
                // medium's spatial coefficient when present, otherwise the
                // explicit configuration coefficient.  A zero config never
                // enters this branch, so it remains an unambiguous lossless run.
                let medium_alpha_db_cm = medium.absorption(ix, iy, iz);
                let alpha_db_cm = if medium_alpha_db_cm.abs() > 0.0 {
                    medium_alpha_db_cm
                } else {
                    alpha_coeff_db
                };
                let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_db_cm, alpha_power);
                let c0_local = medium.sound_speed(ix, iy, iz);
                tau_v[flat] = (-2.0 * alpha_0_si * c0_local.powf(y - 1.0)) as f32;
                eta_v[flat] = (2.0 * alpha_0_si * c0_local.powf(y) * (PI * y / 2.0).tan()) as f32;
            }
            (n1, n2, tau_v, eta_v)
        } else {
            (
                vec![0.0f32; total],
                vec![0.0f32; total],
                vec![0.0f32; total],
                vec![0.0f32; total],
            )
        };

    let mut solver = GpuPstdSolver::<P>::with_auto_device(
        grid,
        MediumArrays {
            c0_flat: &c0_flat,
            rho0_flat: &rho0_flat,
        },
        SolverParams {
            dt,
            nt: time_steps,
            c_ref,
            nonlinear: has_nonlinear,
            absorbing: has_absorption,
        },
        PmlArrays {
            x: &pml_x_3d,
            y: &pml_y_3d,
            z: &pml_z_3d,
            sgx: &pml_sgx_3d,
            sgy: &pml_sgy_3d,
            sgz: &pml_sgz_3d,
        },
        AbsorptionArrays {
            bon_a_flat: &bon_a_flat,
            nabla1: &absorb_nabla1_flat,
            nabla2: &absorb_nabla2_flat,
            tau: &absorb_tau_flat,
            eta: &absorb_eta_flat,
        },
    )
    .map_err(|e| KwaversError::InvalidInput(format!("GPU device init failed: {e}")))?;

    Ok(solver.run(PstdRunInputs {
        sensor_indices: &sensor_indices,
        source_indices: &pressure_source.indices,
        source_signals: &pressure_source.signals,
        pressure_source_correction: pressure_source.uses_kspace_correction,
        vel_x_indices: &vel_x_indices,
        vel_x_signals: &vel_x_signals,
        velocity_source_correction,
        output_request,
    }))
}

/// Whether any packed `B/A / 2` coefficient enables the nonlinear equation.
fn has_nonlinear_coefficient(coefficients: &[f32]) -> bool {
    coefficients.iter().any(|&coefficient| coefficient != 0.0)
}

/// Validate whether the explicit GPU configuration enables fractional absorption.
///
/// The `y = 1` power law has a singular dispersion coefficient
/// `tan(πy/2)`, so it is invalid only when the caller actually enables the
/// fractional-Laplacian model.  This mirrors the CPU PSTD PowerLaw validation.
fn power_law_absorption_enabled(alpha_coeff_db: f64, alpha_power: f64) -> KwaversResult<bool> {
    if !alpha_coeff_db.is_finite() || alpha_coeff_db < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "GPU PSTD alpha_coeff_db must be finite and non-negative; got {alpha_coeff_db}"
        )));
    }
    if alpha_coeff_db == 0.0 {
        return Ok(false);
    }
    if !alpha_power.is_finite() || (alpha_power - 1.0).abs() < 1e-12 {
        return Err(KwaversError::InvalidInput(format!(
            "GPU PSTD alpha_power must be finite and must not equal 1.0 for enabled fractional absorption; got {alpha_power}"
        )));
    }
    Ok(true)
}

fn collect_sensor_indices(sensor_mask: &LetoArray3<bool>) -> KwaversResult<Vec<u32>> {
    let flat = sensor_mask.as_slice_memory_order().ok_or_else(|| {
        KwaversError::InvalidInput("sensor_mask must be dense row-major leto::Array3".into())
    })?;
    Ok(flat
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i as u32) } else { None })
        .collect())
}

/// Minimum active axis length → admissible CPML thickness.
///
/// Returns `(default_thickness, max_allowed)` where `default_thickness` is the
/// conventional 20-cell choice clipped to `max_allowed` (and floored at 2).
#[must_use]
pub fn cpml_thickness_limits(nx: usize, ny: usize, nz: usize) -> (usize, usize) {
    let mut min_dim = usize::MAX;
    for dim in [nx, ny, nz] {
        if dim > 1 {
            min_dim = min_dim.min(dim);
        }
    }
    let min_dim = if min_dim == usize::MAX { 1 } else { min_dim };
    let max_allowed = (min_dim.saturating_sub(2)) / 2;
    let default_thickness = 20_usize.min(max_allowed).max(2);
    (default_thickness, max_allowed)
}

#[cfg(test)]
mod tests {
    use super::{
        collect_sensor_indices, has_nonlinear_coefficient, power_law_absorption_enabled, LetoArray3,
    };

    #[test]
    fn collect_sensor_indices_preserves_row_major_positions() {
        let mask = LetoArray3::from_shape_vec(
            [2, 2, 2],
            vec![false, true, false, false, true, false, false, true],
        )
        .expect("test mask shape matches storage");

        let indices = collect_sensor_indices(&mask).expect("dense Leto mask is valid");

        assert_eq!(indices, vec![1, 4, 7]);
    }

    #[test]
    fn zero_explicit_absorption_is_lossless_at_singular_default_exponent() {
        assert!(!power_law_absorption_enabled(0.0, 1.0)
            .expect("zero coefficient disables fractional absorption"));
    }

    #[test]
    fn enabled_absorption_rejects_singular_power_law_exponent() {
        let error = power_law_absorption_enabled(0.0022, 1.0)
            .expect_err("enabled fractional absorption cannot use y=1");
        assert_eq!(
            error.to_string(),
            "Invalid input: GPU PSTD alpha_power must be finite and must not equal 1.0 for enabled fractional absorption; got 1"
        );
    }

    #[test]
    fn nonlinear_selection_scans_every_packed_medium_cell() {
        assert!(!has_nonlinear_coefficient(&[0.0, 0.0, 0.0]));
        assert!(has_nonlinear_coefficient(&[0.0, 0.0, 2.6]));
    }
}
