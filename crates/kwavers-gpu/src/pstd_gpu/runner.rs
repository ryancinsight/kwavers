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
//! - Per-axis dimension ≤ 256.
//! - Single precision throughout the GPU pipeline; sensor data widened to
//!   `f64` on return.
//!
//! # Output
//! Returns `Array2<f64>` of shape `(num_sensors, time_steps)` with the
//! pressure recorded at each sensor index per step.

use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_boundary::cpml::{CPMLConfig, CPMLProfiles};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_domain::source::GridSource;
use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
use crate::pstd_gpu::{
    AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays, SolverParams,
};
use ndarray::{Array2, Array3};
use std::f64::consts::PI;

/// GPU PSTD acquisition settings.
#[derive(Clone, Copy, Debug)]
pub struct GpuPstdRunConfig {
    /// Number of time steps to integrate.
    pub time_steps: usize,
    /// Time step [s].
    pub dt: f64,
    /// Power-law absorption coefficient [dB/(MHz·cm)]; `0.0` disables.
    pub alpha_coeff_db: f64,
    /// Power-law absorption exponent `y` (typically 1.0–1.5).
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
/// # Errors
/// - GPU PSTD requires power-of-2 grid dimensions with each axis ≤ 256.
/// - GPU device acquisition failures bubble up via the `wgpu` error path.
/// - Invalid medium, source, or sensor inputs return [`KwaversError::InvalidInput`].
pub fn run_gpu_pstd(
    grid: &Grid,
    medium: &dyn Medium,
    source: &GridSource,
    sensor_mask: &Array3<bool>,
    config: GpuPstdRunConfig,
) -> KwaversResult<Array2<f64>> {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let total = nx * ny * nz;
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

    if !nx.is_power_of_two() || !ny.is_power_of_two() || !nz.is_power_of_two() {
        return Err(KwaversError::InvalidInput(format!(
            "GPU PSTD requires power-of-2 grid dimensions; got {nx}x{ny}x{nz}"
        )));
    }
    if nx > 256 || ny > 256 || nz > 256 {
        return Err(KwaversError::InvalidInput(format!(
            "GPU PSTD supports per-axis N <= 256; got {nx}x{ny}x{nz}"
        )));
    }
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
        let exp_half = |sigma: &ndarray::Array1<f64>| -> Vec<f32> {
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

    let sensor_indices: Vec<u32> = {
        let flat = sensor_mask
            .as_slice()
            .ok_or_else(|| KwaversError::InvalidInput("sensor_mask must be C-contiguous".into()))?;
        flat.iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i as u32) } else { None })
            .collect()
    };

    let n_dim_active = [nx > 1, ny > 1, nz > 1]
        .iter()
        .filter(|&&d| d)
        .count()
        .max(1);
    let dx_min = grid.dx.min(grid.dy).min(grid.dz);
    let mass_source_scale = 2.0 * dt / (n_dim_active as f64 * c_ref * dx_min);
    let density_scale = n_dim_active as f64 / 3.0;
    let combined_scale = (mass_source_scale * density_scale) as f32;

    let mut source_indices: Vec<u32> = Vec::new();
    let mut source_signals: Vec<f32> = Vec::new();

    if let (Some(p_mask), Some(p_signal)) = (&source.p_mask, &source.p_signal) {
        let mask_flat = p_mask
            .as_slice()
            .ok_or_else(|| KwaversError::InvalidInput("p_mask must be C-contiguous".into()))?;
        for (i, &v) in mask_flat.iter().enumerate() {
            if v != 0.0 {
                source_indices.push(i as u32);
            }
        }
        let n_src = source_indices.len();
        let n_sig_rows = p_signal.shape()[0];
        let n_sig_cols = p_signal.shape()[1].min(time_steps);
        source_signals = vec![0.0f32; n_src * time_steps];
        for (src_idx, _) in source_indices.iter().enumerate() {
            let sig_row = if n_sig_rows == 1 {
                0
            } else {
                src_idx.min(n_sig_rows - 1)
            };
            for step in 0..n_sig_cols {
                source_signals[src_idx * time_steps + step] =
                    (p_signal[[sig_row, step]] * combined_scale as f64) as f32;
            }
        }
    }

    let mut vel_x_indices: Vec<u32> = Vec::new();
    let mut vel_x_signals: Vec<f32> = Vec::new();

    if let (Some(u_mask), Some(u_signal)) = (&source.u_mask, &source.u_signal) {
        let mask_flat = u_mask
            .as_slice()
            .ok_or_else(|| KwaversError::InvalidInput("u_mask must be C-contiguous".into()))?;
        for (i, &v) in mask_flat.iter().enumerate() {
            if v != 0.0 {
                vel_x_indices.push(i as u32);
            }
        }
        let n_vel = vel_x_indices.len();
        let n_sig_srcs = u_signal.shape()[1];
        let n_sig_cols = u_signal.shape()[2].min(time_steps);
        vel_x_signals = vec![0.0f32; n_vel * time_steps];
        for src_idx in 0..n_vel {
            let sig_row = src_idx.min(n_sig_srcs.saturating_sub(1));
            for step in 0..n_sig_cols {
                vel_x_signals[src_idx * time_steps + step] = u_signal[[0, sig_row, step]] as f32;
            }
        }
    }

    let has_nonlinear = medium.nonlinearity(0, 0, 0) > 0.0;
    let has_absorption =
        alpha_coeff_db > 0.0 || medium.alpha_coefficient(0.0, 0.0, 0.0, grid) > 0.0;

    let bon_a_flat: Vec<f32> = if has_nonlinear {
        (0..total)
            .map(|flat| {
                let ix = flat / (ny * nz);
                let iy = (flat % (ny * nz)) / nz;
                let iz = flat % nz;
                (medium.nonlinearity(ix, iy, iz) / 2.0) as f32
            })
            .collect()
    } else {
        vec![0.0f32; total]
    };

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

                let alpha_db_cm = medium.absorption(ix, iy, iz);
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

    let mut solver = GpuPstdSolver::with_auto_device(
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

    let sensor_data_f32 = solver.run(
        &sensor_indices,
        &source_indices,
        &source_signals,
        &vel_x_indices,
        &vel_x_signals,
    );

    let n_sensors = sensor_indices.len();
    let mut out = Array2::<f64>::zeros((n_sensors, time_steps));
    for s in 0..n_sensors {
        for t in 0..time_steps {
            out[[s, t]] = sensor_data_f32[s * time_steps + t] as f64;
        }
    }

    Ok(out)
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
