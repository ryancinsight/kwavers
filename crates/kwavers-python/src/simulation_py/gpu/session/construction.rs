use numpy::PyReadonlyArray3;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::GpuPstdSession;

#[pymethods]
impl GpuPstdSession {
    /// Create a persistent GPU PSTD session.
    #[new]
    #[pyo3(signature = (
        grid, sound_speed, density,
        dt, time_steps,
        absorption=None, nonlinearity=None,
        pml_size_xyz=None, alpha_power=1.5
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _py: Python<'_>,
        grid: &crate::grid_py::Grid,
        sound_speed: PyReadonlyArray3<f64>,
        density: PyReadonlyArray3<f64>,
        dt: f64,
        time_steps: usize,
        absorption: Option<PyReadonlyArray3<f64>>,
        nonlinearity: Option<PyReadonlyArray3<f64>>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        alpha_power: f64,
    ) -> PyResult<Self> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (
                _py,
                grid,
                sound_speed,
                density,
                dt,
                time_steps,
                absorption,
                nonlinearity,
                pml_size_xyz,
                alpha_power,
            );
            Err(PyRuntimeError::new_err(
                "GpuPstdSession requires the 'gpu' feature.  \
                 Rebuild pykwavers with --features gpu.",
            ))
        }

        #[cfg(feature = "gpu")]
        {
            use kwavers_boundary::cpml::{CPMLConfig, CPMLProfiles};
            use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
            use kwavers_gpu::pstd_gpu::{
                AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays, SolverParams,
            };

            let kgrid = &grid.inner;
            let nx = kgrid.nx;
            let ny = kgrid.ny;
            let nz = kgrid.nz;
            let total = nx * ny * nz;

            if !nx.is_power_of_two() || !ny.is_power_of_two() || !nz.is_power_of_two() {
                return Err(PyValueError::new_err(format!(
                    "GpuPstdSession requires power-of-2 grid; got {}x{}x{}",
                    nx, ny, nz
                )));
            }
            if nx > 256 || ny > 256 || nz > 256 {
                return Err(PyValueError::new_err(format!(
                    "GpuPstdSession: grid axis max 256 pts; got {}x{}x{}",
                    nx, ny, nz
                )));
            }

            let ss_arr = sound_speed.as_array();
            let rho_arr = density.as_array();
            let c0_flat: Vec<f32> = ss_arr.iter().map(|&v| v as f32).collect();
            let rho0_flat: Vec<f32> = rho_arr.iter().map(|&v| v as f32).collect();
            let c_ref = c0_flat.iter().cloned().fold(0.0f32, f32::max) as f64;

            let bon_a_flat: Vec<f32> = if let Some(ref nl) = nonlinearity {
                nl.as_array().iter().map(|&v| (v / 2.0) as f32).collect()
            } else {
                vec![0.0f32; total]
            };

            let has_absorption = absorption.is_some();
            let (absorb_nabla1_flat, absorb_nabla2_flat, absorb_tau_flat, absorb_eta_flat) =
                build_absorption_kernels(
                    has_absorption,
                    absorption.as_ref(),
                    &c0_flat,
                    nx,
                    ny,
                    nz,
                    kgrid.dx,
                    kgrid.dy,
                    kgrid.dz,
                    alpha_power,
                )?;

            if has_absorption {
                let tau_max = absorb_tau_flat
                    .iter()
                    .cloned()
                    .fold(0.0f32, |a, b| a.abs().max(b.abs()));
                let eta_max = absorb_eta_flat
                    .iter()
                    .cloned()
                    .fold(0.0f32, |a, b| a.abs().max(b.abs()));
                let nabla2_max = absorb_nabla2_flat
                    .iter()
                    .cloned()
                    .fold(0.0f32, |a, b| a.max(b));
                eprintln!("[pykwavers-diag] GpuPstdSession absorbing=true: tau_max={tau_max:.3e}, eta_max={eta_max:.3e}, nabla2_max={nabla2_max:.3e}");
            } else {
                eprintln!("[pykwavers-diag] GpuPstdSession absorbing=false (lossless)");
            }

            let (pml_x_3d, pml_y_3d, pml_z_3d, pml_sgx_3d, pml_sgy_3d, pml_sgz_3d) =
                build_pml_arrays(pml_size_xyz, kgrid, c_ref, dt, nx, ny, nz)?;

            let solver = GpuPstdSolver::with_auto_device(
                kgrid,
                MediumArrays {
                    c0_flat: &c0_flat,
                    rho0_flat: &rho0_flat,
                },
                SolverParams {
                    dt,
                    nt: time_steps,
                    c_ref,
                    nonlinear: nonlinearity.is_some(),
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
            .map_err(|e| PyRuntimeError::new_err(format!("GPU solver init failed: {e}")))?;

            Ok(Self {
                solver,
                nx,
                ny,
                nz,
                bon_a_flat,
                absorb_tau_flat,
                absorb_eta_flat,
                has_absorption,
                time_steps,
                sensor_indices: Vec::new(),
                vel_x_indices: Vec::new(),
                vel_x_signals: Vec::new(),
                last_medium_upload_ns: 0,
                last_medium_variable_upload_ns: 0,
                last_medium_static_upload_ns: 0,
                last_solver_run_ns: 0,
                last_materialize_ns: 0,
                last_total_ns: 0,
            })
        }
    }
}

#[cfg(feature = "gpu")]
fn build_absorption_kernels(
    has_absorption: bool,
    absorption: Option<&numpy::PyReadonlyArray3<f64>>,
    c0_flat: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    alpha_power: f64,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
    let total = nx * ny * nz;
    if !has_absorption {
        return Ok((
            vec![0.0f32; total],
            vec![0.0f32; total],
            vec![0.0f32; total],
            vec![0.0f32; total],
        ));
    }

    use std::f64::consts::PI;
    let dk_x = 2.0 * PI / (nx as f64 * dx);
    let dk_y = 2.0 * PI / (ny as f64 * dy);
    let dk_z = 2.0 * PI / (nz as f64 * dz);
    let singularity_thresh: f64 = 1e-8;
    let y = alpha_power;

    let mut n1 = vec![0.0f32; total];
    let mut n2 = vec![0.0f32; total];
    let mut tau_v = vec![0.0f32; total];
    let mut eta_v = vec![0.0f32; total];

    let ab_arr = absorption.unwrap().as_array();

    if (y - 1.0).abs() < 1e-12 && ab_arr.iter().any(|&v| v > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "alpha_power must not be 1.0 for fractional Laplacian absorption",
        ));
    }

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

        let alpha_db_cm = ab_arr[[ix, iy, iz]];
        let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_db_cm, alpha_power);
        let c0_local = c0_flat[flat] as f64;
        tau_v[flat] = (-2.0 * alpha_0_si * c0_local.powf(y - 1.0)) as f32;
        eta_v[flat] = (2.0 * alpha_0_si * c0_local.powf(y) * (PI * y / 2.0).tan()) as f32;
    }
    Ok((n1, n2, tau_v, eta_v))
}

#[cfg(feature = "gpu")]
fn build_pml_arrays(
    pml_size_xyz: Option<(usize, usize, usize)>,
    kgrid: &kwavers_grid::Grid,
    c_ref: f64,
    dt: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    use kwavers_boundary::cpml::{CPMLConfig, CPMLProfiles};
    use pyo3::exceptions::PyRuntimeError;

    let total = nx * ny * nz;
    let (pml_x_sz, pml_y_sz, pml_z_sz) = pml_size_xyz.unwrap_or((10, 10, 10));
    let pml_config = CPMLConfig::with_per_dimension_thickness(pml_x_sz, pml_y_sz, pml_z_sz);
    let profiles = CPMLProfiles::new(&pml_config, kgrid, c_ref, dt)
        .map_err(|e| PyRuntimeError::new_err(format!("PML init failed: {e}")))?;

    let pml_sgx_1d: Vec<f32> = profiles
        .sigma_x_sgx
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();
    let pml_sgy_1d: Vec<f32> = profiles
        .sigma_y_sgy
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();
    let pml_sgz_1d: Vec<f32> = profiles
        .sigma_z_sgz
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();
    let pml_x_1d: Vec<f32> = profiles
        .sigma_x
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();
    let pml_y_1d: Vec<f32> = profiles
        .sigma_y
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();
    let pml_z_1d: Vec<f32> = profiles
        .sigma_z
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();

    let mut pml_x_3d = vec![1.0f32; total];
    let mut pml_y_3d = vec![1.0f32; total];
    let mut pml_z_3d = vec![1.0f32; total];
    let mut pml_sgx_3d = vec![1.0f32; total];
    let mut pml_sgy_3d = vec![1.0f32; total];
    let mut pml_sgz_3d = vec![1.0f32; total];
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

    Ok((
        pml_x_3d, pml_y_3d, pml_z_3d, pml_sgx_3d, pml_sgy_3d, pml_sgz_3d,
    ))
}
