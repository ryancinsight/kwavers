use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "gpu")]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::GpuPstdSession;
#[cfg(feature = "gpu")]
use super::{absorption::build_absorption_kernels, pml::build_pml_arrays};

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
