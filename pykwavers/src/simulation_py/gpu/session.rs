use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// Persistent GPU PSTD session for efficient B-mode scan-line loops.
///
/// Creating a new `GpuPstdSolver` per scan line is expensive (~500 ms) because
/// wgpu must compile ~13 WGSL compute pipelines from scratch.  `GpuPstdSession`
/// creates the solver **once** and re-uses compiled pipelines.  Between scan
/// lines you only re-upload the medium arrays via `run_scan_line()`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
#[pyclass(unsendable)]
pub struct GpuPstdSession {
    #[cfg(feature = "gpu")]
    pub(crate) solver: kwavers::solver::forward::pstd::gpu_pstd::GpuPstdSolver,

    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,

    pub(crate) bon_a_flat: Vec<f32>,
    pub(crate) absorb_tau_flat: Vec<f32>,
    pub(crate) absorb_eta_flat: Vec<f32>,
    pub(crate) has_absorption: bool,

    pub(crate) time_steps: usize,

    pub(crate) sensor_indices: Vec<u32>,
    pub(crate) vel_x_indices: Vec<u32>,
    pub(crate) vel_x_signals: Vec<f32>,
    pub(crate) last_medium_upload_ns: u64,
    pub(crate) last_medium_variable_upload_ns: u64,
    pub(crate) last_medium_static_upload_ns: u64,
    pub(crate) last_solver_run_ns: u64,
    pub(crate) last_materialize_ns: u64,
    pub(crate) last_total_ns: u64,
}

impl GpuPstdSession {
    pub(crate) fn rebuild_source_sensor_indices(
        &mut self,
        mask_arr: ndarray::ArrayView3<'_, f64>,
    ) -> PyResult<()> {
        if mask_arr.shape() != [self.nx, self.ny, self.nz] {
            return Err(PyValueError::new_err(format!(
                "mask shape {:?} must match session grid ({}, {}, {})",
                mask_arr.shape(), self.nx, self.ny, self.nz
            )));
        }

        self.vel_x_indices.clear();
        for ix in 0..self.nx {
            for iy in 0..self.ny {
                for iz in 0..self.nz {
                    if mask_arr[[ix, iy, iz]] != 0.0 {
                        let flat = ix * self.ny * self.nz + iy * self.nz + iz;
                        self.vel_x_indices.push(flat as u32);
                    }
                }
            }
        }

        self.sensor_indices.clear();
        for iz in 0..self.nz {
            for iy in 0..self.ny {
                for ix in 0..self.nx {
                    if mask_arr[[ix, iy, iz]] != 0.0 {
                        let flat = ix * self.ny * self.nz + iy * self.nz + iz;
                        self.sensor_indices.push(flat as u32);
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn update_velocity_signal_rows(
        &mut self,
        sig_arr: ndarray::ArrayView2<'_, f64>,
    ) -> PyResult<()> {
        let time_steps = self.time_steps;
        let signal_shape = sig_arr.shape();
        if signal_shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "ux_signals must be 2D, got shape {:?}",
                signal_shape
            )));
        }

        let n_vel = self.vel_x_indices.len();
        let n_sig_srcs = signal_shape[0];
        let n_sig_cols = signal_shape[1].min(time_steps);

        if n_vel > 0 && n_sig_srcs == 0 {
            return Err(PyValueError::new_err(
                "ux_signals must contain at least one source row for a non-empty mask",
            ));
        }

        self.vel_x_signals.clear();
        self.vel_x_signals.resize(n_vel * time_steps, 0.0f32);
        for src_idx in 0..n_vel {
            let sig_row = src_idx.min(n_sig_srcs.saturating_sub(1));
            for step in 0..n_sig_cols {
                self.vel_x_signals[src_idx * time_steps + step] = sig_arr[[sig_row, step]] as f32;
            }
        }

        Ok(())
    }
}

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
    fn new(
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
            let _ = (_py, grid, sound_speed, density, dt, time_steps,
                     absorption, nonlinearity, pml_size_xyz, alpha_power);
            Err(PyRuntimeError::new_err(
                "GpuPstdSession requires the 'gpu' feature.  \
                 Rebuild pykwavers with --features gpu.",
            ))
        }

        #[cfg(feature = "gpu")]
        {
            use kwavers::domain::boundary::cpml::{CPMLConfig, CPMLProfiles};
            use kwavers::physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
            use kwavers::solver::forward::pstd::gpu_pstd::GpuPstdSolver;

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
                if has_absorption {
                    use std::f64::consts::PI;
                    let dk_x = 2.0 * PI / (nx as f64 * kgrid.dx);
                    let dk_y = 2.0 * PI / (ny as f64 * kgrid.dy);
                    let dk_z = 2.0 * PI / (nz as f64 * kgrid.dz);
                    let singularity_thresh: f64 = 1e-8;
                    let y = alpha_power;

                    let mut n1 = vec![0.0f32; total];
                    let mut n2 = vec![0.0f32; total];
                    let mut tau_v = vec![0.0f32; total];
                    let mut eta_v = vec![0.0f32; total];

                    let ab_arr = absorption.as_ref().unwrap().as_array();

                    if (y - 1.0).abs() < 1e-12 && ab_arr.iter().any(|&v| v > 0.0) {
                        return Err(PyValueError::new_err(
                            "alpha_power must not be 1.0 for fractional Laplacian absorption",
                        ));
                    }

                    for flat in 0..total {
                        let ix = flat / (ny * nz);
                        let iy = (flat % (ny * nz)) / nz;
                        let iz = flat % nz;

                        let kix = if ix <= nx / 2 { ix as f64 } else { (nx - ix) as f64 } * dk_x;
                        let kiy = if iy <= ny / 2 { iy as f64 } else { (ny - iy) as f64 } * dk_y;
                        let kiz = if iz <= nz / 2 { iz as f64 } else { (nz - iz) as f64 } * dk_z;
                        let k_mag = (kix * kix + kiy * kiy + kiz * kiz).sqrt();

                        if k_mag > singularity_thresh {
                            n1[flat] = k_mag.powf(y - 2.0) as f32;
                            n2[flat] = k_mag.powf(y - 1.0) as f32;
                        }

                        let alpha_db_cm = ab_arr[[ix, iy, iz]];
                        let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_db_cm, alpha_power);
                        let c0_local = c0_flat[flat] as f64;
                        tau_v[flat] = (-2.0 * alpha_0_si * c0_local.powf(y - 1.0)) as f32;
                        eta_v[flat] =
                            (2.0 * alpha_0_si * c0_local.powf(y) * (PI * y / 2.0).tan()) as f32;
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

            if has_absorption {
                let tau_max = absorb_tau_flat.iter().cloned().fold(0.0f32, |a, b| a.abs().max(b.abs()));
                let eta_max = absorb_eta_flat.iter().cloned().fold(0.0f32, |a, b| a.abs().max(b.abs()));
                let nabla2_max = absorb_nabla2_flat.iter().cloned().fold(0.0f32, |a, b| a.max(b));
                eprintln!("[pykwavers-diag] GpuPstdSession absorbing=true: tau_max={tau_max:.3e}, eta_max={eta_max:.3e}, nabla2_max={nabla2_max:.3e}");
            } else {
                eprintln!("[pykwavers-diag] GpuPstdSession absorbing=false (lossless)");
            }

            let (pml_x_sz, pml_y_sz, pml_z_sz) = pml_size_xyz.unwrap_or((10, 10, 10));
            let pml_config = CPMLConfig::with_per_dimension_thickness(pml_x_sz, pml_y_sz, pml_z_sz);
            let profiles = CPMLProfiles::new(&pml_config, kgrid, c_ref, dt)
                .map_err(|e| PyRuntimeError::new_err(format!("PML init failed: {e}")))?;

            let pml_sgx_1d: Vec<f32> = profiles.sigma_x_sgx.iter().map(|&s| (-s * dt * 0.5).exp() as f32).collect();
            let pml_sgy_1d: Vec<f32> = profiles.sigma_y_sgy.iter().map(|&s| (-s * dt * 0.5).exp() as f32).collect();
            let pml_sgz_1d: Vec<f32> = profiles.sigma_z_sgz.iter().map(|&s| (-s * dt * 0.5).exp() as f32).collect();
            let pml_x_1d: Vec<f32> = profiles.sigma_x.iter().map(|&s| (-s * dt * 0.5).exp() as f32).collect();
            let pml_y_1d: Vec<f32> = profiles.sigma_y.iter().map(|&s| (-s * dt * 0.5).exp() as f32).collect();
            let pml_z_1d: Vec<f32> = profiles.sigma_z.iter().map(|&s| (-s * dt * 0.5).exp() as f32).collect();

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

            let solver = GpuPstdSolver::with_auto_device(
                kgrid, &c0_flat, &rho0_flat, dt, time_steps, c_ref,
                &pml_x_3d, &pml_y_3d, &pml_z_3d, &pml_sgx_3d, &pml_sgy_3d, &pml_sgz_3d,
                &bon_a_flat, &absorb_nabla1_flat, &absorb_nabla2_flat, &absorb_tau_flat, &absorb_eta_flat,
                nonlinearity.is_some(), has_absorption,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("GPU solver init failed: {e}")))?;

            Ok(Self {
                solver, nx, ny, nz,
                bon_a_flat, absorb_tau_flat, absorb_eta_flat, has_absorption,
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

    /// Set the source and sensor mask for all scan lines (constant per session).
    fn set_source_sensor(
        &mut self,
        _py: Python<'_>,
        mask: PyReadonlyArray3<f64>,
        ux_signals: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let mask_arr = mask.as_array();
        let sig_arr = ux_signals.as_array();
        self.rebuild_source_sensor_indices(mask_arr)?;
        self.update_velocity_signal_rows(sig_arr)
    }

    /// Cache the source/sensor mask when the geometry is invariant across runs.
    fn set_source_sensor_mask(
        &mut self,
        _py: Python<'_>,
        mask: PyReadonlyArray3<f64>,
    ) -> PyResult<()> {
        self.rebuild_source_sensor_indices(mask.as_array())
    }

    /// Update only the x-velocity source signals for a previously cached mask.
    fn set_velocity_signals(
        &mut self,
        _py: Python<'_>,
        ux_signals: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        self.update_velocity_signal_rows(ux_signals.as_array())
    }

    /// Disable the k-space source correction (sets source_kappa = 1 everywhere).
    fn disable_source_correction(&self, _py: Python<'_>) -> PyResult<()> {
        #[cfg(feature = "gpu")]
        {
            self.solver.disable_source_correction();
            Ok(())
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err(
                "GpuPstdSession requires the 'gpu' feature.",
            ))
        }
    }

    /// Return the timing profile from the most recent scan-line execution.
    #[getter]
    fn last_run_profile<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let profile = PyDict::new(py);
        profile.set_item("medium_upload_ns", self.last_medium_upload_ns)?;
        profile.set_item("medium_variable_upload_ns", self.last_medium_variable_upload_ns)?;
        profile.set_item("medium_static_upload_ns", self.last_medium_static_upload_ns)?;
        profile.set_item("solver_run_ns", self.last_solver_run_ns)?;
        profile.set_item("materialize_ns", self.last_materialize_ns)?;
        profile.set_item("total_ns", self.last_total_ns)?;
        profile.set_item("n_sensors", self.sensor_indices.len())?;
        profile.set_item("n_velocity_sources", self.vel_x_indices.len())?;
        Ok(profile)
    }

    /// Return the most recent scan-line timing profile as a compact tuple.
    #[getter]
    fn last_run_profile_ns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                self.last_medium_upload_ns,
                self.last_medium_variable_upload_ns,
                self.last_medium_static_upload_ns,
                self.last_solver_run_ns,
                self.last_materialize_ns,
                self.last_total_ns,
            ],
        )
    }

    /// Run one scan line with updated medium (sound_speed, density).
    fn run_scan_line<'py>(
        &mut self,
        _py: Python<'py>,
        _sound_speed: PyReadonlyArray3<f64>,
        _density: PyReadonlyArray3<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("GPU feature not enabled"))
        }

        #[cfg(feature = "gpu")]
        {
            use std::borrow::Cow;

            let total_t0 = std::time::Instant::now();
            let ss_arr = _sound_speed.as_array();
            let rho_arr = _density.as_array();

            let c0_flat: Cow<'_, [f64]> = match ss_arr.as_slice() {
                Some(slice) => Cow::Borrowed(slice),
                None => Cow::Owned(ss_arr.iter().copied().collect()),
            };
            let rho0_flat: Cow<'_, [f64]> = match rho_arr.as_slice() {
                Some(slice) => Cow::Borrowed(slice),
                None => Cow::Owned(rho_arr.iter().copied().collect()),
            };

            let upload_t0 = std::time::Instant::now();
            self.solver.update_medium_variable(c0_flat.as_ref(), rho0_flat.as_ref());
            let medium_upload_ns = upload_t0.elapsed().as_nanos() as u64;

            let result = self.run_scan_line_cached(_py);
            self.last_medium_variable_upload_ns = medium_upload_ns;
            self.last_medium_static_upload_ns = 0;
            self.last_medium_upload_ns = medium_upload_ns;
            self.last_total_ns = total_t0.elapsed().as_nanos() as u64;
            result
        }
    }

    /// Run one scan line using the currently resident medium buffers.
    fn run_scan_line_cached<'py>(
        &mut self,
        _py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("GPU feature not enabled"))
        }

        #[cfg(feature = "gpu")]
        {
            let total_t0 = std::time::Instant::now();
            self.last_medium_upload_ns = 0;
            self.last_medium_variable_upload_ns = 0;
            self.last_medium_static_upload_ns = 0;
            let time_steps = self.time_steps;

            let source_indices: Vec<u32> = Vec::new();
            let source_signals: Vec<f32> = Vec::new();

            let solver_t0 = std::time::Instant::now();
            let sensor_data_f32 = self.solver.run(
                &self.sensor_indices, &source_indices, &source_signals,
                &self.vel_x_indices, &self.vel_x_signals,
            );
            self.last_solver_run_ns = solver_t0.elapsed().as_nanos() as u64;

            let materialize_t0 = std::time::Instant::now();
            let n_sensors = self.sensor_indices.len();
            let out_flat: Vec<f64> = sensor_data_f32.iter().map(|&v| v as f64).collect();
            let out = ndarray::Array2::from_shape_vec((n_sensors, time_steps), out_flat)
                .expect("sensor_data shape mismatch");
            self.last_materialize_ns = materialize_t0.elapsed().as_nanos() as u64;
            if self.last_medium_upload_ns == 0 {
                self.last_total_ns = total_t0.elapsed().as_nanos() as u64;
            }

            Ok(PyArray2::from_owned_array(_py, out))
        }
    }
}
