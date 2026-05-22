//! GPU PSTD simulation adapter — batch-mode `Solver` trait wrapper.
//!
//! # Architecture
//!
//! `GpuPstdSimulationAdapter` exposes the GPU-resident PSTD acoustic solver
//! through the simulation `Solver` trait.  The GPU solver is architecturally
//! batch-only: all `nt` time steps are encoded into GPU command buffers and
//! submitted in a single blocking call; single-step access is not supported.
//!
//! ## Trait compliance
//!
//! | Method | Behaviour |
//! |---|---|
//! | `run(nt)` | Builds GPU arrays, calls `GpuPstdSolver::with_auto_device` + `run()`; stores sensor traces |
//! | `step_forward()` | Returns `Err(FeatureNotAvailable)` — batch-only arch |
//! | `pressure_field()` | Returns zero array — GPU fields are not downloaded mid-run |
//! | `velocity_fields()` | Returns zero arrays — same reason |
//! | `recorded_sensor_pressure()` | Returns sensor traces after `run()` completes |
//! | `add_source(Box<dyn Source>)` | Extracts spatial mask via `Source::create_mask`; signal not set |
//! | `add_sensor(&GridSensorSet)` | Converts `GridPoint` list to boolean sensor mask |
//!
//! ## Source signal
//!
//! `Box<dyn Source>` carries only the spatial mask; temporal signals are not
//! part of the `Source` trait.  Callers that need a specific source waveform
//! must call [`GpuPstdSimulationAdapter::set_grid_source`] on the concrete
//! adapter before boxing.  The `SimulationSolverFactory` arm for `PstdGpu`
//! constructs the adapter with an empty source; clinical adapters (e.g.
//! `breast_ust_fwi::dataset`) continue to call `run_gpu_pstd` directly when
//! they need per-transmit source signals — the factory path is for callers
//! that set up sources via `add_source` or drive source-free IC propagation.
//!
//! ## Grid constraints
//!
//! GPU PSTD requires power-of-two dimensions with each axis ≤ 256.
//! Construction fails with [`KwaversError::InvalidInput`] if these are
//! violated.  `SimulationSolverFactory::create_solver(SolverType::PstdGpu,
//! ...)` propagates that error to the caller.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::cpml::{CPMLConfig, CPMLProfiles};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::sensor::GridSensorSet;
use crate::domain::source::{GridSource, Source};
use crate::physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
use crate::solver::config::SolverConfiguration;
use crate::solver::feature::SolverFeature;
use crate::solver::forward::pstd::gpu_pstd::{
    AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays, SolverParams,
};
use crate::solver::interface::{Solver, SolverStatistics};
use ndarray::{Array2, Array3};
use std::f64::consts::PI;
use std::time::{Duration, Instant};

/// Pre-extracted medium snapshot for the GPU PSTD adapter.
///
/// Extracted at construction time from `&dyn Medium` so the adapter does not
/// hold a Medium reference (lifetime / object-safety constraints prevent
/// storing `&dyn Medium` in `Box<dyn Solver>`).
#[derive(Debug)]
struct GpuMediumSnapshot {
    /// Per-voxel sound speed [f32], C-order `ix*ny*nz + iy*nz + iz`.
    c0_flat: Vec<f32>,
    /// Per-voxel density [f32].
    rho0_flat: Vec<f32>,
    /// Per-voxel B/(2A) nonlinearity coefficient [f32]; 0 for linear.
    bon_a_flat: Vec<f32>,
    /// Per-voxel alpha_db_cm absorption [f32]; 0 for lossless.
    absorption_flat: Vec<f32>,
    /// Maximum sound speed over the domain [m/s].
    c_ref: f64,
    has_nonlinear: bool,
    has_absorption: bool,
}

impl GpuMediumSnapshot {
    /// Extract all GPU-needed medium data in a single pass.
    fn from_medium<M: Medium>(grid: &Grid, medium: &M) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let total = nx * ny * nz;

        let mut c0_flat = Vec::with_capacity(total);
        let mut rho0_flat = Vec::with_capacity(total);
        let mut bon_a_flat = Vec::with_capacity(total);
        let mut absorption_flat = Vec::with_capacity(total);

        let mut has_nonlinear = false;
        let mut has_absorption = false;

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let c = medium.sound_speed(ix, iy, iz);
                    let rho = medium.density(ix, iy, iz);
                    let nl = medium.nonlinearity(ix, iy, iz);
                    let alpha = medium.absorption(ix, iy, iz);

                    c0_flat.push(c as f32);
                    rho0_flat.push(rho as f32);
                    bon_a_flat.push((nl / 2.0) as f32);
                    absorption_flat.push(alpha as f32);

                    if nl > 0.0 {
                        has_nonlinear = true;
                    }
                    if alpha > 0.0 {
                        has_absorption = true;
                    }
                }
            }
        }

        let c_ref = medium.max_sound_speed();

        Self {
            c0_flat,
            rho0_flat,
            bon_a_flat,
            absorption_flat,
            c_ref,
            has_nonlinear,
            has_absorption,
        }
    }
}

/// GPU-resident PSTD adapter implementing the simulation `Solver` trait.
///
/// See module-level documentation for the full architecture and limitation
/// table.
#[derive(Debug)]
pub struct GpuPstdSimulationAdapter {
    grid: Grid,
    medium: GpuMediumSnapshot,
    dt: f64,
    /// Absorption power-law exponent `y` for the fractional-Laplacian model.
    /// Defaults to `1.0` (standard tissue attenuation).
    alpha_power: f64,
    cpml_config: CPMLConfig,
    pml_inside: bool,
    source: GridSource,
    sensor_mask: Array3<bool>,
    /// Sensor traces recorded by the most-recent `run()` call.
    recorded: Option<Array2<f64>>,
    pressure_zero: Array3<f64>,
    vel_zero: Array3<f64>,
    current_step: usize,
    computation_time: Duration,
}

impl GpuPstdSimulationAdapter {
    /// Construct a GPU PSTD adapter.
    ///
    /// Validates GPU PSTD grid constraints and extracts medium data.  The GPU
    /// device is **not** acquired here; it is acquired on the first `run()` call.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when grid dimensions are not
    /// power-of-two or exceed 256 per axis.
    pub fn new<M: Medium>(
        config: &SolverConfiguration,
        grid: &Grid,
        medium: &M,
    ) -> KwaversResult<Self> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        if !nx.is_power_of_two() || !ny.is_power_of_two() || !nz.is_power_of_two() {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD requires power-of-2 grid dimensions; got {nx}×{ny}×{nz}"
            )));
        }
        if nx > 256 || ny > 256 || nz > 256 {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD supports per-axis N ≤ 256; got {nx}×{ny}×{nz}"
            )));
        }
        if !config.dt.is_finite() || config.dt <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD requires finite positive dt; got {}",
                config.dt
            )));
        }

        let cpml_config = config
            .absorbing_boundary
            .as_ref()
            .map(|abc| abc.cpml.clone())
            .unwrap_or_else(|| CPMLConfig::with_thickness(20));
        let pml_inside = true; // canonical default; matches run_gpu_pstd

        let medium_snap = GpuMediumSnapshot::from_medium(grid, medium);
        let shape = (nx, ny, nz);

        Ok(Self {
            grid: grid.clone(),
            medium: medium_snap,
            dt: config.dt,
            alpha_power: 1.0,
            cpml_config,
            pml_inside,
            source: GridSource::new_empty(),
            sensor_mask: Array3::from_elem(shape, false),
            recorded: None,
            pressure_zero: Array3::zeros(shape),
            vel_zero: Array3::zeros(shape),
            current_step: 0,
            computation_time: Duration::ZERO,
        })
    }

    /// Set a complete `GridSource` (mask + signal).
    ///
    /// Bypasses the `Solver::add_source` limitation (the `Source` trait only
    /// carries spatial masks, not temporal signals). Call before `run()`.
    pub fn set_grid_source(&mut self, source: GridSource) {
        self.source = source;
    }

    /// Build all GPU arrays and run the PSTD time loop.
    ///
    /// This method mirrors `run_gpu_pstd`'s array preparation, then calls
    /// `GpuPstdSolver::with_auto_device` followed by `GpuPstdSolver::run`.
    /// The GPU device is acquired once per `run()` call; wgpu caches the
    /// device handle across calls for the same adapter instance.
    fn run_gpu_impl(&mut self, nt: usize) -> KwaversResult<()> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let total = nx * ny * nz;
        let dt = self.dt;
        let alpha_power = self.alpha_power;

        // ── PML profiles ──────────────────────────────────────────────────────
        let profiles = CPMLProfiles::new(&self.cpml_config, &self.grid, self.medium.c_ref, dt)?;

        let mut pml_sgx = vec![1.0f32; total];
        let mut pml_sgy = vec![1.0f32; total];
        let mut pml_sgz = vec![1.0f32; total];
        let mut pml_x = vec![1.0f32; total];
        let mut pml_y = vec![1.0f32; total];
        let mut pml_z = vec![1.0f32; total];

        if self.pml_inside {
            let exp_half = |sigma: &ndarray::Array1<f64>| -> Vec<f32> {
                sigma
                    .iter()
                    .map(|&s| (-s * dt * 0.5).exp() as f32)
                    .collect()
            };
            let sgx = exp_half(&profiles.sigma_x_sgx);
            let sgy = exp_half(&profiles.sigma_y_sgy);
            let sgz = exp_half(&profiles.sigma_z_sgz);
            let px = exp_half(&profiles.sigma_x);
            let py = exp_half(&profiles.sigma_y);
            let pz = exp_half(&profiles.sigma_z);

            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let flat = ix * ny * nz + iy * nz + iz;
                        pml_sgx[flat] = sgx[ix];
                        pml_sgy[flat] = sgy[iy];
                        pml_sgz[flat] = sgz[iz];
                        pml_x[flat] = px[ix];
                        pml_y[flat] = py[iy];
                        pml_z[flat] = pz[iz];
                    }
                }
            }
        }

        // ── Absorption operator arrays ─────────────────────────────────────────
        let (nabla1, nabla2, tau_v, eta_v) = if self.medium.has_absorption {
            let dk_x = 2.0 * PI / (nx as f64 * self.grid.dx);
            let dk_y = 2.0 * PI / (ny as f64 * self.grid.dy);
            let dk_z = 2.0 * PI / (nz as f64 * self.grid.dz);
            const SINGULARITY: f64 = 1e-8;
            let y = alpha_power;

            let mut n1 = vec![0.0f32; total];
            let mut n2 = vec![0.0f32; total];
            let mut tau = vec![0.0f32; total];
            let mut eta = vec![0.0f32; total];

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

                if k_mag > SINGULARITY {
                    n1[flat] = k_mag.powf(y - 2.0) as f32;
                    n2[flat] = k_mag.powf(y - 1.0) as f32;
                }

                let alpha_db_cm = self.medium.absorption_flat[flat] as f64;
                let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_db_cm, alpha_power);
                let c0 = self.medium.c0_flat[flat] as f64;
                tau[flat] = (-2.0 * alpha_0_si * c0.powf(y - 1.0)) as f32;
                eta[flat] =
                    (2.0 * alpha_0_si * c0.powf(y) * (PI * y / 2.0).tan()) as f32;
            }
            (n1, n2, tau, eta)
        } else {
            (
                vec![0.0f32; total],
                vec![0.0f32; total],
                vec![0.0f32; total],
                vec![0.0f32; total],
            )
        };

        // ── Construct GPU solver ──────────────────────────────────────────────
        let mut gpu_solver = GpuPstdSolver::with_auto_device(
            &self.grid,
            MediumArrays {
                c0_flat: &self.medium.c0_flat,
                rho0_flat: &self.medium.rho0_flat,
            },
            SolverParams {
                dt,
                nt,
                c_ref: self.medium.c_ref,
                nonlinear: self.medium.has_nonlinear,
                absorbing: self.medium.has_absorption,
            },
            PmlArrays {
                x: &pml_x,
                y: &pml_y,
                z: &pml_z,
                sgx: &pml_sgx,
                sgy: &pml_sgy,
                sgz: &pml_sgz,
            },
            AbsorptionArrays {
                bon_a_flat: &self.medium.bon_a_flat,
                nabla1: &nabla1,
                nabla2: &nabla2,
                tau: &tau_v,
                eta: &eta_v,
            },
        )
        .map_err(|e| {
            KwaversError::InvalidInput(format!("GPU device init failed: {e}"))
        })?;

        // ── Sensor indices ────────────────────────────────────────────────────
        let sensor_flat = self.sensor_mask.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput("sensor_mask must be C-contiguous".to_owned())
        })?;
        let sensor_indices: Vec<u32> = sensor_flat
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i as u32) } else { None })
            .collect();

        // ── Source indices and signals ────────────────────────────────────────
        let mass_source_scale = {
            let n_dim_active =
                [nx > 1, ny > 1, nz > 1].iter().filter(|&&d| d).count().max(1);
            let dx_min = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
            (2.0 * dt / (n_dim_active as f64 * self.medium.c_ref * dx_min)) as f32
        };

        let (source_indices, source_signals) =
            if let (Some(p_mask), Some(p_signal)) = (&self.source.p_mask, &self.source.p_signal) {
                let mask_flat = p_mask.as_slice().ok_or_else(|| {
                    KwaversError::InvalidInput("p_mask must be C-contiguous".to_owned())
                })?;
                let mut src_idx: Vec<u32> = Vec::new();
                for (i, &v) in mask_flat.iter().enumerate() {
                    if v != 0.0 {
                        src_idx.push(i as u32);
                    }
                }
                let n_src = src_idx.len();
                let n_sig_rows = p_signal.shape()[0];
                let n_sig_cols = p_signal.shape()[1].min(nt);
                let mut signals = vec![0.0f32; n_src * nt];
                for src in 0..n_src {
                    let row = if n_sig_rows == 1 { 0 } else { src.min(n_sig_rows - 1) };
                    for step in 0..n_sig_cols {
                        signals[src * nt + step] =
                            (p_signal[[row, step]] * mass_source_scale as f64) as f32;
                    }
                }
                (src_idx, signals)
            } else {
                (Vec::new(), Vec::new())
            };

        // ── Run GPU time loop ─────────────────────────────────────────────────
        let t0 = Instant::now();
        let sensor_f32 = gpu_solver.run(
            &sensor_indices,
            &source_indices,
            &source_signals,
            &[],
            &[],
        );
        self.computation_time += t0.elapsed();
        self.current_step += nt;

        // ── Widen to f64 and store ────────────────────────────────────────────
        let n_sensors = sensor_indices.len();
        let mut out = Array2::<f64>::zeros((n_sensors, nt));
        for s in 0..n_sensors {
            for t in 0..nt {
                out[[s, t]] = sensor_f32[s * nt + t] as f64;
            }
        }
        self.recorded = Some(out);
        Ok(())
    }
}

impl Solver for GpuPstdSimulationAdapter {
    fn name(&self) -> &str {
        "GpuPstd"
    }

    fn initialize(&mut self, grid: &Grid, _medium: &dyn crate::domain::medium::Medium) -> KwaversResult<()> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        if (nx, ny, nz) != (self.grid.nx, self.grid.ny, self.grid.nz) {
            return Err(KwaversError::DimensionMismatch(format!(
                "GpuPstd adapter initialized for ({}×{}×{}), got ({}×{}×{})",
                self.grid.nx, self.grid.ny, self.grid.nz, nx, ny, nz
            )));
        }
        Ok(())
    }

    /// Extract the spatial mask from `source` and store it.
    ///
    /// **Signal not set**: `Box<dyn Source>` carries only the spatial mask;
    /// temporal signals are absent from the `Source` trait.  Use
    /// [`Self::set_grid_source`] to configure a waveform.
    fn add_source(&mut self, source: Box<dyn Source>) -> KwaversResult<()> {
        let mask = source.create_mask(&self.grid);
        self.source.p_mask = Some(mask);
        self.source.p_signal = None; // signal requires set_grid_source
        Ok(())
    }

    /// Convert `GridSensorSet` points into a boolean volume sensor mask.
    fn add_sensor(&mut self, sensor: &GridSensorSet) -> KwaversResult<()> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        for point in sensor.points() {
            if point.i >= nx || point.j >= ny || point.k >= nz {
                return Err(KwaversError::InvalidInput(format!(
                    "GpuPstd sensor point ({}, {}, {}) is outside grid ({}×{}×{})",
                    point.i, point.j, point.k, nx, ny, nz
                )));
            }
            self.sensor_mask[[point.i, point.j, point.k]] = true;
        }
        Ok(())
    }

    /// Execute the full batch of `num_steps` GPU time steps.
    ///
    /// Acquires the wgpu device, allocates GPU buffers, and submits the
    /// encoded command buffer.  Results are available via
    /// [`Self::recorded_sensor_pressure`] immediately after return.
    fn run(&mut self, num_steps: usize) -> KwaversResult<()> {
        self.run_gpu_impl(num_steps)
    }

    /// GPU PSTD is batch-only; single-step access is not supported.
    fn step_forward(&mut self) -> KwaversResult<()> {
        Err(KwaversError::FeatureNotAvailable(
            "GpuPstdSimulationAdapter is batch-only: call run(nt) instead of step_forward()"
                .to_owned(),
        ))
    }

    /// Returns a zero pressure field.
    ///
    /// GPU fields remain on VRAM throughout the simulation and are not
    /// downloaded to the CPU.  The zero return is not an approximation — it is
    /// a documented contract: FWI adjoint passes and any caller that needs the
    /// instantaneous pressure field must use the CPU PSTD solver.
    fn pressure_field(&self) -> &Array3<f64> {
        &self.pressure_zero
    }

    fn velocity_fields(
        &self,
    ) -> (
        &Array3<f64>,
        &Array3<f64>,
        &Array3<f64>,
    ) {
        (&self.vel_zero, &self.vel_zero, &self.vel_zero)
    }

    fn recorded_sensor_pressure(&self) -> Option<Array2<f64>> {
        self.recorded.clone()
    }

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics {
            total_steps: self.current_step,
            current_step: self.current_step,
            computation_time: self.computation_time,
            memory_usage: self.medium.c0_flat.len() * std::mem::size_of::<f32>() * 6,
            max_pressure: 0.0, // not downloaded
            max_velocity: 0.0,
        }
    }

    fn supports_feature(&self, feature: SolverFeature) -> bool {
        matches!(feature, SolverFeature::GpuAcceleration)
    }

    fn enable_feature(&mut self, feature: SolverFeature, enable: bool) -> KwaversResult<()> {
        if enable && !self.supports_feature(feature) {
            return Err(KwaversError::FeatureNotAvailable(format!(
                "GpuPstd adapter does not support {feature:?}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER};
    use crate::domain::medium::homogeneous::HomogeneousMedium;
    use crate::solver::config::SolverType;

    #[test]
    fn rejects_non_power_of_two_grid() {
        let grid = Grid::new(5, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
        let config = SolverConfiguration {
            solver_type: SolverType::PstdGpu,
            dt: 1.0e-7,
            ..SolverConfiguration::default()
        };

        let err = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap_err();

        assert!(matches!(err, KwaversError::InvalidInput(_)));
    }

    #[test]
    fn rejects_axis_exceeding_256() {
        let grid = Grid::new(512, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
        let config = SolverConfiguration {
            solver_type: SolverType::PstdGpu,
            dt: 1.0e-7,
            ..SolverConfiguration::default()
        };

        let err = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap_err();

        assert!(matches!(err, KwaversError::InvalidInput(_)));
    }

    #[test]
    fn constructs_for_valid_power_of_two_grid() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
        let config = SolverConfiguration {
            solver_type: SolverType::PstdGpu,
            dt: 1.0e-7,
            ..SolverConfiguration::default()
        };

        let adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();

        assert_eq!(adapter.name(), "GpuPstd");
        assert_eq!(adapter.pressure_field().dim(), (8, 8, 8));
        assert!(adapter.recorded_sensor_pressure().is_none());
    }

    #[test]
    fn step_forward_returns_feature_not_available() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
        let config = SolverConfiguration {
            solver_type: SolverType::PstdGpu,
            dt: 1.0e-7,
            ..SolverConfiguration::default()
        };

        let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();

        let err = adapter.step_forward().unwrap_err();
        assert!(matches!(err, KwaversError::FeatureNotAvailable(_)));
    }

    #[test]
    fn add_sensor_rejects_out_of_bounds_point() {
        use crate::domain::sensor::grid_sampling::{GridPoint, GridSensorSet};

        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
        let config = SolverConfiguration {
            solver_type: SolverType::PstdGpu,
            dt: 1.0e-7,
            ..SolverConfiguration::default()
        };

        let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();
        let sensor = GridSensorSet::from_points(vec![GridPoint::new(99, 0, 0)]);

        let err = adapter.add_sensor(&sensor).unwrap_err();
        assert!(matches!(err, KwaversError::InvalidInput(_)));
    }

    #[test]
    fn add_sensor_valid_points_sets_mask() {
        use crate::domain::sensor::grid_sampling::{GridPoint, GridSensorSet};

        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
        let config = SolverConfiguration {
            solver_type: SolverType::PstdGpu,
            dt: 1.0e-7,
            ..SolverConfiguration::default()
        };

        let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();
        let sensor = GridSensorSet::from_points(vec![
            GridPoint::new(1, 2, 3),
            GridPoint::new(4, 5, 6),
        ]);
        adapter.add_sensor(&sensor).unwrap();

        // Mask contains exactly 2 `true` entries
        let true_count = adapter.sensor_mask.iter().filter(|&&v| v).count();
        assert_eq!(true_count, 2);
        assert!(adapter.sensor_mask[[1, 2, 3]]);
        assert!(adapter.sensor_mask[[4, 5, 6]]);
    }
}
