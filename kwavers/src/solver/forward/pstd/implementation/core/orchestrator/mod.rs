//! PSTD Solver Orchestrator

use crate::core::error::KwaversResult;
use crate::domain::boundary::Boundary;
use crate::domain::field::wave::WaveFields;
use crate::domain::grid::Grid;
use crate::domain::medium::MaterialFields;
use crate::domain::sensor::recorder::simple::SensorRecorder;
use crate::domain::source::{Source, SourceInjectionMode};
use crate::math::fft::{Complex64, ProcessorFft3d};
use crate::solver::fdtd::SourceHandler;
use crate::solver::forward::pstd::implementation::k_space::PSTDKSOperators;
use crate::solver::forward::pstd::physics::absorption::AbsorptionKernel;
use crate::solver::forward::pstd::propagator::axisymmetric::AsContext;
use ndarray::{Array1, Array2, Array3, ArrayView2};
use std::env;
use std::sync::Arc;

mod checkpoint;
mod construction;
mod interface;
mod source;
mod stepping;

/// Core PSTD solver implementing the pseudospectral method
pub struct PSTDSolver {
    pub(crate) config: crate::solver::forward::pstd::config::PSTDConfig,
    pub(crate) grid: Arc<Grid>,
    pub sensor_recorder: SensorRecorder,
    pub(crate) source_handler: SourceHandler,
    pub(crate) dynamic_sources: Vec<(Arc<dyn Source>, Array3<f64>)>,
    pub(crate) source_injection_modes: Vec<SourceInjectionMode>,
    /// Spectral gradient masks for velocity sources, indexed parallel to `dynamic_sources`.
    ///
    /// Entry `i` is `Some(∂mask/∂α)` when `dynamic_sources[i]` is a `VelocityX/Y/Z` source
    /// **and** `FullKSpace` operators were available at registration time; `None` otherwise.
    /// Used in `step_forward_kspace` to inject the pressure-equivalent `−c²·amp·∂mask/∂α`.
    pub(crate) velocity_source_grad_masks: Vec<Option<Array3<f64>>>,
    pub(crate) time_step_index: usize,
    pub(crate) fft: Arc<ProcessorFft3d>,
    pub(crate) kappa: Array3<f64>,
    pub(crate) source_kappa: Array3<f64>,
    pub(crate) filter: Option<Array3<f64>>,
    pub(crate) c_ref: f64,
    /// Precomputed additive mass-source scale: 2·Δt / (N·c₀·Δx_min).
    pub(crate) mass_source_scale: f64,
    /// Construction-time diagnostic source time shift in samples.
    pub(crate) source_time_shift_samples: isize,
    /// Construction-time diagnostic pressure-source gain.
    pub(crate) source_gain: f64,
    pub(crate) k_max: f64,
    pub(crate) boundary: Option<Box<dyn Boundary>>,
    pub fields: WaveFields,
    pub rhox: Array3<f64>,
    pub rhoy: Array3<f64>,
    pub rhoz: Array3<f64>,
    pub(crate) p_k: Array3<Complex64>,
    pub(crate) ux_k: Array3<Complex64>,
    pub(crate) uy_k: Array3<Complex64>,
    pub(crate) uz_k: Array3<Complex64>,
    /// Single k-space gradient scratch, reused for all three spatial axes sequentially.
    pub(crate) grad_k: Array3<Complex64>,
    pub(crate) materials: MaterialFields,
    pub(crate) bon: Array3<f64>,
    /// Precomputed absorption kernel — `None` for lossless simulations.
    pub(crate) absorption: Option<AbsorptionKernel>,
    pub(crate) kspace_operators: Option<PSTDKSOperators>,
    pub(crate) ddx_k_shift_pos: Array1<Complex64>,
    pub(crate) ddy_k_shift_pos: Array1<Complex64>,
    pub(crate) ddz_k_shift_pos: Array1<Complex64>,
    pub(crate) ddx_k_shift_neg: Array1<Complex64>,
    pub(crate) ddy_k_shift_neg: Array1<Complex64>,
    pub(crate) ddz_k_shift_neg: Array1<Complex64>,
    pub(crate) dpx: Array3<f64>,
    pub(crate) dpy: Array3<f64>,
    pub(crate) dpz: Array3<f64>,
    pub(crate) div_u: Array3<f64>,
    /// Axisymmetric WSWA-FFT context — `Some` when `config.geometry == CylindricalAS`.
    pub(crate) as_ctx: Option<AsContext>,
}

impl PSTDSolver {
    /// Compute total density (rhox + rhoy + rhoz) into the provided buffer
    pub fn fill_rho_sum(&self, dest: &mut Array3<f64>) {
        ndarray::Zip::from(dest)
            .and(&self.rhox)
            .and(&self.rhoy)
            .and(&self.rhoz)
            .for_each(|rho_sum, &rx, &ry, &rz| {
                *rho_sum = rx + ry + rz;
            });
    }

    pub fn sensor_indices(&self) -> &[(usize, usize, usize)] {
        self.sensor_recorder.sensor_indices()
    }

    pub fn extract_pressure_data(&self) -> Option<Array2<f64>> {
        self.sensor_recorder.extract_pressure_data()
    }

    /// Borrow the full allocated sensor pressure buffer without cloning.
    #[must_use]
    pub fn pressure_data_view(&self) -> Option<ArrayView2<'_, f64>> {
        self.sensor_recorder.pressure_data_view()
    }

    /// Borrow only recorded sensor pressure samples without cloning.
    #[must_use]
    pub fn recorded_pressure_view(&self) -> Option<ArrayView2<'_, f64>> {
        self.sensor_recorder.recorded_pressure_view()
    }

    pub fn get_timestep(&self) -> f64 {
        self.config.dt
    }

    pub fn pressure_field(&self) -> &Array3<f64> {
        &self.fields.p
    }

    pub fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.fields.ux, &self.fields.uy, &self.fields.uz)
    }

    pub fn run_orchestrated(&mut self, steps: usize) -> KwaversResult<Option<Array2<f64>>> {
        if self.time_step_index == 0 {
            self.sensor_recorder.record_step(&self.fields.p)?;
        }
        for _ in 0..steps {
            self.step_forward()?;
        }
        Ok(self.sensor_recorder.extract_pressure_data())
    }
}

pub(super) fn pstd_source_time_shift_samples() -> isize {
    match env::var("KWAVERS_PSTD_SOURCE_TIME_SHIFT") {
        Ok(value) => value.trim().parse::<isize>().unwrap_or(0),
        Err(_) => 0,
    }
}

pub(super) fn pstd_source_gain() -> f64 {
    match env::var("KWAVERS_PSTD_SOURCE_GAIN") {
        Ok(value) => value.trim().parse::<f64>().unwrap_or(1.0),
        Err(_) => 1.0,
    }
}

impl std::fmt::Debug for PSTDSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PSTDSolver")
            .field("config", &self.config)
            .field("grid", &"Grid { ... }")
            .field("k_max", &self.k_max)
            .finish()
    }
}
