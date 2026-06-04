//! PSTD Solver Orchestrator

use kwavers_core::error::KwaversResult;
use kwavers_boundary::{Boundary, PmlExpFactors};
use kwavers_field::wave::WaveFields;
use kwavers_grid::Grid;
use kwavers_medium::MaterialFields;
use kwavers_domain::sensor::recorder::simple::SensorRecorder;
use kwavers_domain::source::{Source, SourceInjectionMode};
use kwavers_math::fft::{Complex64, Fft3d};
use crate::fdtd::SourceHandler;
use crate::forward::pstd::implementation::k_space::PSTDKSOperators;
use crate::forward::pstd::physics::absorption::AbsorptionKernel;
use crate::forward::pstd::physics::residual_gas_absorption::ResidualGasAbsorption;
use crate::forward::pstd::propagator::axisymmetric::AsContext;
use ndarray::{Array1, Array2, Array3, ArrayView2};
use std::env;
use std::sync::Arc;

mod checkpoint;
mod construction;
mod interface;
mod source;
mod stepping;
pub mod thermal;

/// Core PSTD solver implementing the pseudospectral method
pub struct PSTDSolver {
    pub(crate) config: crate::forward::pstd::config::PSTDConfig,
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
    pub(crate) fft: Arc<Fft3d>,
    /// k-space correction sinc(c_ref·|k|·dt) in r2c half-spectrum order.
    ///
    /// Shape: `(nx, ny, nz_c)` where `nz_c = nz/2 + 1` — pre-truncated at
    /// construction to save `nx·ny·(nz/2-1)·8` bytes vs. the full (nx,ny,nz)
    /// layout. All usage sites (velocity.rs, density_cartesian.rs) operate
    /// on the r2c half-spectrum, so the upper half was always unused.
    pub(crate) kappa: Array3<f64>,
    /// Source k-space correction cos(c_ref·|k|·dt/2) in r2c half-spectrum order.
    ///
    /// Shape: `(nx, ny, nz_c)` — pre-truncated after `set_velocity_source_kappa()`
    /// extracts per-source-point values from the full array. Saves
    /// `nx·ny·(nz/2-1)·8` bytes vs. the full layout.
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
    /// Precomputed split-field PML factors for fused velocity and density updates.
    ///
    /// `Some` when the boundary is `CPMLBoundary`; `None` otherwise (the slow
    /// `apply_pml_to_velocity` / `apply_pml_to_density` paths remain active).
    /// Contains `exp(-σ·Δt/2)` for each grid index per axis, populated once at
    /// construction so per-step cost is O(N) multiplications (no `exp` calls).
    pub(crate) pml_exp: Option<PmlExpFactors>,
    pub fields: WaveFields,
    pub rhox: Array3<f64>,
    pub rhoy: Array3<f64>,
    pub rhoz: Array3<f64>,
    pub(crate) p_k: Array3<Complex64>,
    /// Shared velocity k-space scratch — reused for all three spatial axes sequentially.
    ///
    /// Used as: (a) FFT output for u_x / u_y / u_z in the density update; (b) IFFT
    /// workspace (3rd arg of `inverse_c2r_into`) in the velocity update, absorption, and
    /// filter passes.  Because all axes are processed sequentially (never concurrently),
    /// a single buffer is sufficient — eliminating the former `uy_k` and `uz_k` fields
    /// (Opt-8: saves 2 × N_complex × 16 B of solver memory).
    pub(crate) ux_k: Array3<Complex64>,
    /// Single k-space gradient scratch, reused for all three spatial axes sequentially.
    pub(crate) grad_k: Array3<Complex64>,
    pub(crate) materials: MaterialFields,
    /// Nonlinearity parameter B/(2A) per voxel.
    ///
    /// `Some` iff `config.nonlinearity == true`; `None` for linear simulations,
    /// saving N×8 bytes per solver instance (2 MB at 64³, 16 MB at 128³).
    /// Invariant: `self.bon.is_some() ↔ self.config.nonlinearity`.
    pub(crate) bon: Option<Array3<f64>>,
    /// Precomputed absorption kernel — `None` for lossless simulations.
    pub(crate) absorption: Option<AbsorptionKernel>,
    /// Raw spectral wavenumber magnitude `|k|` in r2c half-spectrum order
    /// `(nx, ny, nz_c)`, captured before the kappa cosine transform. Used to
    /// build the broadband residual-gas absorption operator's spectral shape.
    pub(crate) k_mag_half: Array3<f64>,
    /// Broadband residual-gas (bubble-cloud) absorption operator — `None` until
    /// installed via `set_residual_gas_absorption`. Applies the true frequency-
    /// dependent Commander–Prosperetti attenuation spectrum each step.
    pub(crate) residual_gas_absorption: Option<ResidualGasAbsorption>,
    pub(crate) kspace_operators: Option<PSTDKSOperators>,
    pub(crate) ddx_k_shift_pos: Array1<Complex64>,
    pub(crate) ddy_k_shift_pos: Array1<Complex64>,
    pub(crate) ddz_k_shift_pos: Array1<Complex64>,
    pub(crate) ddx_k_shift_neg: Array1<Complex64>,
    pub(crate) ddy_k_shift_neg: Array1<Complex64>,
    pub(crate) ddz_k_shift_neg: Array1<Complex64>,
    /// Shared real-valued gradient IFFT scratch — reused for all three velocity axes
    /// sequentially and for the absorption L1 accumulator.
    ///
    /// **Usage pattern (per step):**
    /// 1. Velocity update: each axis writes its pressure gradient (IFFT output) into `dpx`
    ///    and the Zip immediately reads it — sequential axes never overlap (Opt-12).
    /// 2. Absorption Step 1: writes ρ₀·(div_ux + div_uy + div_uz) into `dpx`.
    /// 3. Absorption Step 3: FFT(dpx) → grad_k; scale by nabla1; IFFT → dpx (L1).
    ///    Safe: Step 1 content is consumed by the FFT before the IFFT overwrites it.
    /// 4. Absorption Step 5: reads `dpx` (L1) and `dpy` (L2) simultaneously.
    ///
    /// `dpz` was eliminated (Opt-12) because its role (velocity z-gradient scratch and
    /// absorption Step 1 accumulator) is covered by `dpx` without temporal overlap.
    pub(crate) dpx: Array3<f64>,
    /// L2 scratch: holds `IFFT(|k|^(y−1)·FFT(ρ_total))` during absorption Step 4.
    ///
    /// Also used as velocity y-axis gradient IFFT buffer (fallback and fused paths).
    /// Must be a separate allocation from `dpx` because absorption Step 5 reads
    /// `dpx` (L1) and `dpy` (L2) simultaneously.
    pub(crate) dpy: Array3<f64>,
    pub(crate) div_u: Array3<f64>,
    /// Cached kappa-corrected velocity divergences written by `update_density_cartesian`
    /// and read by `apply_absorption_to_pressure`.  `apply_pressure_sources` zeroes `dpx`
    /// between those two calls; caching avoids 3 forward + 3 inverse FFT recomputations
    /// per step on absorbing simulations.  `div_uy` and `div_uz` are zero when the
    /// corresponding axis is singleton (ny=1 or nz=1).  Not checkpointed — recomputed
    /// from velocity fields on the first step after restore.
    pub(crate) div_ux: Array3<f64>,
    pub(crate) div_uy: Array3<f64>,
    pub(crate) div_uz: Array3<f64>,
    /// Axisymmetric WSWA-FFT context — `Some` when `config.geometry == CylindricalAS`.
    pub(crate) as_ctx: Option<AsContext>,
    /// Per-cell acoustic absorption coefficient α(ω_c) [Np/m] at the simulation center frequency.
    ///
    /// `Some` after `populate_alpha_np_m_at_frequency()` or `set_alpha_np_m()` is called.
    /// `None` at construction and for lossless simulations (no thermal heating).
    /// `compute_acoustic_heat_source()` returns zeros when `None`.
    /// Invariant: `None` ⟺ thermal coupling has not been requested.
    /// Memory savings: N×8 bytes avoided for non-thermal simulations (2 MB at 64³, 16 MB at 128³).
    pub(crate) alpha_np_m: Option<Array3<f64>>,
    /// X-row indices at which split-field PML is bypassed during TR Dirichlet reconstruction.
    ///
    /// When `enforce_pressure_dirichlet` forces p[sensor] = data[t], the sensor cell must
    /// emit outward waves into the domain.  The split-field PML at x=0 applies
    /// exp(-sigma_max * dt/2) to velocity and density twice per step, suppressing the
    /// driven amplitude by exp(-sigma_max * dt) ≈ 0.66 per step and collapsing the
    /// reconstruction amplitude to ~0 after many steps.  Setting this list to the sensor
    /// x-indices causes apply_pml_to_velocity and apply_pml_to_density to skip PML at
    /// those rows (save before, restore after), matching KWave.jl's CPML bypass at TR
    /// source cells and allowing the Dirichlet pressure to drive waves normally.
    pub(crate) dirichlet_pml_bypass_x: Vec<usize>,
    /// Reusable yz-plane scratch for preserving Dirichlet-PML bypass rows.
    ///
    /// Shape is `(dirichlet_pml_bypass_x.len(), ny, nz)`. Velocity and density
    /// components reuse the same buffer sequentially, eliminating per-step
    /// `Array2` allocations in the bypass path.
    pub(crate) pml_bypass_plane_scratch: Array3<f64>,
}

impl PSTDSolver {
    /// Compute total density (rhox + rhoy + rhoz) into the provided buffer
    pub fn fill_rho_sum(&self, dest: &mut Array3<f64>) {
        ndarray::Zip::from(dest)
            .and(&self.rhox)
            .and(&self.rhoy)
            .and(&self.rhoz)
            .par_for_each(|rho_sum, &rx, &ry, &rz| {
                *rho_sum = rx + ry + rz;
            });
    }

    #[must_use]
    pub fn sensor_indices(&self) -> &[(usize, usize, usize)] {
        self.sensor_recorder.sensor_indices()
    }

    #[must_use]
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

    #[must_use]
    pub fn get_timestep(&self) -> f64 {
        self.config.dt
    }

    /// Register x-row indices at which the split-field PML is bypassed.
    ///
    /// Use during TR Dirichlet reconstruction: sensor cells at the PML boundary
    /// must emit waves into the domain; bypassing PML at those rows allows the
    /// forced pressure to drive velocity and density without split-field damping.
    pub fn set_dirichlet_pml_bypass_x(&mut self, rows: Vec<usize>) {
        self.dirichlet_pml_bypass_x = rows;
        self.resize_pml_bypass_scratch();
    }

    /// Pressure field.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn pressure_field(&self) -> &Array3<f64> {
        &self.fields.p
    }
    /// Velocity fields.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.fields.ux, &self.fields.uy, &self.fields.uz)
    }
    /// Run orchestrated.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn run_orchestrated(&mut self, steps: usize) -> KwaversResult<Option<Array2<f64>>> {
        // Record the initial state only for IVP (p0) sources, matching k-Wave's convention:
        // k-Wave's step t=0 overrides the computed pressure with p0 before recording, so
        // sensor column 0 = p0.  For time-varying sources k-Wave records p(dt) at column 0
        // (state after the first update), so we must NOT emit the zero initial state here —
        // doing so would shift every time-varying source output by +1 step vs k-Wave.
        if self.time_step_index == 0 && self.source_handler.has_initial_pressure() {
            self.sensor_recorder.record_step(&self.fields.p)?;
        }
        for _ in 0..steps {
            self.step_forward()?;
        }
        Ok(self.sensor_recorder.extract_pressure_data())
    }
}

/// Pstd source time shift samples.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(super) fn pstd_source_time_shift_samples() -> isize {
    match env::var("KWAVERS_PSTD_SOURCE_TIME_SHIFT") {
        Ok(value) => value.trim().parse::<isize>().unwrap_or(0),
        Err(_) => 0,
    }
}

/// Pstd source gain.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
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
