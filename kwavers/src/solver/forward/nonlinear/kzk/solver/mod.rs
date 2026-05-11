//! KZK equation solver using Strang operator splitting.
//!
//! # References
//!
//! - Christopher PT, Parker KJ (1991). "New approaches to nonlinear diffractive
//!   field propagation." J. Acoust. Soc. Am. 90(1), 488–499.
//!   DOI: 10.1121/1.401277
//! - Tavakkoli J et al. (1998). "A new algorithm for computational simulation
//!   of focused ultrasound in inhomogeneous tissue."
//!   IEEE Trans. Ultrason. Ferroelectr. Freq. Control 45(4), 1069–1079.
//! - Strang G (1968). SIAM J. Numer. Anal. 5(3), 506–517. DOI:10.1137/0705041
//!
//! # Module layout
//!
//! - `stepping`: operator-split propagation (`step`, `solve`,
//!   `apply_diffraction`, `apply_absorption`, `apply_nonlinearity`) and the
//!   conservation-diagnostics dispatch.
//! - `observables`: real-valued physical observables derived from the
//!   internal complex pressure field (`get_pressure`, `get_time_signal`,
//!   `get_intensity`, `get_peak_pressure`).
//! - `conservation`: the `ConservationDiagnostics` trait implementation
//!   (volumetric energy / momentum / mass).
//! - `traits`: the physics-layer `KZKSolverTrait` bridge (RMS field,
//!   peak-pressure delegation, `step_forward(dz)`).

mod conservation;
mod observables;
mod stepping;
mod traits;

#[cfg(test)]
mod tests;

use ndarray::{Array2, Array3};
use std::f64::consts::PI;

use super::absorption::AbsorptionOperator;
use super::complex_parabolic_diffraction::ParabolicDiffractionOperator;
use super::nonlinearity::NonlinearOperator;
use super::KZKConfig;
use crate::math::fft::Complex64;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances, ConservationTracker,
};

/// KZK equation solver.
///
/// # Complex-field representation
///
/// The pressure field is stored as `Array3<Complex64>` with shape `[nx, ny, nt]`.
/// The real part is the physical acoustic pressure `p(x,y,τ)`.  The imaginary
/// part carries the accumulated spatial phase from the parabolic diffraction
/// propagator `H = exp(−i k_T² Δz/(2k₀))`.
///
/// Maintaining the full complex field eliminates the ~28% beam-width error that
/// arose in earlier versions when the imaginary part was discarded after each
/// diffraction half-step.  The complex-field path achieves <0.1% error on the
/// Gaussian beam Rayleigh-distance spreading test.
///
/// Physical observables (`get_intensity`, `get_peak_pressure`, `get_time_signal`,
/// `current_field`) all return real-valued quantities computed from `Re[pressure]`.
pub struct KZKSolver {
    pub(super) config: KZKConfig,
    /// Complex pressure field p(x,y,τ) = Re[p] + i·Im[p] at current z-plane.
    /// Re[p] is the physical pressure; Im[p] tracks diffraction phase.
    pub(super) pressure: Array3<Complex64>,
    /// Previous complex pressure (for time derivatives in nonlinear operator)
    pub(super) pressure_prev: Array3<Complex64>,
    /// Complex-field parabolic diffraction operator H = exp(−ik_T²Δz/(2k₀)).
    pub(super) complex_diffraction: ParabolicDiffractionOperator,
    /// Absorption operator (spectral, operates on complex waveform)
    pub(super) absorption: AbsorptionOperator,
    /// Nonlinear operator (operates on Re[p] only)
    pub(super) nonlinear: NonlinearOperator,
    /// Conservation diagnostics tracker
    pub(super) conservation_tracker: Option<ConservationTracker>,
    /// Current z-step (for tracking propagation)
    pub(super) current_z_step: usize,
    /// Current simulation time
    pub(super) current_time: f64,
}

impl std::fmt::Debug for KZKSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KZKSolver")
            .field("config", &self.config)
            .field(
                "pressure",
                &format!(
                    "Array3<Complex64> {}x{}x{}",
                    self.pressure.shape()[0],
                    self.pressure.shape()[1],
                    self.pressure.shape()[2]
                ),
            )
            .field("absorption", &self.absorption)
            .field("nonlinear", &self.nonlinear)
            .field("conservation_tracker", &self.conservation_tracker.is_some())
            .field("current_z_step", &self.current_z_step)
            .finish()
    }
}

impl KZKSolver {
    /// Create new KZK solver.
    ///
    /// Initialises the complex pressure field to zero and constructs the
    /// spectral diffraction, absorption, and nonlinear sub-operators.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: KZKConfig) -> Result<Self, String> {
        super::validate_config(&config)?;

        let pressure = Array3::<Complex64>::zeros((config.nx, config.ny, config.nt));
        let pressure_prev = Array3::<Complex64>::zeros((config.nx, config.ny, config.nt));

        let complex_diffraction = ParabolicDiffractionOperator::new(&config);
        let absorption = AbsorptionOperator::new(&config);
        let nonlinear = NonlinearOperator::new(&config);

        Ok(Self {
            config,
            pressure,
            pressure_prev,
            complex_diffraction,
            absorption,
            nonlinear,
            conservation_tracker: None,
            current_z_step: 0,
            current_time: 0.0,
        })
    }

    /// Enable conservation diagnostics with specified tolerances
    ///
    /// # Arguments
    ///
    /// * `tolerances` - Conservation tolerance parameters (absolute/relative/check_interval)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kwavers::solver::forward::nonlinear::conservation::ConservationTolerances;
    ///
    /// let mut solver = KZKSolver::new(config)?;
    /// solver.enable_conservation_diagnostics(ConservationTolerances::default());
    /// ```
    pub fn enable_conservation_diagnostics(&mut self, tolerances: ConservationTolerances) {
        let initial_energy = self.calculate_total_energy();
        let initial_momentum = self.calculate_total_momentum();
        let initial_mass = self.calculate_total_mass();

        self.conservation_tracker = Some(ConservationTracker::new(
            initial_energy,
            initial_momentum,
            initial_mass,
            tolerances,
        ));
    }

    /// Disable conservation diagnostics
    pub fn disable_conservation_diagnostics(&mut self) {
        self.conservation_tracker = None;
    }

    /// Get conservation diagnostic summary
    ///
    /// Returns a summary of all conservation checks performed,
    /// including maximum severity and error magnitudes.
    #[must_use] 
    pub fn get_conservation_summary(&self) -> Option<String> {
        self.conservation_tracker
            .as_ref()
            .map(|tracker| tracker.summary().to_string())
    }

    /// Check if solution satisfies conservation constraints
    ///
    /// Returns `true` if all conservation violations are within acceptable limits.
    #[must_use] 
    pub fn is_solution_valid(&self) -> bool {
        self.conservation_tracker
            .as_ref()
            .map_or_else(|| true, |tracker| tracker.is_solution_valid())
    }

    /// Set initial condition (source plane at z=0).
    ///
    /// Initialises the complex pressure field with a time-harmonic signal:
    ///   `Re[p(x,y,τ)] = source(x,y) · sin(ω₀τ)`, `Im[p] = 0` at z = 0.
    ///
    /// The imaginary component is zero at the source plane; it accumulates
    /// non-zero values as the field propagates through diffraction steps.
    pub fn set_source(&mut self, source: Array2<f64>, frequency: f64) {
        // Store frequency in config for all operators.
        self.config.frequency = frequency;
        // Re-initialize operators with updated frequency.
        self.complex_diffraction = ParabolicDiffractionOperator::new(&self.config);
        self.absorption = AbsorptionOperator::new(&self.config);

        // Set source as time-harmonic signal (real-valued at z=0).
        let omega = 2.0 * PI * frequency;
        let dt = self.config.dt;

        for t in 0..self.config.nt {
            let time = t as f64 * dt;
            let temporal = (omega * time).sin();

            for j in 0..self.config.ny {
                for i in 0..self.config.nx {
                    self.pressure[[i, j, t]] = Complex64::new(source[[i, j]] * temporal, 0.0);
                }
            }
        }

        self.pressure_prev.assign(&self.pressure);
    }
}
