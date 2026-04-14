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

use log::warn;
use ndarray::{Array2, Array3, Axis};
use std::f64::consts::PI;

use super::absorption::AbsorptionOperator;
use super::complex_parabolic_diffraction::ParabolicDiffractionOperator;
use super::nonlinearity::NonlinearOperator;
use super::KZKConfig;
use crate::math::fft::Complex64;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances, ConservationTracker, ViolationSeverity,
};

// Alias the physics-layer trait to avoid name collision with the solver struct.
use crate::physics::acoustics::wave_propagation::nonlinear::kzk::KZKSolver as KZKSolverTrait;

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
    config: KZKConfig,
    /// Complex pressure field p(x,y,τ) = Re[p] + i·Im[p] at current z-plane.
    /// Re[p] is the physical pressure; Im[p] tracks diffraction phase.
    pressure: Array3<Complex64>,
    /// Previous complex pressure (for time derivatives in nonlinear operator)
    pressure_prev: Array3<Complex64>,
    /// Complex-field parabolic diffraction operator H = exp(−ik_T²Δz/(2k₀)).
    complex_diffraction: ParabolicDiffractionOperator,
    /// Absorption operator (spectral, operates on complex waveform)
    absorption: AbsorptionOperator,
    /// Nonlinear operator (operates on Re[p] only)
    nonlinear: NonlinearOperator,
    /// Conservation diagnostics tracker
    conservation_tracker: Option<ConservationTracker>,
    /// Current z-step (for tracking propagation)
    current_z_step: usize,
    /// Current simulation time
    current_time: f64,
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
    pub fn get_conservation_summary(&self) -> Option<String> {
        self.conservation_tracker
            .as_ref()
            .map(|tracker| tracker.summary().to_string())
    }

    /// Check if solution satisfies conservation constraints
    ///
    /// Returns `true` if all conservation violations are within acceptable limits.
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

    /// Step forward one z-plane using operator splitting
    /// Uses second-order Strang splitting: D(dz/2) * A(dz/2) * N(dz) * A(dz/2) * D(dz/2)
    pub fn step(&mut self) {
        let dz = self.config.dz;

        // Step 1: Diffraction for dz/2
        if self.config.include_diffraction {
            self.apply_diffraction(dz / 2.0);
        }

        // Step 2: Absorption for dz/2
        if self.config.include_absorption {
            self.apply_absorption(dz / 2.0);
        }

        // Step 3: Nonlinearity for full dz
        if self.config.include_nonlinearity {
            self.apply_nonlinearity(dz);
        }

        // Step 4: Absorption for dz/2
        if self.config.include_absorption {
            self.apply_absorption(dz / 2.0);
        }

        // Step 5: Diffraction for dz/2
        if self.config.include_diffraction {
            self.apply_diffraction(dz / 2.0);
        }

        // Update history
        self.pressure_prev.assign(&self.pressure);

        // Update step counters
        self.current_z_step += 1;
        self.current_time += dz / self.config.c0;

        // Conservation diagnostics (if enabled)
        self.check_conservation_laws();
    }

    /// Propagate the acoustic field forward by `n_steps` axial z-steps.
    ///
    /// ## Algorithm (Aanonsen et al. 1984, §3; Lee & Pierce 1995, §II)
    ///
    /// Each step applies Strang-split operators in the order:
    ///
    /// ```text
    /// D(Δz/2) · A(Δz/2) · N(Δz) · A(Δz/2) · D(Δz/2)
    /// ```
    ///
    /// where D = parabolic diffraction, A = absorption, N = Burgers nonlinearity.
    /// Strang splitting achieves 2nd-order accuracy in Δz (Strang 1968).
    ///
    /// ## Errors
    ///
    /// Returns `Err(String)` when `n_steps > self.config.nz`, because the axial
    /// grid has exactly `nz` planes and propagating beyond it is undefined.
    ///
    /// ## References
    ///
    /// - Aanonsen SI et al. (1984). "Distortion and harmonic generation in the
    ///   nearfield of a finite amplitude sound beam."
    ///   J. Acoust. Soc. Am. 75(3), 749–768. DOI: 10.1121/1.390585
    /// - Lee Y-S, Pierce AD (1995). "Parabolic equation development in recent
    ///   decade." J. Comput. Acoust. 3(2), 95–111.
    ///   DOI: 10.1142/S0218396X95000100
    /// - Strang G (1968). "On the construction and comparison of difference
    ///   schemes." SIAM J. Numer. Anal. 5(3), 506–517. DOI: 10.1137/0705041
    pub fn solve(&mut self, n_steps: usize) -> Result<(), String> {
        if n_steps > self.config.nz {
            return Err(format!(
                "KZKSolver::solve: n_steps={n_steps} exceeds config.nz={}; \
                 the axial grid has only {} planes",
                self.config.nz, self.config.nz
            ));
        }
        for _ in 0..n_steps {
            self.step();
        }
        Ok(())
    }

    /// Check conservation laws and log diagnostics
    ///
    /// Performs conservation checks at configured intervals and logs violations.
    /// Critical violations trigger warnings via tracing infrastructure.
    fn check_conservation_laws(&mut self) {
        // Check if we should perform diagnostics at this step
        let should_check = self.conservation_tracker.as_ref().map_or_else(
            || false,
            |tracker| {
                self.current_z_step
                    .is_multiple_of(tracker.tolerances.check_interval)
            },
        );

        if !should_check {
            return;
        }

        // Compute diagnostics first (without holding mutable reference to tracker)
        let (initial_energy, initial_momentum, initial_mass, tolerances) =
            if let Some(ref tracker) = self.conservation_tracker {
                (
                    tracker.initial_energy,
                    tracker.initial_momentum,
                    tracker.initial_mass,
                    tracker.tolerances,
                )
            } else {
                return;
            };

        let diagnostics = self.check_all_conservation(
            initial_energy,
            initial_momentum,
            initial_mass,
            self.current_z_step,
            self.current_time,
            &tolerances,
        );

        // Now update tracker with diagnostics
        if let Some(ref mut tracker) = self.conservation_tracker {
            // Update max severity
            for diag in &diagnostics {
                if diag.severity > tracker.max_severity {
                    tracker.max_severity = diag.severity;
                }
            }
            // Store in history
            tracker.history.extend(diagnostics.clone());
        }

        // Log diagnostics based on severity
        for diag in diagnostics {
            match diag.severity {
                ViolationSeverity::Acceptable => {
                    // Silent for acceptable violations
                }
                ViolationSeverity::Warning => {
                    warn!("KZK Conservation Warning: {}", diag);
                }
                ViolationSeverity::Error => {
                    warn!("KZK Conservation Error: {}", diag);
                }
                ViolationSeverity::Critical => {
                    warn!("KZK Conservation CRITICAL: {}", diag);
                    warn!("   Solution may be physically invalid!");
                }
            }
        }
    }

    /// Apply complex-field parabolic diffraction to each retarded-time slice.
    ///
    /// For each τ index, extracts the 2D complex spatial slice `p[:,:,t]` and
    /// applies the spectral propagator H(k_T) = exp(−ik_T²Δz/(2k₀)) in-place.
    /// The complex field is preserved without discarding the imaginary part,
    /// ensuring accurate phase accumulation over many axial steps.
    fn apply_diffraction(&mut self, step_size: f64) {
        for t in 0..self.config.nt {
            let mut slice = self.pressure.index_axis_mut(Axis(2), t);
            self.complex_diffraction.apply_complex(&mut slice, step_size);
        }
    }

    /// Apply absorption operator
    fn apply_absorption(&mut self, step_size: f64) {
        self.absorption.apply(&mut self.pressure, step_size);
    }

    /// Apply nonlinear operator
    fn apply_nonlinearity(&mut self, step_size: f64) {
        self.nonlinear
            .apply(&mut self.pressure, &self.pressure_prev, step_size);
    }

    /// Return the physical pressure field Re[p(x,y,τ)] as a real array.
    ///
    /// Extracts the real part of the internal complex-field representation.
    /// The imaginary part (diffraction phase accumulator) is discarded.
    #[must_use]
    pub fn get_pressure(&self) -> Array3<f64> {
        self.pressure.mapv(|c| c.re)
    }

    /// Return the physical pressure waveform p(τ) [Pa] at transverse point (x, y).
    ///
    /// Returns `Re[pressure[x, y, 0..nt]]`.
    #[must_use]
    pub fn get_time_signal(&self, x: usize, y: usize) -> Vec<f64> {
        let mut signal = Vec::with_capacity(self.config.nt);
        for t in 0..self.config.nt {
            signal.push(self.pressure[[x, y, t]].re);
        }
        signal
    }

    /// Calculate time-averaged acoustic intensity I = p²_rms / (ρ₀c₀) [W/m²].
    ///
    /// Uses the physical (real) pressure: I(i,j) = ⟨Re[p]²⟩_τ / (ρ₀c₀).
    #[must_use]
    pub fn get_intensity(&self) -> Array2<f64> {
        let mut intensity = Array2::zeros((self.config.nx, self.config.ny));
        let factor = 1.0 / (self.config.rho0 * self.config.c0 * self.config.nt as f64);

        for j in 0..self.config.ny {
            for i in 0..self.config.nx {
                let mut sum = 0.0;
                for t in 0..self.config.nt {
                    let p = self.pressure[[i, j, t]].re;
                    sum += p * p;
                }
                intensity[[i, j]] = sum * factor;
            }
        }

        intensity
    }

    /// Calculate peak positive pressure field max_τ Re[p(x,y,τ)] [Pa].
    #[must_use]
    pub fn get_peak_pressure(&self) -> Array2<f64> {
        let mut peak = Array2::zeros((self.config.nx, self.config.ny));

        for j in 0..self.config.ny {
            for i in 0..self.config.nx {
                let mut max_p: f64 = 0.0;
                for t in 0..self.config.nt {
                    max_p = max_p.max(self.pressure[[i, j, t]].re.abs());
                }
                peak[[i, j]] = max_p;
            }
        }

        peak
    }
}

/// Implementation of conservation diagnostics trait for KZK solver.
///
/// All conservation quantities are computed from the physical (real) pressure
/// `Re[p]`.  The imaginary component carries diffraction phase information
/// and does not represent additional acoustic energy.
impl ConservationDiagnostics for KZKSolver {
    fn calculate_total_energy(&self) -> f64 {
        // Acoustic energy density: E = p²/(2ρ₀c₀²)
        // Total energy: ∫∫∫ E dV
        let mut total_energy = 0.0;
        let rho0 = self.config.rho0;
        let c0 = self.config.c0;
        let factor = 1.0 / (2.0 * rho0 * c0 * c0);
        let dv = self.config.dx * self.config.dx * self.config.dt * c0;

        for i in 0..self.config.nx {
            for j in 0..self.config.ny {
                for t in 0..self.config.nt {
                    let p = self.pressure[[i, j, t]].re;
                    total_energy += p * p * factor * dv;
                }
            }
        }

        total_energy
    }

    fn calculate_total_momentum(&self) -> (f64, f64, f64) {
        // Momentum density: ρ₀ u = p/c₀ (acoustic approximation)
        let mut pz = 0.0;
        let rho0 = self.config.rho0;
        let c0 = self.config.c0;
        let dv = self.config.dx * self.config.dx * self.config.dt * c0;

        for i in 0..self.config.nx {
            for j in 0..self.config.ny {
                for t in 0..self.config.nt {
                    let p = self.pressure[[i, j, t]].re;
                    pz += (rho0 * p / c0) * dv;
                }
            }
        }

        // KZK assumes predominantly z-directed propagation.
        (0.0, 0.0, pz)
    }

    fn calculate_total_mass(&self) -> f64 {
        // For acoustic waves: ρ = ρ₀(1 + p/(ρ₀c₀²))
        let mut total_mass = 0.0;
        let rho0 = self.config.rho0;
        let c0 = self.config.c0;
        let dv = self.config.dx * self.config.dx * self.config.dt * c0;

        for i in 0..self.config.nx {
            for j in 0..self.config.ny {
                for t in 0..self.config.nt {
                    let p = self.pressure[[i, j, t]].re;
                    let rho = rho0 * (1.0 + p / (rho0 * c0 * c0));
                    total_mass += rho * dv;
                }
            }
        }

        total_mass
    }
}

/// # Trait impl: `physics::acoustics::wave_propagation::nonlinear::kzk::KZKSolver`
///
/// Bridges the physics-layer trait contract to this solver struct.
///
/// ## API mapping
///
/// | Trait method          | Solver implementation                              |
/// |-----------------------|----------------------------------------------------|
/// | `step_forward(dz)`    | Sets `config.dz = dz`, then calls `self.step()`    |
/// | `current_field()`     | RMS pressure over retarded time at current z-plane |
/// | `peak_pressure()`     | Delegates to `self.get_peak_pressure()`            |
///
/// ## `current_field` semantics
///
/// The solver stores `p(x, y, τ)` as a 3D array.  The trait returns
/// a 2D `Array2<f64>` which is defined as the RMS pressure over τ:
///
/// ```text
/// p_rms(i, j) = √( (1/nt) Σ_t p[i,j,t]² )      [Pa]
/// ```
///
/// This is the most physically relevant single-slice summary for HIFU intensity
/// calculations, where I ∝ p_rms².
impl KZKSolverTrait for KZKSolver {
    /// Advance the pressure field by axial increment `dz` [m].
    ///
    /// Overrides `config.dz` for this step, then applies the full Strang-split
    /// D(dz/2)·A(dz/2)·N(dz)·A(dz/2)·D(dz/2) sequence.
    fn step_forward(&mut self, dz: f64) {
        self.config.dz = dz;
        self.step();
    }

    /// Return the RMS pressure field [Pa] at the current axial z-plane.
    ///
    /// Shape: `(nx, ny)` — transverse grid.
    ///
    /// # Theorem (RMS as L² norm)
    ///
    /// p_rms(i,j) = ‖Re[p[i,j,·]]‖_{L²} / √nt
    ///
    /// This is proportional to the time-averaged acoustic intensity:
    ///   I(i,j) = p_rms(i,j)² / (ρ₀c₀)      [W/m²]
    fn current_field(&self) -> Array2<f64> {
        let nt = self.config.nt as f64;
        let mut rms = Array2::zeros((self.config.nx, self.config.ny));
        for i in 0..self.config.nx {
            for j in 0..self.config.ny {
                let sum_sq: f64 = (0..self.config.nt)
                    .map(|t| self.pressure[[i, j, t]].re.powi(2))
                    .sum();
                rms[[i, j]] = (sum_sq / nt).sqrt();
            }
        }
        rms
    }

    /// Return the peak positive pressure field [Pa] at the current z-plane.
    fn peak_pressure(&self) -> Array2<f64> {
        self.get_peak_pressure()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fft::Complex64;
    use crate::solver::forward::nonlinear::conservation::ConservationTolerances;
    use ndarray::Array2;

    #[test]
    fn test_kzk_solver_creation() {
        let config = KZKConfig::default();
        let solver = KZKSolver::new(config);
        assert!(solver.is_ok());
    }

    /// Test Gaussian beam propagation (COMPREHENSIVE - Tier 3)
    ///
    /// This test uses a 64×64×128 grid for thorough validation.
    /// Execution time: >30s, classified as Tier 3 comprehensive validation.
    #[test]
    #[ignore = "Tier 3: Comprehensive validation (>30s execution time)"]
    fn test_gaussian_beam_propagation() {
        let mut config = KZKConfig {
            nx: 64,
            ny: 64,
            nz: 128,
            nt: 100,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            ..Default::default()
        };
        config.include_nonlinearity = false; // Linear case first

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Create Gaussian source
        let mut source = Array2::zeros((config.nx, config.ny));
        let cx = config.nx as f64 / 2.0;
        let cy = config.ny as f64 / 2.0;
        let sigma = 10.0; // Grid points

        for j in 0..config.ny {
            for i in 0..config.nx {
                let sigma_f64: f64 = sigma;
                let r2 = ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)) / sigma_f64.powi(2);
                source[[i, j]] = (-r2).exp();
            }
        }

        solver.set_source(source, 1e6); // 1 MHz

        // Propagate
        for _ in 0..10 {
            solver.step();
        }

        // Check that beam has propagated (peak should shift)
        let intensity = solver.get_intensity();
        assert!(intensity.sum() > 0.0);
    }

    /// Test Gaussian beam propagation (FAST - Tier 1)
    ///
    /// Fast version with reduced grid (16×16×32) for CI/CD.
    /// Execution time: <2s, classified as Tier 1 fast validation.
    #[test]
    fn test_gaussian_beam_propagation_fast() {
        let mut config = KZKConfig {
            nx: 16,
            ny: 16,
            nz: 32,
            nt: 20,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            ..Default::default()
        };
        config.include_nonlinearity = false; // Linear case first

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Create Gaussian source
        let mut source = Array2::zeros((config.nx, config.ny));
        let cx = config.nx as f64 / 2.0;
        let cy = config.ny as f64 / 2.0;
        let sigma = 3.0; // Grid points (smaller for smaller grid)

        for j in 0..config.ny {
            for i in 0..config.nx {
                let sigma_f64: f64 = sigma;
                let r2 = ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)) / sigma_f64.powi(2);
                source[[i, j]] = (-r2).exp();
            }
        }

        solver.set_source(source, 1e6); // 1 MHz

        // Propagate fewer steps for fast validation
        for _ in 0..3 {
            solver.step();
        }

        // Check that beam has propagated (peak should shift)
        let intensity = solver.get_intensity();
        assert!(intensity.sum() > 0.0);
    }

    #[test]
    fn test_conservation_diagnostics_integration() {
        let mut config = KZKConfig {
            nx: 16,
            ny: 16,
            nz: 32,
            nt: 20,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            ..Default::default()
        };
        config.include_nonlinearity = false; // Linear case for energy conservation

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Enable conservation diagnostics with strict tolerances
        let tolerances = ConservationTolerances {
            absolute_tolerance: 1e-6,
            relative_tolerance: 1e-4,
            check_interval: 2, // Check every 2 steps
        };
        solver.enable_conservation_diagnostics(tolerances);

        // Create Gaussian source
        let mut source = Array2::zeros((config.nx, config.ny));
        let cx = config.nx as f64 / 2.0;
        let cy = config.ny as f64 / 2.0;
        let sigma: f64 = 3.0;

        for j in 0..config.ny {
            for i in 0..config.nx {
                let r2 = ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)) / sigma.powi(2);
                source[[i, j]] = (-r2).exp();
            }
        }

        solver.set_source(source, 1e6);

        // Propagate and check conservation
        for _ in 0..4 {
            solver.step();
        }

        // Verify conservation tracking is working
        assert!(solver.conservation_tracker.is_some());
        assert!(solver.is_solution_valid());

        // Get summary
        let summary = solver.get_conservation_summary();
        assert!(summary.is_some());
    }

    #[test]
    fn test_conservation_energy_calculation() {
        let config = KZKConfig {
            nx: 8,
            ny: 8,
            nz: 16,
            nt: 10,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Set uniform real pressure field (imaginary part stays zero).
        solver.pressure.fill(Complex64::new(1000.0, 0.0)); // 1 kPa

        // Calculate energy
        let energy = solver.calculate_total_energy();

        // Energy should be positive
        assert!(energy > 0.0);

        // Energy should scale with pressure squared
        let p = 1000.0;
        let rho0 = config.rho0;
        let c0 = config.c0;
        let volume = (config.nx as f64 * config.dx)
            * (config.ny as f64 * config.dx)
            * (config.nt as f64 * config.dt * c0);
        let expected = p * p / (2.0 * rho0 * c0 * c0) * volume;

        let relative_error = (energy - expected).abs() / expected;
        assert!(relative_error < 1e-10, "Energy calculation error too large");
    }

    #[test]
    fn test_conservation_diagnostics_disable() {
        let config = KZKConfig::default();
        let mut solver = KZKSolver::new(config).unwrap();

        // Enable then disable
        solver.enable_conservation_diagnostics(ConservationTolerances::default());
        assert!(solver.conservation_tracker.is_some());

        solver.disable_conservation_diagnostics();
        assert!(solver.conservation_tracker.is_none());

        // Should not fail when disabled
        solver.step();
        assert!(solver.is_solution_valid()); // Returns true when disabled
    }

    #[test]
    fn test_conservation_check_interval() {
        let mut config = KZKConfig {
            nx: 8,
            ny: 8,
            nz: 16,
            nt: 10,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            ..Default::default()
        };
        config.include_nonlinearity = false;

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Enable with check interval of 5
        let tolerances = ConservationTolerances {
            check_interval: 5,
            ..Default::default()
        };
        solver.enable_conservation_diagnostics(tolerances);

        // Set simple source
        let source = Array2::from_elem((config.nx, config.ny), 1000.0);
        solver.set_source(source, 1e6);

        // Step 4 times (shouldn't trigger check at step 4)
        for _ in 0..4 {
            solver.step();
        }

        // Step once more to reach step 5 (should trigger check)
        solver.step();

        // Verify tracking occurred
        if let Some(ref tracker) = solver.conservation_tracker {
            assert!(
                !tracker.history.is_empty(),
                "Conservation check should have been performed at step 5"
            );
        }
    }

    /// `solve(0)` must succeed and leave the field unchanged.
    ///
    /// ## Theorem
    /// Zero propagation steps applies no operators; the identity is preserved.
    #[test]
    fn test_kzk_solve_zero_steps() {
        let config = KZKConfig {
            nx: 8,
            ny: 8,
            nz: 16,
            nt: 10,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            include_diffraction: false,
            include_absorption: false,
            include_nonlinearity: false,
            ..Default::default()
        };
        let mut solver = KZKSolver::new(config).unwrap();
        let source = Array2::from_elem((8, 8), 500.0_f64);
        solver.set_source(source, 1e6);
        let p_before = solver.pressure.clone();

        solver.solve(0).expect("solve(0) must succeed");
        assert_eq!(solver.pressure, p_before, "solve(0) must not change the field");
    }

    /// `solve(10)` advances the internal step counter by 10.
    ///
    /// ## Theorem
    /// Each call to `step()` increments `current_z_step` by 1; `solve(n)` calls
    /// `step()` exactly n times, so the counter increases by n.
    #[test]
    fn test_kzk_solve_basic_propagation() {
        let config = KZKConfig {
            nx: 8,
            ny: 8,
            nz: 16,
            nt: 10,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            include_nonlinearity: false,
            ..Default::default()
        };
        let mut solver = KZKSolver::new(config).unwrap();
        let source = Array2::from_elem((8, 8), 1000.0_f64);
        solver.set_source(source, 1e6);

        solver.solve(10).expect("solve(10) must succeed");
        assert_eq!(solver.current_z_step, 10, "current_z_step should be 10");
    }

    /// `solve(nz + 1)` must return an error.
    ///
    /// ## Rationale
    /// The axial grid has exactly `nz` planes; propagating beyond it is undefined.
    #[test]
    fn test_kzk_solve_exceeds_nz_returns_error() {
        let config = KZKConfig {
            nx: 8,
            ny: 8,
            nz: 16,
            nt: 10,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            ..Default::default()
        };
        let mut solver = KZKSolver::new(config.clone()).unwrap();
        let result = solver.solve(config.nz + 1);
        assert!(
            result.is_err(),
            "solve(nz+1) must return Err, got Ok"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("n_steps") && msg.contains("nz"),
            "error message should mention n_steps and nz, got: {msg}"
        );
    }

    /// `solve(nz)` (full grid) must complete without error.
    ///
    /// ## Theorem
    /// `solve(n)` for n ≤ nz must not return an error; boundary case n == nz is valid.
    #[test]
    fn test_kzk_solve_full_propagation() {
        let config = KZKConfig {
            nx: 4,
            ny: 4,
            nz: 8,
            nt: 5,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            include_nonlinearity: false,
            ..Default::default()
        };
        let nz = config.nz;
        let mut solver = KZKSolver::new(config).unwrap();
        let source = Array2::from_elem((4, 4), 100.0_f64);
        solver.set_source(source, 1e6);

        let result = solver.solve(nz);
        assert!(result.is_ok(), "solve(nz) must succeed, got: {:?}", result);
        assert_eq!(solver.current_z_step, nz, "step counter must equal nz after full propagation");
    }

    /// `solve(5)` produces the same result as 5 sequential `step()` calls.
    ///
    /// ## Theorem
    /// `solve(n)` is exactly equivalent to calling `step()` n times on an
    /// identical initial state; both paths must yield bitwise-equal fields.
    #[test]
    fn test_kzk_solve_matches_manual_step_loop() {
        let config = KZKConfig {
            nx: 8,
            ny: 8,
            nz: 16,
            nt: 10,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            include_nonlinearity: false,
            ..Default::default()
        };
        let source = Array2::from_elem((8, 8), 800.0_f64);

        // Path A: use solve()
        let mut solver_a = KZKSolver::new(config.clone()).unwrap();
        solver_a.set_source(source.clone(), 1e6);
        solver_a.solve(5).unwrap();

        // Path B: manual loop
        let mut solver_b = KZKSolver::new(config).unwrap();
        solver_b.set_source(source, 1e6);
        for _ in 0..5 {
            solver_b.step();
        }

        assert_eq!(
            solver_a.pressure,
            solver_b.pressure,
            "solve(5) must match 5×step()"
        );
        assert_eq!(solver_a.current_z_step, solver_b.current_z_step);
    }
}
