//! KZK Strang-split propagation: per-step operator application and
//! conservation-diagnostic dispatch.

use log::warn;
use ndarray::Axis;

use super::KZKSolver;
use crate::solver::forward::nonlinear::conservation::{ConservationDiagnostics, ViolationSeverity};

impl KZKSolver {
    /// Step forward one z-plane using operator splitting.
    /// Uses second-order Strang splitting:
    /// `D(dz/2) · A(dz/2) · N(dz) · A(dz/2) · D(dz/2)`.
    pub fn step(&mut self) {
        let dz = self.config.dz;

        if self.config.include_diffraction {
            self.apply_diffraction(dz / 2.0);
        }
        if self.config.include_absorption {
            self.apply_absorption(dz / 2.0);
        }
        if self.config.include_nonlinearity {
            self.apply_nonlinearity(dz);
        }
        if self.config.include_absorption {
            self.apply_absorption(dz / 2.0);
        }
        if self.config.include_diffraction {
            self.apply_diffraction(dz / 2.0);
        }

        self.pressure_prev.assign(&self.pressure);

        self.current_z_step += 1;
        self.current_time += dz / self.config.c0;

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

    /// Apply complex-field parabolic diffraction to each retarded-time slice.
    ///
    /// For each τ index, extracts the 2D complex spatial slice `p[:,:,t]` and
    /// applies the spectral propagator H(k_T) = exp(−ik_T²Δz/(2k₀)) in-place.
    /// The complex field is preserved without discarding the imaginary part,
    /// ensuring accurate phase accumulation over many axial steps.
    pub(super) fn apply_diffraction(&mut self, step_size: f64) {
        for t in 0..self.config.nt {
            let mut slice = self.pressure.index_axis_mut(Axis(2), t);
            self.complex_diffraction
                .apply_complex(&mut slice, step_size);
        }
    }

    /// Apply absorption operator
    pub(super) fn apply_absorption(&mut self, step_size: f64) {
        self.absorption.apply(&mut self.pressure, step_size);
    }

    /// Apply nonlinear operator
    pub(super) fn apply_nonlinearity(&mut self, step_size: f64) {
        self.nonlinear
            .apply(&mut self.pressure, &self.pressure_prev, step_size);
    }

    /// Check conservation laws and log diagnostics.
    ///
    /// Performs conservation checks at configured intervals and logs violations.
    /// Critical violations trigger warnings via tracing infrastructure.
    fn check_conservation_laws(&mut self) {
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

        if let Some(ref mut tracker) = self.conservation_tracker {
            for diag in &diagnostics {
                if diag.severity > tracker.max_severity {
                    tracker.max_severity = diag.severity;
                }
            }
            tracker.history.extend(diagnostics.clone());
        }

        for diag in diagnostics {
            match diag.severity {
                ViolationSeverity::Acceptable => {}
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
}
