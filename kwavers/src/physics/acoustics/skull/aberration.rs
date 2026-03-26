//! Aberration correction for transcranial ultrasound
//!
//! # Theorem: Time-Reversal Reciprocity (Fink 1992)
//!
//! The linear acoustic wave equation is invariant under time reversal `t → −t`.
//! If `p(x, t)` is a solution, then `p*(x, t) = p(x, T − t)` (the time-reversed field)
//! is also a solution. This gives the **time-reversal mirror (TRM)** principle:
//!
//! 1. **Forward pass**: Transmit from a point source at the focus; record the
//!    distorted wavefield `p(xᵢ, t)` at each transducer element `i`.
//! 2. **Phase conjugation**: In the frequency domain, conjugate the received
//!    spectrum: `P_back(xᵢ, ω) = P*(xᵢ, ω)`. This reverses the phase accumulated
//!    by skull aberration.
//! 3. **Back-propagation**: Re-transmit `p*(xᵢ, t)`. Reciprocity guarantees that
//!    the back-propagated field refocuses at the original source location despite
//!    the intervening heterogeneous skull.
//!
//! ## Phase Aberration Model
//!
//! For continuous-wave (CW) single-frequency operation, the skull introduces a
//! spatially varying phase shift:
//! ```text
//!   φ_skull(x) = (k_water − k_skull(x)) · d(x)
//! ```
//! where `k = 2πf/c` is the local wavenumber and `d(x)` is the skull thickness.
//! The corrected drive signal for element `i` acquires a pre-compensation phase
//! `−φ_skull(xᵢ)` to cancel the aberration at the focal point.
//!
//! ## Discretization
//!
//! 1. Compute `c_local(i,j,k)` from the CT-derived skull model.
//! 2. Local wavenumber: `k_local = 2πf / c_local`.
//! 3. Phase delay along propagation path of length `r`:
//!    `φ(x) = (k_local − k_water) · r`.
//! 4. Correction: `φ_corr(x) = −φ(x)` (applied to transducer drive signals).
//!
//! ## References
//! - Fink, M. (1992). Time reversal of ultrasonic fields — Part I: Basic principles.
//!   IEEE Trans. Ultrason. Ferroelectr. Freq. Control 39(5), 555–566.
//! - Aubry, J.-F., Tanter, M., Pernot, M., Thomas, J.-L. & Fink, M. (2003).
//!   Experimental demonstration of noninvasive transskull adaptive focusing based
//!   on prior computed tomography scans. J. Acoust. Soc. Am. 113(1), 84–93.
//! - Tanter, M., Thomas, J.-L. & Fink, M. (1998). Focusing and steering through
//!   absorbing and aberrating layers: Application to ultrasonic propagation through
//!   the skull. J. Acoust. Soc. Am. 103(5), 2403–2410.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::physics::acoustics::skull::HeterogeneousSkull;
use ndarray::Array3;
use std::f64::consts::PI;

/// Aberration correction using time-reversal methods
/// TODO_AUDIT: P1 - Advanced Skull Aberration Correction - Implement full time-reversal focusing with adaptive optics and patient-specific optimization
/// DEPENDS ON: physics/acoustics/skull/aberration/adaptive_optics.rs, physics/acoustics/skull/aberration/time_reversal.rs, physics/acoustics/skull/aberration/optimization.rs
/// MISSING: Full time-reversal mirror implementation with iterative focusing
/// MISSING: Adaptive optics with deformable mirror integration
/// MISSING: Patient-specific aberration profile characterization
/// MISSING: Multi-frequency aberration correction for broadband pulses
/// MISSING: Real-time aberration tracking during therapy
/// THEOREM: Time-reversal reciprocity: Wave equation is invariant under t → -t, x → -x
/// THEOREM: Phase conjugation: u*(t) compensates for aberrated u(t)
/// REFERENCES: Aubry et al. (2003) JASA 113, 84; Fink et al. (2003) IEEE Trans Ultrason Ferroelectr Freq Control
#[derive(Debug)]
pub struct AberrationCorrection<'a> {
    grid: &'a Grid,
    skull: &'a HeterogeneousSkull,
}

impl<'a> AberrationCorrection<'a> {
    /// Create new aberration correction calculator
    pub fn new(grid: &'a Grid, skull: &'a HeterogeneousSkull) -> Self {
        Self { grid, skull }
    }

    /// Compute time-reversal phase corrections
    ///
    /// Returns phase corrections in radians for each grid point
    pub fn compute_time_reversal_phases(&self, frequency: f64) -> KwaversResult<Array3<f64>> {
        use crate::core::constants::thermodynamic::ROOM_TEMPERATURE_C;
        use crate::core::constants::water::WaterProperties;

        let water_c = WaterProperties::sound_speed(ROOM_TEMPERATURE_C);
        let k = 2.0 * PI * frequency / water_c; // Water wavenumber

        let mut phases = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        // Simplified phase aberration model
        // In practice, this would use ray tracing or full wave simulation
        let center = (self.grid.nx / 2, self.grid.ny / 2, self.grid.nz / 2);

        for ((i, j, k_idx), phase) in phases.indexed_iter_mut() {
            let dx = (i as f64 - center.0 as f64) * self.grid.dx;
            let dy = (j as f64 - center.1 as f64) * self.grid.dy;
            let dz = (k_idx as f64 - center.2 as f64) * self.grid.dz;

            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            let c_local = self.skull.sound_speed[[i, j, k_idx]];

            // Phase delay due to sound speed variation
            let k_local = 2.0 * PI * frequency / c_local;
            *phase = (k_local - k) * distance;
        }

        Ok(phases)
    }
}
