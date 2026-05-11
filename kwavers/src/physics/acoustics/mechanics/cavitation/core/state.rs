//! Cavitation state tracking and dose accumulation

/// Cavitation state at a point
#[derive(Debug, Clone, Copy)]
pub struct CavitationState {
    /// Whether cavitation is occurring
    pub is_cavitating: bool,
    /// Cavitation intensity (0-1)
    pub intensity: f64,
    /// Time since cavitation onset (s)
    pub duration: f64,
    /// Peak negative pressure reached (Pa)
    pub peak_negative_pressure: f64,
    /// Mechanical index value
    pub mechanical_index: f64,
}

impl Default for CavitationState {
    fn default() -> Self {
        Self {
            is_cavitating: false,
            intensity: 0.0,
            duration: 0.0,
            peak_negative_pressure: 0.0,
            mechanical_index: 0.0,
        }
    }
}

/// Cavitation dose accumulation using O(1) running statistics
///
/// # Mathematical Specification
/// The dose integral $D(t) = \int_0^t I(\tau) d\tau$ is accumulated
/// incrementally via the trapezoidal approximation $D_{n+1} = D_n + I_n \cdot \Delta t$.
/// Running statistics (count, sum, max) replace unbounded history vectors,
/// guaranteeing constant memory regardless of simulation duration.
#[derive(Debug, Clone)]
pub struct CavitationDose {
    /// Accumulated dose value: $\sum I_i \cdot \Delta t_i$
    pub total_dose: f64,
    /// Number of cavitation samples recorded
    pub sample_count: u64,
    /// Running sum of intensities for mean computation
    intensity_sum: f64,
    /// Peak intensity observed across entire simulation
    pub peak_intensity: f64,
    /// Latest timestamp of a cavitation event (s)
    pub last_event_time: f64,
}

impl CavitationDose {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_dose: 0.0,
            sample_count: 0,
            intensity_sum: 0.0,
            peak_intensity: 0.0,
            last_event_time: 0.0,
        }
    }

    /// Update dose with new cavitation event
    pub fn update(&mut self, intensity: f64, dt: f64, time: f64) {
        self.total_dose += intensity * dt;
        self.sample_count += 1;
        self.intensity_sum += intensity;
        if intensity > self.peak_intensity {
            self.peak_intensity = intensity;
        }
        self.last_event_time = time;
    }

    /// Calculate time-weighted average intensity from running statistics
    ///
    /// Returns $\bar{I} = \frac{\sum I_i}{N}$ where $N$ is the sample count.
    #[must_use]
    pub fn average_intensity(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            self.intensity_sum / self.sample_count as f64
        }
    }
}

impl Default for CavitationDose {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CavitationDose::new` produces a fully zeroed accumulator.
    #[test]
    fn cavitation_dose_new_is_fully_zeroed() {
        let d = CavitationDose::new();
        assert_eq!(d.total_dose, 0.0);
        assert_eq!(d.sample_count, 0);
        assert_eq!(d.peak_intensity, 0.0);
        assert_eq!(d.last_event_time, 0.0);
    }

    /// `average_intensity` returns 0 when no samples have been recorded.
    #[test]
    fn average_intensity_zero_with_no_samples() {
        let d = CavitationDose::new();
        assert_eq!(d.average_intensity(), 0.0);
    }

    /// `update` accumulates total_dose as Σ(I·dt).
    ///
    /// Three calls with intensity=0.5 and dt=1e-4 s:
    /// total_dose = 3 × 0.5 × 1e-4 = 1.5e-4.
    #[test]
    fn dose_update_accumulates_total_dose() {
        let mut d = CavitationDose::new();
        for step in 0..3 {
            d.update(0.5, 1e-4, step as f64 * 1e-4);
        }
        assert!(
            (d.total_dose - 1.5e-4).abs() < 1e-18,
            "total_dose must be 1.5e-4 (got {:.6e})",
            d.total_dose
        );
        assert_eq!(d.sample_count, 3);
    }

    /// `average_intensity` divides the running sum by sample_count.
    ///
    /// Three calls with intensities [0.2, 0.5, 0.8]: avg = 0.5.
    #[test]
    fn average_intensity_equals_sum_divided_by_count() {
        let mut d = CavitationDose::new();
        d.update(0.2, 1e-4, 0.0);
        d.update(0.5, 1e-4, 1e-4);
        d.update(0.8, 1e-4, 2e-4);
        assert!(
            (d.average_intensity() - 0.5).abs() < 1e-15,
            "average_intensity must be 0.5 (got {:.6})",
            d.average_intensity()
        );
    }

    /// `peak_intensity` stores the maximum intensity seen across all updates.
    #[test]
    fn peak_intensity_is_maximum_across_all_updates() {
        let mut d = CavitationDose::new();
        d.update(0.3, 1e-4, 0.0);
        d.update(0.9, 1e-4, 1e-4); // peak
        d.update(0.1, 1e-4, 2e-4);
        assert!(
            (d.peak_intensity - 0.9).abs() < 1e-15,
            "peak_intensity must be 0.9 (got {:.3})",
            d.peak_intensity
        );
    }

    /// `CavitationState::default` produces a non-cavitating zero-intensity state.
    #[test]
    fn cavitation_state_default_not_cavitating_and_zero_fields() {
        let s = CavitationState::default();
        assert!(!s.is_cavitating);
        assert_eq!(s.intensity, 0.0);
        assert_eq!(s.duration, 0.0);
        assert_eq!(s.mechanical_index, 0.0);
    }
}
