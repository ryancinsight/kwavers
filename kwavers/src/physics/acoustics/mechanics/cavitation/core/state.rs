//! Cavitation state tracking and dose accumulation

/// Cavitation state at a point
#[derive(Debug, Clone, Copy)]
pub struct CavitationState {
    /// Whether cavitation is occurring
    pub is_cavitating: bool,
    /// Cavitation intensity (0-1)
    pub intensity: f64,
    /// Time since cavitation onset [s]
    pub duration: f64,
    /// Peak negative pressure reached [Pa]
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
    /// Latest timestamp of a cavitation event [s]
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
