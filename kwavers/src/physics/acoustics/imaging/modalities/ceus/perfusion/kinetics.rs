//! Flow kinetics, perfusion parameters, and tissue uptake types.

/// Flow kinetics model
#[derive(Debug)]
pub struct FlowKinetics {
    /// Arterial input function
    pub arterial_input: Vec<f64>,
    /// Tissue residue function
    pub residue_function: Vec<f64>,
    /// Mean transit time (s)
    pub mean_transit_time: f64,
}

impl FlowKinetics {
    /// Create new flow kinetics model
    #[must_use]
    pub fn new() -> Self {
        Self {
            arterial_input: Vec::new(),
            residue_function: Vec::new(),
            mean_transit_time: 10.0,
        }
    }

    /// Compute perfusion parameters from time-intensity curve
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[must_use]
    pub fn analyze_tic(&self, tic: &[f64], frame_rate: f64) -> PerfusionParameters {
        if tic.is_empty() {
            return PerfusionParameters::default();
        }

        let peak_idx = tic
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map_or(0, |(i, _)| i);

        let peak_intensity = tic[peak_idx];
        let time_to_peak = peak_idx as f64 / frame_rate;

        let mut auc = 0.0;
        let dt = 1.0 / frame_rate;
        for i in 0..tic.len().saturating_sub(1) {
            auc += (tic[i] + tic[i + 1]) * dt / 2.0;
        }

        let wash_in_rate = if peak_idx > 1 {
            (tic[1] - tic[0]) / dt
        } else {
            0.0
        };

        let wash_out_rate = if tic.len() > peak_idx + 2 {
            (tic[tic.len() - 1] - tic[tic.len() - 2]) / dt
        } else {
            0.0
        };

        PerfusionParameters {
            peak_intensity,
            time_to_peak,
            area_under_curve: auc,
            wash_in_rate,
            wash_out_rate,
            mean_transit_time: self.mean_transit_time,
        }
    }
}

impl Default for FlowKinetics {
    fn default() -> Self {
        Self::new()
    }
}

/// Perfusion parameters from time-intensity curve analysis
#[derive(Debug, Clone)]
pub struct PerfusionParameters {
    /// Peak intensity (dB)
    pub peak_intensity: f64,
    /// Time to peak (s)
    pub time_to_peak: f64,
    /// Area under curve (dB·s)
    pub area_under_curve: f64,
    /// Wash-in rate (dB/s)
    pub wash_in_rate: f64,
    /// Wash-out rate (dB/s)
    pub wash_out_rate: f64,
    /// Mean transit time (s)
    pub mean_transit_time: f64,
}

impl Default for PerfusionParameters {
    fn default() -> Self {
        Self {
            peak_intensity: 0.0,
            time_to_peak: 0.0,
            area_under_curve: 0.0,
            wash_in_rate: 0.0,
            wash_out_rate: 0.0,
            mean_transit_time: 10.0,
        }
    }
}

/// Tissue uptake model
#[derive(Debug)]
pub struct TissueUptake {
    /// Uptake rate constant (1/s)
    pub uptake_rate: f64,
    /// Clearance rate constant (1/s)
    pub clearance_rate: f64,
    /// Partition coefficient
    pub partition_coefficient: f64,
}

impl TissueUptake {
    /// Create new tissue uptake model
    #[must_use]
    pub fn new() -> Self {
        Self {
            uptake_rate: 0.1,
            clearance_rate: 0.05,
            partition_coefficient: 0.2,
        }
    }

    /// Compute tissue concentration over time
    #[must_use]
    pub fn tissue_concentration(&self, plasma_concentration: f64, time: f64) -> f64 {
        let k1 = self.uptake_rate;
        let k2 = self.clearance_rate;
        let v = self.partition_coefficient;

        if time <= 0.0 {
            0.0
        } else {
            v * k1 * plasma_concentration * (1.0 - (-k2 * time).exp()) / k2
        }
    }
}

impl Default for TissueUptake {
    fn default() -> Self {
        Self::new()
    }
}
