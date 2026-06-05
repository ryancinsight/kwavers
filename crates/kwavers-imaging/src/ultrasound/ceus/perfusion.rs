//! `PerfusionMap` and `PerfusionStatistics` — quantitative perfusion analysis.

use ndarray::Array3;

/// Perfusion map containing quantitative perfusion parameters
#[derive(Debug, Clone)]
pub struct PerfusionMap {
    /// Peak intensity (dB)
    pub peak_intensity: Array3<f64>,
    /// Time to peak intensity (s)
    pub time_to_peak: Array3<f64>,
    /// Area under the time-intensity curve (dB·s)
    pub area_under_curve: Array3<f64>,
}

impl PerfusionMap {
    /// Get perfusion statistics for a region of interest
    #[must_use]
    pub fn roi_statistics(
        &self,
        x_range: (usize, usize),
        y_range: (usize, usize),
        z_range: (usize, usize),
    ) -> PerfusionStatistics {
        let mut peak_values = Vec::new();
        let mut ttp_values = Vec::new();
        let mut auc_values = Vec::new();

        for i in x_range.0..=x_range.1 {
            for j in y_range.0..=y_range.1 {
                for k in z_range.0..=z_range.1 {
                    if self.peak_intensity[[i, j, k]] > 0.0 {
                        peak_values.push(self.peak_intensity[[i, j, k]]);
                        ttp_values.push(self.time_to_peak[[i, j, k]]);
                        auc_values.push(self.area_under_curve[[i, j, k]]);
                    }
                }
            }
        }

        PerfusionStatistics::from_samples(&peak_values, &ttp_values, &auc_values)
    }
}

/// Perfusion statistics for a region of interest
#[derive(Debug, Clone)]
pub struct PerfusionStatistics {
    /// Mean peak intensity (dB)
    pub mean_peak: f64,
    /// Standard deviation of peak intensity
    pub std_peak: f64,
    /// Mean time to peak (s)
    pub mean_ttp: f64,
    /// Standard deviation of time to peak
    pub std_ttp: f64,
    /// Mean area under curve (dB·s)
    pub mean_auc: f64,
    /// Standard deviation of area under curve
    pub std_auc: f64,
}

impl PerfusionStatistics {
    #[must_use]
    pub fn from_samples(peaks: &[f64], ttp: &[f64], auc: &[f64]) -> Self {
        let n = peaks.len() as f64;
        let mean_peak = peaks.iter().sum::<f64>() / n;
        let std_peak = (peaks.iter().map(|x| (x - mean_peak).powi(2)).sum::<f64>() / n).sqrt();
        let mean_ttp = ttp.iter().sum::<f64>() / n;
        let std_ttp = (ttp.iter().map(|x| (x - mean_ttp).powi(2)).sum::<f64>() / n).sqrt();
        let mean_auc = auc.iter().sum::<f64>() / n;
        let std_auc = (auc.iter().map(|x| (x - mean_auc).powi(2)).sum::<f64>() / n).sqrt();

        Self {
            mean_peak,
            std_peak,
            mean_ttp,
            std_ttp,
            mean_auc,
            std_auc,
        }
    }
}
