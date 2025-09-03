//! Cavitation detection and analysis

use ndarray::Array3;

/// Cavitation detector
#[derive(Debug)]
pub struct CavitationDetector {
    threshold: f64,
    detected_regions: Vec<CavitationRegion>,
}

/// Detected cavitation region
#[derive(Debug, Clone)]
pub struct CavitationRegion {
    pub center: (usize, usize, usize),
    pub volume: f64,
    pub min_pressure: f64,
    pub time_step: usize,
}

impl CavitationDetector {
    /// Create detector with threshold
    #[must_use]
    pub fn create(threshold: f64) -> Self {
        Self {
            threshold,
            detected_regions: Vec::new(),
        }
    }

    /// Detect cavitation in pressure field
    pub fn detect(&mut self, pressure: &Array3<f64>, time_step: usize, dx: f64) {
        let mut regions = Vec::new();

        for ((i, j, k), &p) in pressure.indexed_iter() {
            if p < self.threshold {
                // Simple detection - could be enhanced with region growing
                regions.push(CavitationRegion {
                    center: (i, j, k),
                    volume: dx * dx * dx,
                    min_pressure: p,
                    time_step,
                });
            }
        }

        self.detected_regions.extend(regions);
    }

    /// Get detected regions
    #[must_use]
    pub fn regions(&self) -> &[CavitationRegion] {
        &self.detected_regions
    }

    /// Clear detected regions
    pub fn clear(&mut self) {
        self.detected_regions.clear();
    }

    /// Get statistics
    pub fn statistics(&self) -> CavitationStatistics {
        if self.detected_regions.is_empty() {
            return CavitationStatistics::default();
        }

        let total_volume: f64 = self.detected_regions.iter().map(|r| r.volume).sum();

        let min_pressure = self
            .detected_regions
            .iter()
            .map(|r| r.min_pressure)
            .fold(f64::INFINITY, f64::min);

        CavitationStatistics {
            num_regions: self.detected_regions.len(),
            total_volume,
            min_pressure,
        }
    }
}

/// Cavitation statistics
#[derive(Debug, Default)]
pub struct CavitationStatistics {
    pub num_regions: usize,
    pub total_volume: f64,
    pub min_pressure: f64,
}
