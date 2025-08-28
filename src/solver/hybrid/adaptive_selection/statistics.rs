// adaptive_selection/statistics.rs - Selection statistics tracking

use super::selector::SelectedMethod;
use ndarray::Array3;

/// Statistics for method selection
#[derive(Debug, Clone, Default)]
pub struct SelectionStatistics {
    pub spectral_count: usize,
    pub fd_count: usize,
    pub dg_count: usize,
    pub total_switches: usize,
    pub time_steps: usize,
}

impl SelectionStatistics {
    /// Update statistics with new selection
    pub fn update(
        &mut self,
        selection: &Array3<SelectedMethod>,
        previous: Option<&Array3<SelectedMethod>>,
    ) {
        // Count method usage
        self.spectral_count = 0;
        self.fd_count = 0;
        self.dg_count = 0;

        for &method in selection.iter() {
            match method {
                SelectedMethod::Spectral => self.spectral_count += 1,
                SelectedMethod::FiniteDifference => self.fd_count += 1,
                SelectedMethod::DiscontinuousGalerkin => self.dg_count += 1,
            }
        }

        // Count switches
        if let Some(prev) = previous {
            for (curr, prev) in selection.iter().zip(prev.iter()) {
                if curr != prev {
                    self.total_switches += 1;
                }
            }
        }

        self.time_steps += 1;
    }

    /// Get method distribution as percentages
    pub fn get_distribution(&self) -> (f64, f64, f64) {
        let total = (self.spectral_count + self.fd_count + self.dg_count) as f64;

        if total > 0.0 {
            (
                100.0 * self.spectral_count as f64 / total,
                100.0 * self.fd_count as f64 / total,
                100.0 * self.dg_count as f64 / total,
            )
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// Get average switches per time step
    pub fn switches_per_step(&self) -> f64 {
        if self.time_steps > 0 {
            self.total_switches as f64 / self.time_steps as f64
        } else {
            0.0
        }
    }

    /// Print summary
    pub fn print_summary(&self) {
        let (spectral_pct, fd_pct, dg_pct) = self.get_distribution();

        println!("=== Adaptive Selection Statistics ===");
        println!("Time steps: {}", self.time_steps);
        println!("Method distribution:");
        println!("  Spectral: {:.1}%", spectral_pct);
        println!("  Finite Difference: {:.1}%", fd_pct);
        println!("  Discontinuous Galerkin: {:.1}%", dg_pct);
        println!("Total switches: {}", self.total_switches);
        println!("Switches per step: {:.2}", self.switches_per_step());
    }
}
