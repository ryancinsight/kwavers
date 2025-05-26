// src/physics/mechanics/acoustic_wave/nonlinear/performance.rs
use super::config::NonlinearWave;
use log::debug;

impl NonlinearWave {
    /// Reports performance metrics for the nonlinear wave computation.
    ///
    /// This method logs the average time spent in different parts of the `update_wave`
    /// method over all calls. It includes total time per call and breakdowns for
    /// nonlinear term calculation, FFT operations, source application, and field combination.
    /// If `update_wave` has not been called, it logs a message indicating that.
    /// The output is directed to the debug log.
    pub fn report_performance(&self) {
        if self.call_count == 0 {
            debug!("No calls to NonlinearWave::update_wave yet");
            return;
        }

        let total_time = self.nonlinear_time + self.fft_time + self.source_time + self.combination_time;
        // Ensure call_count is not zero to prevent division by zero, though already checked.
        let avg_time = if self.call_count > 0 { total_time / self.call_count as f64 } else { 0.0 };

        debug!(
            "NonlinearWave performance (avg over {} calls):",
            self.call_count
        );
        debug!("  Total time per call:   {:.3e} s", avg_time);
        
        if total_time > 0.0 { // Avoid division by zero if total_time is zero
            debug!(
                "  Nonlinear term calc:    {:.3e} s ({:.1}%)",
                self.nonlinear_time / self.call_count as f64,
                100.0 * self.nonlinear_time / total_time
            );
            debug!(
                "  FFT operations:         {:.3e} s ({:.1}%)",
                self.fft_time / self.call_count as f64,
                100.0 * self.fft_time / total_time
            );
            debug!(
                "  Source application:     {:.3e} s ({:.1}%)",
                self.source_time / self.call_count as f64,
                100.0 * self.source_time / total_time
            );
            debug!(
                "  Field combination:      {:.3e} s ({:.1}%)",
                self.combination_time / self.call_count as f64,
                100.0 * self.combination_time / total_time
            );
        } else {
            // This case handles when total_time is 0 (e.g. if all time components are 0)
            // and call_count > 0.
            debug!("  Detailed breakdown not available (total_time or call_count is zero leading to no meaningful percentages).");
            debug!("    Nonlinear term calc:    {:.3e} s", self.nonlinear_time / self.call_count.max(1) as f64);
            debug!("    FFT operations:         {:.3e} s", self.fft_time / self.call_count.max(1) as f64);
            debug!("    Source application:     {:.3e} s", self.source_time / self.call_count.max(1) as f64);
            debug!("    Field combination:      {:.3e} s", self.combination_time / self.call_count.max(1) as f64);

        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid; // Use the actual Grid struct

    // Helper to create a basic NonlinearWave instance for testing performance reporting
    fn create_test_wave_for_performance() -> NonlinearWave {
        // Use the actual Grid constructor
        let test_grid = Grid::new(1, 1, 1, 0.1, 0.1, 0.1); // Minimal grid
        NonlinearWave::new(&test_grid)
    }

    #[test]
    fn test_report_performance_no_calls() {
        let wave = create_test_wave_for_performance();
        // The primary goal is to ensure this doesn't panic.
        // Log output itself is not easily captured/asserted in standard unit tests
        // without external crates or more complex setup.
        wave.report_performance(); 
        // If we reach here, the test passes (no panic).
    }

    #[test]
    fn test_report_performance_with_calls_zero_time() {
        let mut wave = create_test_wave_for_performance();
        
        // Manually set performance metrics
        wave.call_count = 5;
        wave.nonlinear_time = 0.0;
        wave.fft_time = 0.0;
        wave.source_time = 0.0;
        wave.combination_time = 0.0;
        
        // Test for panic safety
        wave.report_performance();
        // If we reach here, the test passes.
    }

    #[test]
    fn test_report_performance_with_calls_non_zero_time() {
        let mut wave = create_test_wave_for_performance();
        
        // Manually set performance metrics
        wave.call_count = 10;
        wave.nonlinear_time = 0.5;
        wave.fft_time = 0.2;
        wave.source_time = 0.1;
        wave.combination_time = 0.3;
        
        // Test for panic safety
        wave.report_performance();
        // If we reach here, the test passes.
    }
}
