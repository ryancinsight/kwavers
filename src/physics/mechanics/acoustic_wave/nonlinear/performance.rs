// src/physics/mechanics/acoustic_wave/nonlinear/performance.rs
// The NonlinearWave::report_performance method has been moved to trait_impls.rs
// as part of the AcousticWaveModel trait implementation.

// This file is now empty or can be removed if no other performance-related logic resides here.
// For now, leaving it empty.

#[cfg(test)]
mod tests {
    // Original tests for report_performance might need to be adapted or moved.
    // If NonlinearWave is constructed and the trait method is called,
    // the tests might still be relevant.

    // Example of how tests might need to change:
    /*
    use super::super::config::NonlinearWave; // To get NonlinearWave struct
    use crate::physics::traits::AcousticWaveModel; // To use the trait method
    use crate::grid::Grid;

    fn create_test_wave_for_performance_trait() -> NonlinearWave {
        let test_grid = Grid::new(1, 1, 1, 0.1, 0.1, 0.1);
        NonlinearWave::new(&test_grid)
    }

    #[test]
    fn test_report_performance_no_calls_via_trait() {
        let wave = create_test_wave_for_performance_trait();
        AcousticWaveModel::report_performance(&wave); // Call via trait
    }
    */
    // For now, the original tests from performance.rs are preserved below.
    // They should still pass if NonlinearWave is instantiated and its
    // (now trait) methods are called.
    use super::super::config::NonlinearWave; // To get NonlinearWave struct
    use crate::physics::traits::AcousticWaveModel; // To use the trait method
    use crate::grid::Grid;


    fn create_test_wave_for_performance() -> NonlinearWave {

        let test_grid = Grid::new(1, 1, 1, 0.1, 0.1, 0.1);
        NonlinearWave::new(&test_grid)
    }

    #[test]
    fn test_report_performance_no_calls() {
        let wave = create_test_wave_for_performance();

        AcousticWaveModel::report_performance(&wave);

    }

    #[test]
    fn test_report_performance_with_calls_zero_time() {
        let mut wave = create_test_wave_for_performance();
        

        wave.call_count = 5;
        wave.nonlinear_time = 0.0;
        wave.fft_time = 0.0;
        wave.source_time = 0.0;
        wave.combination_time = 0.0;
        

        AcousticWaveModel::report_performance(&wave);

    }

    #[test]
    fn test_report_performance_with_calls_non_zero_time() {
        let mut wave = create_test_wave_for_performance();
        

        wave.call_count = 10;
        wave.nonlinear_time = 0.5;
        wave.fft_time = 0.2;
        wave.source_time = 0.1;
        wave.combination_time = 0.3;
        

        AcousticWaveModel::report_performance(&wave);

    }
}
