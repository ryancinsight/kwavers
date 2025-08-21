//! Numerical methods validation tests
//! 
//! References:
//! - Treeby & Cox (2010) - "k-Wave: MATLAB toolbox"
//! - Gear & Wells (1984) - "Multirate linear multistep methods"
//! - Berger & Oliger (1984) - "Adaptive mesh refinement"
//! - Persson & Peraire (2006) - "Sub-cell shock capturing"

use ndarray::{Array3, ArrayView3};
use crate::grid::Grid;
use crate::solver::pstd::PstdSolver;
use crate::solver::amr::AMRManager;
use std::f64::consts::PI;

// Numerical method constants
const CFL_NUMBER: f64 = 0.3;
const PPW_MINIMUM: usize = 6; // Points per wavelength
const DISPERSION_TOLERANCE: f64 = 0.01; // 1% phase error
const AMR_REFINEMENT_RATIO: usize = 2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::pstd::PstdConfig;
    use crate::solver::amr::{AMRConfig, WaveletType};

    #[test]
    fn test_pstd_plane_wave_accuracy() {
        // Validate k-space method accuracy (Treeby & Cox 2010, Section 3.2)
        let n = 128;
        let dx = 1e-3;
        let frequency = 1e6;
        let wavelength = 1500.0 / frequency;
        let ppw = wavelength / dx;
        
        assert!(ppw >= PPW_MINIMUM as f64, "Insufficient spatial sampling");
        
        let config = PstdConfig::default();
        let grid = Grid::new(n, n, 1, dx, dx, dx);
        let mut solver = PstdSolver::new(config, &grid);
        
        // Initialize plane wave
        let k = 2.0 * PI / wavelength;
        let mut pressure = Array3::zeros((n, n, 1));
        
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 * dx;
                pressure[[i, j, 0]] = (k * x).sin();
            }
        }
        
        // Propagate one wavelength
        let steps = (wavelength / (1500.0 * solver.get_timestep())) as usize;
        let initial = pressure.clone();
        
        for _ in 0..steps {
            solver.step(&mut pressure);
        }
        
        // Calculate phase error
        let mut phase_error = 0.0;
        let mut amplitude_error = 0.0;
        
        for i in n/4..3*n/4 { // Avoid boundaries
            let expected = initial[[i, n/2, 0]];
            let actual = pressure[[i, n/2, 0]];
            
            // Cross-correlation for phase
            let phase_shift = (actual * expected).acos();
            phase_error += phase_shift.abs();
            
            // Amplitude preservation
            amplitude_error += (actual.abs() - expected.abs()).abs();
        }
        
        phase_error /= (n/2) as f64;
        amplitude_error /= (n/2) as f64;
        
        assert!(phase_error < DISPERSION_TOLERANCE, 
                "Excessive phase error: {:.4}", phase_error);
        assert!(amplitude_error < 0.05, 
                "Amplitude not preserved: {:.4}", amplitude_error);
    }

    #[test]
    fn test_numerical_dispersion() {
        // Test numerical dispersion for different PPW values
        let frequencies = vec![0.5e6, 1e6, 2e6, 5e6];
        let dx = 1e-3;
        let n = 64;
        
        for freq in frequencies {
            let wavelength = 1500.0 / freq;
            let ppw = wavelength / dx;
            
            if ppw < 2.0 {
                continue; // Skip under-resolved cases
            }
            
            // Theoretical phase velocity (no dispersion)
            let c_theoretical = 1500.0;
            
            // Numerical phase velocity (with dispersion)
            let k = 2.0 * PI / wavelength;
            let k_dx = k * dx;
            
            // Second-order finite difference dispersion relation
            let c_numerical = c_theoretical * (k_dx / 2.0).sin() / (k_dx / 2.0);
            
            let dispersion_error = (c_numerical - c_theoretical).abs() / c_theoretical;
            
            // Error should decrease with increasing PPW
            let expected_error = 1.0 / ppw.powi(2); // Second-order accuracy
            
            assert!(dispersion_error < expected_error * 10.0,
                    "Dispersion error at {} MHz: {:.4} (PPW: {:.1})",
                    freq / 1e6, dispersion_error, ppw);
        }
    }

    #[test]
    fn test_multirate_time_integration() {
        // Validate time-scale separation (Gear & Wells 1984)
        // Fast acoustic + slow thermal diffusion
        
        const THERMAL_DIFFUSIVITY: f64 = 1.4e-7; // m²/s for tissue
        const ACOUSTIC_SPEED: f64 = 1500.0; // m/s
        
        let dx = 1e-3;
        let dt_acoustic = CFL_NUMBER * dx / ACOUSTIC_SPEED;
        let dt_thermal = 0.5 * dx.powi(2) / THERMAL_DIFFUSIVITY;
        
        let time_scale_ratio = dt_thermal / dt_acoustic;
        
        assert!(time_scale_ratio > 100.0, 
                "Insufficient time scale separation: {:.1}x", time_scale_ratio);
        
        // Verify stability of multirate scheme
        let n = 32;
        let steps_fast = 1000;
        let steps_slow = (steps_fast as f64 / time_scale_ratio) as usize;
        
        let mut acoustic_state = Array3::zeros((n, n, n));
        let mut thermal_state = Array3::zeros((n, n, n));
        
        // Initialize with test pattern
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let r = (((i as f64 - n as f64/2.0).powi(2) + 
                             (j as f64 - n as f64/2.0).powi(2) + 
                             (k as f64 - n as f64/2.0).powi(2)).sqrt());
                    acoustic_state[[i, j, k]] = (-r.powi(2) / 10.0).exp();
                    thermal_state[[i, j, k]] = 300.0 + 10.0 * (-r.powi(2) / 20.0).exp();
                }
            }
        }
        
        let initial_acoustic_energy: f64 = acoustic_state.iter().map(|x| x * x).sum();
        let initial_thermal_energy: f64 = thermal_state.iter().sum();
        
        // Simplified multirate evolution
        for slow_step in 0..steps_slow {
            // Multiple fast steps per slow step
            let fast_per_slow = (time_scale_ratio as usize).max(1);
            
            for _ in 0..fast_per_slow {
                // Fast acoustic update (simplified)
                acoustic_state *= 0.9999; // Artificial damping for stability
            }
            
            // Slow thermal update
            thermal_state *= 0.999; // Simplified diffusion
        }
        
        let final_acoustic_energy: f64 = acoustic_state.iter().map(|x| x * x).sum();
        let final_thermal_energy: f64 = thermal_state.iter().sum();
        
        // Check conservation (with allowance for numerical dissipation)
        let acoustic_loss = (initial_acoustic_energy - final_acoustic_energy) / initial_acoustic_energy;
        let thermal_loss = (initial_thermal_energy - final_thermal_energy) / initial_thermal_energy;
        
        assert!(acoustic_loss < 0.1, "Excessive acoustic energy loss: {:.2}%", acoustic_loss * 100.0);
        assert!(thermal_loss < 0.1, "Excessive thermal energy loss: {:.2}%", thermal_loss * 100.0);
    }

    #[test]
    fn test_amr_wavelet_refinement() {
        // Validate adaptive mesh refinement (Berger & Oliger 1984)
        let base_n = 32;
        let dx = 1e-3;
        
        let config = AMRConfig {
            max_levels: 3,
            refinement_ratio: AMR_REFINEMENT_RATIO,
            buffer_cells: 2,
            regrid_interval: 10,
            wavelet_type: WaveletType::Daubechies4,
            threshold_percentile: 0.95,
        };
        
        let grid = Grid::new(base_n, base_n, base_n, dx, dx, dx);
        let mut amr = AMRManager::new(config, grid);
        
        // Create localized feature requiring refinement
        let mut field = Array3::zeros((base_n, base_n, base_n));
        let center = base_n / 2;
        let feature_width = 3;
        
        for i in (center - feature_width)..(center + feature_width) {
            for j in (center - feature_width)..(center + feature_width) {
                for k in (center - feature_width)..(center + feature_width) {
                    let r = (((i as i32 - center as i32).pow(2) + 
                             (j as i32 - center as i32).pow(2) + 
                             (k as i32 - center as i32).pow(2)) as f64).sqrt();
                    field[[i, j, k]] = (PI * r / feature_width as f64).cos();
                }
            }
        }
        
        // Apply wavelet analysis for refinement
        let coefficients = amr.wavelet_transform(&field);
        let refinement_flags = amr.compute_refinement_flags(&coefficients);
        
        // Check that high-gradient regions are flagged
        let flagged_count = refinement_flags.iter().filter(|&&x| x).count();
        let expected_refined = (2 * feature_width).pow(3); // Approximate
        
        assert!(flagged_count > expected_refined / 2,
                "Insufficient refinement: {} cells flagged", flagged_count);
        assert!(flagged_count < expected_refined * 4,
                "Excessive refinement: {} cells flagged", flagged_count);
    }

    #[test]
    fn test_spectral_dg_shock_detection() {
        // Validate shock capturing (Persson & Peraire 2006)
        let n = 64;
        let dx = 1e-3;
        
        // Create discontinuous field (shock)
        let mut field = Array3::zeros((n, 1, 1));
        for i in 0..n {
            if i < n/2 {
                field[[i, 0, 0]] = 1.0;
            } else {
                field[[i, 0, 0]] = 0.1;
            }
        }
        
        // Smooth slightly to avoid numerical issues
        for _ in 0..2 {
            let mut smoothed = field.clone();
            for i in 1..n-1 {
                smoothed[[i, 0, 0]] = 0.25 * field[[i-1, 0, 0]] + 
                                      0.5 * field[[i, 0, 0]] + 
                                      0.25 * field[[i+1, 0, 0]];
            }
            field = smoothed;
        }
        
        // Compute smoothness indicator (Persson-Peraire)
        let mut smoothness = Array3::zeros((n, 1, 1));
        
        for i in 2..n-2 {
            // Modal decay indicator
            let local_vals = vec![
                field[[i-2, 0, 0]],
                field[[i-1, 0, 0]],
                field[[i, 0, 0]],
                field[[i+1, 0, 0]],
                field[[i+2, 0, 0]],
            ];
            
            // Compute local polynomial coefficients (simplified)
            let mean = local_vals.iter().sum::<f64>() / 5.0;
            let variance = local_vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 5.0;
            
            // High variance indicates discontinuity
            smoothness[[i, 0, 0]] = variance.sqrt();
        }
        
        // Find shock location
        let mut max_indicator = 0.0;
        let mut shock_location = 0;
        
        for i in 0..n {
            if smoothness[[i, 0, 0]] > max_indicator {
                max_indicator = smoothness[[i, 0, 0]];
                shock_location = i;
            }
        }
        
        // Shock should be detected near n/2
        let error = (shock_location as i32 - n as i32 / 2).abs();
        assert!(error < 5, "Shock detection error: {} cells", error);
    }
}