//! Individual theorem validation methods.

use super::{TheoremValidation, TheoremValidator};
use crate::core::constants::fundamental::{AVOGADRO, BOLTZMANN};
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

impl TheoremValidator {
    pub(super) fn calculate_confidence(error_bound: f64, measured_error: f64, passed: bool) -> f64 {
        if !passed {
            return 0.2;
        }
        if error_bound <= 0.0 {
            return 0.5;
        }
        let error_ratio = measured_error / error_bound;
        if error_ratio < 0.1 {
            0.95
        } else if error_ratio < 0.3 {
            0.85
        } else if error_ratio < 0.7 {
            0.7
        } else {
            0.5
        }
    }

    /// Validate Beer-Lambert law for attenuation
    pub fn validate_beer_lambert_law(
        initial_intensity: f64,
        absorption_coeff: f64,
        distances: &[f64],
        measured_intensities: &[f64],
    ) -> TheoremValidation {
        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;

        for (&distance, &measured) in distances.iter().zip(measured_intensities.iter()) {
            let theoretical = initial_intensity * (-absorption_coeff * distance).exp();
            let error = (measured - theoretical).abs() / theoretical.abs().max(1e-10);
            max_error = max_error.max(error);
            total_error += error;
        }

        let _avg_error = total_error / distances.len() as f64;
        let passed = max_error < 0.05;

        let max_distance = distances.iter().copied().fold(0.0, f64::max);
        let theoretical_bound = absorption_coeff * max_distance * max_distance / 2.0
            * (-absorption_coeff * max_distance).exp();

        TheoremValidation {
            theorem: "Beer-Lambert Law".to_string(),
            passed,
            error_bound: theoretical_bound,
            measured_error: max_error,
            confidence: Self::calculate_confidence(theoretical_bound, max_error, passed),
            details: format!(
                "Validated over {} points, max error: {:.2e}, theoretical bound: {:.2e}",
                distances.len(),
                max_error,
                theoretical_bound
            ),
        }
    }

    /// Validate Courant-Friedrichs-Lewy (CFL) condition for FDTD
    pub fn validate_cfl_condition(
        time_step: f64,
        spatial_step: f64,
        wave_speed: f64,
        dimensions: usize,
    ) -> TheoremValidation {
        let cfl_number = time_step * wave_speed / spatial_step;
        let cfl_limit = 1.0 / (dimensions as f64).sqrt();

        let passed = cfl_number <= cfl_limit * 0.95;
        let stability_margin = cfl_limit - cfl_number;

        TheoremValidation {
            theorem: "CFL Condition".to_string(),
            passed,
            error_bound: cfl_limit,
            measured_error: cfl_number,
            confidence: if passed { 0.99 } else { 0.1 },
            details: format!(
                "CFL number: {:.3}, limit: {:.3}, stability margin: {:.3}",
                cfl_number, cfl_limit, stability_margin
            ),
        }
    }

    /// Validate Parseval's theorem: ∑|x[n]|² = (1/N) ∑|X[k]|²
    pub fn validate_parsevals_theorem(
        time_domain: &Array1<f64>,
        freq_domain: &Array1<Complex64>,
        _sampling_rate: f64,
    ) -> TheoremValidation {
        let n_samples = time_domain.len() as f64;
        let time_energy: f64 = time_domain.iter().map(|&x| x * x).sum::<f64>();
        let freq_energy: f64 = freq_domain
            .iter()
            .map(|&x: &Complex64| x.norm_sqr())
            .sum::<f64>()
            / n_samples;

        let error = (time_energy - freq_energy).abs() / time_energy.abs().max(1e-10);
        let passed = error < 1e-10;

        TheoremValidation {
            theorem: "Parseval's Theorem".to_string(),
            passed,
            error_bound: 1e-12,
            measured_error: error,
            confidence: Self::calculate_confidence(1e-12, error, passed),
            details: format!(
                "N: {}, Time energy: {:.6e}, Freq energy: {:.6e}, Relative error: {:.2e}",
                n_samples as usize, time_energy, freq_energy, error
            ),
        }
    }

    /// Validate MUSIC algorithm resolution theorem (Rayleigh criterion + Cramér-Rao bound)
    pub fn validate_music_resolution(
        array_length: f64,
        wavelength: f64,
        snr_db: f64,
        measured_resolution: f64,
    ) -> TheoremValidation {
        let rayleigh_limit = 0.886 * wavelength / array_length;
        let snr_linear = 10.0f64.powf(snr_db / 10.0);
        let cramer_rao_limit = rayleigh_limit / snr_linear.sqrt();

        let passed = measured_resolution >= cramer_rao_limit * 0.9;

        TheoremValidation {
            theorem: "MUSIC Resolution Theorem".to_string(),
            passed,
            error_bound: cramer_rao_limit,
            measured_error: measured_resolution,
            confidence: if passed { 0.9 } else { 0.3 },
            details: format!(
                "Rayleigh limit: {:.2e} rad, CR bound: {:.2e} rad, Measured: {:.2e} rad",
                rayleigh_limit, cramer_rao_limit, measured_resolution
            ),
        }
    }

    /// Validate PINN convergence theorem: ‖u − u_PINN‖_H1 ≤ C (log N / N)^{1/4} + C W^{-1/2}
    pub fn validate_pinn_convergence(
        n_collocation: usize,
        network_width: usize,
        measured_error: f64,
        solution_smoothness: f64,
    ) -> TheoremValidation {
        let n_term = (n_collocation as f64).ln() / (n_collocation as f64);
        let convergence_term = n_term.powf(0.25);
        let width_term = 1.0 / (network_width as f64).sqrt();
        let theoretical_bound = solution_smoothness * (convergence_term + width_term);

        let passed = measured_error <= theoretical_bound * 2.0;
        let relative_error = measured_error / theoretical_bound;

        TheoremValidation {
            theorem: "PINN Convergence Theorem".to_string(),
            passed,
            error_bound: theoretical_bound,
            measured_error,
            confidence: if passed { 0.8 } else { 0.4 },
            details: format!(
                "N: {}, Width: {}, Theoretical bound: {:.2e}, Measured: {:.2e}, Ratio: {:.2}",
                n_collocation, network_width, theoretical_bound, measured_error, relative_error
            ),
        }
    }

    /// Validate Kramers-Kronig relations for power-law absorption/dispersion
    pub fn validate_kramers_kronig(
        frequencies: &[f64],
        absorption: &[f64],
        dispersion: &[f64],
        sound_speed: f64,
    ) -> TheoremValidation {
        let alpha_0 = absorption[0] / frequencies[0].powf(1.5);
        let alpha_power = 1.5;

        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;

        for (i, (&omega, &measured_disp)) in frequencies.iter().zip(dispersion.iter()).enumerate() {
            let tan_factor = (alpha_power * PI / 2.0).tan();
            let theoretical_disp =
                sound_speed + (2.0 * alpha_0 / PI) * omega.powf(alpha_power) * tan_factor;

            let error = (theoretical_disp - measured_disp).abs() / sound_speed.abs().max(1e-10);
            max_error = max_error.max(error);
            total_error += error;

            let theoretical_alpha = alpha_0 * omega.powf(alpha_power);
            let measured_alpha = absorption[i];
            let alpha_error =
                (theoretical_alpha - measured_alpha).abs() / measured_alpha.abs().max(1e-10);
            max_error = max_error.max(alpha_error);
        }

        let avg_error = total_error / frequencies.len() as f64;
        let passed = max_error < 0.1;

        TheoremValidation {
            theorem: "Kramers-Kronig Relations".to_string(),
            passed,
            error_bound: 0.05,
            measured_error: max_error,
            confidence: if passed { 0.9 } else { 0.4 },
            details: format!(
                "Power-law fit (y={:.1}): α₀={:.2e}, Max error: {:.2e}, Avg error: {:.2e}",
                alpha_power, alpha_0, max_error, avg_error
            ),
        }
    }

    /// Validate ultrasound axial resolution: Δr = c / (2 · bandwidth)
    pub fn validate_sa_resolution(
        bandwidth: f64,
        sound_speed: f64,
        _snr_db: f64,
        measured_resolution: f64,
    ) -> TheoremValidation {
        let theoretical_resolution = sound_speed / (2.0 * bandwidth);
        let passed = measured_resolution <= theoretical_resolution * 2.0;

        TheoremValidation {
            theorem: "Ultrasound Axial Resolution".to_string(),
            passed,
            error_bound: theoretical_resolution,
            measured_error: measured_resolution,
            confidence: if passed { 0.9 } else { 0.5 },
            details: format!(
                "Bandwidth: {:.1} MHz, c: {:.0} m/s, Theoretical: {:.2e} m, Measured: {:.2e} m, Ratio: {:.1}x",
                bandwidth / 1e6, sound_speed, theoretical_resolution, measured_resolution,
                measured_resolution / theoretical_resolution
            ),
        }
    }

    /// Validate coded excitation SNR: SNR_improved = SNR_original · √L · η
    pub fn validate_coded_excitation_snr(
        code_length: usize,
        compression_ratio: f64,
        measured_snr_improvement_db: f64,
    ) -> TheoremValidation {
        let processing_gain_db = 5.0 * (code_length as f64).log10();
        let compression_gain_db = 10.0 * compression_ratio.log10();
        let theoretical_snr_db = processing_gain_db + compression_gain_db;

        let passed = measured_snr_improvement_db >= theoretical_snr_db * 0.7;
        let efficiency = measured_snr_improvement_db / theoretical_snr_db;

        TheoremValidation {
            theorem: "Coded Excitation SNR Theorem".to_string(),
            passed,
            error_bound: theoretical_snr_db,
            measured_error: measured_snr_improvement_db,
            confidence: if passed { 0.85 } else { 0.4 },
            details: format!(
                "Code length: {}, Processing gain: {:.1} dB, Compression: {:.1}x ({:.1} dB), Theoretical: {:.1} dB, Measured: {:.1} dB, Efficiency: {:.1}%",
                code_length, processing_gain_db, compression_ratio, compression_gain_db,
                theoretical_snr_db, measured_snr_improvement_db, efficiency * 100.0
            ),
        }
    }

    /// Validate acoustic reciprocity: p(r1|r2) = p(r2|r1)
    pub fn validate_reciprocity(
        pressure_12: f64,
        pressure_21: f64,
        tolerance: f64,
    ) -> TheoremValidation {
        let ratio = pressure_12 / pressure_21;
        let passed = (ratio - 1.0).abs() < tolerance;
        let error = (ratio - 1.0).abs();

        TheoremValidation {
            theorem: "Acoustic Reciprocity Theorem".to_string(),
            passed,
            error_bound: tolerance,
            measured_error: error,
            confidence: if passed { 0.95 } else { 0.3 },
            details: format!(
                "p12/p21 = {:.6}, Expected: 1.0, Error: {:.2e}, Tolerance: {:.2e}",
                ratio, error, tolerance
            ),
        }
    }

    /// Validate plane wave reflection: R = (Z2 - Z1) / (Z2 + Z1)
    pub fn validate_impedance_reflection(
        rho1: f64,
        c1: f64,
        rho2: f64,
        c2: f64,
        measured_reflection_coeff: f64,
    ) -> TheoremValidation {
        let z1 = rho1 * c1;
        let z2 = rho2 * c2;
        let theoretical_r = (z2 - z1) / (z2 + z1);

        let error =
            (measured_reflection_coeff - theoretical_r).abs() / theoretical_r.abs().max(1e-10);
        let passed = error < 0.05;

        TheoremValidation {
            theorem: "Acoustic Impedance Reflection".to_string(),
            passed,
            error_bound: theoretical_r.abs(),
            measured_error: measured_reflection_coeff,
            confidence: if passed { 0.9 } else { 0.4 },
            details: format!(
                "Z1: {:.0} Pa·s/m, Z2: {:.0} Pa·s/m, Theoretical R: {:.4}, Measured R: {:.4}, Error: {:.2e}",
                z1, z2, theoretical_r, measured_reflection_coeff, error
            ),
        }
    }

    /// Validate speed of sound in ideal gas: c = √(γ kT / m)
    pub fn validate_ideal_gas_speed(
        temperature_k: f64,
        molecular_mass: f64,
        gamma: f64,
        measured_speed: f64,
    ) -> TheoremValidation {
        let theoretical_speed =
            (gamma * BOLTZMANN * temperature_k / (molecular_mass / AVOGADRO)).sqrt();

        let error = (measured_speed - theoretical_speed).abs() / theoretical_speed;
        let passed = error < 0.01;

        TheoremValidation {
            theorem: "Ideal Gas Speed of Sound".to_string(),
            passed,
            error_bound: theoretical_speed,
            measured_error: measured_speed,
            confidence: if passed { 0.95 } else { 0.6 },
            details: format!(
                "T: {:.0} K, M: {:.2e} kg/mol, γ: {:.2}, Theoretical c: {:.0} m/s, Measured c: {:.0} m/s, Error: {:.2e}",
                temperature_k, molecular_mass, gamma, theoretical_speed, measured_speed, error
            ),
        }
    }

    /// Validate Rayleigh-Sommerfeld far-field 1/r spherical wave decay
    pub fn validate_rayleigh_sommerfeld_decay(
        distances: &[f64],
        pressures: &[f64],
        _source_pressure: f64,
    ) -> TheoremValidation {
        let mut passed = true;
        let mut max_error: f64 = 0.0;

        for i in 1..distances.len() {
            let r1: f64 = distances[i - 1];
            let r2: f64 = distances[i];
            let p1: f64 = pressures[i - 1].abs();
            let p2: f64 = pressures[i].abs();

            if p1 > 1e-12 && p2 > 1e-12 {
                let theoretical_ratio = r1 / r2;
                let measured_ratio = p2 / p1;
                let error = (measured_ratio - theoretical_ratio).abs() / theoretical_ratio;

                if error > 0.1 {
                    passed = false;
                }
                max_error = max_error.max(error);
            }
        }

        TheoremValidation {
            theorem: "Rayleigh-Sommerfeld Diffraction".to_string(),
            passed,
            error_bound: 0.05,
            measured_error: max_error,
            confidence: if passed { 0.85 } else { 0.4 },
            details: format!(
                "Spherical wave decay validation: {} points, Max error: {:.2e}",
                distances.len(),
                max_error
            ),
        }
    }
}
