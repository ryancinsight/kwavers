//! Material properties validation tests
//!
//! References:
//! - Szabo (1994) - "Time domain wave equations for lossy media"
//! - Duck (1990) - "Physical Properties of Tissue"
//! - Royer & Dieulesaint (2000) - "Elastic Waves in Solids"

use ndarray::Array3;

// Test-specific tissue properties structure
#[derive(Debug, Clone)]
pub struct TissueProperties {
    pub density: Array3<f64>,
    pub sound_speed: Array3<f64>,
    pub attenuation: Array3<f64>,
}

impl TissueProperties {
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            density: Array3::zeros(shape),
            sound_speed: Array3::zeros(shape),
            attenuation: Array3::zeros(shape),
        }
    }
}

// Tissue property constants from Duck (1990)
const LIVER_DENSITY: f64 = 1060.0; // kg/m³
const LIVER_SOUND_SPEED: f64 = 1595.0; // m/s
const LIVER_ATTENUATION_COEFF: f64 = 0.5; // dB/cm/MHz
const LIVER_ATTENUATION_POWER: f64 = 1.1;

// Bone properties from Royer & Dieulesaint
const BONE_DENSITY: f64 = 1900.0; // kg/m³
const BONE_P_WAVE_SPEED: f64 = 4080.0; // m/s
const BONE_S_WAVE_SPEED: f64 = 2000.0; // m/s

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractional_absorption_power_law() {
        // Validate Szabo's fractional derivative model
        let frequencies = vec![1e6, 2e6, 5e6, 10e6]; // Hz
        let distance = 0.1; // 10 cm

        for freq in frequencies {
            // Calculate attenuation using power law
            // α = α₀ * f^n where α₀ is in dB/cm/MHz^n
            let attenuation_db_per_cm =
                LIVER_ATTENUATION_COEFF * (freq / 1e6_f64).powf(LIVER_ATTENUATION_POWER);
            let attenuation_db = attenuation_db_per_cm * (distance * 100.0); // distance in cm

            // Convert dB to Neper: 1 dB = ln(10)/20 Np ≈ 0.115129 Np
            // But we have dB total, not dB/m, so no additional conversion needed
            let attenuation_np = attenuation_db / 8.686; // 20/ln(10)

            // Theoretical amplitude reduction
            let amplitude_ratio = (-attenuation_np).exp();

            // Debug output
            eprintln!(
                "Freq: {} MHz, Atten_dB: {:.2}, Atten_Np: {:.4}, Ratio: {:.3}",
                freq / 1e6,
                attenuation_db,
                attenuation_np,
                amplitude_ratio
            );

            // Verify against calculated values for liver tissue
            // For liver: α = 0.5 dB/cm/MHz^1.1
            // At 10 cm distance, amplitude ratio = exp(-α*d/8.686)
            let expected_ratio = match (freq / 1e6) as i32 {
                1 => 0.562,  // At 1 MHz, 10 cm (5.0 dB total)
                2 => 0.291,  // At 2 MHz, 10 cm (10.7 dB total)
                5 => 0.034,  // At 5 MHz, 10 cm (29.4 dB total)
                10 => 0.007, // At 10 MHz, 10 cm (63.1 dB total)
                _ => continue,
            };

            let error = (amplitude_ratio - expected_ratio).abs() / expected_ratio;
            assert!(
                error < 0.1,
                "Power law absorption error at {} MHz: {:.2}%",
                freq / 1e6,
                error * 100.0
            );
        }
    }

    #[test]
    fn test_anisotropic_christoffel_equation() {
        // Validate wave speeds in anisotropic bone using Christoffel equation
        // For isotropic approximation of cortical bone

        // Calculate Lamé parameters from wave speeds
        // For elastic waves: v_p² = (λ + 2μ)/ρ and v_s² = μ/ρ
        let shear_modulus = BONE_DENSITY * BONE_S_WAVE_SPEED.powi(2);
        let lambda = BONE_DENSITY * BONE_P_WAVE_SPEED.powi(2) - 2.0 * shear_modulus;
        let bulk_modulus = lambda + 2.0 * shear_modulus / 3.0;

        // Verify P and S wave speeds from Lamé parameters
        let calc_p_speed = ((lambda + 2.0 * shear_modulus) / BONE_DENSITY).sqrt();
        let calc_s_speed = (shear_modulus / BONE_DENSITY).sqrt();

        let p_error = (calc_p_speed - BONE_P_WAVE_SPEED).abs() / BONE_P_WAVE_SPEED;
        let s_error = (calc_s_speed - BONE_S_WAVE_SPEED).abs() / BONE_S_WAVE_SPEED;

        assert!(
            p_error < 0.01,
            "P-wave speed error: {:.2}%",
            p_error * 100.0
        );
        assert!(
            s_error < 0.01,
            "S-wave speed error: {:.2}%",
            s_error * 100.0
        );

        // Test Christoffel determinant for wave propagation
        // For propagation along x-axis: det(Γ - ρv²I) = 0
        let propagation_dir = [1.0, 0.0, 0.0];

        // Christoffel matrix for isotropic medium
        let gamma_11 = lambda + 2.0 * shear_modulus;
        let gamma_22 = shear_modulus;
        let gamma_33 = shear_modulus;

        // Eigenvalues should give wave speeds
        let eigenval_p = gamma_11 / BONE_DENSITY;
        let eigenval_s = gamma_22 / BONE_DENSITY;

        assert!((eigenval_p.sqrt() - BONE_P_WAVE_SPEED).abs() < 1.0);
        assert!((eigenval_s.sqrt() - BONE_S_WAVE_SPEED).abs() < 1.0);
    }

    #[test]
    fn test_tissue_heterogeneity() {
        // Test proper handling of tissue interfaces
        let nx = 100;
        let tissue_map: Array3<f64> = Array3::zeros((nx, 1, 1));

        // Create layered medium: water -> liver -> bone
        let mut properties = TissueProperties::new((nx, 1, 1));

        for i in 0..nx {
            if i < nx / 3 {
                // Water
                properties.density[[i, 0, 0]] = 1000.0;
                properties.sound_speed[[i, 0, 0]] = 1500.0;
                properties.attenuation[[i, 0, 0]] = 0.0022;
            } else if i < 2 * nx / 3 {
                // Liver
                properties.density[[i, 0, 0]] = LIVER_DENSITY;
                properties.sound_speed[[i, 0, 0]] = LIVER_SOUND_SPEED;
                properties.attenuation[[i, 0, 0]] = LIVER_ATTENUATION_COEFF;
            } else {
                // Bone
                properties.density[[i, 0, 0]] = BONE_DENSITY;
                properties.sound_speed[[i, 0, 0]] = BONE_P_WAVE_SPEED;
                properties.attenuation[[i, 0, 0]] = 10.0; // High attenuation
            }
        }

        // Calculate impedance and transmission coefficients
        // Acoustic impedance Z = ρ * c
        let z_water = 1000.0 * 1500.0; // 1.5 MRayl
        let z_liver = LIVER_DENSITY * LIVER_SOUND_SPEED; // ~1.69 MRayl
        let z_bone = BONE_DENSITY * BONE_P_WAVE_SPEED; // ~7.75 MRayl

        // Transmission coefficient T = 2*Z2/(Z1+Z2)
        let t_water_liver = 2.0 * z_liver / (z_water + z_liver);
        let t_liver_bone = 2.0 * z_bone / (z_liver + z_bone);

        // Log calculated values for debugging
        eprintln!(
            "Z_water: {:.2e}, Z_liver: {:.2e}, Z_bone: {:.2e}",
            z_water, z_liver, z_bone
        );
        eprintln!(
            "T_water-liver: {:.3}, T_liver-bone: {:.3}",
            t_water_liver, t_liver_bone
        );

        // Verify transmission coefficients
        // Expected: T_water-liver ≈ 1.06 (Z_liver/Z_water ≈ 1.13)
        // Expected: T_liver-bone ≈ 1.64 (Z_bone/Z_liver ≈ 4.6)
        assert!(
            (t_water_liver - 1.06).abs() < 0.05,
            "Water-liver transmission error: {:.3} (expected ~1.06)",
            t_water_liver
        );
        assert!(
            (t_liver_bone - 1.64).abs() < 0.05,
            "Liver-bone transmission error: {:.3} (expected ~1.64)",
            t_liver_bone
        );
    }
}
