//! Absorption and dispersion models for acoustic wave propagation
//!
//! This module implements state-of-the-art absorption models including:
//! - Power-law frequency dependence
//! - Fractional derivative formulations
//! - Tissue-specific absorption coefficients
//!
//! # Theory
//!
//! Acoustic absorption in biological tissues and other media follows a
//! power-law frequency dependence:
//! ```text
//! α(f) = α₀ * f^y
//! ```
//! where:
//! - α is the absorption coefficient [Np/m]
//! - α₀ is the absorption coefficient at 1 MHz
//! - f is frequency [MHz]
//! - y is the power law exponent (typically 1.0-1.5 for tissues)
//!
//! This can be modeled using fractional derivatives in the time domain
//! or efficiently in the frequency domain for spectral methods.
//!
//! # References
//!
//! - Treeby, B. E., & Cox, B. T. (2010). "Modeling power law absorption and
//!   dispersion for acoustic propagation using the fractional Laplacian."
//!   JASA, 127(5), 2741-2748.
//! - Szabo, T. L. (2004). "Diagnostic ultrasound imaging: inside out."
//!   Academic Press.

use ndarray::{Array3, Zip};
use num_complex::Complex;
use std::f64::consts::PI;
use serde::{Serialize, Deserialize};
use crate::grid::Grid;

/// Power-law absorption model configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PowerLawAbsorption {
    /// Absorption coefficient at 1 MHz [dB/(MHz^y cm)]
    pub alpha_0: f64,
    /// Power law exponent (typically 1.0-1.5)
    pub y: f64,
    /// Reference frequency for alpha_0 [Hz]
    pub f_ref: f64,
    /// Enable dispersion correction
    pub dispersion_correction: bool,
}

impl Default for PowerLawAbsorption {
    fn default() -> Self {
        Self {
            alpha_0: 0.75,  // Typical for soft tissue
            y: 1.05,        // Typical for soft tissue
            f_ref: 1e6,     // 1 MHz reference
            dispersion_correction: true,
        }
    }
}

impl PowerLawAbsorption {
    /// Create absorption model for specific tissue type
    pub fn for_tissue(tissue_type: TissueType) -> Self {
        match tissue_type {
            TissueType::Water => Self {
                alpha_0: 0.0022,
                y: 2.0,
                f_ref: 1e6,
                dispersion_correction: true,
            },
            TissueType::Blood => Self {
                alpha_0: 0.18,
                y: 1.2,
                f_ref: 1e6,
                dispersion_correction: true,
            },
            TissueType::Fat => Self {
                alpha_0: 0.63,
                y: 1.0,
                f_ref: 1e6,
                dispersion_correction: true,
            },
            TissueType::Muscle => Self {
                alpha_0: 1.3,
                y: 1.0,
                f_ref: 1e6,
                dispersion_correction: true,
            },
            TissueType::Liver => Self {
                alpha_0: 0.9,
                y: 1.0,
                f_ref: 1e6,
                dispersion_correction: true,
            },
            TissueType::Brain => Self {
                alpha_0: 0.85,
                y: 1.21,
                f_ref: 1e6,
                dispersion_correction: true,
            },
            TissueType::Bone => Self {
                alpha_0: 20.0,
                y: 1.0,
                f_ref: 1e6,
                dispersion_correction: true,
            },
            TissueType::SoftTissue => Self {
                alpha_0: 0.75,
                y: 1.05,
                f_ref: 1e6,
                dispersion_correction: true,
            },
            TissueType::Skin => Self {
                alpha_0: 2.1,
                y: 1.75,
                f_ref: 1e6,
                dispersion_correction: true,
            },
            TissueType::Custom(alpha_0, y) => Self {
                alpha_0,
                y,
                f_ref: 1e6,
                dispersion_correction: true,
            },
        }
    }
    
    /// Compute absorption coefficient at given frequency
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        // Convert to dB/(MHz^y cm) to Np/m
        let f_mhz = frequency / 1e6;
        let alpha_db = self.alpha_0 * f_mhz.powf(self.y);
        // Convert from dB/cm to Np/m
        // 1 dB/cm = 1.1513 Np/m
        alpha_db * 1.1513
    }
    
    /// Compute phase velocity from dispersion relation
    pub fn phase_velocity(&self, frequency: f64, c0: f64) -> f64 {
        if !self.dispersion_correction {
            return c0;
        }
        
        // Kramers-Kronig relation for causality
        // c(ω) = c₀ / (1 - α₀ * tan(πy/2) * ω^(y-1))
        let omega = 2.0 * PI * frequency;
        let omega_ref = 2.0 * PI * self.f_ref;
        let tan_term = (PI * self.y / 2.0).tan();
        
        // Normalized frequency
        let omega_norm = omega / omega_ref;
        
        // Phase velocity with dispersion
        c0 / (1.0 + self.alpha_0 * tan_term * omega_norm.powf(self.y - 1.0) / (2.0 * PI))
    }
}

/// Tissue types with known absorption properties
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TissueType {
    Water,
    Blood,
    Fat,
    Muscle,
    Liver,
    Brain,
    Bone,
    SoftTissue,
    Skin,
    Custom(f64, f64),  // (alpha_0, y)
}

// Manual implementation of Eq for TissueType
impl Eq for TissueType {}

// Manual implementation of Hash for TissueType
impl std::hash::Hash for TissueType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use std::mem::discriminant;
        discriminant(self).hash(state);
        
        // For Custom variant, hash the bit representation of the floats
        if let TissueType::Custom(a, b) = self {
            a.to_bits().hash(state);
            b.to_bits().hash(state);
        }
    }
}

/// Fractional Laplacian operator for power-law absorption
///
/// Implements the fractional Laplacian (-∇²)^(y/2) for modeling
/// power-law absorption and dispersion in the frequency domain.
pub struct FractionalLaplacian {
    /// Power law exponent
    pub y: f64,
    /// Absorption coefficient
    pub alpha_coeff: f64,
    /// Sound speed
    pub c0: f64,
    /// Enable tan correction for causality
    pub use_tan_correction: bool,
}

impl FractionalLaplacian {
    /// Create a new fractional Laplacian operator
    pub fn new(absorption: &PowerLawAbsorption, c0: f64) -> Self {
        // Convert absorption coefficient to appropriate units
        // α₀ in dB/(MHz^y cm) to absorption coefficient for fractional Laplacian
        let alpha_coeff = absorption.alpha_0 * 1.1513e-2 / (2.0 * c0);
        
        Self {
            y: absorption.y,
            alpha_coeff,
            c0,
            use_tan_correction: absorption.dispersion_correction,
        }
    }
    
    /// Apply fractional Laplacian absorption in k-space
    ///
    /// This implements the absorption-dispersion term:
    /// ```text
    /// L_α = -2α₀c₀^(y-1) * (-k²)^(y/2)
    /// ```
    pub fn apply_absorption_kspace(
        &self,
        field_k: &mut Array3<Complex<f64>>,
        k_squared: &Array3<f64>,
        dt: f64,
    ) {
        let tan_factor = if self.use_tan_correction {
            (PI * self.y / 2.0).tan()
        } else {
            0.0
        };
        
        Zip::from(field_k)
            .and(k_squared)
            .for_each(|f, &k2| {
                if k2 > 0.0 {
                    // Fractional power of k²
                    let k_frac = k2.powf(self.y / 2.0);
                    
                    // Absorption term
                    let absorption = 2.0 * self.alpha_coeff * self.c0.powf(self.y - 1.0) * k_frac;
                    
                    // Dispersion term (imaginary part for phase shift)
                    let dispersion = if self.use_tan_correction {
                        absorption * tan_factor
                    } else {
                        0.0
                    };
                    
                    // Apply absorption and dispersion
                    // exp(-absorption*dt) ≈ 1 - absorption*dt for small dt
                    let decay = (-absorption * dt).exp();
                    let phase = dispersion * dt;
                    
                    *f *= Complex::new(decay * phase.cos(), -decay * phase.sin());
                }
            });
    }
    
    /// Compute absorption loss for a field
    pub fn compute_absorption_loss(
        &self,
        field: &Array3<f64>,
        k_squared: &Array3<f64>,
        frequency: f64,
    ) -> Array3<f64> {
        let mut loss = Array3::zeros(field.raw_dim());
        
        // Power-law absorption coefficient at this frequency
        let f_mhz = frequency / 1e6;
        let alpha = self.alpha_coeff * 2.0 * self.c0 * f_mhz.powf(self.y);
        
        Zip::from(&mut loss)
            .and(field)
            .and(k_squared)
            .for_each(|l, &f, &k2| {
                if k2 > 0.0 {
                    // Fractional Laplacian absorption
                    let k_frac = k2.powf(self.y / 2.0);
                    *l = -alpha * k_frac * f;
                }
            });
        
        loss
    }
}

/// Acoustic diffusivity model (correct physical implementation)
///
/// Acoustic diffusivity δ is related to both thermal and viscous effects:
/// ```text
/// δ = (2/ρ₀c₀³) * [4μ/3 + μ_B + κ(γ-1)/C_p]
/// ```
/// where:
/// - μ is shear viscosity
/// - μ_B is bulk viscosity
/// - κ is thermal conductivity
/// - γ is specific heat ratio
/// - C_p is specific heat at constant pressure
pub struct AcousticDiffusivity {
    /// Shear viscosity [Pa·s]
    pub shear_viscosity: f64,
    /// Bulk viscosity [Pa·s]
    pub bulk_viscosity: f64,
    /// Thermal conductivity [W/(m·K)]
    pub thermal_conductivity: f64,
    /// Specific heat ratio
    pub gamma: f64,
    /// Specific heat at constant pressure [J/(kg·K)]
    pub cp: f64,
    /// Reference density [kg/m³]
    pub rho0: f64,
    /// Reference sound speed [m/s]
    pub c0: f64,
}

impl AcousticDiffusivity {
    /// Create acoustic diffusivity for water at 20°C
    pub fn water_20c() -> Self {
        Self {
            shear_viscosity: 1.002e-3,
            bulk_viscosity: 2.81e-3,
            thermal_conductivity: 0.598,
            gamma: 1.0,  // Nearly incompressible
            cp: 4182.0,
            rho0: 998.2,
            c0: 1482.0,
        }
    }
    
    /// Create acoustic diffusivity for soft tissue
    pub fn soft_tissue() -> Self {
        Self {
            shear_viscosity: 1.5e-3,
            bulk_viscosity: 3.0e-3,
            thermal_conductivity: 0.5,
            gamma: 1.1,
            cp: 3600.0,
            rho0: 1050.0,
            c0: 1540.0,
        }
    }
    
    /// Compute total acoustic diffusivity coefficient
    pub fn total_diffusivity(&self) -> f64 {
        let viscous_term = (4.0 / 3.0) * self.shear_viscosity + self.bulk_viscosity;
        let thermal_term = self.thermal_conductivity * (self.gamma - 1.0) / self.cp;
        
        (2.0 / (self.rho0 * self.c0.powi(3))) * (viscous_term + thermal_term)
    }
    
    /// Apply diffusivity term in Kuznetsov equation
    ///
    /// This implements the term: -(δ/c₀⁴)∂³p/∂t³
    pub fn apply_diffusivity(
        &self,
        d3p_dt3: &Array3<f64>,
    ) -> Array3<f64> {
        let delta = self.total_diffusivity();
        let c0_4 = self.c0.powi(4);
        let coefficient = -delta / c0_4;
        
        d3p_dt3.mapv(|val| coefficient * val)
    }
}

/// Apply power-law absorption to a field in the frequency domain
///
/// This is the main function for applying absorption in PSTD/k-space methods
pub fn apply_power_law_absorption(
    field_k: &mut Array3<Complex<f64>>,
    k_squared: &Array3<f64>,
    absorption: &PowerLawAbsorption,
    c0: f64,
    dt: f64,
) {
    let frac_laplacian = FractionalLaplacian::new(absorption, c0);
    frac_laplacian.apply_absorption_kspace(field_k, k_squared, dt);
}

/// Compute tissue-specific absorption coefficient
pub fn tissue_specific_absorption(
    tissue_type: TissueType,
    frequency: f64,
) -> f64 {
    let absorption = PowerLawAbsorption::for_tissue(tissue_type);
    absorption.absorption_at_frequency(frequency)
}

/// Helper function for computing absorption coefficient
/// This is for backward compatibility with existing code
pub fn absorption_coefficient(frequency: f64, _temperature: f64, _bubble_radius: Option<f64>) -> f64 {
    // Default soft tissue absorption
    let absorption = PowerLawAbsorption::default();
    absorption.absorption_at_frequency(frequency)
}

/// Module for power law absorption (backward compatibility)
pub mod power_law_absorption {
    use super::*;
    
    /// Compute power-law absorption coefficient
    pub fn power_law_absorption_coefficient(frequency: f64, alpha0: f64, y: f64) -> f64 {
        let absorption = PowerLawAbsorption {
            alpha_0: alpha0,
            y,
            f_ref: 1e6,
            dispersion_correction: false,
        };
        absorption.absorption_at_frequency(frequency)
    }
}

/// Module for tissue-specific properties (backward compatibility)
pub mod tissue_specific {
    use super::*;
    use std::collections::HashMap;
    
    /// Tissue properties structure
    #[derive(Debug, Clone, Copy)]
    pub struct TissueProperties {
        pub density: f64,
        pub sound_speed: f64,
        pub alpha0: f64,
        pub delta: f64,
        pub lame_lambda: f64,
        pub lame_mu: f64,
        pub specific_heat: f64,
        pub thermal_conductivity: f64,
        pub b_a: f64,  // nonlinearity coefficient
        pub shear_sound_speed: f64,
        pub shear_viscosity_coeff: f64,
        pub bulk_viscosity_coeff: f64,
    }
    
    /// Get tissue database
    pub fn tissue_database() -> HashMap<TissueType, TissueProperties> {
        let mut db = HashMap::new();
        
        db.insert(TissueType::SoftTissue, TissueProperties {
            density: 1050.0,
            sound_speed: 1540.0,
            alpha0: 0.75,
            delta: 1.05,
            lame_lambda: 2.28e9,
            lame_mu: 1.0e4,
            specific_heat: 3600.0,  // J/kg·K
            thermal_conductivity: 0.52,  // W/m·K
            b_a: 6.0,  // typical for soft tissue
            shear_sound_speed: 3.08,  // m/s (calculated from lame_mu/density)
            shear_viscosity_coeff: 0.0,  // Pa·s
            bulk_viscosity_coeff: 0.0,  // Pa·s
        });
        
        db.insert(TissueType::Skin, TissueProperties {
            density: 1100.0,
            sound_speed: 1600.0,
            alpha0: 2.1,
            delta: 1.75,
            lame_lambda: 2.5e9,
            lame_mu: 2.0e4,
            specific_heat: 3700.0,  // J/kg·K
            thermal_conductivity: 0.5,  // W/m·K
            b_a: 1.0,  // typical for skin
            shear_sound_speed: 1.5,  // m/s (calculated from lame_mu/density)
            shear_viscosity_coeff: 1.0e-3,  // Pa·s
            bulk_viscosity_coeff: 1.0e-3,  // Pa·s
        });
        
        db
    }
}

/// Export for compatibility
pub use self::apply_power_law_absorption as power_law_absorption;
pub use self::tissue_specific_absorption as tissue_specific_fn;
pub use self::FractionalLaplacian as fractional_derivative;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_power_law_absorption() {
        let absorption = PowerLawAbsorption::for_tissue(TissueType::Muscle);
        
        // Test at 1 MHz
        let alpha_1mhz = absorption.absorption_at_frequency(1e6);
        assert!(alpha_1mhz > 0.0);
        
        // Test frequency dependence
        let alpha_2mhz = absorption.absorption_at_frequency(2e6);
        let ratio = alpha_2mhz / alpha_1mhz;
        assert!((ratio - 2.0_f64.powf(absorption.y)).abs() < 0.01);
    }
    
    #[test]
    fn test_acoustic_diffusivity() {
        let diffusivity = AcousticDiffusivity::water_20c();
        let delta = diffusivity.total_diffusivity();
        
        // Check that diffusivity is positive and reasonable
        assert!(delta > 0.0);
        assert!(delta < 1e-6);  // Should be small for water
    }
    
    #[test]
    fn test_phase_velocity_dispersion() {
        let absorption = PowerLawAbsorption::default();
        let c0 = 1500.0;
        
        // Phase velocity should change with frequency due to dispersion
        let c_1mhz = absorption.phase_velocity(1e6, c0);
        let c_5mhz = absorption.phase_velocity(5e6, c0);
        
        assert!(c_1mhz != c_5mhz);
        // Higher frequencies typically travel faster in dispersive media
        assert!(c_5mhz > c_1mhz);
    }
}