// medium/absorption/tissue_specific.rs
use log::{debug, info, trace};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Tissue type enum for different biological tissues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TissueType {
    BloodVessel,
    Bone,
    BoneCortical,
    BoneMarrow,
    Brain,
    Fat,
    Kidney,
    Liver,
    Lung,
    Muscle,
    Skin,
    SoftTissue, // General soft tissue
    Tumor,      // Generic tumor tissue
    // Add more tissue types as needed
}

/// Tissue properties for acoustic simulations
#[derive(Debug, Clone, Copy)]
pub struct TissueProperties {
    /// Name of the tissue (for logging and display)
    pub name: &'static str,
    /// Density (kg/m³)
    pub density: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Absorption coefficient at 1MHz (Np/m/MHz^y)
    pub alpha0: f64,
    /// Power law exponent for frequency-dependent absorption
    pub y: f64,
    /// Nonlinearity parameter B/A
    pub b_a: f64,
    /// Specific heat capacity (J/kg/K)
    pub specific_heat: f64,
    /// Thermal conductivity (W/m/K)
    pub thermal_conductivity: f64,
    /// Acoustic impedance (kg/m²/s)
    pub impedance: f64,
    /// Shear wave speed (m/s) - typically low for soft tissues, higher for bone
    pub shear_sound_speed: f64,
    /// Shear viscosity coefficient (Pa·s) - related to shear wave damping
    pub shear_viscosity_coeff: f64,
    /// Bulk viscosity coefficient (Pa·s) - related to compressional wave damping not covered by other mechanisms
    pub bulk_viscosity_coeff: f64,
}

/// Get tissue database singleton
pub fn tissue_database() -> &'static HashMap<TissueType, TissueProperties> {
    static TISSUE_DB: OnceLock<HashMap<TissueType, TissueProperties>> = OnceLock::new();
    
    TISSUE_DB.get_or_init(|| {
        let mut db = HashMap::new();
        
        // Add tissue properties from literature
        // Values from:
        // - Szabo, "Diagnostic Ultrasound Imaging: Inside Out" (2014)
        // - Duck, "Physical Properties of Tissue" (1990)
        // - Bamber, "Acoustic characteristics of biological media" (1986)
        
        db.insert(TissueType::BloodVessel, TissueProperties {
            name: "Blood Vessel",
            density: 1060.0,
            sound_speed: 1580.0,
            alpha0: 0.18,
            y: 1.3,
            b_a: 6.1,
            specific_heat: 3770.0,
            thermal_conductivity: 0.51,
            impedance: 1.68e6,
            shear_sound_speed: 10.0,
            shear_viscosity_coeff: 0.1,
            bulk_viscosity_coeff: 0.1,
        });
        
        db.insert(TissueType::Bone, TissueProperties {
            name: "Bone (General)",
            density: 1908.0,
            sound_speed: 3476.0,
            alpha0: 5.5,
            y: 1.1,
            b_a: 15.0, 
            specific_heat: 1313.0,
            thermal_conductivity: 0.32,
            impedance: 6.63e6,
            shear_sound_speed: 1000.0,
            shear_viscosity_coeff: 1.0,
            bulk_viscosity_coeff: 1.0,
        });
        
        db.insert(TissueType::BoneCortical, TissueProperties {
            name: "Bone (Cortical)",
            density: 1975.0,
            sound_speed: 4080.0,
            alpha0: 6.9,
            y: 1.0,
            b_a: 22.0,
            specific_heat: 1300.0,
            thermal_conductivity: 0.38,
            impedance: 8.06e6,
            shear_sound_speed: 1500.0,
            shear_viscosity_coeff: 1.5,
            bulk_viscosity_coeff: 1.5,
        });
        
        db.insert(TissueType::BoneMarrow, TissueProperties {
            name: "Bone Marrow",
            density: 970.0,
            sound_speed: 1700.0,
            alpha0: 0.9,
            y: 1.5,
            b_a: 8.0,
            specific_heat: 2700.0,
            thermal_conductivity: 0.22,
            impedance: 1.65e6,
            shear_sound_speed: 5.0, // Lower than typical fat, more fluid-like
            shear_viscosity_coeff: 0.05,
            bulk_viscosity_coeff: 0.05,
        });
        
        db.insert(TissueType::Brain, TissueProperties {
            name: "Brain Tissue",
            density: 1040.0,
            sound_speed: 1550.0,
            alpha0: 0.6,
            y: 1.3,
            b_a: 6.8,
            specific_heat: 3630.0,
            thermal_conductivity: 0.51,
            impedance: 1.61e6,
            shear_sound_speed: 8.0,
            shear_viscosity_coeff: 0.1,
            bulk_viscosity_coeff: 0.1,
        });
        
        db.insert(TissueType::Fat, TissueProperties {
            name: "Fat",
            density: 950.0,
            sound_speed: 1450.0,
            alpha0: 0.63,
            y: 1.5,
            b_a: 9.6,
            specific_heat: 2348.0,
            thermal_conductivity: 0.21,
            impedance: 1.38e6,
            shear_sound_speed: 7.0,
            shear_viscosity_coeff: 0.08,
            bulk_viscosity_coeff: 0.08,
        });
        
        db.insert(TissueType::Kidney, TissueProperties {
            name: "Kidney",
            density: 1050.0,
            sound_speed: 1560.0,
            alpha0: 1.0,
            y: 1.3,
            b_a: 7.4,
            specific_heat: 3763.0,
            thermal_conductivity: 0.53,
            impedance: 1.64e6,
            shear_sound_speed: 10.0,
            shear_viscosity_coeff: 0.1,
            bulk_viscosity_coeff: 0.1,
        });
        
        db.insert(TissueType::Liver, TissueProperties {
            name: "Liver",
            density: 1060.0,
            sound_speed: 1590.0,
            alpha0: 0.9,
            y: 1.15,
            b_a: 6.7,
            specific_heat: 3540.0,
            thermal_conductivity: 0.51,
            impedance: 1.69e6,
            shear_sound_speed: 9.0,
            shear_viscosity_coeff: 0.1,
            bulk_viscosity_coeff: 0.1,
        });
        
        db.insert(TissueType::Lung, TissueProperties {
            name: "Lung",
            density: 394.0, // Varies greatly with inflation
            sound_speed: 650.0, // Varies with inflation
            alpha0: 9.5,
            y: 1.2,
            b_a: 9.0,
            specific_heat: 3886.0,
            thermal_conductivity: 0.39,
            impedance: 0.26e6,
            shear_sound_speed: 0.0, // Effectively no shear wave propagation in air
            shear_viscosity_coeff: 1.81e-5, // Air viscosity (approx)
            bulk_viscosity_coeff: 0.0, // Negligible for air in this context
        });
        
        db.insert(TissueType::Muscle, TissueProperties {
            name: "Muscle",
            density: 1050.0,
            sound_speed: 1580.0,
            alpha0: 1.1,
            y: 1.1,
            b_a: 7.2,
            specific_heat: 3421.0,
            thermal_conductivity: 0.49,
            impedance: 1.66e6,
            shear_sound_speed: 12.0,
            shear_viscosity_coeff: 0.12,
            bulk_viscosity_coeff: 0.12,
        });
        
        db.insert(TissueType::Skin, TissueProperties {
            name: "Skin",
            density: 1109.0,
            sound_speed: 1624.0,
            alpha0: 0.65,
            y: 1.35,
            b_a: 6.5,
            specific_heat: 3500.0, 
            thermal_conductivity: 0.37,
            impedance: 1.80e6,
            shear_sound_speed: 10.0,
            shear_viscosity_coeff: 0.1,
            bulk_viscosity_coeff: 0.1,
        });
        
        db.insert(TissueType::SoftTissue, TissueProperties {
            name: "Soft Tissue (Average)",
            density: 1040.0,
            sound_speed: 1540.0,
            alpha0: 0.75,
            y: 1.2,
            b_a: 6.5,
            specific_heat: 3600.0,
            thermal_conductivity: 0.5,
            impedance: 1.63e6,
            shear_sound_speed: 10.0,
            shear_viscosity_coeff: 0.1,
            bulk_viscosity_coeff: 0.1,
        });
        
        db.insert(TissueType::Tumor, TissueProperties {
            name: "Tumor (Generic)",
            density: 1070.0,
            sound_speed: 1620.0,
            alpha0: 0.85,
            y: 1.1,
            b_a: 7.2,
            specific_heat: 3770.0,
            thermal_conductivity: 0.55,
            impedance: 1.73e6,
            shear_sound_speed: 11.0,
            shear_viscosity_coeff: 0.11,
            bulk_viscosity_coeff: 0.11,
        });
        
        info!("Initialized tissue property database with {} tissues", db.len());
        db
    })
}

/// Calculate the tissue-specific absorption coefficient using relevant physical models
///
/// Includes:
/// - Frequency-dependent power law absorption
/// - Temperature dependence
/// - Optional adjustments for pressure (when in nonlinear regime)
pub fn tissue_absorption_coefficient(
    tissue_type: TissueType,
    frequency: f64,
    temperature: f64,
    pressure_amplitude: Option<f64>,
) -> f64 {
    debug!(
        "Computing tissue absorption for {:?}: freq = {:.2e} Hz, temp = {:.2} K",
        tissue_type, frequency, temperature
    );
    
    let db = tissue_database();
    let props = db.get(&tissue_type).unwrap_or_else(|| {
        // Default to soft tissue if tissue type not found
        debug!("Tissue type {:?} not found, using SoftTissue properties", tissue_type);
        db.get(&TissueType::SoftTissue).unwrap()
    });
    
    // Base absorption: power law α = α₀·fʸ, with f in MHz
    let f_mhz = frequency / 1e6;
    let mut alpha = props.alpha0 * f_mhz.powf(props.y);
    
    // Temperature dependence:
    // Most tissues show ~1-2% increase in absorption per degree above 37°C
    let temp_c = temperature - 273.15;
    let temp_factor = 1.0 + 0.015 * (temp_c - 37.0);
    alpha *= temp_factor.max(0.5); // Limit reduction to 50% at very low temperatures
    
    // High intensity effects (optional)
    // In high-intensity fields, the absorption can increase due to nonlinear effects
    if let Some(p_amp) = pressure_amplitude {
        // For very high pressures, absorption increases
        // This is a simplified model that approximates additional nonlinear losses
        let p_threshold = 1.0e6; // 1 MPa threshold for nonlinear effects
        if p_amp > p_threshold {
            let nonlinear_factor = 1.0 + 0.05 * (p_amp - p_threshold) / 1.0e6;
            alpha *= nonlinear_factor.min(2.0); // Cap at doubling the absorption
        }
    }
    
    trace!("Tissue absorption coefficient: α = {:.6e} Np/m", alpha);
    alpha
}

/// Calculate the frequency-dependent acoustic impedance for a specific tissue
/// This is important for reflection/transmission calculations at tissue interfaces
pub fn tissue_impedance(tissue_type: TissueType, frequency: f64) -> f64 {
    let db = tissue_database();
    let props = db.get(&tissue_type).unwrap_or_else(|| {
        db.get(&TissueType::SoftTissue).unwrap()
    });
    
    // Acoustic impedance is typically weakly frequency dependent
    // This model includes a small frequency dependence term
    let f_mhz = frequency / 1e6;
    let impedance = props.impedance * (1.0 + 0.002 * (f_mhz - 1.0));
    
    trace!("Tissue impedance for {:?}: Z = {:.3e} kg/m²/s at f = {:.2e} Hz", 
           tissue_type, impedance, frequency);
    
    impedance
}

/// Calculate reflection coefficient between two tissue types
/// Useful for modeling tissue boundaries in the simulation
pub fn reflection_coefficient(tissue1: TissueType, tissue2: TissueType, frequency: f64) -> f64 {
    let z1 = tissue_impedance(tissue1, frequency);
    let z2 = tissue_impedance(tissue2, frequency);
    
    let r = ((z2 - z1) / (z2 + z1)).powi(2);
    
    debug!("Reflection coefficient between {:?} and {:?}: {:.4} at {:.2e} Hz",
           tissue1, tissue2, r, frequency);
    
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tissue_properties_struct_has_new_fields() {
        // This test mainly checks if the struct can be instantiated with new fields,
        // confirming their existence.
        let props = TissueProperties {
            name: "TestTissue",
            density: 1000.0,
            sound_speed: 1500.0,
            alpha0: 0.5,
            y: 1.1,
            b_a: 5.0,
            specific_heat: 3600.0,
            thermal_conductivity: 0.5,
            impedance: 1.5e6,
            shear_sound_speed: 10.0,
            shear_viscosity_coeff: 0.1,
            bulk_viscosity_coeff: 0.2,
        };
        assert_eq!(props.shear_sound_speed, 10.0);
        assert_eq!(props.shear_viscosity_coeff, 0.1);
        assert_eq!(props.bulk_viscosity_coeff, 0.2);
    }

    #[test]
    fn test_tissue_database_entries_contain_shear_properties() {
        let db = tissue_database();
        assert!(!db.is_empty(), "Tissue database should not be empty");

        for (tissue_type, props) in db.iter() {
            // Check that the new fields are present and have some value.
            // We don't assert specific values here as they are placeholders,
            // but we check they are not uninitialized or default zero for all,
            // unless that's the intended placeholder (like for Lung shear speed).
            assert!(props.shear_sound_speed >= 0.0, "Shear sound speed should be non-negative for {:?}", tissue_type);
            assert!(props.shear_viscosity_coeff >= 0.0, "Shear viscosity coeff should be non-negative for {:?}", tissue_type);
            assert!(props.bulk_viscosity_coeff >= 0.0, "Bulk viscosity coeff should be non-negative for {:?}", tissue_type);

            // Example check for a specific tissue type if needed for more detailed validation
            if *tissue_type == TissueType::BoneCortical {
                assert_eq!(props.shear_sound_speed, 1500.0, "Cortical bone shear speed mismatch");
                assert_eq!(props.shear_viscosity_coeff, 1.5, "Cortical bone shear viscosity mismatch");
                assert_eq!(props.bulk_viscosity_coeff, 1.5, "Cortical bone bulk viscosity mismatch");
            }
            if *tissue_type == TissueType::Lung {
                 assert_eq!(props.shear_sound_speed, 0.0, "Lung shear speed should be ~0");
            }
        }
    }

    #[test]
    fn test_all_tissue_types_in_database() {
        // Ensure all enum variants of TissueType are keys in the database.
        // This is a meta-test to catch if new tissue types are added to the enum
        // but not to the database initialization.
        let db = tissue_database();
        let all_types = [
            TissueType::BloodVessel,
            TissueType::Bone,
            TissueType::BoneCortical,
            TissueType::BoneMarrow,
            TissueType::Brain,
            TissueType::Fat,
            TissueType::Kidney,
            TissueType::Liver,
            TissueType::Lung,
            TissueType::Muscle,
            TissueType::Skin,
            TissueType::SoftTissue,
            TissueType::Tumor,
        ];

        for t_type in all_types.iter() {
            assert!(db.contains_key(t_type), "Tissue database is missing entry for {:?}", t_type);
        }
        assert_eq!(db.len(), all_types.len(), "Mismatch between number of tissue types in enum and database entries. DB has {}, enum has {}.", db.len(), all_types.len());
    }
}