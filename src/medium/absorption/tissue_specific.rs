use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Tissue type enumeration for medical imaging applications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    SoftTissue,
    Tumor,
    // Add more tissue types as needed
}

impl TissueType {
    /// Get the shear modulus (Pa) for this tissue type
    pub fn get_shear_modulus(&self) -> f64 {
        match self {
            TissueType::BloodVessel => 0.0,      // Fluid-like
            TissueType::Bone => 5.0e6,           // 5 MPa
            TissueType::BoneCortical => 6.0e6,   // 6 MPa
            TissueType::BoneMarrow => 0.5e3,     // 0.5 kPa
            TissueType::Brain => 2.5e3,          // 2.5 kPa
            TissueType::Fat => 2.0e3,            // 2 kPa
            TissueType::Kidney => 2.5e3,         // 2.5 kPa
            TissueType::Liver => 2.0e3,          // 2 kPa
            TissueType::Lung => 1.0e3,           // 1 kPa
            TissueType::Muscle => 12.0e3,        // 12 kPa
            TissueType::Skin => 15.0e3,          // 15 kPa
            TissueType::SoftTissue => 3.0e3,     // 3 kPa
            TissueType::Tumor => 20.0e3,         // 20 kPa
        }
    }
}

/// Tissue properties for acoustic simulations
#[derive(Debug, Clone, Copy)]
pub struct TissueProperties {
    /// Density (kg/m³)
    pub density: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Absorption coefficient at reference frequency (dB/MHz/cm)
    pub absorption_coefficient: f64,
    /// Power law exponent for frequency dependence
    pub power_law_exponent: f64,
    /// Nonlinearity parameter B/A
    pub b_a: f64,
    /// Specific heat (J/kg/K)
    pub specific_heat: f64,
    /// Thermal conductivity (W/m/K)
    pub thermal_conductivity: f64,
    /// Acoustic impedance (kg/m²/s)
    pub impedance: f64,
    /// Shear modulus (Pa)
    pub shear_modulus: f64,
}

/// Get tissue database singleton
pub fn tissue_database() -> &'static HashMap<TissueType, TissueProperties> {
    static TISSUE_DB: OnceLock<HashMap<TissueType, TissueProperties>> = OnceLock::new();
    
    TISSUE_DB.get_or_init(|| {
        let mut db = HashMap::new();
        
        // Blood vessel properties
        db.insert(TissueType::BloodVessel, TissueProperties {
            density: 1060.0,
            sound_speed: 1584.0,
            absorption_coefficient: 0.15,
            power_law_exponent: 1.2,
            b_a: 6.1,
            specific_heat: 3770.0,
            thermal_conductivity: 0.51,
            impedance: 1.70e6,
            shear_modulus: 0.0,  // Fluid-like
        });
        
        db.insert(TissueType::Bone, TissueProperties {
            density: 1908.0,
            sound_speed: 4080.0,
            absorption_coefficient: 9.94,
            power_law_exponent: 2.2,
            b_a: 7.4,
            specific_heat: 1313.0,
            thermal_conductivity: 0.32,
            impedance: 6.63e6,
            shear_modulus: 5.0e6,  // 5 MPa
        });
        
        db.insert(TissueType::BoneCortical, TissueProperties {
            density: 2175.0,
            sound_speed: 4000.0,
            absorption_coefficient: 6.9,
            power_law_exponent: 2.2,
            b_a: 7.4,
            specific_heat: 1300.0,
            thermal_conductivity: 0.38,
            impedance: 8.06e6,
            shear_modulus: 6.0e6,  // 6 MPa
        });
        
        db.insert(TissueType::BoneMarrow, TissueProperties {
            density: 1100.0,
            sound_speed: 1500.0,
            absorption_coefficient: 0.6,
            power_law_exponent: 1.1,
            b_a: 6.1,
            specific_heat: 2700.0,
            thermal_conductivity: 0.22,
            impedance: 1.65e6,
            shear_modulus: 0.5e3,  // 0.5 kPa (soft marrow)
        });
        
        db.insert(TissueType::Brain, TissueProperties {
            density: 1046.0,
            sound_speed: 1546.0,
            absorption_coefficient: 0.6,
            power_law_exponent: 1.1,
            b_a: 6.6,
            specific_heat: 3630.0,
            thermal_conductivity: 0.51,
            impedance: 1.61e6,
            shear_modulus: 2.5e3,  // 2.5 kPa
        });
        
        db.insert(TissueType::Fat, TissueProperties {
            density: 911.0,
            sound_speed: 1450.0,
            absorption_coefficient: 0.48,
            power_law_exponent: 1.1,
            b_a: 10.0,
            specific_heat: 2348.0,
            thermal_conductivity: 0.21,
            impedance: 1.38e6,
            shear_modulus: 2.0e3,  // 2 kPa
        });
        
        db.insert(TissueType::Kidney, TissueProperties {
            density: 1050.0,
            sound_speed: 1561.0,
            absorption_coefficient: 1.0,
            power_law_exponent: 1.1,
            b_a: 6.8,
            specific_heat: 3763.0,
            thermal_conductivity: 0.53,
            impedance: 1.64e6,
            shear_modulus: 2.5e3,  // 2.5 kPa
        });
        
        db.insert(TissueType::Liver, TissueProperties {
            density: 1079.0,
            sound_speed: 1570.0,
            absorption_coefficient: 0.5,
            power_law_exponent: 1.1,
            b_a: 6.8,
            specific_heat: 3540.0,
            thermal_conductivity: 0.52,
            impedance: 1.69e6,
            shear_modulus: 2.0e3,  // 2 kPa
        });
        
        db.insert(TissueType::Lung, TissueProperties {
            density: 394.0,
            sound_speed: 650.0,
            absorption_coefficient: 40.0,
            power_law_exponent: 1.0,
            b_a: 9.0,
            specific_heat: 3886.0,
            thermal_conductivity: 0.39,
            impedance: 0.52e6,
            shear_modulus: 1.0e3,  // 1 kPa
        });
        
        db.insert(TissueType::Muscle, TissueProperties {
            density: 1090.0,
            sound_speed: 1580.0,
            absorption_coefficient: 0.54,
            power_law_exponent: 1.1,
            b_a: 7.4,
            specific_heat: 3421.0,
            thermal_conductivity: 0.49,
            impedance: 1.72e6,
            shear_modulus: 12.0e3,  // 12 kPa
        });
        
        db.insert(TissueType::Skin, TissueProperties {
            density: 1109.0,
            sound_speed: 1498.0,
            absorption_coefficient: 0.2,
            power_law_exponent: 1.1,
            b_a: 6.1,
            specific_heat: 3391.0,
            thermal_conductivity: 0.37,
            impedance: 1.66e6,
            shear_modulus: 15.0e3,  // 15 kPa
        });
        
        db.insert(TissueType::SoftTissue, TissueProperties {
            density: 1050.0,
            sound_speed: 1540.0,
            absorption_coefficient: 0.54,
            power_law_exponent: 1.1,
            b_a: 6.5,
            specific_heat: 3600.0,
            thermal_conductivity: 0.5,
            impedance: 1.63e6,
            shear_modulus: 3.0e3,  // 3 kPa
        });
        
        db.insert(TissueType::Tumor, TissueProperties {
            density: 1080.0,
            sound_speed: 1550.0,
            absorption_coefficient: 0.7,
            power_law_exponent: 1.1,
            b_a: 7.0,
            specific_heat: 3500.0,
            thermal_conductivity: 0.48,
            impedance: 1.67e6,
            shear_modulus: 20.0e3,  // 20 kPa
        });
        
        db
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_tissue_database() {
        let db = tissue_database();
        
        // Test that all tissue types are present
        assert!(db.contains_key(&TissueType::Brain));
        assert!(db.contains_key(&TissueType::Muscle));
        assert!(db.contains_key(&TissueType::Fat));
        
        // Test specific properties
        let brain = &db[&TissueType::Brain];
        assert_relative_eq!(brain.density, 1046.0, max_relative = 1e-10);
        assert_relative_eq!(brain.sound_speed, 1546.0, max_relative = 1e-10);
    }
    
    #[test]
    fn test_shear_modulus() {
        assert_relative_eq!(TissueType::Brain.get_shear_modulus(), 2.5e3);
        assert_relative_eq!(TissueType::Muscle.get_shear_modulus(), 12.0e3);
        assert_relative_eq!(TissueType::BloodVessel.get_shear_modulus(), 0.0);
    }
}