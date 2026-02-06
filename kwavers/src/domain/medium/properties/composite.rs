//! Composite material properties for multi-physics simulations
//!
//! # Domain Rule: Single Source of Truth
//!
//! This is the **canonical composition point** for all material properties in kwavers.
//! Physics modules should extract only the properties they need from this composite.
//!
//! # Design Pattern: Builder + Optional Properties
//!
//! - **Acoustic**: Always present (base requirement for wave propagation)
//! - **Elastic**: Optional (only for solid media)
//! - **Electromagnetic**: Optional (only for EM simulations)
//! - **Optical**: Optional (only for light propagation)
//! - **Strength**: Optional (only for damage/fracture mechanics)
//! - **Thermal**: Optional (only for thermal effects)

use super::{
    AcousticPropertyData, ElasticPropertyData, ElectromagneticPropertyData, OpticalPropertyData,
    StrengthPropertyData, ThermalPropertyData,
};
use std::fmt;

/// Composite material properties for multi-physics simulations
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialProperties {
    /// Acoustic properties (always present)
    pub acoustic: AcousticPropertyData,

    /// Elastic properties (optional, for solids)
    pub elastic: Option<ElasticPropertyData>,

    /// Electromagnetic properties (optional, for EM waves)
    pub electromagnetic: Option<ElectromagneticPropertyData>,

    /// Optical properties (optional, for light propagation)
    pub optical: Option<OpticalPropertyData>,

    /// Strength properties (optional, for damage/fracture)
    pub strength: Option<StrengthPropertyData>,

    /// Thermal properties (optional, for thermal effects)
    pub thermal: Option<ThermalPropertyData>,
}

impl MaterialProperties {
    /// Create acoustic-only material (e.g., water, air)
    pub fn acoustic_only(acoustic: AcousticPropertyData) -> Self {
        Self {
            acoustic,
            elastic: None,
            electromagnetic: None,
            optical: None,
            strength: None,
            thermal: None,
        }
    }

    /// Create elastic material with acoustic coupling
    pub fn elastic(acoustic: AcousticPropertyData, elastic: ElasticPropertyData) -> Self {
        Self {
            acoustic,
            elastic: Some(elastic),
            electromagnetic: None,
            optical: None,
            strength: None,
            thermal: None,
        }
    }

    /// Create builder for multi-physics composition
    pub fn builder() -> MaterialPropertiesBuilder {
        MaterialPropertiesBuilder::default()
    }

    /// Water at 20°C (acoustic + thermal)
    pub fn water() -> Self {
        Self::builder()
            .acoustic(AcousticPropertyData::water())
            .thermal(ThermalPropertyData::water())
            .build()
    }

    /// Soft tissue (acoustic + thermal + EM)
    pub fn soft_tissue() -> Self {
        Self::builder()
            .acoustic(AcousticPropertyData::soft_tissue())
            .thermal(ThermalPropertyData::soft_tissue())
            .electromagnetic(ElectromagneticPropertyData::tissue())
            .build()
    }

    /// Cortical bone (acoustic + elastic + thermal + strength)
    pub fn bone() -> Self {
        Self::builder()
            .acoustic(AcousticPropertyData {
                density: 1850.0,
                sound_speed: 3500.0,
                absorption_coefficient: 0.8,
                absorption_power: 1.2,
                nonlinearity: 5.5,
            })
            .elastic(ElasticPropertyData::bone())
            .thermal(ThermalPropertyData::bone())
            .strength(StrengthPropertyData::bone())
            .build()
    }

    /// Steel (acoustic + elastic + strength)
    pub fn steel() -> Self {
        Self::builder()
            .acoustic(AcousticPropertyData {
                density: 7850.0,
                sound_speed: 5960.0,
                absorption_coefficient: 0.001,
                absorption_power: 1.0,
                nonlinearity: 4.0,
            })
            .elastic(ElasticPropertyData::steel())
            .strength(StrengthPropertyData::steel())
            .build()
    }
}

impl fmt::Display for MaterialProperties {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MaterialProperties {{")?;
        writeln!(f, "  {}", self.acoustic)?;
        if let Some(ref elastic) = self.elastic {
            writeln!(f, "  {}", elastic)?;
        }
        if let Some(ref em) = self.electromagnetic {
            writeln!(f, "  {}", em)?;
        }
        if let Some(ref optical) = self.optical {
            writeln!(f, "  {}", optical)?;
        }
        if let Some(ref strength) = self.strength {
            writeln!(f, "  {}", strength)?;
        }
        if let Some(ref thermal) = self.thermal {
            writeln!(f, "  {}", thermal)?;
        }
        write!(f, "}}")
    }
}

/// Builder for constructing `MaterialProperties` with optional components
#[derive(Debug, Default)]
pub struct MaterialPropertiesBuilder {
    acoustic: Option<AcousticPropertyData>,
    elastic: Option<ElasticPropertyData>,
    electromagnetic: Option<ElectromagneticPropertyData>,
    optical: Option<OpticalPropertyData>,
    strength: Option<StrengthPropertyData>,
    thermal: Option<ThermalPropertyData>,
}

impl MaterialPropertiesBuilder {
    /// Set acoustic properties (required)
    pub fn acoustic(mut self, acoustic: AcousticPropertyData) -> Self {
        self.acoustic = Some(acoustic);
        self
    }

    /// Set elastic properties (optional)
    pub fn elastic(mut self, elastic: ElasticPropertyData) -> Self {
        self.elastic = Some(elastic);
        self
    }

    /// Set electromagnetic properties (optional)
    pub fn electromagnetic(mut self, em: ElectromagneticPropertyData) -> Self {
        self.electromagnetic = Some(em);
        self
    }

    /// Set optical properties (optional)
    pub fn optical(mut self, optical: OpticalPropertyData) -> Self {
        self.optical = Some(optical);
        self
    }

    /// Set strength properties (optional)
    pub fn strength(mut self, strength: StrengthPropertyData) -> Self {
        self.strength = Some(strength);
        self
    }

    /// Set thermal properties (optional)
    pub fn thermal(mut self, thermal: ThermalPropertyData) -> Self {
        self.thermal = Some(thermal);
        self
    }

    /// Build the composite material properties
    ///
    /// # Panics
    ///
    /// Panics if acoustic properties are not set
    pub fn build(self) -> MaterialProperties {
        MaterialProperties {
            acoustic: self.acoustic.expect("Acoustic properties are required"),
            elastic: self.elastic,
            electromagnetic: self.electromagnetic,
            optical: self.optical,
            strength: self.strength,
            thermal: self.thermal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_acoustic_only() {
        let water = MaterialProperties::acoustic_only(AcousticPropertyData::water());
        assert!(water.elastic.is_none());
        assert!(water.thermal.is_none());
    }

    #[test]
    fn test_material_builder() {
        let props = MaterialProperties::builder()
            .acoustic(AcousticPropertyData::water())
            .thermal(ThermalPropertyData::water())
            .build();

        assert!(props.thermal.is_some());
        assert!(props.elastic.is_none());
    }

    #[test]
    fn test_material_presets() {
        let water = MaterialProperties::water();
        let tissue = MaterialProperties::soft_tissue();
        let bone = MaterialProperties::bone();

        assert!(water.thermal.is_some());
        assert!(tissue.electromagnetic.is_some());
        assert!(bone.elastic.is_some());
        assert!(bone.strength.is_some());
    }

    #[test]
    #[should_panic(expected = "Acoustic properties are required")]
    fn test_material_builder_missing_acoustic() {
        MaterialPropertiesBuilder::default().build();
    }

    #[test]
    fn test_property_physical_bounds() {
        let props = MaterialProperties::builder()
            .acoustic(AcousticPropertyData {
                density: 1000.0,
                sound_speed: 1500.0,
                absorption_coefficient: 0.5,
                absorption_power: 1.1,
                nonlinearity: 5.0,
            })
            .build();

        assert_eq!(props.acoustic.density, 1000.0);
        assert_eq!(props.acoustic.sound_speed, 1500.0);
    }
}
