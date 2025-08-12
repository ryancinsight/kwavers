//! Unified field mapping system - Single Source of Truth for field indices
//!
//! This module provides a centralized, type-safe way to map between field types
//! and their indices in the global fields array. This prevents data corruption
//! from incorrect field indexing.

use crate::physics::state::field_indices;
use std::fmt;

/// Unified field type enum that maps directly to field indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnifiedFieldType {
    Pressure,
    Temperature,
    BubbleRadius,
    BubbleVelocity,
    Density,
    SoundSpeed,
    VelocityX,
    VelocityY,
    VelocityZ,
    StressXX,
    StressYY,
    StressZZ,
    StressXY,
    StressXZ,
    StressYZ,
    LightFluence,
    ChemicalConcentration,
    Light,       // For light/optical field
    Cavitation,  // For cavitation field
}

impl UnifiedFieldType {
    /// Get the array index for this field type
    /// This is the ONLY place where field indices should be defined
    pub fn index(&self) -> usize {
        match self {
            Self::Pressure => field_indices::PRESSURE_IDX,
            Self::Temperature => field_indices::TEMPERATURE_IDX,
            Self::BubbleRadius => field_indices::BUBBLE_RADIUS_IDX,
            Self::BubbleVelocity => field_indices::BUBBLE_VELOCITY_IDX,
            Self::Density => field_indices::DENSITY_IDX,
            Self::SoundSpeed => field_indices::SOUND_SPEED_IDX,
            Self::VelocityX => field_indices::VX_IDX,
            Self::VelocityY => field_indices::VY_IDX,
            Self::VelocityZ => field_indices::VZ_IDX,
            Self::StressXX => field_indices::STRESS_XX_IDX,
            Self::StressYY => field_indices::STRESS_YY_IDX,
            Self::StressZZ => field_indices::STRESS_ZZ_IDX,
            Self::StressXY => field_indices::STRESS_XY_IDX,
            Self::StressXZ => field_indices::STRESS_XZ_IDX,
            Self::StressYZ => field_indices::STRESS_YZ_IDX,
            Self::LightFluence => field_indices::LIGHT_IDX,
            Self::ChemicalConcentration => field_indices::CHEMICAL_IDX,
            Self::Light => field_indices::LIGHT_IDX,  // Map to same as LightFluence
            Self::Cavitation => field_indices::BUBBLE_RADIUS_IDX,  // Map to bubble radius field
        }
    }
    
    /// Get human-readable name for this field
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pressure => "Pressure",
            Self::Temperature => "Temperature",
            Self::BubbleRadius => "Bubble Radius",
            Self::BubbleVelocity => "Bubble Velocity",
            Self::Density => "Density",
            Self::SoundSpeed => "Sound Speed",
            Self::VelocityX => "Velocity X",
            Self::VelocityY => "Velocity Y",
            Self::VelocityZ => "Velocity Z",
            Self::StressXX => "Stress XX",
            Self::StressYY => "Stress YY",
            Self::StressZZ => "Stress ZZ",
            Self::StressXY => "Stress XY",
            Self::StressXZ => "Stress XZ",
            Self::StressYZ => "Stress YZ",
            Self::LightFluence => "Light Fluence",
            Self::ChemicalConcentration => "Chemical Concentration",
            Self::Light => "Light",
            Self::Cavitation => "Cavitation",
        }
    }
    
    /// Get unit string for this field
    pub fn unit(&self) -> &'static str {
        match self {
            Self::Pressure => "Pa",
            Self::Temperature => "K",
            Self::BubbleRadius => "m",
            Self::BubbleVelocity => "m/s",
            Self::Density => "kg/m³",
            Self::SoundSpeed => "m/s",
            Self::VelocityX | Self::VelocityY | Self::VelocityZ => "m/s",
            Self::StressXX | Self::StressYY | Self::StressZZ |
            Self::StressXY | Self::StressXZ | Self::StressYZ => "Pa",
            Self::LightFluence => "J/m²",
            Self::ChemicalConcentration => "mol/m³",
            Self::Light => "W/m²",
            Self::Cavitation => "m",
        }
    }
    
    /// Get all field types
    pub fn all() -> Vec<Self> {
        vec![
            Self::Pressure,
            Self::Temperature,
            Self::BubbleRadius,
            Self::BubbleVelocity,
            Self::Density,
            Self::SoundSpeed,
            Self::VelocityX,
            Self::VelocityY,
            Self::VelocityZ,
            Self::StressXX,
            Self::StressYY,
            Self::StressZZ,
            Self::StressXY,
            Self::StressXZ,
            Self::StressYZ,
            Self::LightFluence,
            Self::ChemicalConcentration,
            Self::Light,
            Self::Cavitation,
        ]
    }
    
    /// Create from index (for backward compatibility)
    pub fn from_index(index: usize) -> Option<Self> {
        Self::all().into_iter().find(|f| f.index() == index)
    }
}

impl fmt::Display for UnifiedFieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name(), self.unit())
    }
}

/// Type-safe field accessor to prevent index confusion
pub struct FieldAccessor<'a> {
    fields: &'a ndarray::Array4<f64>,
}

impl<'a> FieldAccessor<'a> {
    pub fn new(fields: &'a ndarray::Array4<f64>) -> Self {
        Self { fields }
    }
    
    /// Get a specific field by type
    pub fn get(&self, field_type: UnifiedFieldType) -> ndarray::ArrayView3<'a, f64> {
        self.fields.index_axis(ndarray::Axis(0), field_type.index())
    }
    
    /// Get pressure field
    pub fn pressure(&self) -> ndarray::ArrayView3<'a, f64> {
        self.get(UnifiedFieldType::Pressure)
    }
    
    /// Get temperature field
    pub fn temperature(&self) -> ndarray::ArrayView3<'a, f64> {
        self.get(UnifiedFieldType::Temperature)
    }
    
    /// Get density field
    pub fn density(&self) -> ndarray::ArrayView3<'a, f64> {
        self.get(UnifiedFieldType::Density)
    }
}

/// Type-safe mutable field accessor
pub struct FieldAccessorMut<'a> {
    fields: &'a mut ndarray::Array4<f64>,
}

impl<'a> FieldAccessorMut<'a> {
    pub fn new(fields: &'a mut ndarray::Array4<f64>) -> Self {
        Self { fields }
    }
    
    /// Get a specific field mutably by type
    pub fn get_mut(&mut self, field_type: UnifiedFieldType) -> ndarray::ArrayViewMut3<'a, f64> {
        self.fields.index_axis_mut(ndarray::Axis(0), field_type.index())
    }
    
    /// Get pressure field mutably
    pub fn pressure_mut(&mut self) -> ndarray::ArrayViewMut3<'a, f64> {
        self.fields.index_axis_mut(ndarray::Axis(0), UnifiedFieldType::Pressure.index())
    }
    
    /// Get temperature field mutably
    pub fn temperature_mut(&mut self) -> ndarray::ArrayViewMut3<'a, f64> {
        self.fields.index_axis_mut(ndarray::Axis(0), UnifiedFieldType::Temperature.index())
    }
    
    /// Get density field mutably
    pub fn density_mut(&mut self) -> ndarray::ArrayViewMut3<'a, f64> {
        self.fields.index_axis_mut(ndarray::Axis(0), UnifiedFieldType::Density.index())
    }
}

// Migration helper removed - composable module has been deprecated and removed

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_field_indices_are_unique() {
        let all_fields = UnifiedFieldType::all();
        let mut indices = std::collections::HashSet::new();
        
        for field in &all_fields {
            let idx = field.index();
            assert!(indices.insert(idx), "Duplicate index {} for field {:?}", idx, field);
        }
    }
    
    #[test]
    fn test_field_index_consistency() {
        // Ensure indices match the constants in state.rs
        assert_eq!(UnifiedFieldType::Pressure.index(), 0);
        assert_eq!(UnifiedFieldType::Temperature.index(), 1);
        assert_eq!(UnifiedFieldType::BubbleRadius.index(), 2);
        assert_eq!(UnifiedFieldType::BubbleVelocity.index(), 3);
        assert_eq!(UnifiedFieldType::Density.index(), 4);
        assert_eq!(UnifiedFieldType::SoundSpeed.index(), 5);
    }
    
    #[test]
    fn test_from_index() {
        assert_eq!(UnifiedFieldType::from_index(0), Some(UnifiedFieldType::Pressure));
        assert_eq!(UnifiedFieldType::from_index(1), Some(UnifiedFieldType::Temperature));
        assert_eq!(UnifiedFieldType::from_index(100), None);
    }
}