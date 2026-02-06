//! Unified field mapping system - Single Source of Truth for field indices
//!
//! This module provides a centralized, type-safe way to map between field types
//! and their indices in the global fields array. This prevents data corruption
//! from incorrect field indexing.

use crate::domain::field::indices as field_indices;
use std::fmt;

/// Unified field type enum that maps directly to field indices
/// Uses repr(usize) for O(1) array indexing instead of `HashMap` lookups
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum UnifiedFieldType {
    Pressure = 0,
    Temperature = 1,
    BubbleRadius = 2,
    BubbleVelocity = 3,
    Density = 4,
    SoundSpeed = 5,
    VelocityX = 6,
    VelocityY = 7,
    VelocityZ = 8,
    StressXX = 9,
    StressYY = 10,
    StressZZ = 11,
    StressXY = 12,
    StressXZ = 13,
    StressYZ = 14,
    LightFluence = 15,
    ChemicalConcentration = 16,
}

impl UnifiedFieldType {
    /// Total number of field types - used for sizing arrays
    pub const COUNT: usize = 17;

    /// Get the array index for this field type
    /// Now simply returns the enum's numeric value for O(1) access
    #[must_use]
    pub fn index(&self) -> usize {
        *self as usize
    }

    /// Legacy compatibility - maps to old field indices
    #[must_use]
    pub fn legacy_index(&self) -> usize {
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
        }
    }

    /// Get human-readable name for this field
    #[must_use]
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
        }
    }

    /// Get unit string for this field
    #[must_use]
    pub fn unit(&self) -> &'static str {
        match self {
            Self::Pressure => "Pa",
            Self::Temperature => "K",
            Self::BubbleRadius => "m",
            Self::BubbleVelocity => "m/s",
            Self::Density => "kg/m³",
            Self::SoundSpeed => "m/s",
            Self::VelocityX | Self::VelocityY | Self::VelocityZ => "m/s",
            Self::StressXX
            | Self::StressYY
            | Self::StressZZ
            | Self::StressXY
            | Self::StressXZ
            | Self::StressYZ => "Pa",
            Self::LightFluence => "J/m²",
            Self::ChemicalConcentration => "mol/m³",
        }
    }

    /// Get all field types
    #[must_use]
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
        ]
    }

    /// Create from index (efficient constant-time lookup)
    #[must_use]
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            field_indices::PRESSURE_IDX => Some(Self::Pressure),
            field_indices::TEMPERATURE_IDX => Some(Self::Temperature),
            field_indices::BUBBLE_RADIUS_IDX => Some(Self::BubbleRadius),
            field_indices::BUBBLE_VELOCITY_IDX => Some(Self::BubbleVelocity),
            field_indices::DENSITY_IDX => Some(Self::Density),
            field_indices::SOUND_SPEED_IDX => Some(Self::SoundSpeed),
            field_indices::VX_IDX => Some(Self::VelocityX),
            field_indices::VY_IDX => Some(Self::VelocityY),
            field_indices::VZ_IDX => Some(Self::VelocityZ),
            field_indices::STRESS_XX_IDX => Some(Self::StressXX),
            field_indices::STRESS_YY_IDX => Some(Self::StressYY),
            field_indices::STRESS_ZZ_IDX => Some(Self::StressZZ),
            field_indices::STRESS_XY_IDX => Some(Self::StressXY),
            field_indices::STRESS_XZ_IDX => Some(Self::StressXZ),
            field_indices::STRESS_YZ_IDX => Some(Self::StressYZ),
            field_indices::LIGHT_IDX => Some(Self::LightFluence),
            field_indices::CHEMICAL_IDX => Some(Self::ChemicalConcentration),
            _ => None,
        }
    }
}

impl fmt::Display for UnifiedFieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name(), self.unit())
    }
}
