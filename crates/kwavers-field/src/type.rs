//! Unified field mapping system - Single Source of Truth for field indices
//!
//! This module provides a centralized, type-safe way to map between field types
//! and their indices in the global fields array. This prevents data corruption
//! from incorrect field indexing.

use crate::indices as field_indices;
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
    /// Volumetric thermal deposition in watts per cubic metre.
    VolumetricHeatSource = 17,
}

impl UnifiedFieldType {
    /// Total number of field types - used for sizing arrays
    pub const COUNT: usize = 18;

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
            // This field was added to the unified layout and has no slot in
            // the historical field-index table.
            Self::VolumetricHeatSource => Self::VolumetricHeatSource.index(),
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
            Self::VolumetricHeatSource => "Volumetric Heat Source",
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
            Self::VolumetricHeatSource => "W/m³",
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
            Self::VolumetricHeatSource,
        ]
    }

    /// Create from index (efficient constant-time lookup)
    #[must_use]
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Pressure),
            1 => Some(Self::Temperature),
            2 => Some(Self::BubbleRadius),
            3 => Some(Self::BubbleVelocity),
            4 => Some(Self::Density),
            5 => Some(Self::SoundSpeed),
            6 => Some(Self::VelocityX),
            7 => Some(Self::VelocityY),
            8 => Some(Self::VelocityZ),
            9 => Some(Self::StressXX),
            10 => Some(Self::StressYY),
            11 => Some(Self::StressZZ),
            12 => Some(Self::StressXY),
            13 => Some(Self::StressXZ),
            14 => Some(Self::StressYZ),
            15 => Some(Self::LightFluence),
            16 => Some(Self::ChemicalConcentration),
            17 => Some(Self::VolumetricHeatSource),
            _ => None,
        }
    }
}

impl fmt::Display for UnifiedFieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name(), self.unit())
    }
}
