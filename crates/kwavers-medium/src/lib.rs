// src/medium/mod.rs

// Module declarations
pub mod absorption;
pub mod acoustic;
pub mod adapters;
pub mod analytical_properties;
pub mod anisotropic;
pub mod bubble;
pub mod config;
pub mod core;
pub mod elastic;
pub mod error;
pub mod frequency_dependent;
pub mod heterogeneous;

pub mod homogeneous;
pub mod interface;
pub mod material_fields;
pub mod optical;
pub mod optical_map;
pub mod properties;
pub mod thermal;
pub mod traits;
pub mod viscoelastic;
pub mod viscous;
pub mod wrapper;

// pub mod simulation_config;
pub mod builder;
pub mod validation_simulation;

// Re-export types from submodules
pub use absorption::{AbsorptionTissueType, PowerLawAbsorption};
pub use anisotropic::{
    AnisotropicStiffnessTensor, AnisotropyType, ChristoffelEquation, MuscleFiberModel,
};
pub use config::{DomainMediumParameters, InterfaceTypeParameters, LayerParameters, MediumType};
pub use frequency_dependent::{FrequencyDependentProperties, TissueFrequencyModels};
pub use homogeneous::HomogeneousMedium;
pub use material_fields::MaterialFields;
pub use optical_map::{Layer, OpticalPropertyMap, OpticalPropertyMapBuilder, Region};
// pub use simulation_config::MediumConfig;
pub use builder::MediumBuilder;
pub use validation_simulation::MediumValidator;

// Re-export new modular traits
pub use acoustic::AcousticProperties;
pub use analytical_properties::AnalyticalMediumProperties;
pub use bubble::{BubbleProperties, BubbleState};
pub use core::{ArrayAccess, CoreMedium};
pub use elastic::{ElasticArrayAccess, ElasticProperties};
pub use optical::MediumOpticalProperties;
pub use thermal::{ThermalField, ThermalProperties};

// Re-export canonical property data structures (SSOT)
pub use properties::{
    AcousticMaterialProperties, AcousticPropertyData, ElasticPropertyData,
    ElectromagneticPropertyData, MaterialPropertiesBuilder, OpticalPropertyData,
    StrengthPropertyData, ThermalPropertyData,
};
pub use traits::Medium;
pub use viscous::ViscousProperties;

// Re-export utility functions and types
pub use core::{continuous_to_discrete, max_sound_speed, max_sound_speed_pointwise};
pub use interface::{find_interfaces, MediumInterfacePoint};
pub use wrapper::{
    absorption_at, absorption_at_core, density_at, density_at_core, nonlinearity_at,
    nonlinearity_at_core, sound_speed_at, sound_speed_at_core,
};

/// Custom iterators for medium property traversal
pub mod iterators;
