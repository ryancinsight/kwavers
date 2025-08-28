//! Thermal constants and properties

/// Stefan-Boltzmann constant [W/(m²·K⁴)]
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Thermal dose threshold for tissue damage (CEM43)
pub const THERMAL_DOSE_THRESHOLD: f64 = 240.0; // minutes at 43°C

/// Reference temperature for thermal dose calculation [°C]
pub const THERMAL_DOSE_REF_TEMP: f64 = 43.0;

/// Thermal dose rate constant above reference
pub const THERMAL_DOSE_RATE_ABOVE: f64 = 0.5;

/// Thermal dose rate constant below reference
pub const THERMAL_DOSE_RATE_BELOW: f64 = 0.25;

/// Blood perfusion rate [kg/(m³·s)]
pub const BLOOD_PERFUSION: f64 = 0.5;

/// Blood specific heat [J/(kg·K)]
pub const BLOOD_SPECIFIC_HEAT: f64 = 3617.0;

/// Blood temperature [°C]
pub const BLOOD_TEMPERATURE: f64 = 37.0;

/// Tissue thermal properties
pub const TISSUE_THERMAL_CONDUCTIVITY: f64 = 0.5; // [W/(m·K)]
pub const TISSUE_SPECIFIC_HEAT: f64 = 3500.0; // [J/(kg·K)]
pub const TISSUE_THERMAL_DIFFUSIVITY: f64 = 1.4e-7; // [m²/s]

/// Pennes bioheat equation parameters
pub const METABOLIC_HEAT_RATE: f64 = 400.0; // [W/m³]
pub const PERFUSION_COEFFICIENT: f64 = BLOOD_PERFUSION * BLOOD_SPECIFIC_HEAT;