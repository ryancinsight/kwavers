//! Validation Configuration System
//!
//! This module provides a centralized configuration system for validation parameters,
//! replacing hardcoded magic numbers throughout the codebase to improve maintainability
//! and enable runtime configuration.

use serde::{Deserialize, Serialize};

/// Field validation configuration - single source of truth for all field limits
/// 
/// This configuration type provides a simpler alternative to ValidationConfig
/// for cases where only basic min/max limits are needed.
/// Follows SSOT and DRY principles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldValidationConfig {
    /// Pressure field limits
    pub pressure: FieldLimits,
    /// Temperature field limits  
    pub temperature: FieldLimits,
    /// Light intensity limits
    pub light: FieldLimits,
    /// Velocity field limits
    pub velocity: FieldLimits,
    /// Stress field limits
    pub stress: FieldLimits,
}

/// Field validation limits with optional warning thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldLimits {
    pub min: f64,
    pub max: f64,
    pub warn_min: Option<f64>,
    pub warn_max: Option<f64>,
}

impl Default for FieldValidationConfig {
    fn default() -> Self {
        Self {
            pressure: FieldLimits {
                min: -1e9,  // -1 GPa
                max: 1e9,   // 1 GPa
                warn_min: Some(-1e8),
                warn_max: Some(1e8),
            },
            temperature: FieldLimits {
                min: 0.0,     // Absolute zero
                max: 10000.0, // 10,000 K
                warn_min: Some(200.0),
                warn_max: Some(5000.0),
            },
            light: FieldLimits {
                min: 0.0,
                max: 1e12,  // 1 TW/m²
                warn_min: None,
                warn_max: Some(1e10),
            },
            velocity: FieldLimits {
                min: -10000.0,  // -10 km/s
                max: 10000.0,   // 10 km/s
                warn_min: Some(-5000.0),
                warn_max: Some(5000.0),
            },
            stress: FieldLimits {
                min: -1e10,  // -10 GPa
                max: 1e10,   // 10 GPa
                warn_min: Some(-1e9),
                warn_max: Some(1e9),
            },
        }
    }
}

impl FieldValidationConfig {
    /// Load from file
    pub fn from_file(path: &str) -> Result<Self, std::io::Error> {
        let contents = std::fs::read_to_string(path)?;
        toml::from_str(&contents).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }
    
    /// Save to file  
    pub fn to_file(&self, path: &str) -> Result<(), std::io::Error> {
        let contents = toml::to_string_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        std::fs::write(path, contents)
    }
}

/// Comprehensive validation configuration for all physics simulations
/// 
/// This structure centralizes all validation limits and thresholds, providing
/// a single source of truth for numerical bounds and stability criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Pressure field validation
    pub pressure: PressureValidation,
    /// Temperature field validation
    pub temperature: ThermalValidation,
    /// Light field validation
    pub light: LightValidation,
    /// Velocity field validation
    pub velocity: VelocityValidation,
    /// Density field validation
    pub density: DensityValidation,
    /// Numerical stability settings
    pub stability: StabilityValidation,
    /// Time step validation
    pub time_step: TimeStepValidation,
}

/// Pressure field validation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureValidation {
    /// Minimum allowable pressure [Pa]
    pub min_pressure: f64,
    /// Maximum allowable pressure [Pa]
    pub max_pressure: f64,
    /// Pressure gradient limit for stability [Pa/m]
    pub max_gradient: f64,
    /// Pressure change rate limit [Pa/s]
    pub max_rate: f64,
}

/// Temperature field validation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalValidation {
    /// Minimum allowable temperature [K]
    pub min_temperature: f64,
    /// Maximum allowable temperature [K]
    pub max_temperature: f64,
    /// Temperature gradient limit [K/m]
    pub max_gradient: f64,
    /// Temperature change rate limit [K/s]
    pub max_rate: f64,
}

/// Light field validation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightValidation {
    /// Minimum light intensity [W/m²]
    pub min_intensity: f64,
    /// Maximum light intensity [W/m²]
    pub max_intensity: f64,
    /// Light gradient limit [W/m³]
    pub max_gradient: f64,
    /// Absorption coefficient limits [1/m]
    pub max_absorption: f64,
}

/// Velocity field validation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityValidation {
    /// Maximum velocity magnitude [m/s]
    pub max_velocity: f64,
    /// Maximum acceleration [m/s²]
    pub max_acceleration: f64,
    /// Velocity divergence limit [1/s]
    pub max_divergence: f64,
}

/// Density field validation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityValidation {
    /// Minimum density [kg/m³]
    pub min_density: f64,
    /// Maximum density [kg/m³]
    pub max_density: f64,
    /// Density gradient limit [kg/m⁴]
    pub max_gradient: f64,
}

/// Numerical stability validation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityValidation {
    /// CFL safety factor
    pub cfl_safety_factor: f64,
    /// Maximum CFL number allowed
    pub max_cfl: f64,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Maximum iterations for implicit solvers
    pub max_iterations: usize,
    /// NaN/Inf detection threshold
    pub finite_threshold: f64,
}

/// Time step validation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeStepValidation {
    /// Minimum time step [s]
    pub min_dt: f64,
    /// Maximum time step [s]
    pub max_dt: f64,
    /// Time step change factor limit
    pub max_dt_change_factor: f64,
    /// Maximum simulation time [s]
    pub max_simulation_time: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            pressure: PressureValidation::default(),
            temperature: ThermalValidation::default(),
            light: LightValidation::default(),
            velocity: VelocityValidation::default(),
            density: DensityValidation::default(),
            stability: StabilityValidation::default(),
            time_step: TimeStepValidation::default(),
        }
    }
}

impl Default for PressureValidation {
    fn default() -> Self {
        Self {
            min_pressure: -1e9,    // -1 GPa
            max_pressure: 1e9,     // 1 GPa  
            max_gradient: 1e15,    // 1 PPa/m
            max_rate: 1e12,        // 1 TPa/s
        }
    }
}

impl Default for ThermalValidation {
    fn default() -> Self {
        Self {
            min_temperature: 273.15,  // 0°C
            max_temperature: 1000.0,  // 1000K
            max_gradient: 1e6,        // 1 MK/m
            max_rate: 1e6,            // 1 MK/s
        }
    }
}

impl Default for LightValidation {
    fn default() -> Self {
        Self {
            min_intensity: 0.0,
            max_intensity: 1e10,      // 10 GW/m²
            max_gradient: 1e15,       // 1 PW/m³
            max_absorption: 1e6,      // 1 Mm⁻¹
        }
    }
}

impl Default for VelocityValidation {
    fn default() -> Self {
        Self {
            max_velocity: 1e4,        // 10 km/s
            max_acceleration: 1e9,    // 1 Gm/s²
            max_divergence: 1e6,      // 1 MHz
        }
    }
}

impl Default for DensityValidation {
    fn default() -> Self {
        Self {
            min_density: 1e-3,        // 1 mg/m³
            max_density: 1e5,         // 100 g/cm³
            max_gradient: 1e8,        // 100 kg/m⁴
        }
    }
}

impl Default for StabilityValidation {
    fn default() -> Self {
        Self {
            cfl_safety_factor: 0.95,
            max_cfl: 1.0,
            convergence_tolerance: 1e-12,
            max_iterations: 1000,
            finite_threshold: 1e-100,
        }
    }
}

impl Default for TimeStepValidation {
    fn default() -> Self {
        Self {
            min_dt: 1e-12,            // 1 ps
            max_dt: 1e-3,             // 1 ms
            max_dt_change_factor: 2.0,
            max_simulation_time: 1e3, // 1000 s
        }
    }
}

impl ValidationConfig {
    /// Create a new validation configuration from file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save validation configuration to file
    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Create validation configuration for ultrasound applications
    pub fn ultrasound() -> Self {
        let mut config = Self::default();
        config.pressure.max_pressure = 1e8;  // 100 MPa
        config.pressure.min_pressure = -1e8; // -100 MPa
        config.temperature.max_temperature = 350.0; // 77°C
        config
    }
    
    /// Create validation configuration for high-intensity applications
    pub fn high_intensity() -> Self {
        let mut config = Self::default();
        config.pressure.max_pressure = 1e10;  // 10 GPa
        config.pressure.min_pressure = -1e10; // -10 GPa
        config.temperature.max_temperature = 2000.0; // 2000K
        config.light.max_intensity = 1e12;    // 1 TW/m²
        config
    }
    
    /// Create validation configuration for micro-scale simulations
    pub fn microscale() -> Self {
        let mut config = Self::default();
        config.time_step.min_dt = 1e-15;     // 1 fs
        config.time_step.max_dt = 1e-9;      // 1 ns
        config.velocity.max_velocity = 1e6;  // 1000 km/s
        config
    }
    
    /// Validate a configuration for self-consistency
    pub fn validate(&self) -> Result<(), String> {
        // Pressure validation
        if self.pressure.min_pressure >= self.pressure.max_pressure {
            return Err("min_pressure must be less than max_pressure".to_string());
        }
        
        // Temperature validation
        if self.temperature.min_temperature >= self.temperature.max_temperature {
            return Err("min_temperature must be less than max_temperature".to_string());
        }
        
        // Time step validation
        if self.time_step.min_dt >= self.time_step.max_dt {
            return Err("min_dt must be less than max_dt".to_string());
        }
        
        // Stability validation
        if self.stability.cfl_safety_factor <= 0.0 || self.stability.cfl_safety_factor > 1.0 {
            return Err("cfl_safety_factor must be between 0 and 1".to_string());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_validation_config() {
        let config = ValidationConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_ultrasound_config() {
        let config = ValidationConfig::ultrasound();
        assert!(config.validate().is_ok());
        assert_eq!(config.pressure.max_pressure, 1e8);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = ValidationConfig::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: ValidationConfig = toml::from_str(&serialized).unwrap();
        assert!(deserialized.validate().is_ok());
    }
    
    #[test]
    fn test_invalid_pressure_range() {
        let mut config = ValidationConfig::default();
        config.pressure.min_pressure = 100.0;
        config.pressure.max_pressure = 50.0;
        assert!(config.validate().is_err());
    }
}