//! Unified configuration system implementing SSOT principles
//!
//! This module consolidates all configuration into a single hierarchical structure,
//! eliminating the 80+ redundant Config structs that violated SSOT.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Master configuration structure - Single Source of Truth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Configuration {
    /// Simulation parameters
    pub simulation: SimulationParameters,
    /// Grid discretization
    pub grid: GridParameters,
    /// Medium properties
    pub medium: MediumParameters,
    /// Source configuration
    pub source: SourceParameters,
    /// Boundary conditions
    pub boundary: BoundaryParameters,
    /// Solver settings
    pub solver: SolverParameters,
    /// Output control
    pub output: OutputParameters,
    /// Performance tuning
    pub performance: PerformanceParameters,
    /// Validation settings
    pub validation: ValidationParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParameters {
    /// Simulation duration in seconds
    pub duration: f64,
    /// Time step (auto-calculated if None)
    pub dt: Option<f64>,
    /// CFL number for stability
    pub cfl: f64,
    /// Reference frequency in Hz
    pub frequency: f64,
    /// Enable nonlinear effects
    pub nonlinear: bool,
    /// Temperature in Kelvin
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridParameters {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Spatial steps (auto-calculated if None)
    pub dx: Option<f64>,
    pub dy: Option<f64>,
    pub dz: Option<f64>,
    /// Points per wavelength (for auto-sizing)
    pub ppw: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediumParameters {
    /// Density in kg/m³
    pub density: f64,
    /// Sound speed in m/s
    pub sound_speed: f64,
    /// Absorption coefficient in dB/cm/MHz^y
    pub absorption: f64,
    /// Absorption power law exponent
    pub absorption_power: f64,
    /// Nonlinearity parameter B/A
    pub nonlinearity: f64,
    /// Heterogeneous properties (if any)
    pub heterogeneous: Option<HeterogeneousParameters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneousParameters {
    /// Path to density map
    pub density_map: Option<PathBuf>,
    /// Path to sound speed map
    pub sound_speed_map: Option<PathBuf>,
    /// Inline density values
    pub density_values: Option<Vec<f64>>,
    /// Inline sound speed values
    pub sound_speed_values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceParameters {
    /// Source type
    pub source_type: SourceType,
    /// Source frequency in Hz
    pub frequency: f64,
    /// Source amplitude in Pa
    pub amplitude: f64,
    /// Source position (grid indices)
    pub position: Option<(usize, usize, usize)>,
    /// Multiple source positions
    pub positions: Option<Vec<(usize, usize, usize)>>,
    /// Phase delays for array sources
    pub phase_delays: Option<Vec<f64>>,
    /// Apodization weights
    pub apodization: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceType {
    Point,
    Line,
    Plane,
    Focused,
    Array,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryParameters {
    /// Boundary type
    pub boundary_type: BoundaryType,
    /// PML/CPML thickness in grid points
    pub thickness: usize,
    /// Maximum theoretical reflection coefficient
    pub reflection_coefficient: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryType {
    Periodic,
    PML,
    CPML,
    Rigid,
    Free,
    Mixed(HashMap<String, BoundaryType>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverParameters {
    /// Solver method
    pub method: SolverMethod,
    /// Order of accuracy
    pub order: usize,
    /// Adaptive time stepping
    pub adaptive: bool,
    /// Tolerance for adaptive methods
    pub tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolverMethod {
    FDTD,
    PSTD,
    SpectralDG,
    Hybrid,
    KSpace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputParameters {
    /// Output directory
    pub directory: PathBuf,
    /// Save interval (time steps)
    pub save_interval: usize,
    /// Fields to save
    pub fields: Vec<FieldType>,
    /// Output format
    pub format: OutputFormat,
    /// Compression
    pub compress: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    Pressure,
    Velocity,
    Density,
    Temperature,
    Intensity,
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    HDF5,
    NPZ,
    Binary,
    Text,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceParameters {
    /// Number of threads (0 = auto)
    pub threads: usize,
    /// Enable GPU acceleration
    pub gpu: bool,
    /// GPU device index
    pub gpu_device: usize,
    /// Cache size in MB
    pub cache_size: usize,
    /// Enable SIMD
    pub simd: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationParameters {
    /// Enable validation
    pub enabled: bool,
    /// Strict mode (fail on warnings)
    pub strict: bool,
    /// Check energy conservation
    pub energy_conservation: bool,
    /// Check CFL condition
    pub cfl_check: bool,
    /// Validate against analytical solutions
    pub analytical_validation: bool,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            simulation: SimulationParameters {
                duration: 1e-3,
                dt: None,
                cfl: 0.3,
                frequency: 1e6,
                nonlinear: false,
                temperature: 293.15,
            },
            grid: GridParameters {
                nx: 128,
                ny: 128,
                nz: 128,
                dx: None,
                dy: None,
                dz: None,
                ppw: 4.0,
            },
            medium: MediumParameters {
                density: 1000.0,
                sound_speed: 1500.0,
                absorption: 0.75,
                absorption_power: 1.05,
                nonlinearity: 3.5,
                heterogeneous: None,
            },
            source: SourceParameters {
                source_type: SourceType::Point,
                frequency: 1e6,
                amplitude: 1e6,
                position: Some((64, 64, 64)),
                positions: None,
                phase_delays: None,
                apodization: None,
            },
            boundary: BoundaryParameters {
                boundary_type: BoundaryType::CPML,
                thickness: 10,
                reflection_coefficient: 1e-6,
            },
            solver: SolverParameters {
                method: SolverMethod::FDTD,
                order: 2,
                adaptive: false,
                tolerance: 1e-6,
            },
            output: OutputParameters {
                directory: PathBuf::from("output"),
                save_interval: 100,
                fields: vec![FieldType::Pressure],
                format: OutputFormat::HDF5,
                compress: true,
            },
            performance: PerformanceParameters {
                threads: 0,
                gpu: false,
                gpu_device: 0,
                cache_size: 1024,
                simd: true,
            },
            validation: ValidationParameters {
                enabled: true,
                strict: false,
                energy_conservation: true,
                cfl_check: true,
                analytical_validation: false,
            },
        }
    }
}

impl Configuration {
    /// Load configuration from TOML file
    pub fn from_file(path: &PathBuf) -> crate::error::KwaversResult<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self =
            toml::from_str(&contents).map_err(|e| crate::error::ConfigError::ParseError {
                line: 0,
                message: e.to_string(),
            })?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn to_file(&self, path: &PathBuf) -> crate::error::KwaversResult<()> {
        let contents =
            toml::to_string_pretty(self).map_err(|e| crate::error::ConfigError::ParseError {
                line: 0,
                message: e.to_string(),
            })?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Validate configuration for consistency
    pub fn validate(&self) -> crate::error::KwaversResult<()> {
        // Validate grid parameters
        if self.grid.nx == 0 || self.grid.ny == 0 || self.grid.nz == 0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "grid dimensions".to_string(),
                value: format!("{}x{}x{}", self.grid.nx, self.grid.ny, self.grid.nz),
                constraint: "must be positive".to_string(),
            }
            .into());
        }

        // Validate CFL condition
        if self.simulation.cfl <= 0.0 || self.simulation.cfl > 1.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "CFL number".to_string(),
                value: self.simulation.cfl.to_string(),
                constraint: "must be in (0, 1]".to_string(),
            }
            .into());
        }

        // Validate medium parameters
        if self.medium.density <= 0.0 || self.medium.sound_speed <= 0.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "medium properties".to_string(),
                value: format!(
                    "density={}, sound_speed={}",
                    self.medium.density, self.medium.sound_speed
                ),
                constraint: "must be positive".to_string(),
            }
            .into());
        }

        // Validate boundary thickness
        if self.boundary.thickness == 0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "boundary thickness".to_string(),
                value: self.boundary.thickness.to_string(),
                constraint: "must be positive".to_string(),
            }
            .into());
        }

        Ok(())
    }

    /// Calculate derived parameters
    pub fn calculate_derived(&mut self) -> crate::error::KwaversResult<()> {
        // Calculate spatial steps if not provided
        if self.grid.dx.is_none() {
            let wavelength = self.medium.sound_speed / self.simulation.frequency;
            let dx = wavelength / self.grid.ppw;
            self.grid.dx = Some(dx);
            self.grid.dy = Some(dx);
            self.grid.dz = Some(dx);
        }

        // Calculate time step from CFL condition
        if self.simulation.dt.is_none() {
            let dx = self.grid.dx.unwrap();
            let dt = self.simulation.cfl * dx / self.medium.sound_speed;
            self.simulation.dt = Some(dt);
        }

        Ok(())
    }

    /// Get a specific parameter by path (e.g., "simulation.frequency")
    pub fn get(&self, path: &str) -> Option<String> {
        // This would use serde_json to navigate the structure
        // Implementation left as exercise for actual deployment
        None
    }

    /// Set a specific parameter by path
    pub fn set(&mut self, path: &str, value: &str) -> crate::error::KwaversResult<()> {
        // This would use serde_json to navigate and update
        // Implementation left as exercise for actual deployment
        Ok(())
    }
}

/// Builder pattern for Configuration
pub struct ConfigurationBuilder {
    config: Configuration,
}

impl ConfigurationBuilder {
    pub fn new() -> Self {
        Self {
            config: Configuration::default(),
        }
    }

    pub fn simulation(mut self, params: SimulationParameters) -> Self {
        self.config.simulation = params;
        self
    }

    pub fn grid(mut self, params: GridParameters) -> Self {
        self.config.grid = params;
        self
    }

    pub fn medium(mut self, params: MediumParameters) -> Self {
        self.config.medium = params;
        self
    }

    pub fn source(mut self, params: SourceParameters) -> Self {
        self.config.source = params;
        self
    }

    pub fn boundary(mut self, params: BoundaryParameters) -> Self {
        self.config.boundary = params;
        self
    }

    pub fn solver(mut self, params: SolverParameters) -> Self {
        self.config.solver = params;
        self
    }

    pub fn output(mut self, params: OutputParameters) -> Self {
        self.config.output = params;
        self
    }

    pub fn performance(mut self, params: PerformanceParameters) -> Self {
        self.config.performance = params;
        self
    }

    pub fn validation(mut self, params: ValidationParameters) -> Self {
        self.config.validation = params;
        self
    }

    pub fn build(mut self) -> crate::error::KwaversResult<Configuration> {
        self.config.calculate_derived()?;
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configuration() {
        let config = Configuration::default();
        assert_eq!(config.simulation.frequency, 1e6);
        assert_eq!(config.medium.sound_speed, 1500.0);
    }

    #[test]
    fn test_configuration_builder() {
        let config = ConfigurationBuilder::new()
            .simulation(SimulationParameters {
                duration: 2e-3,
                dt: None,
                cfl: 0.5,
                frequency: 2e6,
                nonlinear: true,
                temperature: 300.0,
            })
            .build()
            .unwrap();

        assert_eq!(config.simulation.frequency, 2e6);
        assert!(config.simulation.nonlinear);
    }

    #[test]
    fn test_validation_fails_on_invalid_cfl() {
        let mut config = Configuration::default();
        config.simulation.cfl = 1.5; // Invalid
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_derived_calculations() {
        let mut config = Configuration::default();
        config.calculate_derived().unwrap();

        assert!(config.grid.dx.is_some());
        assert!(config.simulation.dt.is_some());

        // Verify CFL condition
        let dx = config.grid.dx.unwrap();
        let dt = config.simulation.dt.unwrap();
        let cfl_actual = config.medium.sound_speed * dt / dx;
        assert!((cfl_actual - config.simulation.cfl).abs() < 1e-10);
    }
}
