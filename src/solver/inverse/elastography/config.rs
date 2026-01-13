//! Elastography Inversion Configuration
//!
//! Configuration types for elasticity reconstruction algorithms, including
//! linear and nonlinear inversion method selection.

use crate::domain::imaging::ultrasound::elastography::{InversionMethod, NonlinearInversionMethod};

/// Configuration for shear wave inversion
#[derive(Debug, Clone)]
pub struct ShearWaveInversionConfig {
    /// Selected inversion method
    pub method: InversionMethod,
    /// Tissue density for elasticity calculation (kg/m³)
    pub density: f64,
}

impl ShearWaveInversionConfig {
    /// Create new configuration with default parameters
    ///
    /// # Arguments
    ///
    /// * `method` - Inversion algorithm to use
    pub fn new(method: InversionMethod) -> Self {
        Self {
            method,
            density: 1000.0, // Typical soft tissue density
        }
    }

    /// Set tissue density
    ///
    /// # Arguments
    ///
    /// * `density` - Tissue density in kg/m³
    pub fn with_density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }

    /// Validate configuration parameters
    ///
    /// # Returns
    ///
    /// `Ok(())` if configuration is valid, `Err` with message otherwise
    pub fn validate(&self) -> Result<(), String> {
        if self.density <= 0.0 {
            return Err(format!(
                "Density must be positive, got: {} kg/m³",
                self.density
            ));
        }

        if self.density < 100.0 || self.density > 10000.0 {
            return Err(format!(
                "Density outside physiological range (100-10000 kg/m³): {} kg/m³",
                self.density
            ));
        }

        Ok(())
    }
}

impl Default for ShearWaveInversionConfig {
    fn default() -> Self {
        Self::new(InversionMethod::TimeOfFlight)
    }
}

/// Configuration for nonlinear parameter inversion
#[derive(Debug, Clone)]
pub struct NonlinearInversionConfig {
    /// Selected nonlinear inversion method
    pub method: NonlinearInversionMethod,
    /// Tissue density (kg/m³)
    pub density: f64,
    /// Acoustic speed in tissue (m/s)
    pub acoustic_speed: f64,
    /// Maximum iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl NonlinearInversionConfig {
    /// Create new configuration with default parameters
    ///
    /// # Arguments
    ///
    /// * `method` - Nonlinear inversion algorithm to use
    pub fn new(method: NonlinearInversionMethod) -> Self {
        Self {
            method,
            density: 1000.0,        // kg/m³
            acoustic_speed: 1540.0, // m/s (typical for soft tissue)
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }

    /// Set tissue properties
    ///
    /// # Arguments
    ///
    /// * `density` - Tissue density in kg/m³
    /// * `acoustic_speed` - Sound speed in m/s
    pub fn with_tissue_properties(mut self, density: f64, acoustic_speed: f64) -> Self {
        self.density = density;
        self.acoustic_speed = acoustic_speed;
        self
    }

    /// Set convergence parameters
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance
    pub fn with_convergence(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }

    /// Validate configuration parameters
    ///
    /// # Returns
    ///
    /// `Ok(())` if configuration is valid, `Err` with message otherwise
    pub fn validate(&self) -> Result<(), String> {
        if self.density <= 0.0 {
            return Err(format!(
                "Density must be positive, got: {} kg/m³",
                self.density
            ));
        }

        if self.density < 100.0 || self.density > 10000.0 {
            return Err(format!(
                "Density outside physiological range (100-10000 kg/m³): {} kg/m³",
                self.density
            ));
        }

        if self.acoustic_speed <= 0.0 {
            return Err(format!(
                "Acoustic speed must be positive, got: {} m/s",
                self.acoustic_speed
            ));
        }

        if self.acoustic_speed < 300.0 || self.acoustic_speed > 4000.0 {
            return Err(format!(
                "Acoustic speed outside reasonable range (300-4000 m/s): {} m/s",
                self.acoustic_speed
            ));
        }

        if self.max_iterations == 0 {
            return Err("Max iterations must be at least 1".to_string());
        }

        if self.tolerance <= 0.0 {
            return Err(format!(
                "Tolerance must be positive, got: {}",
                self.tolerance
            ));
        }

        Ok(())
    }
}

impl Default for NonlinearInversionConfig {
    fn default() -> Self {
        Self::new(NonlinearInversionMethod::HarmonicRatio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shear_wave_config_default() {
        let config = ShearWaveInversionConfig::default();
        assert_eq!(config.density, 1000.0);
        assert!(matches!(config.method, InversionMethod::TimeOfFlight));
    }

    #[test]
    fn test_shear_wave_config_builder() {
        let config =
            ShearWaveInversionConfig::new(InversionMethod::PhaseGradient).with_density(1050.0);

        assert_eq!(config.density, 1050.0);
        assert!(matches!(config.method, InversionMethod::PhaseGradient));
    }

    #[test]
    fn test_shear_wave_config_validation() {
        let config = ShearWaveInversionConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config =
            ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight).with_density(-100.0);
        assert!(invalid_config.validate().is_err());

        let invalid_config2 =
            ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight).with_density(20000.0);
        assert!(invalid_config2.validate().is_err());
    }

    #[test]
    fn test_nonlinear_config_default() {
        let config = NonlinearInversionConfig::default();
        assert_eq!(config.density, 1000.0);
        assert_eq!(config.acoustic_speed, 1540.0);
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.tolerance, 1e-6);
        assert!(matches!(
            config.method,
            NonlinearInversionMethod::HarmonicRatio
        ));
    }

    #[test]
    fn test_nonlinear_config_builder() {
        let config = NonlinearInversionConfig::new(NonlinearInversionMethod::NonlinearLeastSquares)
            .with_tissue_properties(1050.0, 1500.0)
            .with_convergence(200, 1e-8);

        assert_eq!(config.density, 1050.0);
        assert_eq!(config.acoustic_speed, 1500.0);
        assert_eq!(config.max_iterations, 200);
        assert_eq!(config.tolerance, 1e-8);
    }

    #[test]
    fn test_nonlinear_config_validation() {
        let config = NonlinearInversionConfig::default();
        assert!(config.validate().is_ok());

        let invalid_density =
            NonlinearInversionConfig::default().with_tissue_properties(-100.0, 1540.0);
        assert!(invalid_density.validate().is_err());

        let invalid_speed =
            NonlinearInversionConfig::default().with_tissue_properties(1000.0, -1540.0);
        assert!(invalid_speed.validate().is_err());

        let invalid_iterations = NonlinearInversionConfig {
            method: NonlinearInversionMethod::HarmonicRatio,
            density: 1000.0,
            acoustic_speed: 1540.0,
            max_iterations: 0,
            tolerance: 1e-6,
        };
        assert!(invalid_iterations.validate().is_err());

        let invalid_tolerance = NonlinearInversionConfig::default().with_convergence(100, -1e-6);
        assert!(invalid_tolerance.validate().is_err());
    }

    #[test]
    fn test_all_inversion_methods() {
        for method in [
            InversionMethod::TimeOfFlight,
            InversionMethod::PhaseGradient,
            InversionMethod::DirectInversion,
            InversionMethod::VolumetricTimeOfFlight,
            InversionMethod::DirectionalPhaseGradient,
        ] {
            let config = ShearWaveInversionConfig::new(method);
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn test_all_nonlinear_methods() {
        for method in [
            NonlinearInversionMethod::HarmonicRatio,
            NonlinearInversionMethod::NonlinearLeastSquares,
            NonlinearInversionMethod::BayesianInversion,
        ] {
            let config = NonlinearInversionConfig::new(method);
            assert!(config.validate().is_ok());
        }
    }
}
