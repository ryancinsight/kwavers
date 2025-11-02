//! Acoustic Wave Physics Domain for PINN
//!
//! This module implements acoustic wave equations for ultrasound physics using
//! Physics-Informed Neural Networks. Supports linear and nonlinear acoustic
//! wave propagation in homogeneous and heterogeneous media.
//!
//! ## Mathematical Formulation
//!
//! ### Linear Acoustic Wave Equation
//! ∂²p/∂t² = c²∇²p
//!
//! Where:
//! - p: acoustic pressure (Pa)
//! - c: speed of sound (m/s)
//! - t: time (s)
//! - ∇²: Laplacian operator
//!
//! ### Nonlinear Acoustic Wave Equation (Kuznetsov)
//! ∂²p/∂t² = c²∇²p + (β/ρ₀c⁴)∂²p²/∂t²
//!
//! Where:
//! - β: nonlinearity coefficient (dimensionless)
//! - ρ₀: equilibrium density (kg/m³)

use crate::error::{KwaversError, KwaversResult};
use crate::ml::pinn::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface,
    CouplingType, InitialConditionSpec, PhysicsDomain, PhysicsLossWeights,
    PhysicsParameters, PhysicsValidationMetric,
};
use burn::tensor::{backend::AutodiffBackend, Tensor};
use std::collections::HashMap;

/// Acoustic wave problem type
#[derive(Debug, Clone, PartialEq)]
pub enum AcousticProblemType {
    /// Linear acoustic wave equation
    Linear,
    /// Nonlinear acoustic wave equation (Kuznetsov)
    Nonlinear,
}

/// Acoustic source specification
#[derive(Debug, Clone)]
pub struct AcousticSource {
    /// Source position (x, y, z)
    pub position: (f64, f64, f64),
    /// Source type
    pub source_type: AcousticSourceType,
    /// Source parameters
    pub parameters: AcousticSourceParameters,
}

/// Acoustic source types
#[derive(Debug, Clone)]
pub enum AcousticSourceType {
    /// Monopole source (pressure source)
    Monopole,
    /// Dipole source (velocity source)
    Dipole,
    /// Focused transducer
    FocusedTransducer,
}

/// Acoustic source parameters
#[derive(Debug, Clone)]
pub struct AcousticSourceParameters {
    /// Frequency (Hz)
    pub frequency: f64,
    /// Amplitude (Pa for pressure, m/s for velocity)
    pub amplitude: f64,
    /// Phase (radians)
    pub phase: f64,
    /// Focal length for focused sources (m)
    pub focal_length: Option<f64>,
}

/// Acoustic boundary condition specification
#[derive(Debug, Clone)]
pub struct AcousticBoundarySpec {
    /// Boundary position
    pub position: BoundaryPosition,
    /// Boundary condition type
    pub condition_type: AcousticBoundaryType,
    /// Boundary parameters
    pub parameters: HashMap<String, f64>,
}

/// Acoustic boundary condition types
#[derive(Debug, Clone)]
pub enum AcousticBoundaryType {
    /// Sound-soft (pressure = 0)
    SoundSoft,
    /// Sound-hard (normal velocity = 0)
    SoundHard,
    /// Absorbing boundary
    Absorbing,
    /// Impedance boundary
    Impedance,
}

/// Acoustic wave physics domain implementation
pub struct AcousticWaveDomain {
    /// Problem type
    problem_type: AcousticProblemType,
    /// Wave speed (m/s)
    wave_speed: f64,
    /// Density (kg/m³)
    density: f64,
    /// Nonlinearity coefficient β (dimensionless)
    nonlinearity_coefficient: Option<f64>,
    /// Boundary conditions
    boundary_conditions: Vec<AcousticBoundarySpec>,
    /// Sources
    sources: Vec<AcousticSource>,
}

impl AcousticWaveDomain {
    /// Create new acoustic wave domain
    pub fn new(
        problem_type: AcousticProblemType,
        wave_speed: f64,
        density: f64,
        nonlinearity_coefficient: Option<f64>,
    ) -> Self {
        Self {
            problem_type,
            wave_speed,
            density,
            nonlinearity_coefficient,
            boundary_conditions: Vec::new(),
            sources: Vec::new(),
        }
    }

    /// Add boundary condition
    pub fn add_boundary_condition(&mut self, boundary: AcousticBoundarySpec) {
        self.boundary_conditions.push(boundary);
    }

    /// Add acoustic source
    pub fn add_source(&mut self, source: AcousticSource) {
        self.sources.push(source);
    }

    /// Get wave speed
    pub fn wave_speed(&self) -> f64 {
        self.wave_speed
    }

    /// Get density
    pub fn density(&self) -> f64 {
        self.density
    }

    /// Get nonlinearity coefficient
    pub fn nonlinearity_coefficient(&self) -> Option<f64> {
        self.nonlinearity_coefficient
    }
}

impl<B: AutodiffBackend> PhysicsDomain<B> for AcousticWaveDomain {
    fn domain_name(&self) -> &'static str {
        "acoustic_wave"
    }

    fn pde_residual(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PhysicsParameters,
    ) -> Tensor<B, 2> {
        // Get predictions from model
        let p = model.forward(x, y, t);

        // Compute second derivatives using automatic differentiation
        let p_t = p.backward_grad(t);
        let p_tt = p_t.backward_grad(t);

        let p_x = p.backward_grad(x);
        let p_xx = p_x.backward_grad(x);

        let p_y = p.backward_grad(y);
        let p_yy = p_y.backward_grad(y);

        // Laplacian: ∇²p = ∂²p/∂x² + ∂²p/∂y²
        let laplacian = p_xx + p_yy;

        // Base linear wave equation residual: ∂²p/∂t² - c²∇²p
        let c = physics_params.get_float("wave_speed").unwrap_or(self.wave_speed);
        let c_squared = c * c;
        let mut residual = p_tt - c_squared * laplacian;

        // Add nonlinear term if specified
        if let AcousticProblemType::Nonlinear = self.problem_type {
            if let Some(beta) = self.nonlinearity_coefficient {
                // Nonlinear term: (β/ρ₀c⁴)∂²p²/∂t²
                let rho_0 = physics_params.get_float("density").unwrap_or(self.density);
                let coeff = beta / (rho_0 * c_squared * c_squared);

                // Compute p² and its second time derivative
                let p_squared = p.clone() * p.clone();
                let p2_t = p_squared.backward_grad(t);
                let p2_tt = p2_t.backward_grad(t);

                residual = residual + coeff * p2_tt;
            }
        }

        residual
    }

    fn boundary_conditions(&self) -> Vec<BoundaryConditionSpec> {
        self.boundary_conditions
            .iter()
            .map(|bc| {
                let components = match bc.condition_type {
                    AcousticBoundaryType::SoundSoft => {
                        // p = 0
                        vec![BoundaryComponent {
                            variable: "pressure".to_string(),
                            value: 0.0,
                            derivative_order: 0,
                        }]
                    }
                    AcousticBoundaryType::SoundHard => {
                        // ∂p/∂n = 0 (normal derivative = 0)
                        vec![BoundaryComponent {
                            variable: "pressure".to_string(),
                            value: 0.0,
                            derivative_order: 1,
                        }]
                    }
                    AcousticBoundaryType::Absorbing => {
                        // Radiation boundary condition approximation
                        vec![BoundaryComponent {
                            variable: "pressure".to_string(),
                            value: 0.0,
                            derivative_order: 1,
                        }]
                    }
                    AcousticBoundaryType::Impedance => {
                        // Impedance boundary: ∂p/∂n = -Z ∂p/∂t
                        // This would need more complex implementation
                        vec![BoundaryComponent {
                            variable: "pressure".to_string(),
                            value: 0.0,
                            derivative_order: 1,
                        }]
                    }
                };

                BoundaryConditionSpec {
                    position: bc.position.clone(),
                    components,
                }
            })
            .collect()
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        // Initial conditions: p(x,y,0) = 0, ∂p/∂t(x,y,0) = 0
        // (or specific initial pressure distribution based on sources)
        vec![
            InitialConditionSpec {
                time: 0.0,
                variable: "pressure".to_string(),
                expression: "0.0".to_string(), // Could be more complex based on sources
                derivative_order: 0,
            },
            InitialConditionSpec {
                time: 0.0,
                variable: "pressure".to_string(),
                expression: "0.0".to_string(),
                derivative_order: 1,
            },
        ]
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        PhysicsLossWeights {
            pde_weight: 1.0,
            boundary_weight: 10.0,
            initial_weight: 10.0,
        }
    }

    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric> {
        vec![
            PhysicsValidationMetric {
                name: "wave_speed_accuracy".to_string(),
                description: "Accuracy of predicted wave speed".to_string(),
                unit: "m/s".to_string(),
            },
            PhysicsValidationMetric {
                name: "dispersion_error".to_string(),
                description: "Numerical dispersion error".to_string(),
                unit: "percent".to_string(),
            },
            PhysicsValidationMetric {
                name: "nonlinearity_accuracy".to_string(),
                description: "Accuracy of nonlinear effects".to_string(),
                unit: "dimensionless".to_string(),
            },
        ]
    }

    fn supports_coupling(&self) -> bool {
        true
    }

    fn coupling_interfaces(&self) -> Vec<CouplingInterface> {
        vec![
            CouplingInterface {
                name: "acoustic_solid".to_string(),
                coupling_type: CouplingType::Interface,
                variables: vec!["pressure".to_string(), "velocity".to_string()],
                description: "Acoustic-solid coupling for medical ultrasound".to_string(),
            },
            CouplingInterface {
                name: "acoustic_thermal".to_string(),
                coupling_type: CouplingType::MultiPhysics,
                variables: vec!["pressure".to_string(), "temperature".to_string()],
                description: "Acoustic-thermal coupling for HIFU".to_string(),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_acoustic_wave_domain_creation() {
        let domain = AcousticWaveDomain::new(
            AcousticProblemType::Linear,
            1500.0, // m/s (typical for soft tissue)
            1000.0, // kg/m³ (water density)
            None,   // No nonlinearity
        );

        assert_eq!(domain.domain_name(), "acoustic_wave");
        assert_eq!(domain.wave_speed(), 1500.0);
        assert_eq!(domain.density(), 1000.0);
        assert!(domain.nonlinearity_coefficient().is_none());
        assert!(domain.supports_coupling());
    }

    #[test]
    fn test_nonlinear_acoustic_domain() {
        let domain = AcousticWaveDomain::new(
            AcousticProblemType::Nonlinear,
            1500.0,
            1000.0,
            Some(3.5), // Typical β for water
        );

        assert_eq!(domain.problem_type, AcousticProblemType::Nonlinear);
        assert_eq!(domain.nonlinearity_coefficient(), Some(3.5));
    }

    #[test]
    fn test_boundary_conditions() {
        let mut domain = AcousticWaveDomain::new(
            AcousticProblemType::Linear,
            1500.0,
            1000.0,
            None,
        );

        domain.add_boundary_condition(AcousticBoundarySpec {
            position: BoundaryPosition::Top,
            condition_type: AcousticBoundaryType::SoundSoft,
            parameters: HashMap::new(),
        });

        let bcs = domain.boundary_conditions();
        assert_eq!(bcs.len(), 1);
        assert_eq!(bcs[0].position, BoundaryPosition::Top);
    }

    #[test]
    fn test_validation_metrics() {
        let domain = AcousticWaveDomain::new(
            AcousticProblemType::Linear,
            1500.0,
            1000.0,
            None,
        );

        let metrics = domain.validation_metrics();
        assert_eq!(metrics.len(), 3);
        assert_eq!(metrics[0].name, "wave_speed_accuracy");
        assert_eq!(metrics[1].name, "dispersion_error");
        assert_eq!(metrics[2].name, "nonlinearity_accuracy");
    }

    #[test]
    fn test_coupling_interfaces() {
        let domain = AcousticWaveDomain::new(
            AcousticProblemType::Linear,
            1500.0,
            1000.0,
            None,
        );

        let interfaces = domain.coupling_interfaces();
        assert_eq!(interfaces.len(), 2);
        assert_eq!(interfaces[0].name, "acoustic_solid");
        assert_eq!(interfaces[1].name, "acoustic_thermal");
    }
}
