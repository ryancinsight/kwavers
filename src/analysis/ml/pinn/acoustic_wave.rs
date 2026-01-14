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

use crate::analysis::ml::pinn::adapters::source::PinnAcousticSource;
use crate::analysis::ml::pinn::physics::{
    BoundaryComponent, BoundaryConditionSpec, BoundaryPosition, CouplingInterface, CouplingType,
    InitialConditionSpec, PhysicsDomain, PhysicsLossWeights, PhysicsParameters,
    PhysicsValidationMetric,
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
#[derive(Debug)]
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
    /// Sources (adapted from domain layer)
    sources: Vec<PinnAcousticSource>,
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

    /// Add acoustic source (adapted from domain source)
    pub fn add_source(&mut self, source: PinnAcousticSource) {
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
        let _p = model.forward(x.clone(), y.clone(), t.clone());

        // Compute second derivatives using automatic differentiation
        // Proper implementation with Burn autodiff for acoustic wave equation

        // Enable gradients for input coordinates
        let x_grad = x.clone().require_grad();
        let y_grad = y.clone().require_grad();
        let t_grad = t.clone().require_grad();

        // Forward pass with gradient tracking
        let p = model.forward(x_grad.clone(), y_grad.clone(), t_grad.clone());

        // First derivatives
        let grad_p = p.backward();
        let _p_x = x_grad
            .grad(&grad_p)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| x.zeros_like());
        let _p_y = y_grad
            .grad(&grad_p)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| y.zeros_like());
        let _p_t = t_grad
            .grad(&grad_p)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| t.zeros_like());

        // Second derivatives using nested autodiff
        let x_grad_2 = x.clone().require_grad();
        let y_grad_2 = y.clone().require_grad();
        let t_grad_2 = t.clone().require_grad();

        // Second derivative w.r.t. x (p_xx)
        let p_x_for_xx = model.forward(x_grad_2.clone(), y.clone(), t.clone());
        let grad_p_x = p_x_for_xx.backward();
        let p_xx = x_grad_2
            .grad(&grad_p_x)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| x.zeros_like());

        // Second derivative w.r.t. y (p_yy)
        let p_y_for_yy = model.forward(x.clone(), y_grad_2.clone(), t.clone());
        let grad_p_y = p_y_for_yy.backward();
        let p_yy = y_grad_2
            .grad(&grad_p_y)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| y.zeros_like());

        // Second derivative w.r.t. t (p_tt)
        let p_t_for_tt = model.forward(x.clone(), y.clone(), t_grad_2.clone());
        let grad_p_t = p_t_for_tt.backward();
        let p_tt = t_grad_2
            .grad(&grad_p_t)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| t.zeros_like());

        // Laplacian: ∇²p = ∂²p/∂x² + ∂²p/∂y²
        let laplacian = p_xx + p_yy;

        // Base linear wave equation residual: ∂²p/∂t² - c²∇²p
        let c = physics_params
            .material_properties
            .get("wave_speed")
            .copied()
            .unwrap_or(self.wave_speed);
        let c_squared = c * c;
        let mut residual = p_tt - c_squared * laplacian;

        // Add nonlinear term if specified
        if let AcousticProblemType::Nonlinear = self.problem_type {
            if let Some(beta) = self.nonlinearity_coefficient {
                // Nonlinear term: (β/ρ₀c⁴)∂²p²/∂t²
                let rho_0 = physics_params
                    .material_properties
                    .get("density")
                    .copied()
                    .unwrap_or(self.density);
                let coeff = beta / (rho_0 * c_squared * c_squared);

                // TODO_AUDIT: P1 - PINN Acoustic Nonlinearity Term - Placeholder Zero Gradient
                //
                // PROBLEM:
                // The second time derivative of p² (p2_tt) is hardcoded to zero, bypassing nonlinear
                // acoustic wave equation enforcement. The nonlinearity term β/(ρ₀c⁴) · ∂²(p²)/∂t² is
                // always zero regardless of the actual pressure field.
                //
                // IMPACT:
                // - PINN cannot learn nonlinear acoustic propagation (shock waves, harmonic generation)
                // - Training loss underestimates true PDE residual for high-amplitude fields
                // - Predictions are limited to linear acoustics even when nonlinearity is configured
                // - Blocks applications: histotripsy, oncotripsy, shock wave lithotripsy
                //
                // REQUIRED IMPLEMENTATION:
                // 1. Compute p² from pressure field: p_squared = p * p
                // 2. Compute first time derivative: ∂(p²)/∂t using autodiff on p_squared w.r.t. t
                // 3. Compute second time derivative: ∂²(p²)/∂t² using autodiff on ∂(p²)/∂t w.r.t. t
                // 4. Scale by coefficient: β/(ρ₀c⁴) and add to residual
                // 5. Use Burn's gradient API:
                //    ```
                //    let p_squared = p.clone().mul(p.clone());
                //    let p2_t_grad = t_grad.grad(&p_squared).unwrap();
                //    let p2_tt = t_grad_2.grad(&p2_t_grad).unwrap();
                //    ```
                //
                // MATHEMATICAL SPECIFICATION:
                // Nonlinear wave equation (Westervelt):
                //   ∇²p - (1/c²)∂²p/∂t² = β/(ρ₀c⁴) · ∂²(p²)/∂t²
                // where:
                //   - β = B/2A is the nonlinearity coefficient
                //   - B/A is the parameter of nonlinearity (material property)
                //   - ρ₀ is the ambient density
                //   - c is the sound speed
                //
                // VALIDATION CRITERIA:
                // 1. Unit test: known nonlinear solution (e.g., Fubini solution for plane wave)
                // 2. Property test: p2_tt should be proportional to amplitude squared
                // 3. Convergence test: residual decreases during training with nonlinearity
                // 4. Compare with analytical harmonic generation for sinusoidal source
                // 5. Verify gradient flow: check that p2_tt.backward() updates model parameters
                //
                // REFERENCES:
                // - Westervelt, P.J. (1963). "Parametric Acoustic Array"
                // - Hamilton & Blackstock (2008). "Nonlinear Acoustics", Ch. 3
                // - backlog.md: Sprint 212-213 Advanced Features
                //
                // EFFORT: ~12-16 hours (gradient computation, testing, validation)
                // SPRINT: Sprint 212 (nonlinear acoustics enhancement)
                //
                // Compute p² and its second time derivative
                let _p_squared = p.clone() * p.clone();
                // Placeholder for nonlinear wave equation gradients
                let _p2_t = Tensor::zeros_like(t);
                let p2_tt = Tensor::zeros_like(t);

                residual = residual + coeff * p2_tt;
            }
        }

        residual
    }

    fn boundary_conditions(&self) -> Vec<BoundaryConditionSpec> {
        self.boundary_conditions
            .iter()
            .map(|bc| {
                match bc.condition_type {
                    AcousticBoundaryType::SoundSoft => {
                        // p = 0 (Dirichlet condition)
                        BoundaryConditionSpec::Dirichlet {
                            boundary: bc.position.clone(),
                            value: vec![0.0],
                            component: BoundaryComponent::Scalar,
                        }
                    }
                    AcousticBoundaryType::SoundHard => {
                        // ∂p/∂n = 0 (Neumann condition)
                        BoundaryConditionSpec::Neumann {
                            boundary: bc.position.clone(),
                            flux: vec![0.0],
                            component: BoundaryComponent::Scalar,
                        }
                    }
                    AcousticBoundaryType::Absorbing => {
                        // Radiation boundary condition (Robin approximation)
                        BoundaryConditionSpec::Robin {
                            boundary: bc.position.clone(),
                            alpha: 1.0,
                            beta: 0.0,
                            component: BoundaryComponent::Scalar,
                        }
                    }
                    AcousticBoundaryType::Impedance => {
                        // Impedance boundary: Z ∂p/∂n + p = 0
                        let z = bc.parameters.get("impedance").copied().unwrap_or(1.0);
                        BoundaryConditionSpec::Robin {
                            boundary: bc.position.clone(),
                            alpha: z,
                            beta: 1.0,
                            component: BoundaryComponent::Scalar,
                        }
                    }
                }
            })
            .collect()
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        // Initial conditions: p(x,y,0) = 0, ∂p/∂t(x,y,0) = 0
        // (or specific initial pressure distribution based on sources)
        vec![
            // Initial pressure: p(x,y,0) = 0
            InitialConditionSpec::DirichletConstant {
                value: vec![0.0],
                component: BoundaryComponent::Scalar,
            },
            // Initial velocity: ∂p/∂t(x,y,0) = 0
            InitialConditionSpec::NeumannConstant {
                flux: vec![0.0],
                component: BoundaryComponent::Scalar,
            },
        ]
    }

    fn loss_weights(&self) -> PhysicsLossWeights {
        PhysicsLossWeights {
            pde_weight: 1.0,
            boundary_weight: 10.0,
            initial_weight: 10.0,
            physics_weights: HashMap::new(),
        }
    }

    fn validation_metrics(&self) -> Vec<PhysicsValidationMetric> {
        vec![
            PhysicsValidationMetric {
                name: "wave_speed_accuracy".to_string(),
                value: 0.0,
                acceptable_range: (-0.01, 0.01), // ±1% accuracy
                description: "Accuracy of predicted vs expected wave speed".to_string(),
            },
            PhysicsValidationMetric {
                name: "energy_conservation".to_string(),
                value: 0.0,
                acceptable_range: (-0.001, 0.001), // ±0.1% energy conservation
                description: "Acoustic energy conservation error".to_string(),
            },
            PhysicsValidationMetric {
                name: "nonlinearity_error".to_string(),
                value: 0.0,
                acceptable_range: (-0.01, 0.01), // ±1% error
                description: "Error in nonlinear acoustic effects".to_string(),
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
                position: BoundaryPosition::CustomRectangular {
                    x_min: 0.0,
                    x_max: 1.0,
                    y_min: 0.0,
                    y_max: 1.0,
                },
                coupled_domains: vec!["acoustic".to_string(), "solid".to_string()],
                coupling_type: CouplingType::Conjugate,
                coupling_params: {
                    let mut params = HashMap::new();
                    params.insert("pressure_continuity".to_string(), 1.0);
                    params.insert("velocity_continuity".to_string(), 1.0);
                    params
                },
            },
            CouplingInterface {
                name: "acoustic_thermal".to_string(),
                position: BoundaryPosition::CustomRectangular {
                    x_min: 0.0,
                    x_max: 1.0,
                    y_min: 0.0,
                    y_max: 1.0,
                },
                coupled_domains: vec!["acoustic".to_string(), "thermal".to_string()],
                coupling_type: CouplingType::FluxContinuity,
                coupling_params: {
                    let mut params = HashMap::new();
                    params.insert("heat_generation".to_string(), 1.0);
                    params.insert("temperature_coupling".to_string(), 1.0);
                    params
                },
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = burn::backend::Autodiff<NdArray<f32>>;

    #[test]
    fn test_acoustic_wave_domain_creation() {
        let domain = AcousticWaveDomain::new(
            AcousticProblemType::Linear,
            1500.0, // m/s (typical for soft tissue)
            1000.0, // kg/m³ (water density)
            None,   // No nonlinearity
        );

        assert_eq!(
            <AcousticWaveDomain as PhysicsDomain<B>>::domain_name(&domain),
            "acoustic_wave"
        );
        assert_eq!(domain.wave_speed(), 1500.0);
        assert_eq!(domain.density(), 1000.0);
        assert!(domain.nonlinearity_coefficient().is_none());
        assert!(<AcousticWaveDomain as PhysicsDomain<B>>::supports_coupling(
            &domain
        ));
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
        let mut domain = AcousticWaveDomain::new(AcousticProblemType::Linear, 1500.0, 1000.0, None);

        domain.add_boundary_condition(AcousticBoundarySpec {
            position: BoundaryPosition::Top,
            condition_type: AcousticBoundaryType::SoundSoft,
            parameters: HashMap::new(),
        });

        let bcs = <AcousticWaveDomain as PhysicsDomain<B>>::boundary_conditions(&domain);
        assert_eq!(bcs.len(), 1);
        // Check that boundary conditions are defined (exact structure depends on enum variants)
        assert!(!bcs.is_empty());
    }

    #[test]
    fn test_validation_metrics() {
        let domain = AcousticWaveDomain::new(AcousticProblemType::Linear, 1500.0, 1000.0, None);

        let metrics = <AcousticWaveDomain as PhysicsDomain<B>>::validation_metrics(&domain);
        assert_eq!(metrics.len(), 3);
        assert_eq!(metrics[0].name, "wave_speed_accuracy");
        assert_eq!(metrics[1].name, "energy_conservation");
        assert_eq!(metrics[2].name, "nonlinearity_error");
    }

    #[test]
    fn test_coupling_interfaces() {
        let domain = AcousticWaveDomain::new(AcousticProblemType::Linear, 1500.0, 1000.0, None);

        let interfaces = <AcousticWaveDomain as PhysicsDomain<B>>::coupling_interfaces(&domain);
        assert_eq!(interfaces.len(), 2);
        assert_eq!(interfaces[0].name, "acoustic_solid");
        assert_eq!(interfaces[1].name, "acoustic_thermal");
    }
}
