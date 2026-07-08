//! Acoustic wave physics domain implementation.

use super::types::{AcousticBoundarySpec, AcousticProblemType, PinnAcousticBoundaryType};
use crate::inverse::pinn::ml::adapters::source::PinnAcousticSource;
use crate::inverse::pinn::ml::physics::{
    BoundaryPosition, InitialConditionSpec, PhysicsLossWeights, PhysicsValidationMetric,
    PinnBoundaryComponent, PinnBoundaryConditionSpec, PinnCouplingInterface,
    PinnDomainPhysicsParameters, PinnPhysicsCouplingType, SimulationPhysicsDomain,
};
use coeus_autograd::Var;
use std::collections::HashMap;

/// Acoustic wave physics domain implementation
#[derive(Debug)]
pub struct AcousticWaveDomain {
    /// Problem type
    pub(super) problem_type: AcousticProblemType,
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

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> SimulationPhysicsDomain<B>
    for AcousticWaveDomain
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    fn domain_name(&self) -> &'static str {
        "acoustic_wave"
    }

    fn pde_residual(
        &self,
        model: &crate::inverse::pinn::ml::PinnWave2D<B>,
        x: &Var<f32, B>,
        y: &Var<f32, B>,
        t: &Var<f32, B>,
        physics_params: &PinnDomainPhysicsParameters,
    ) -> Var<f32, B> {
        let backend = B::default();
        let zeros_like = |v: &Var<f32, B>| {
            Var::new(
                coeus_tensor::Tensor::zeros_on(v.tensor.shape(), &backend),
                false,
            )
        };

        // First derivatives are not needed by the residual itself (only the
        // second derivatives below and the raw field `p` for the nonlinear
        // term), so this is a plain forward pass.
        let p = model.forward(x, y, t);

        let x_grad_2 = Var::new(x.tensor.clone(), true);
        let p_x_for_xx = model.forward(&x_grad_2, y, t);
        p_x_for_xx.backward();
        let p_xx = x_grad_2
            .grad()
            .map(|g| Var::new(g, false))
            .unwrap_or_else(|| zeros_like(x));

        let y_grad_2 = Var::new(y.tensor.clone(), true);
        let p_y_for_yy = model.forward(x, &y_grad_2, t);
        p_y_for_yy.backward();
        let p_yy = y_grad_2
            .grad()
            .map(|g| Var::new(g, false))
            .unwrap_or_else(|| zeros_like(y));

        let t_grad_2 = Var::new(t.tensor.clone(), true);
        let p_t_for_tt = model.forward(x, y, &t_grad_2);
        p_t_for_tt.backward();
        let p_tt = t_grad_2
            .grad()
            .map(|g| Var::new(g, false))
            .unwrap_or_else(|| zeros_like(t));

        let laplacian = coeus_autograd::add(&p_xx, &p_yy);

        let c = physics_params
            .material_properties
            .get("wave_speed")
            .copied()
            .unwrap_or(self.wave_speed);
        let c_squared = (c * c) as f32;
        let mut residual =
            coeus_autograd::sub(&p_tt, &coeus_autograd::scalar_mul(&laplacian, c_squared));

        if let AcousticProblemType::Nonlinear = self.problem_type {
            if let Some(beta) = self.nonlinearity_coefficient {
                let rho_0 = physics_params
                    .material_properties
                    .get("density")
                    .copied()
                    .unwrap_or(self.density);
                // Westervelt equation (Hamilton & Blackstock 1998 Eq. 3.27):
                //   p_tt - c²∇²p = β/(ρ₀c²) * ∂²(p²)/∂t²
                // Residual = 0 form:
                //   R = p_tt - c²∇²p - β/(ρ₀c²) * ∂²(p²)/∂t²
                // coeff = β/(ρ₀c²), not β/(ρ₀c⁴).
                let coeff = (beta / (rho_0 * (c * c))) as f32;

                let t_grad_for_pt = Var::new(t.tensor.clone(), true);
                let p_for_pt = model.forward(x, y, &t_grad_for_pt);
                p_for_pt.backward();
                let p_t = t_grad_for_pt
                    .grad()
                    .map(|g| Var::new(g, false))
                    .unwrap_or_else(|| zeros_like(t));

                // ∂²(p²)/∂t² = 2*(p_t² + p * p_tt)
                let p_t_sq = coeus_autograd::mul(&p_t, &p_t);
                let p_p_tt = coeus_autograd::mul(&p, &p_tt);
                let p2_tt = coeus_autograd::scalar_mul(&coeus_autograd::add(&p_t_sq, &p_p_tt), 2.0);

                residual =
                    coeus_autograd::sub(&residual, &coeus_autograd::scalar_mul(&p2_tt, coeff));
            }
        }

        residual
    }

    fn boundary_conditions(&self) -> Vec<PinnBoundaryConditionSpec> {
        self.boundary_conditions
            .iter()
            .map(|bc| match bc.condition_type {
                PinnAcousticBoundaryType::SoundSoft => PinnBoundaryConditionSpec::Dirichlet {
                    boundary: bc.position.clone(),
                    value: vec![0.0],
                    component: PinnBoundaryComponent::Scalar,
                },
                PinnAcousticBoundaryType::SoundHard => PinnBoundaryConditionSpec::Neumann {
                    boundary: bc.position.clone(),
                    flux: vec![0.0],
                    component: PinnBoundaryComponent::Scalar,
                },
                PinnAcousticBoundaryType::Absorbing => PinnBoundaryConditionSpec::Robin {
                    boundary: bc.position.clone(),
                    alpha: 1.0,
                    beta: 0.0,
                    component: PinnBoundaryComponent::Scalar,
                },
                PinnAcousticBoundaryType::Impedance => {
                    let z = bc.parameters.get("impedance").copied().unwrap_or(1.0);
                    PinnBoundaryConditionSpec::Robin {
                        boundary: bc.position.clone(),
                        alpha: z,
                        beta: 1.0,
                        component: PinnBoundaryComponent::Scalar,
                    }
                }
            })
            .collect()
    }

    fn initial_conditions(&self) -> Vec<InitialConditionSpec> {
        vec![
            InitialConditionSpec::DirichletConstant {
                value: vec![0.0],
                component: PinnBoundaryComponent::Scalar,
            },
            InitialConditionSpec::NeumannConstant {
                flux: vec![0.0],
                component: PinnBoundaryComponent::Scalar,
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
                acceptable_range: (-0.01, 0.01),
                description: "Accuracy of predicted vs expected wave speed".to_string(),
            },
            PhysicsValidationMetric {
                name: "energy_conservation".to_string(),
                value: 0.0,
                acceptable_range: (-0.001, 0.001),
                description: "Acoustic energy conservation error".to_string(),
            },
            PhysicsValidationMetric {
                name: "nonlinearity_error".to_string(),
                value: 0.0,
                acceptable_range: (-0.01, 0.01),
                description: "Error in nonlinear acoustic effects".to_string(),
            },
        ]
    }

    fn supports_coupling(&self) -> bool {
        true
    }

    fn coupling_interfaces(&self) -> Vec<PinnCouplingInterface> {
        vec![
            PinnCouplingInterface {
                name: "acoustic_solid".to_string(),
                position: BoundaryPosition::CustomRectangular {
                    x_min: 0.0,
                    x_max: 1.0,
                    y_min: 0.0,
                    y_max: 1.0,
                },
                coupled_domains: vec!["acoustic".to_string(), "solid".to_string()],
                coupling_type: PinnPhysicsCouplingType::Conjugate,
                coupling_params: {
                    let mut params = HashMap::new();
                    params.insert("pressure_continuity".to_string(), 1.0);
                    params.insert("velocity_continuity".to_string(), 1.0);
                    params
                },
            },
            PinnCouplingInterface {
                name: "acoustic_thermal".to_string(),
                position: BoundaryPosition::CustomRectangular {
                    x_min: 0.0,
                    x_max: 1.0,
                    y_min: 0.0,
                    y_max: 1.0,
                },
                coupled_domains: vec!["acoustic".to_string(), "thermal".to_string()],
                coupling_type: PinnPhysicsCouplingType::FluxContinuity,
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
