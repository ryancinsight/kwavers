//! Multi-Physics Coupling Bounded Context
//!
//! ## Ubiquitous Language
//! - **Coupling Terms**: Interface conditions between different physics domains
//! - **Domain Decomposition**: Spatial partitioning for multi-physics problems
//! - **Interface Conditions**: Continuity requirements at material boundaries
//! - **Conservative Coupling**: Energy/momentum conservation across domains
//! - **Operator Splitting**: Temporal decomposition of coupled PDEs
//! - **Schwarz Methods**: Iterative coupling between physics solvers
//!
//! ## 🎯 Business Value
//! Multi-physics coupling enables:
//! - **Photoacoustic Imaging**: EM absorption → thermal expansion → acoustic waves
//! - **Thermoacoustics**: Thermal diffusion → acoustic pressure waves
//! - **Acousto-optics**: Acoustic modulation of optical properties
//! - **Elasto-acoustics**: Elastic wave coupling with acoustic domains
//! - **Electromagnetic-Acoustic**: Full EM-acoustic wave interactions
//!
//! ## 📐 Mathematical Foundation
//!
//! ### General Coupling Framework
//!
//! For coupled systems of PDEs:
//!
//! ```text
//! ∂u/∂t = L₁[u] + C₁₂[v]     in Ω₁  (Physics 1)
//! ∂v/∂t = L₂[v] + C₂₁[u]     in Ω₂  (Physics 2)
//!
//! Interface conditions:
//! u = g₁(v)                   on Γ₁₂  (Dirichlet-type)
//! ∂u/∂n = g₂(v)               on Γ₁₂  (Neumann-type)
//! ```
//!
//! ### Conservation Properties
//!
//! ```text
//! dE_total/dt = 0            (Energy conservation)
//! dP_total/dt = 0            (Momentum conservation)
//! dM_total/dt = 0            (Mass conservation)
//! ```
//!
//! ## 🏗️ Architecture
//!
//! ### Coupling Strategy Traits
//!
//! ```text
//! MultiPhysicsCoupling (trait)
//! ├── coupling_strength()          ← Coupling coefficient magnitude
//! ├── interface_conditions()       ← Boundary/interface requirements
//! ├── energy_transfer_rate()       ← Power transfer between domains
//! └── stability_criteria()         ← CFL-like conditions for coupling
//! │
//! ├── AcousticElasticCoupling      ← Solid-fluid interfaces
//! │   ├── normal_stress()          ← σ·n continuity
//! │   ├── tangential_velocity()    ← Velocity continuity
//! │   └── acoustic_impedance()     ← Z = ρc matching
//! │
//! ├── AcousticThermalCoupling      ← Thermoacoustic effects
//! │   ├── thermal_expansion()      ← β ∂T/∂t → pressure source
//! │   ├── viscous_dissipation()    ← Heat generation from viscosity
//! │   └── thermal_conductivity()   ← Heat diffusion effects
//! │
//! ├── ElectromagneticAcoustic       ← Photoacoustic/optotoacoustic
//! │   ├── optical_absorption()      ← μ_a(x) absorption coefficient
//! │   ├── gruneisen_parameter()     ← Γ thermoelastic coupling
//! │   └── fluence_to_pressure()     ← Φ → p₀ conversion
//! │
//! └── ElectromagneticThermal       ← Photothermal effects
//!     ├── optical_heating()        ← Q = μ_a Φ optical energy deposition
//!     ├── thermal_diffusion()      ← Heat conduction in tissue
//!     └── perfusion_cooling()      ← Blood flow cooling effects
//! ```
//!
//! ### Solver Integration Patterns
//!
//! #### Monolithic Coupling (Single Solver)
//! ```rust,ignore
//! struct CoupledAcousticElasticSolver {
//!     acoustic_domain: AcousticDomain,
//!     elastic_domain: ElasticDomain,
//!     interface: AcousticElasticInterface,
//! }
//
//! impl CoupledAcousticElasticSolver {
//!     fn step(&mut self, dt: f64) {
//!         // Solve coupled system simultaneously
//!         self.solve_coupled_system(dt);
//!         self.enforce_interface_conditions();
//!     }
//! }
//! ```
//!
//! #### Partitioned Coupling (Multiple Solvers)
//! ```rust,ignore
//! struct PartitionedMultiPhysicsSolver<A, E> {
//!     acoustic_solver: A,    // AcousticWaveEquation
//!     elastic_solver: E,     // ElasticWaveEquation
//!     coupling: AcousticElasticCoupling,
//! }
//
//! impl<A, E> PartitionedMultiPhysicsSolver<A, E> {
//!     fn step_schwarz(&mut self, dt: f64, n_iterations: usize) {
//!         for _ in 0..n_iterations {
//!             // Update interface conditions
//!             let interface_traction = self.elastic_solver.surface_traction();
//!             let interface_velocity = self.acoustic_solver.surface_velocity();
//!
//!             // Solve acoustic domain with elastic BCs
//!             self.acoustic_solver.apply_boundary_traction(&interface_traction);
//!
//!             // Solve elastic domain with acoustic BCs
//!             self.elastic_solver.apply_boundary_velocity(&interface_velocity);
//!         }
//!     }
//! }
//! ```
//!
//! ## 💡 Design Patterns
//!
//! ### Strategy Pattern for Coupling Methods
//! ```rust,ignore
//! enum CouplingStrategy {
//!     Monolithic,        // Single matrix system
//!     SchwarzIterative,  // Alternating Schwarz method
//!     DirichletNeumann,  // DN coupling
//!     OptimizedSchwarz,  // Optimized Schwarz with overlap
//! }
//
//! trait CouplingSolver {
//!     fn couple_domains(&mut self, strategy: CouplingStrategy, dt: f64);
//! }
//! ```
//!
//! ### Observer Pattern for Coupling Events
//! ```rust,ignore
//! // Notify when coupling conditions change
//! trait CouplingObserver {
//!     fn on_interface_update(&mut self, interface_flux: &ArrayD<f64>);
//!     fn on_energy_transfer(&mut self, power_transfer: f64);
//! }
//
//! struct CoupledSolver {
//!     observers: Vec<Box<dyn CouplingObserver>>,
//! }
//
//! impl CoupledSolver {
//!     fn notify_observers(&mut self, event: CouplingEvent) {
//!         for observer in &mut self.observers {
//!             observer.on_event(event);
//!         }
//!     }
//! }
//! ```
//!
//! ## 🔗 Coupling Physics Examples
//!
//! ### Photoacoustic Coupling (EM → Acoustic)
//! ```rust,ignore
//! impl ElectromagneticAcousticCoupling for PhotoacousticSolver {
//!     fn optical_absorption(&self, x: &[f64]) -> f64 {
//!         // μ_a(x) - wavelength-dependent absorption
//!         self.tissue_model.absorption_coefficient(x, self.wavelength)
//!     }
//
//!     fn gruneisen_parameter(&self, x: &[f64]) -> f64 {
//!         // Γ(x) = β c² / C_p - thermoelastic efficiency
//!         let beta = self.tissue_model.thermal_expansion(x);
//!         let c = self.tissue_model.sound_speed(x);
//!         let cp = self.tissue_model.specific_heat(x);
//!         beta * c * c / cp
//!     }
//
//!     fn initial_pressure(&self, fluence: &ArrayD<f64>) -> ArrayD<f64> {
//!         // p₀ = Γ μ_a Φ
//!         fluence.mapv(|phi| {
//!             let gamma = self.gruneisen_parameter(&[0.0, 0.0, 0.0]); // Position-dependent
//!             let mu_a = self.optical_absorption(&[0.0, 0.0, 0.0]);   // Position-dependent
//!             gamma * mu_a * phi
//!         })
//!     }
//! }
//! ```
//!
//! ### Thermoacoustic Coupling (Thermal → Acoustic)
//! ```rust,ignore
//! impl ThermalAcousticCoupling for ThermoacousticSolver {
//!     fn thermal_expansion_coefficient(&self, x: &[f64]) -> f64 {
//!         self.material.thermal_expansion(x)
//!     }
//
//!     fn pressure_source(&self, temperature_rate: &ArrayD<f64>) -> ArrayD<f64> {
//!         // ∂p/∂t = (β/κ) ∂²T/∂t² for thermoacoustic waves
//!         // Simplified: p_source = β ρ c² ∂T/∂t
//!         let beta = self.thermal_expansion_coefficient(&[0.0, 0.0, 0.0]);
//!         let rho = self.material.density(&[0.0, 0.0, 0.0]);
//!         let c = self.material.sound_speed(&[0.0, 0.0, 0.0]);
//!
//!         beta * rho * c * c * temperature_rate
//!     }
//! }
//! ```
//!
//! ## 🧪 Validation Framework
//!
//! ### Conservation Checks
//! - **Energy Conservation**: Total energy across all domains conserved
//! - **Momentum Conservation**: Net momentum transfer at interfaces
//! - **Mass Conservation**: Mass transfer across permeable interfaces
//!
//! ### Interface Condition Verification
//! - **Continuity**: Field continuity at material interfaces
//! - **Flux Balance**: Normal flux continuity (stress, heat flux, etc.)
//!
//! ### Stability Analysis
//! - **Coupling CFL**: Time step limited by coupling strength
//! - **Interface Stability**: No spurious oscillations at boundaries
//! - **Convergence**: Iterative coupling methods converge to solution
//!
//! ## 🔬 Advanced Coupling Techniques
//!
//! ### Immersed Boundary Methods
//! ```rust,ignore
//! // Complex geometries embedded in regular grids
//! struct ImmersedBoundaryCoupling {
//!     level_set: ArrayD<f64>,      // Signed distance to interface
//!     interpolation: IBInterpolation,
//!     spreading: IBSpreading,
//! }
//
//! impl ImmersedBoundaryCoupling {
//!     fn couple_acoustic_elastic(&mut self, acoustic_field: &mut ArrayD<f64>, elastic_field: &mut ArrayD<f64>) {
//!         // Interpolate elastic solution to immersed boundary
//!         let boundary_velocity = self.interpolation.interpolate_velocity(&elastic_field, &self.level_set);
//!
//!         // Spread acoustic boundary conditions to elastic domain
//!         self.spreading.spread_traction(acoustic_field, &boundary_velocity, &self.level_set);
//!     }
//! }
//! ```
//!
//! ### Domain Decomposition Methods
//! ```rust,ignore
//! // Overlapping/non-overlapping domain decomposition
//! struct DomainDecompositionCoupling {
//!     subdomains: Vec<PhysicsDomain>,
//!     transmission_conditions: Vec<TransmissionCondition>,
//!     schwarz_method: SchwarzIteration,
//! }
//
//! impl DomainDecompositionCoupling {
//!     fn solve_schwarz(&mut self, tolerance: f64) {
//!         loop {
//!             // Update transmission conditions
//!             self.update_transmission_conditions();
//
//!             // Solve each subdomain
//!             for subdomain in &mut self.subdomains {
//!                 subdomain.solve_local_problem();
//!             }
//
//!             // Check convergence
//!             if self.convergence_error() < tolerance {
//!                 break;
//!             }
//!         }
//!     }
//! }
//! ```

use ndarray::ArrayD;
use std::fmt::Debug;

/// Coupling strength between physics domains
#[derive(Debug, Clone)]
pub struct CouplingStrength {
    /// Spatial coupling coefficient (dimensionless or with units)
    pub spatial_coefficient: f64,
    /// Temporal coupling coefficient (1/s)
    pub temporal_coefficient: f64,
    /// Energy transfer efficiency (dimensionless)
    pub energy_efficiency: f64,
}

/// Interface condition type
#[derive(Debug, Clone)]
pub enum InterfaceCondition {
    /// Dirichlet-type: field continuity u₁ = u₂
    Dirichlet { field_name: String },
    /// Neumann-type: flux continuity ∂u₁/∂n = ∂u₂/∂n
    Neumann { flux_name: String },
    /// Robin-type: weighted combination αu₁ + β∂u₁/∂n = γu₂ + δ∂u₂/∂n
    Robin {
        alpha: f64,
        beta: f64,
        gamma: f64,
        delta: f64,
    },
    /// Transmission condition for wave propagation
    Transmission { impedance_ratio: f64 },
}

/// Multi-physics coupling trait
pub trait MultiPhysicsCoupling: Send + Sync {
    /// Get coupling strength between domains
    fn coupling_strength(&self) -> CouplingStrength;

    /// Get interface conditions for this coupling
    fn interface_conditions(&self) -> Vec<InterfaceCondition>;

    /// Compute energy transfer rate between domains (W/m³)
    fn energy_transfer_rate(&self, interface_position: &[f64]) -> f64;

    /// Check stability criteria for coupled time stepping
    fn stability_criteria(&self, dt: f64) -> Result<(), String>;

    /// Apply coupling at interface
    fn apply_coupling(&mut self, dt: f64) -> Result<(), String>;
}

/// Acoustic-elastic coupling for fluid-solid interfaces
pub trait AcousticElasticCoupling: MultiPhysicsCoupling {
    /// Normal stress continuity σ·n (Pa)
    fn normal_stress(&self, position: &[f64]) -> f64;

    /// Tangential velocity continuity v_τ (m/s)
    fn tangential_velocity(&self, position: &[f64]) -> [f64; 2]; // 2D tangential

    /// Acoustic impedance matching Z = ρc (kg/m²s)
    fn acoustic_impedance(&self, position: &[f64]) -> f64;

    /// Reflection coefficient R = (Z2 - Z1)/(Z2 + Z1)
    fn reflection_coefficient(&self, position: &[f64]) -> f64 {
        let z1 = self.acoustic_impedance(&[position[0] - 1e-6, position[1], position[2]]); // Domain 1
        let z2 = self.acoustic_impedance(&[position[0] + 1e-6, position[1], position[2]]); // Domain 2
        (z2 - z1) / (z2 + z1)
    }

    /// Transmission coefficient T = 2Z2/(Z1 + Z2)
    fn transmission_coefficient(&self, position: &[f64]) -> f64 {
        let z1 = self.acoustic_impedance(&[position[0] - 1e-6, position[1], position[2]]);
        let z2 = self.acoustic_impedance(&[position[0] + 1e-6, position[1], position[2]]);
        2.0 * z2 / (z1 + z2)
    }
}

/// Acoustic-thermal coupling for thermoacoustic effects
pub trait AcousticThermalCoupling: MultiPhysicsCoupling {
    /// Thermal expansion coefficient β (1/K)
    fn thermal_expansion_coefficient(&self, position: &[f64]) -> f64;

    /// Compute acoustic pressure source from temperature rate ∂T/∂t
    /// ∂p/∂t = β ρ c² ∂T/∂t (simplified thermoacoustic coupling)
    /// TODO_AUDIT: P2 - Multi-Physics Coupling - Implement complete thermoacoustic, acousto-optic, and electromagnetic-acoustic coupling
    /// DEPENDS ON: physics/foundations/coupling/thermoacoustic.rs, physics/foundations/coupling/acoustooptic.rs
    /// MISSING: Full Navier-Stokes thermoacoustic coupling with viscous heating
    /// MISSING: Acousto-optic Bragg diffraction and phase modulation
    /// MISSING: Piezoelectric coupling for transducer modeling
    /// MISSING: Magnetoacoustic coupling for contrast agents
    fn pressure_source_from_temperature(
        &self,
        temperature_rate: &ArrayD<f64>,
        position: &[f64],
    ) -> ArrayD<f64> {
        let beta = self.thermal_expansion_coefficient(position);
        // These would come from material properties in a full implementation
        let rho = 1000.0; // kg/m³ (water density approximation)
        let c = 1500.0; // m/s (sound speed approximation)

        temperature_rate.mapv(|dtdt| beta * rho * c * c * dtdt)
    }

    /// Viscous dissipation heating rate Q = μ (∂v_i/∂x_j + ∂v_j/∂x_i)²/2  (W/m³)
    ///
    /// **Default returns 0.0** — computing the rate-of-strain tensor requires
    /// spatial derivatives of the velocity field and knowledge of the grid spacing,
    /// which are not available from `ArrayD` alone. Override this method in
    /// implementors that can supply proper gradient information.
    fn viscous_heating(&self, _velocity_field: &ArrayD<f64>, _position: &[f64]) -> f64 {
        0.0 // Override with actual strain-rate computation
    }

    /// Thermal conductivity effects on acoustic damping
    fn thermal_conductivity_damping(&self, frequency: f64, _position: &[f64]) -> f64 {
        // Thermal damping coefficient
        let k = 0.6; // W/m·K (thermal conductivity)
        let rho = 1000.0; // kg/m³
        let cp = 4186.0; // J/kg·K (specific heat)
        let alpha = k / (rho * cp); // Thermal diffusivity

        // Acoustic damping due to thermal conduction
        // δ = √(2α/(ω ρ c_p)) (thermal penetration depth)
        let omega = 2.0 * std::f64::consts::PI * frequency;
        (2.0 * alpha / (omega * rho * cp)).sqrt()
    }
}

/// Electromagnetic-acoustic coupling for photoacoustic effects
pub trait ElectromagneticAcousticCoupling: MultiPhysicsCoupling {
    /// Optical absorption coefficient μ_a (m⁻¹)
    fn optical_absorption_coefficient(&self, position: &[f64], wavelength: f64) -> f64;

    /// Reduced scattering coefficient μ_s' (m⁻¹)
    fn reduced_scattering_coefficient(&self, position: &[f64], wavelength: f64) -> f64;

    /// Grüneisen parameter Γ = β c² / C_p (dimensionless)
    fn gruneisen_parameter(&self, position: &[f64]) -> f64;

    /// Anisotropy factor g (dimensionless, -1 to 1)
    fn anisotropy_factor(&self, _position: &[f64]) -> f64 {
        0.9 // Typical for tissue (forward scattering)
    }

    /// Compute initial acoustic pressure from optical fluence
    /// p₀ = Γ μ_a Φ where Φ is fluence (J/m²)
    fn fluence_to_pressure(
        &self,
        fluence: &ArrayD<f64>,
        position: &[f64],
        wavelength: f64,
    ) -> ArrayD<f64> {
        let gamma = self.gruneisen_parameter(position);
        let mu_a = self.optical_absorption_coefficient(position, wavelength);

        fluence.mapv(|phi| gamma * mu_a * phi)
    }

    /// Compute optical fluence from electromagnetic energy density
    fn em_energy_to_fluence(
        &self,
        energy_density: &ArrayD<f64>,
        pulse_duration: f64,
    ) -> ArrayD<f64> {
        // Φ = ∫ u dt ≈ u * τ for short pulses
        energy_density.mapv(|u| u * pulse_duration)
    }

    /// Optical diffusion approximation for fluence
    fn diffuse_fluence(
        &self,
        source_position: &[f64],
        evaluation_position: &[f64],
        wavelength: f64,
    ) -> f64 {
        let r = ((evaluation_position[0] - source_position[0]).powi(2)
            + (evaluation_position[1] - source_position[1]).powi(2)
            + (evaluation_position[2] - source_position[2]).powi(2))
        .sqrt();

        if r == 0.0 {
            return 0.0; // Avoid singularity
        }

        let mu_a = self.optical_absorption_coefficient(evaluation_position, wavelength);
        let mu_s_prime = self.reduced_scattering_coefficient(evaluation_position, wavelength);
        let mu_eff = (3.0 * mu_a * (mu_a + mu_s_prime)).sqrt(); // Effective attenuation

        // Diffuse fluence: Φ(r) ∝ exp(-μ_eff r)/r
        (-mu_eff * r).exp() / r
    }
}

/// Electromagnetic-thermal coupling for photothermal effects
pub trait ElectromagneticThermalCoupling: MultiPhysicsCoupling {
    /// Optical heating rate Q = μ_a Φ (W/m³)
    fn optical_heating_rate(
        &self,
        fluence_rate: &ArrayD<f64>,
        position: &[f64],
        wavelength: f64,
    ) -> ArrayD<f64> {
        let mu_a = self.optical_absorption_coefficient(position, wavelength);
        fluence_rate.mapv(|dphi_dt| mu_a * dphi_dt)
    }

    /// Optical absorption coefficient μ_a (m⁻¹)
    fn optical_absorption_coefficient(&self, position: &[f64], wavelength: f64) -> f64;

    /// Thermal relaxation time τ = ρ C_p / k (s)
    fn thermal_relaxation_time(&self, _position: &[f64]) -> f64 {
        let rho = 1000.0; // kg/m³
        let cp = 4186.0; // J/kg·K
        let k = 0.6; // W/m·K
        rho * cp / k
    }

    /// Perfusion cooling rate (W/m³/K)
    fn perfusion_cooling(&self, temperature: f64, _position: &[f64]) -> f64 {
        let w = 0.01; // perfusion rate (kg/m³/s) - typical tissue value
        let rho_b = 1060.0; // blood density (kg/m³)
        let cp_b = 3860.0; // blood specific heat (J/kg·K)
        let tb = 37.0; // blood temperature (°C)

        w * rho_b * cp_b * (temperature - tb)
    }

    /// Bioheat equation source term (includes perfusion)
    fn bioheat_source(
        &self,
        fluence_rate: &ArrayD<f64>,
        temperature: f64,
        position: &[f64],
        wavelength: f64,
    ) -> ArrayD<f64> {
        let optical_heating = self.optical_heating_rate(fluence_rate, position, wavelength);
        let perfusion = self.perfusion_cooling(temperature, position);

        // Q_bioheat = Q_optical - Q_perfusion
        optical_heating.mapv(|q_opt| q_opt - perfusion)
    }
}

/// Domain decomposition for multi-physics problems
#[derive(Debug, Clone)]
pub struct DomainDecomposition {
    /// Subdomain boundaries
    pub subdomain_bounds: Vec<Vec<f64>>,
    /// Interface conditions between subdomains
    pub interface_conditions: Vec<InterfaceCondition>,
    /// Overlap region thickness (for overlapping Schwarz methods)
    pub overlap_thickness: f64,
    /// Transmission conditions for domain coupling
    pub transmission_conditions: Vec<TransmissionCondition>,
}

/// Transmission condition for domain decomposition
#[derive(Debug, Clone)]
pub enum TransmissionCondition {
    /// Dirichlet transmission: u = g
    Dirichlet { boundary_value: f64 },
    /// Neumann transmission: ∂u/∂n = g
    Neumann { boundary_flux: f64 },
    /// Robin transmission: αu + β∂u/∂n = g
    Robin {
        alpha: f64,
        beta: f64,
        boundary_value: f64,
    },
    /// Optimized Schwarz with optimized interface conditions
    OptimizedSchwarz { optimization_parameter: f64 },
}

/// Schwarz iteration method for domain decomposition
pub trait SchwarzMethod {
    /// Perform one Schwarz iteration
    fn schwarz_iteration(&mut self, dt: f64) -> Result<(), String>;

    /// Check convergence of Schwarz method
    fn check_convergence(&self, tolerance: f64) -> bool;

    /// Get current iteration residual
    fn iteration_residual(&self) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interface_conditions() {
        let dirichlet = InterfaceCondition::Dirichlet {
            field_name: "pressure".to_string(),
        };
        let neumann = InterfaceCondition::Neumann {
            flux_name: "stress".to_string(),
        };

        match dirichlet {
            InterfaceCondition::Dirichlet { field_name } => assert_eq!(field_name, "pressure"),
            _ => panic!("Wrong condition type"),
        }

        match neumann {
            InterfaceCondition::Neumann { flux_name } => assert_eq!(flux_name, "stress"),
            _ => panic!("Wrong condition type"),
        }
    }

    #[test]
    fn test_coupling_strength() {
        let strength = CouplingStrength {
            spatial_coefficient: 1.0,
            temporal_coefficient: 1e6,
            energy_efficiency: 0.9,
        };

        assert_eq!(strength.spatial_coefficient, 1.0);
        assert_eq!(strength.temporal_coefficient, 1e6);
        assert_eq!(strength.energy_efficiency, 0.9);
    }

    #[test]
    fn test_domain_decomposition() {
        let decomposition = DomainDecomposition {
            subdomain_bounds: vec![vec![0.0, 1.0], vec![1.0, 2.0]],
            interface_conditions: vec![InterfaceCondition::Dirichlet {
                field_name: "velocity".to_string(),
            }],
            overlap_thickness: 0.1,
            transmission_conditions: vec![TransmissionCondition::Dirichlet {
                boundary_value: 0.0,
            }],
        };

        assert_eq!(decomposition.subdomain_bounds.len(), 2);
        assert_eq!(decomposition.interface_conditions.len(), 1);
        assert_eq!(decomposition.overlap_thickness, 0.1);
    }
}
