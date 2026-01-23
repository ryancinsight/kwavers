//! Wave Equation Trait Specifications
//!
//! This module defines abstract trait interfaces for wave equation physics.
//! These traits specify the mathematical structure of wave propagation PDEs
//! without committing to a particular numerical method (finite difference,
//! finite element, spectral, or neural network approximation).
//! TODO_AUDIT: P1 - Generalized Wave Physics - Implement complete wave equation hierarchy with nonlinear, dispersive, and multi-physics coupling
//! DEPENDS ON: physics/foundations/wave_equations/nonlinear.rs, physics/foundations/wave_equations/dispersive.rs, physics/foundations/wave_equations/coupled.rs
//! MISSING: Nonlinear wave equations (KZK, Westervelt) with exact dispersion relations
//! MISSING: Dispersive media modeling with Kramers-Kronig relations
//! MISSING: Multi-physics coupling (thermoacoustic, acousto-optic, piezoelectric)
//! MISSING: Fractional wave equations for anomalous dispersion
//! MISSING: Time-reversal acoustics for focusing and imaging
//! MISSING: Quantum acoustic effects for extreme conditions
//!
//! # Design Principle
//!
//! Separate *specification* (what the PDE is) from *implementation* (how to solve it).
//! This allows:
//! - Forward solvers (numerical discretization) and inverse solvers (PINN, optimization)
//!   to share the same physics specification
//! - Validation logic to be reused across solver types
//! - Material properties and boundary conditions to be solver-agnostic
//!
//! # Mathematical Foundation
//!
//! All wave equations in this system are second-order hyperbolic PDEs of the form:
//!
//! ```text
//! ∂²u/∂t² = L[u] + f
//! ```
//!
//! where:
//! - u is the field variable (displacement, pressure, etc.)
//! - L is a spatial differential operator (varies by wave type)
//! - f is a source term
//!
//! Different wave types (acoustic, elastic, electromagnetic) differ in:
//! - The structure of L (scalar vs vector vs tensor)
//! - Material properties (density, sound speed, elastic moduli)
//! - Boundary condition types (Dirichlet, Neumann, absorbing)

use ndarray::ArrayD;

/// Spatial dimension specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialDimension {
    /// 1D problems (x-axis only)
    One,
    /// 2D problems (x-y plane)
    Two,
    /// 3D problems (x-y-z volume)
    Three,
}

/// Time integration scheme specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeIntegration {
    /// Explicit time stepping (CFL-limited)
    Explicit,
    /// Implicit time stepping (unconditionally stable)
    Implicit,
    /// Semi-implicit (mixed explicit/implicit)
    SemiImplicit,
}

/// Boundary condition type
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryCondition {
    /// Dirichlet: prescribed field value u = g
    Dirichlet { value: f64 },
    /// Neumann: prescribed normal derivative ∂u/∂n = g
    Neumann { flux: f64 },
    /// Absorbing: perfectly matched layer or damping
    Absorbing { damping: f64 },
    /// Periodic: u(x₀) = u(x₁)
    Periodic,
    /// Free surface: stress-free condition
    FreeSurface,
    /// Rigid: zero displacement
    Rigid,
}

/// Spatial domain specification
#[derive(Debug, Clone)]
pub struct Domain {
    /// Spatial dimension
    pub dimension: SpatialDimension,
    /// Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    pub bounds: Vec<f64>,
    /// Grid resolution [nx, ny, nz]
    pub resolution: Vec<usize>,
    /// Boundary conditions for each face [x_min, x_max, y_min, y_max, z_min, z_max]
    pub boundaries: Vec<BoundaryCondition>,
}

impl Domain {
    /// Create 1D domain
    pub fn new_1d(xmin: f64, xmax: f64, nx: usize, bc: BoundaryCondition) -> Self {
        Self {
            dimension: SpatialDimension::One,
            bounds: vec![xmin, xmax],
            resolution: vec![nx],
            boundaries: vec![bc.clone(), bc],
        }
    }

    /// Create 2D domain
    pub fn new_2d(
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        nx: usize,
        ny: usize,
        bc: BoundaryCondition,
    ) -> Self {
        Self {
            dimension: SpatialDimension::Two,
            bounds: vec![xmin, xmax, ymin, ymax],
            resolution: vec![nx, ny],
            boundaries: vec![bc.clone(), bc.clone(), bc.clone(), bc],
        }
    }

    /// Create 3D domain
    pub fn new_3d(
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        zmin: f64,
        zmax: f64,
        nx: usize,
        ny: usize,
        nz: usize,
        bc: BoundaryCondition,
    ) -> Self {
        Self {
            dimension: SpatialDimension::Three,
            bounds: vec![xmin, xmax, ymin, ymax, zmin, zmax],
            resolution: vec![nx, ny, nz],
            boundaries: vec![
                bc.clone(),
                bc.clone(),
                bc.clone(),
                bc.clone(),
                bc.clone(),
                bc,
            ],
        }
    }

    /// Get spatial step sizes [dx, dy, dz]
    pub fn spacing(&self) -> Vec<f64> {
        self.resolution
            .iter()
            .enumerate()
            .map(|(i, &n)| (self.bounds[2 * i + 1] - self.bounds[2 * i]) / (n - 1) as f64)
            .collect()
    }
}

/// Abstract wave equation trait for traditional numerical solvers
///
/// This trait defines the mathematical structure common to all wave equations
/// implemented with traditional numerical methods (finite difference, finite element,
/// spectral methods, etc.).
///
/// # Thread Safety
///
/// This trait requires `Send + Sync` to enable parallel validation and testing.
/// Traditional solvers using ndarray and standard Rust data structures satisfy this.
///
/// # See Also
///
/// For neural network and autodiff-based solvers that cannot satisfy `Sync` due to
/// internal framework constraints, see [`AutodiffWaveEquation`].
pub trait WaveEquation: Send + Sync {
    /// Get spatial domain specification
    fn domain(&self) -> &Domain;

    /// Get time integration scheme
    fn time_integration(&self) -> TimeIntegration;

    /// Compute the CFL stability limit (timestep in seconds)
    ///
    /// For explicit schemes: Δt ≤ CFL_factor * min(Δx, Δy, Δz) / c_max
    /// where c_max is the maximum wave speed in the domain.
    fn cfl_timestep(&self) -> f64;

    /// Evaluate the spatial differential operator L[u] at current state
    ///
    /// Returns the right-hand side of the wave equation:
    /// ∂²u/∂t² = spatial_operator(u) + source
    fn spatial_operator(&self, field: &ArrayD<f64>) -> ArrayD<f64>;

    /// Apply boundary conditions to field
    fn apply_boundary_conditions(&mut self, field: &mut ArrayD<f64>);

    /// Check if the current state satisfies physics constraints
    ///
    /// Returns Ok(()) if constraints are satisfied, Err with violation description otherwise.
    fn check_constraints(&self, field: &ArrayD<f64>) -> Result<(), String>;
}

/// Abstract wave equation trait for autodiff-based solvers
///
/// This trait mirrors [`WaveEquation`] but relaxes the `Sync` constraint to accommodate
/// neural network frameworks (e.g., Burn) that use internal cell types for lazy
/// initialization and gradient tracking.
///
/// # Design Rationale
///
/// Automatic differentiation frameworks like Burn use `std::cell::OnceCell` and similar
/// constructs internally for:
/// - Lazy tensor initialization
/// - Gradient accumulation
/// - Computation graph construction
///
/// These types are `!Sync` by design, preventing implementations from satisfying the
/// `Sync` bound required by traditional numerical solvers.
///
/// Rather than compromising either API, we maintain separate trait hierarchies:
/// - `WaveEquation` (with `Sync`) for traditional methods
/// - `AutodiffWaveEquation` (without `Sync`) for neural networks
///
/// This preserves thread safety guarantees where possible while enabling autodiff
/// integration where necessary.
///
/// # Thread Safety
///
/// This trait requires only `Send`, allowing solvers to be moved between threads but
/// not shared. Most PINN training is single-threaded per model instance, making this
/// sufficient.
pub trait AutodiffWaveEquation: Send {
    /// Get spatial domain specification
    fn domain(&self) -> &Domain;

    /// Get time integration scheme
    fn time_integration(&self) -> TimeIntegration;

    /// Compute the CFL stability limit (timestep in seconds)
    ///
    /// For explicit schemes: Δt ≤ CFL_factor * min(Δx, Δy, Δz) / c_max
    /// where c_max is the maximum wave speed in the domain.
    fn cfl_timestep(&self) -> f64;

    /// Evaluate the spatial differential operator L[u] at current state
    ///
    /// Returns the right-hand side of the wave equation:
    /// ∂²u/∂t² = spatial_operator(u) + source
    fn spatial_operator(&self, field: &ArrayD<f64>) -> ArrayD<f64>;

    /// Apply boundary conditions to field
    fn apply_boundary_conditions(&mut self, field: &mut ArrayD<f64>);

    /// Check if the current state satisfies physics constraints
    ///
    /// Returns Ok(()) if constraints are satisfied, Err with violation description otherwise.
    fn check_constraints(&self, field: &ArrayD<f64>) -> Result<(), String>;
}

/// Acoustic wave equation trait (scalar pressure field)
///
/// Governs propagation of pressure waves in fluids:
///
/// ```text
/// ∂²p/∂t² = c²∇²p + f
/// ```
///
/// where:
/// - p(x,t) is acoustic pressure [Pa]
/// - c(x) is sound speed [m/s]
/// - f(x,t) is acoustic source [Pa/s²]
pub trait AcousticWaveEquation: WaveEquation {
    /// Get sound speed field c(x) [m/s]
    fn sound_speed(&self) -> ArrayD<f64>;

    /// Get density field ρ(x) [kg/m³]
    fn density(&self) -> ArrayD<f64>;

    /// Get absorption coefficient α(x) [Np/m]
    fn absorption(&self) -> ArrayD<f64>;

    /// Compute acoustic energy
    ///
    /// E = ∫ (½ρ|∂p/∂t|² + ½|∇p|²/(ρc²)) dV
    fn acoustic_energy(&self, pressure: &ArrayD<f64>, velocity: &ArrayD<f64>) -> f64;
}

/// Elastic wave equation trait for traditional solvers (vector displacement field)
///
/// Governs propagation of elastic waves in solids:
///
/// ```text
/// ρ ∂²u/∂t² = ∇·σ + f
/// σ = C:ε
/// ε = ½(∇u + (∇u)ᵀ)
/// ```
///
/// where:
/// - u(x,t) is displacement vector [m]
/// - σ is stress tensor [Pa]
/// - ε is strain tensor [dimensionless]
/// - C is elastic modulus tensor [Pa]
/// - ρ(x) is density [kg/m³]
/// - f(x,t) is body force [N/m³]
///
/// # See Also
///
/// For autodiff-based implementations (PINN), see [`AutodiffElasticWaveEquation`].
pub trait ElasticWaveEquation: WaveEquation {
    /// Get Lamé first parameter λ(x) [Pa]
    fn lame_lambda(&self) -> ArrayD<f64>;

    /// Get Lamé second parameter μ(x) (shear modulus) [Pa]
    fn lame_mu(&self) -> ArrayD<f64>;

    /// Get density field ρ(x) [kg/m³]
    fn density(&self) -> ArrayD<f64>;

    /// Compute stress tensor from displacement field
    ///
    /// σᵢⱼ = λδᵢⱼ∇·u + μ(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
    fn stress_from_displacement(&self, displacement: &ArrayD<f64>) -> ArrayD<f64>;

    /// Compute strain tensor from displacement field
    ///
    /// εᵢⱼ = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
    fn strain_from_displacement(&self, displacement: &ArrayD<f64>) -> ArrayD<f64>;

    /// Compute elastic energy
    ///
    /// E = ∫ (½ρ|∂u/∂t|² + ½σ:ε) dV
    fn elastic_energy(&self, displacement: &ArrayD<f64>, velocity: &ArrayD<f64>) -> f64;

    /// Get P-wave (longitudinal) speed [m/s]
    fn p_wave_speed(&self) -> ArrayD<f64> {
        let lambda = self.lame_lambda();
        let mu = self.lame_mu();
        let rho = self.density();
        ((lambda + 2.0 * mu) / rho).mapv(f64::sqrt)
    }

    /// Get S-wave (shear) speed [m/s]
    fn s_wave_speed(&self) -> ArrayD<f64> {
        let mu = self.lame_mu();
        let rho = self.density();
        (mu / rho).mapv(f64::sqrt)
    }
}

/// Elastic wave equation trait for autodiff-based solvers
///
/// This trait mirrors [`ElasticWaveEquation`] but extends [`AutodiffWaveEquation`]
/// instead of [`WaveEquation`], relaxing the `Sync` constraint to accommodate
/// neural network frameworks.
///
/// # Use Cases
///
/// - Physics-Informed Neural Networks (PINN)
/// - Neural operator methods (FNO, DeepONet)
/// - Hybrid neural-numerical solvers
///
/// # Mathematical Equivalence
///
/// Despite the different trait bounds, implementations must satisfy the same
/// mathematical constraints as traditional solvers:
/// - Material property bounds (ρ > 0, μ > 0, λ > -2μ/3)
/// - Wave speed relationships (cₚ > cₛ)
/// - PDE satisfaction (residual minimization)
/// - Energy conservation
///
/// The validation framework provides separate functions for each trait hierarchy
/// but enforces identical mathematical requirements.
pub trait AutodiffElasticWaveEquation: AutodiffWaveEquation {
    /// Get Lamé first parameter λ(x) [Pa]
    fn lame_lambda(&self) -> ArrayD<f64>;

    /// Get Lamé second parameter μ(x) (shear modulus) [Pa]
    fn lame_mu(&self) -> ArrayD<f64>;

    /// Get density field ρ(x) [kg/m³]
    fn density(&self) -> ArrayD<f64>;

    /// Compute stress tensor from displacement field
    ///
    /// σᵢⱼ = λδᵢⱼ∇·u + μ(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
    fn stress_from_displacement(&self, displacement: &ArrayD<f64>) -> ArrayD<f64>;

    /// Compute strain tensor from displacement field
    ///
    /// εᵢⱼ = ½(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
    fn strain_from_displacement(&self, displacement: &ArrayD<f64>) -> ArrayD<f64>;

    /// Compute elastic energy
    ///
    /// E = ∫ (½ρ|∂u/∂t|² + ½σ:ε) dV
    fn elastic_energy(&self, displacement: &ArrayD<f64>, velocity: &ArrayD<f64>) -> f64;

    /// Get P-wave (longitudinal) speed [m/s]
    fn p_wave_speed(&self) -> ArrayD<f64> {
        let lambda = self.lame_lambda();
        let mu = self.lame_mu();
        let rho = self.density();
        ((lambda + 2.0 * mu) / rho).mapv(f64::sqrt)
    }

    /// Get S-wave (shear) speed [m/s]
    fn s_wave_speed(&self) -> ArrayD<f64> {
        let mu = self.lame_mu();
        let rho = self.density();
        (mu / rho).mapv(f64::sqrt)
    }
}

/// Source term trait
///
/// Defines time-varying source distributions f(x,t)
pub trait SourceTerm: Send + Sync {
    /// Evaluate source at given time [appropriate units]
    fn evaluate(&self, time: f64) -> ArrayD<f64>;

    /// Get temporal support [t_start, t_end]
    fn time_window(&self) -> (f64, f64);

    /// Check if source is active at given time
    fn is_active(&self, time: f64) -> bool {
        let (t_start, t_end) = self.time_window();
        time >= t_start && time <= t_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_1d() {
        let domain = Domain::new_1d(0.0, 1.0, 101, BoundaryCondition::Periodic);
        assert_eq!(domain.dimension, SpatialDimension::One);
        assert_eq!(domain.resolution.len(), 1);
        assert_eq!(domain.resolution[0], 101);
        let spacing = domain.spacing();
        assert!((spacing[0] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_domain_2d() {
        let domain = Domain::new_2d(
            -1.0,
            1.0,
            -2.0,
            2.0,
            51,
            101,
            BoundaryCondition::Absorbing { damping: 0.1 },
        );
        assert_eq!(domain.dimension, SpatialDimension::Two);
        assert_eq!(domain.resolution.len(), 2);
        let spacing = domain.spacing();
        assert!((spacing[0] - 0.04).abs() < 1e-10);
        assert!((spacing[1] - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_condition_types() {
        let bc1 = BoundaryCondition::Dirichlet { value: 1.0 };
        let bc2 = BoundaryCondition::Neumann { flux: 0.5 };
        let bc3 = BoundaryCondition::Periodic;
        assert!(matches!(bc1, BoundaryCondition::Dirichlet { .. }));
        assert!(matches!(bc2, BoundaryCondition::Neumann { .. }));
        assert_eq!(bc3, BoundaryCondition::Periodic);
    }
}
