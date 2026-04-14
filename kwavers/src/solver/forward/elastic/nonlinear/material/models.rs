use super::energy::{
    compute_strain_energy_derivative_wrt_i1, compute_strain_energy_derivative_wrt_i2,
    compute_strain_energy_derivative_wrt_j, ogden_strain_energy, strain_energy,
};
use super::invariants::{matrix_eigenvalues, principal_stretches};
use super::stress::{cauchy_stress, first_pk_stress};

/// Hyperelastic material models for nonlinear elasticity
///
/// # Theorem Reference
/// Hyperelastic materials are characterized by strain energy density functions W(I₁,I₂,J)
/// where I₁,I₂ are strain invariants and J is the volume ratio.
///
/// **Neo-Hookean Theorem**: For isotropic hyperelastic materials with limited compressibility,
/// W = C₁(I₁ - 3) + D₁(J - 1)² where C₁ = μ/2, D₁ = λ/(4μ) + λ/(4μ)²
/// (Reference: Holzapfel, 2000, Chapter 6)
///
/// **Mooney-Rivlin Theorem**: Extension of Neo-Hookean for better fit to experimental data,
/// W = C₁(I₁ - 3) + C₂(I₂ - 3) + D₁(J - 1)²
/// (Reference: Mooney, 1940; Rivlin, 1948)
///
/// **Ogden Theorem**: Principal stretch-based formulation for anisotropic materials,
/// W = Σᵢ (μᵢ/αᵢ) [(λ₁^αⁱ + λ₂^αⁱ + λ₃^αⁱ - 3)]
/// Provides better accuracy for large deformations.
/// (Reference: Ogden, 1972, 1984)
#[derive(Debug, Clone)]
pub enum HyperelasticModel {
    /// Neo-Hookean model: W = C₁(I₁ - 3) + D₁(J - 1)²
    /// # Theorem: Simplest hyperelastic model for nearly incompressible materials
    /// # Assumptions: Isotropic, small strains, limited compressibility
    /// # Applications: Soft tissues, elastomers at moderate strains
    NeoHookean {
        /// Material parameter C₁ = μ/2 (Pa)
        c1: f64,
        /// Compressibility parameter D₁ (Pa⁻¹)
        d1: f64,
    },
    /// Mooney-Rivlin model: W = C₁(I₁ - 3) + C₂(I₂ - 3) + D₁(J - 1)²
    /// # Theorem: Two-parameter invariant-based model for improved accuracy
    /// # Assumptions: Isotropic, incompressible limit valid, I₂ term accounts for strain history
    /// # Applications: Rubber-like materials, biological tissues
    MooneyRivlin {
        /// Material parameter C₁ (Pa)
        c1: f64,
        /// Material parameter C₂ (Pa)
        c2: f64,
        /// Compressibility parameter D₁ (Pa⁻¹)
        d1: f64,
    },
    /// Ogden model: W = Σᵢ (μᵢ/αᵢ) [(λ₁^αⁱ + λ₂^αⁱ + λ₃^αⁱ - 3)]
    /// # Theorem: Principal stretch formulation for large deformations and anisotropy
    /// # Assumptions: Material behavior depends on principal stretches λᵢ, not invariants
    /// # Advantages: Better accuracy for large strains, handles anisotropy naturally
    /// # Applications: Biological tissues, composites, materials under extreme loading
    Ogden {
        /// Shear moduli μᵢ for each term (Pa)
        mu: Vec<f64>,
        /// Exponents αᵢ for each term (dimensionless)
        alpha: Vec<f64>,
    },
}

impl HyperelasticModel {
    /// Create Neo-Hookean model for soft tissue
    #[must_use]
    pub fn neo_hookean_soft_tissue() -> Self {
        // Typical values for soft tissue (liver, muscle)
        Self::NeoHookean {
            c1: 1e3,  // Pa
            d1: 1e-4, // Pa⁻¹ (nearly incompressible)
        }
    }

    /// Create Mooney-Rivlin model for biological tissue
    #[must_use]
    pub fn mooney_rivlin_biological() -> Self {
        // Enhanced model for biological tissues
        Self::MooneyRivlin {
            c1: 1e3,   // Pa
            c2: 0.1e3, // Pa
            d1: 1e-4,  // Pa⁻¹
        }
    }

    #[must_use]
    pub fn strain_energy(&self, i1: f64, i2: f64, j: f64) -> f64 {
        strain_energy(self, i1, i2, j)
    }

    #[must_use]
    pub fn ogden_strain_energy(&self, f: &[[f64; 3]; 3]) -> f64 {
        ogden_strain_energy(self, f)
    }

    #[must_use]
    pub fn cauchy_stress(&self, f: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        cauchy_stress(self, f)
    }

    #[must_use]
    pub fn first_pk_stress(&self, f: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        first_pk_stress(self, f)
    }

    #[must_use]
    pub fn principal_stretches(&self, f: &[[f64; 3]; 3]) -> [f64; 3] {
        principal_stretches(self, f)
    }

    #[must_use]
    pub fn matrix_eigenvalues(&self, m: &[[f64; 3]; 3]) -> [f64; 3] {
        matrix_eigenvalues(m)
    }

    #[must_use]
    pub fn compute_strain_energy_derivative_wrt_i1(
        &self,
        i1: f64,
        i2: f64,
        j: f64,
        f: Option<&[[f64; 3]; 3]>,
    ) -> f64 {
        compute_strain_energy_derivative_wrt_i1(self, i1, i2, j, f)
    }

    #[must_use]
    pub fn compute_strain_energy_derivative_wrt_i2(&self, i1: f64, i2: f64, j: f64) -> f64 {
        compute_strain_energy_derivative_wrt_i2(self, i1, i2, j)
    }

    #[must_use]
    pub fn compute_strain_energy_derivative_wrt_j(&self, i1: f64, i2: f64, j: f64) -> f64 {
        compute_strain_energy_derivative_wrt_j(self, i1, i2, j)
    }
}
