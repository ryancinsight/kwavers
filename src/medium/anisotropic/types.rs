//! Core types for anisotropic materials

/// Anisotropic material types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnisotropyType {
    /// Isotropic (no directional dependence)
    Isotropic,
    /// Transversely isotropic (fiber-like, e.g., muscle)
    TransverselyIsotropic,
    /// Orthotropic (three orthogonal symmetry planes)
    Orthotropic,
    /// General anisotropic (no symmetry)
    General,
}

impl AnisotropyType {
    /// Number of independent elastic constants
    pub fn num_constants(&self) -> usize {
        match self {
            Self::Isotropic => 2,             // λ and μ (or E and ν)
            Self::TransverselyIsotropic => 5, // C11, C12, C13, C33, C44
            Self::Orthotropic => 9,           // C11, C22, C33, C12, C13, C23, C44, C55, C66
            Self::General => 21,              // Full symmetric 6x6 matrix
        }
    }

    /// Check if material has symmetry
    pub fn has_symmetry(&self) -> bool {
        !matches!(self, Self::General)
    }

    /// Get symmetry planes
    pub fn symmetry_planes(&self) -> Vec<&str> {
        match self {
            Self::Isotropic => vec!["all"],
            Self::TransverselyIsotropic => vec!["xy"],
            Self::Orthotropic => vec!["xy", "xz", "yz"],
            Self::General => vec![],
        }
    }
}
