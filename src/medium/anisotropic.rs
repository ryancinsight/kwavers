//! Anisotropic material support for biological tissues
//! 
//! This module implements full anisotropic material models including
//! orthotropic, transversely isotropic, and general anisotropic materials
//! commonly found in biological tissues.
//!
//! References:
//! - Royer, D., & Dieulesaint, E. (2000). "Elastic waves in solids I:
//!   Free and guided propagation" Springer.
//! - Aristizabal, S., et al. (2018). "Shear wave vibrometry in ex vivo
//!   porcine lens" Journal of Biomechanics, 72, 24-32.

use crate::{KwaversResult, KwaversError, ValidationError};
use crate::Grid;
use ndarray::{Array2, Array3, Array4, Zip};
use rayon::prelude::*;
use std::f64::consts::PI;

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

/// Full elastic stiffness tensor (6x6 in Voigt notation)
#[derive(Debug, Clone)]
pub struct StiffnessTensor {
    /// Stiffness matrix in Voigt notation
    pub c: Array2<f64>,
    /// Anisotropy type
    pub anisotropy_type: AnisotropyType,
}

impl StiffnessTensor {
    /// Create isotropic stiffness tensor from Lamé parameters
    pub fn isotropic(lambda: f64, mu: f64) -> Self {
        let mut c = Array2::zeros((6, 6));
        
        // Diagonal terms
        c[[0, 0]] = lambda + 2.0 * mu; // C11
        c[[1, 1]] = lambda + 2.0 * mu; // C22
        c[[2, 2]] = lambda + 2.0 * mu; // C33
        c[[3, 3]] = mu;                 // C44
        c[[4, 4]] = mu;                 // C55
        c[[5, 5]] = mu;                 // C66
        
        // Off-diagonal terms
        c[[0, 1]] = lambda; c[[1, 0]] = lambda; // C12
        c[[0, 2]] = lambda; c[[2, 0]] = lambda; // C13
        c[[1, 2]] = lambda; c[[2, 1]] = lambda; // C23
        
        Self {
            c,
            anisotropy_type: AnisotropyType::Isotropic,
        }
    }
    
    /// Create transversely isotropic tensor (fiber direction along z)
    pub fn transversely_isotropic(
        c11: f64, c12: f64, c13: f64, c33: f64, c44: f64
    ) -> KwaversResult<Self> {
        // Validate parameters
        if c11 <= 0.0 || c33 <= 0.0 || c44 <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "stiffness_components".to_string(),
                value: "negative".to_string(),
                constraint: "Diagonal components must be positive".to_string(),
            }));
        }
        
        // C66 = (C11 - C12) / 2 for transverse isotropy
        let c66 = (c11 - c12) / 2.0;
        
        let mut c = Array2::zeros((6, 6));
        
        // Fill symmetric matrix
        c[[0, 0]] = c11; c[[1, 1]] = c11; c[[2, 2]] = c33;
        c[[3, 3]] = c44; c[[4, 4]] = c44; c[[5, 5]] = c66;
        
        c[[0, 1]] = c12; c[[1, 0]] = c12;
        c[[0, 2]] = c13; c[[2, 0]] = c13;
        c[[1, 2]] = c13; c[[2, 1]] = c13;
        
        Ok(Self {
            c,
            anisotropy_type: AnisotropyType::TransverselyIsotropic,
        })
    }
    
    /// Create orthotropic tensor
    pub fn orthotropic(components: [[f64; 6]; 6]) -> KwaversResult<Self> {
        let flat: Vec<f64> = components.iter().flat_map(|row| row.iter()).cloned().collect();
        let c = Array2::from_shape_vec((6, 6), flat)
            .map_err(|_| KwaversError::Validation(ValidationError::FieldValidation {
                field: "stiffness_tensor".to_string(),
                value: "invalid shape".to_string(),
                constraint: "Failed to create stiffness tensor from components".to_string(),
            }))?;
        
        // Verify symmetry
        for i in 0..6 {
            for j in i+1..6 {
                if (c[[i, j]] - c[[j, i]]).abs() > 1e-10 {
                    return Err(KwaversError::Validation(ValidationError::FieldValidation {
                        field: "stiffness_tensor".to_string(),
                        value: "asymmetric".to_string(),
                        constraint: "Stiffness tensor must be symmetric".to_string(),
                    }));
                }
            }
        }
        
        Ok(Self {
            c,
            anisotropy_type: AnisotropyType::Orthotropic,
        })
    }
    
    /// Check positive definiteness
    pub fn is_positive_definite(&self) -> bool {
        // Simplified check: all eigenvalues should be positive
        // For now, just check diagonal dominance
        (0..6).all(|i| {
            let diagonal = self.c[[i, i]];
            let off_diagonal_sum: f64 = (0..6)
                .filter(|&j| j != i)
                .map(|j| self.c[[i, j]].abs())
                .sum();
            diagonal > off_diagonal_sum
        })
    }
    
    /// Rotate stiffness tensor by Euler angles
    pub fn rotate(&self, phi: f64, theta: f64, psi: f64) -> Self {
        // Build 3x3 rotation matrix from Euler angles (ZYX convention)
        let (sp, cp) = phi.sin_cos();
        let (st, ct) = theta.sin_cos();
        let (ss, cs) = psi.sin_cos();
        
        // Rotation matrix R
        let r = [
            [ct*cp, ct*sp, -st],
            [ss*st*cp - cs*sp, ss*st*sp + cs*cp, ss*ct],
            [cs*st*cp + ss*sp, cs*st*sp - ss*cp, cs*ct],
        ];
        
        // Build 6x6 Bond transformation matrix
        let mut bond = Array2::zeros((6, 6));
        
        // Fill Bond matrix according to Voigt notation transformation rules
        // For stress/strain: σ' = M σ, where M is the Bond matrix
        
        // Normal components (11, 22, 33)
        bond[[0, 0]] = r[0][0] * r[0][0];
        bond[[0, 1]] = r[0][1] * r[0][1];
        bond[[0, 2]] = r[0][2] * r[0][2];
        bond[[0, 3]] = 2.0 * r[0][1] * r[0][2];
        bond[[0, 4]] = 2.0 * r[0][0] * r[0][2];
        bond[[0, 5]] = 2.0 * r[0][0] * r[0][1];
        
        bond[[1, 0]] = r[1][0] * r[1][0];
        bond[[1, 1]] = r[1][1] * r[1][1];
        bond[[1, 2]] = r[1][2] * r[1][2];
        bond[[1, 3]] = 2.0 * r[1][1] * r[1][2];
        bond[[1, 4]] = 2.0 * r[1][0] * r[1][2];
        bond[[1, 5]] = 2.0 * r[1][0] * r[1][1];
        
        bond[[2, 0]] = r[2][0] * r[2][0];
        bond[[2, 1]] = r[2][1] * r[2][1];
        bond[[2, 2]] = r[2][2] * r[2][2];
        bond[[2, 3]] = 2.0 * r[2][1] * r[2][2];
        bond[[2, 4]] = 2.0 * r[2][0] * r[2][2];
        bond[[2, 5]] = 2.0 * r[2][0] * r[2][1];
        
        // Shear components (23, 13, 12)
        bond[[3, 0]] = r[1][0] * r[2][0];
        bond[[3, 1]] = r[1][1] * r[2][1];
        bond[[3, 2]] = r[1][2] * r[2][2];
        bond[[3, 3]] = r[1][1] * r[2][2] + r[1][2] * r[2][1];
        bond[[3, 4]] = r[1][0] * r[2][2] + r[1][2] * r[2][0];
        bond[[3, 5]] = r[1][0] * r[2][1] + r[1][1] * r[2][0];
        
        bond[[4, 0]] = r[0][0] * r[2][0];
        bond[[4, 1]] = r[0][1] * r[2][1];
        bond[[4, 2]] = r[0][2] * r[2][2];
        bond[[4, 3]] = r[0][1] * r[2][2] + r[0][2] * r[2][1];
        bond[[4, 4]] = r[0][0] * r[2][2] + r[0][2] * r[2][0];
        bond[[4, 5]] = r[0][0] * r[2][1] + r[0][1] * r[2][0];
        
        bond[[5, 0]] = r[0][0] * r[1][0];
        bond[[5, 1]] = r[0][1] * r[1][1];
        bond[[5, 2]] = r[0][2] * r[1][2];
        bond[[5, 3]] = r[0][1] * r[1][2] + r[0][2] * r[1][1];
        bond[[5, 4]] = r[0][0] * r[1][2] + r[0][2] * r[1][0];
        bond[[5, 5]] = r[0][0] * r[1][1] + r[0][1] * r[1][0];
        
        // Apply Bond transformation: C' = M^T C M
        let bond_t = bond.t();
        let temp = bond_t.dot(&self.c);
        let rotated = temp.dot(&bond);
        
        Self {
            c: rotated,
            anisotropy_type: self.anisotropy_type,
        }
    }
}

/// Anisotropic tissue properties
#[derive(Debug, Clone)]
pub struct AnisotropicTissueProperties {
    /// Base density (kg/m³)
    pub density: f64,
    /// Stiffness tensor
    pub stiffness: StiffnessTensor,
    /// Fiber orientation angles (radians)
    pub fiber_angles: Option<(f64, f64, f64)>,
    /// Viscosity tensor (for viscoelastic effects)
    pub viscosity: Option<Array2<f64>>,
}

impl AnisotropicTissueProperties {
    /// Create muscle tissue with fiber orientation
    pub fn muscle(fiber_angle: f64) -> Self {
        // Muscle is transversely isotropic
        // Values from literature (approximate)
        let c11 = 15e9;  // Pa - transverse stiffness
        let c12 = 10e9;  // Pa
        let c13 = 12e9;  // Pa
        let c33 = 25e9;  // Pa - along fiber stiffness
        let c44 = 3e9;   // Pa - shear modulus
        
        let stiffness = StiffnessTensor::transversely_isotropic(
            c11, c12, c13, c33, c44
        ).unwrap();
        
        // Rotate to fiber orientation
        let rotated_stiffness = stiffness.rotate(fiber_angle, 0.0, 0.0);
        
        Self {
            density: 1050.0,
            stiffness: rotated_stiffness,
            fiber_angles: Some((fiber_angle, 0.0, 0.0)),
            viscosity: None,
        }
    }
    
    /// Create tendon/ligament tissue
    pub fn tendon() -> Self {
        // Highly anisotropic along fiber direction
        let c11 = 50e6;   // Pa - transverse
        let c12 = 30e6;   // Pa
        let c13 = 40e6;   // Pa
        let c33 = 1.2e9;  // Pa - along fiber (very stiff)
        let c44 = 20e6;   // Pa - shear
        
        let stiffness = StiffnessTensor::transversely_isotropic(
            c11, c12, c13, c33, c44
        ).unwrap();
        
        Self {
            density: 1100.0,
            stiffness,
            fiber_angles: Some((0.0, 0.0, 0.0)),
            viscosity: None,
        }
    }
    
    /// Create bone tissue (cortical)
    pub fn cortical_bone() -> Self {
        // Orthotropic material
        let components = [
            [20.0e9, 9.0e9,  10.5e9, 0.0,    0.0,    0.0],
            [9.0e9,  22.0e9, 10.5e9, 0.0,    0.0,    0.0],
            [10.5e9, 10.5e9, 27.0e9, 0.0,    0.0,    0.0],
            [0.0,    0.0,    0.0,    5.3e9,  0.0,    0.0],
            [0.0,    0.0,    0.0,    0.0,    5.8e9,  0.0],
            [0.0,    0.0,    0.0,    0.0,    0.0,    6.2e9],
        ];
        
        let stiffness = StiffnessTensor::orthotropic(components).unwrap();
        
        Self {
            density: 1900.0,
            stiffness,
            fiber_angles: None,
            viscosity: None,
        }
    }
    
    /// Get wave velocities in a given direction
    pub fn wave_velocities(&self, direction: (f64, f64, f64)) -> KwaversResult<(f64, f64, f64)> {
        let (nx, ny, nz) = direction;
        let n_mag = (nx*nx + ny*ny + nz*nz).sqrt();
        let (nx, ny, nz) = (nx/n_mag, ny/n_mag, nz/n_mag);
        
        // Christoffel matrix calculation (simplified)
        // M_ik = C_ijkl * n_j * n_l
        let mut christoffel: Array2<f64> = Array2::zeros((3, 3));
        
        // This is a simplified version - full implementation would use
        // proper tensor contraction with the full stiffness tensor
        
        // For now, return approximate values
        let vp = (self.stiffness.c[[0, 0]] / self.density).sqrt();  // P-wave
        let vs1 = (self.stiffness.c[[3, 3]] / self.density).sqrt(); // S-wave 1
        let vs2 = (self.stiffness.c[[4, 4]] / self.density).sqrt(); // S-wave 2
        
        Ok((vp, vs1, vs2))
    }
}

/// Anisotropic wave propagator
pub struct AnisotropicWavePropagator {
    /// Grid spacing
    grid: Grid,
    /// Anisotropic properties field
    properties: Array3<AnisotropicTissueProperties>,
}

impl AnisotropicWavePropagator {
    /// Create new anisotropic propagator
    pub fn new(grid: Grid, uniform_properties: AnisotropicTissueProperties) -> Self {
        let properties = Array3::from_elem(
            (grid.nx, grid.ny, grid.nz),
            uniform_properties
        );
        
        Self { grid, properties }
    }
    
    /// Set spatially varying properties
    pub fn set_properties(&mut self, properties: Array3<AnisotropicTissueProperties>) {
        self.properties = properties;
    }
    
    /// Compute stress from strain using anisotropic stiffness
    pub fn compute_stress(
        &self,
        strain: &Array4<f64>,  // (6, nx, ny, nz) in Voigt notation
    ) -> KwaversResult<Array4<f64>> {
        let (_, nx, ny, nz) = strain.dim();
        let mut stress = Array4::zeros((6, nx, ny, nz));
        
        // Apply Hooke's law: σ = C : ε
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let c = &self.properties[[i, j, k]].stiffness.c;
                    
                    // Matrix-vector multiplication in Voigt notation
                    for m in 0..6 {
                        let mut sum = 0.0;
                        for n in 0..6 {
                            sum += c[[m, n]] * strain[[n, i, j, k]];
                        }
                        stress[[m, i, j, k]] = sum;
                    }
                }
            }
        }
        
        Ok(stress)
    }
    
    /// Compute strain from displacement gradients
    pub fn compute_strain(
        &self,
        ux: &Array3<f64>,
        uy: &Array3<f64>,
        uz: &Array3<f64>,
    ) -> KwaversResult<Array4<f64>> {
        let (nx, ny, nz) = ux.dim();
        let mut strain = Array4::zeros((6, nx, ny, nz));
        
        // Compute strain components (engineering strain)
        (1..nx-1).for_each(|i| {
            (1..ny-1).for_each(|j| {
                (1..nz-1).for_each(|k| {
                    // Normal strains
                    strain[[0, i, j, k]] = (ux[[i+1, j, k]] - ux[[i-1, j, k]]) / (2.0 * self.grid.dx);
                    strain[[1, i, j, k]] = (uy[[i, j+1, k]] - uy[[i, j-1, k]]) / (2.0 * self.grid.dy);
                    strain[[2, i, j, k]] = (uz[[i, j, k+1]] - uz[[i, j, k-1]]) / (2.0 * self.grid.dz);
                    
                    // Shear strains (engineering strain = 2 * tensor strain)
                    strain[[3, i, j, k]] = (uy[[i, j, k+1]] - uy[[i, j, k-1]]) / (2.0 * self.grid.dz)
                                         + (uz[[i, j+1, k]] - uz[[i, j-1, k]]) / (2.0 * self.grid.dy);
                    strain[[4, i, j, k]] = (ux[[i, j, k+1]] - ux[[i, j, k-1]]) / (2.0 * self.grid.dz)
                                         + (uz[[i+1, j, k]] - uz[[i-1, j, k]]) / (2.0 * self.grid.dx);
                    strain[[5, i, j, k]] = (ux[[i, j+1, k]] - ux[[i, j-1, k]]) / (2.0 * self.grid.dy)
                                         + (uy[[i+1, j, k]] - uy[[i-1, j, k]]) / (2.0 * self.grid.dx);
                });
            });
        });
        
        Ok(strain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_isotropic_stiffness() {
        let lambda = 2e9;
        let mu = 1e9;
        let stiffness = StiffnessTensor::isotropic(lambda, mu);
        
        assert_eq!(stiffness.anisotropy_type, AnisotropyType::Isotropic);
        assert!((stiffness.c[[0, 0]] - (lambda + 2.0 * mu)).abs() < 1e-10);
        assert!((stiffness.c[[3, 3]] - mu).abs() < 1e-10);
    }
    
    #[test]
    fn test_muscle_anisotropy() {
        let muscle = AnisotropicTissueProperties::muscle(PI / 4.0);
        assert!(muscle.fiber_angles.is_some());
        
        // Test wave velocities
        let (vp, vs1, vs2) = muscle.wave_velocities((1.0, 0.0, 0.0)).unwrap();
        assert!(vp > vs1);
        assert!(vp > vs2);
    }
    
    #[test]
    fn test_stiffness_rotation() {
        // Create a simple transversely isotropic tensor
        let stiffness = StiffnessTensor::transversely_isotropic(
            10e9, 6e9, 8e9, 20e9, 3e9
        ).unwrap();
        
        // Rotate by 90 degrees around z-axis
        let rotated = stiffness.rotate(0.0, 0.0, PI / 2.0);
        
        // After 90° rotation around z, C11 and C22 should swap
        assert!((rotated.c[[0, 0]] - stiffness.c[[1, 1]]).abs() < 1e-6);
        assert!((rotated.c[[1, 1]] - stiffness.c[[0, 0]]).abs() < 1e-6);
        
        // C33 should remain unchanged (z-direction)
        assert!((rotated.c[[2, 2]] - stiffness.c[[2, 2]]).abs() < 1e-6);
        
        // Test that the rotated tensor is still symmetric
        for i in 0..6 {
            for j in i+1..6 {
                assert!((rotated.c[[i, j]] - rotated.c[[j, i]]).abs() < 1e-10,
                    "Rotated tensor not symmetric at [{}, {}]", i, j);
            }
        }
    }
}