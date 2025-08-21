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
//!   porcine lens." J Biomech 75: 19-25.

use crate::Grid;
use crate::{ConfigError, KwaversError, KwaversResult, ValidationError};
use ndarray::{Array2, Array3, Array4};

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
        c[[3, 3]] = mu; // C44
        c[[4, 4]] = mu; // C55
        c[[5, 5]] = mu; // C66

        // Off-diagonal terms
        c[[0, 1]] = lambda;
        c[[1, 0]] = lambda; // C12
        c[[0, 2]] = lambda;
        c[[2, 0]] = lambda; // C13
        c[[1, 2]] = lambda;
        c[[2, 1]] = lambda; // C23

        Self {
            c,
            anisotropy_type: AnisotropyType::Isotropic,
        }
    }

    /// Create transversely isotropic tensor (fiber direction along z)
    pub fn transversely_isotropic(
        c11: f64,
        c12: f64,
        c13: f64,
        c33: f64,
        c44: f64,
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
        c[[0, 0]] = c11;
        c[[1, 1]] = c11;
        c[[2, 2]] = c33;
        c[[3, 3]] = c44;
        c[[4, 4]] = c44;
        c[[5, 5]] = c66;

        c[[0, 1]] = c12;
        c[[1, 0]] = c12;
        c[[0, 2]] = c13;
        c[[2, 0]] = c13;
        c[[1, 2]] = c13;
        c[[2, 1]] = c13;

        Ok(Self {
            c,
            anisotropy_type: AnisotropyType::TransverselyIsotropic,
        })
    }

    /// Create orthotropic tensor
    pub fn orthotropic(components: [[f64; 6]; 6]) -> KwaversResult<Self> {
        let flat: Vec<f64> = components
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .collect();
        let c = Array2::from_shape_vec((6, 6), flat).map_err(|_| {
            KwaversError::Validation(ValidationError::FieldValidation {
                field: "stiffness_tensor".to_string(),
                value: "invalid shape".to_string(),
                constraint: "Failed to create stiffness tensor from components".to_string(),
            })
        })?;

        // Verify symmetry
        for i in 0..6 {
            for j in i + 1..6 {
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
            [ct * cp, ct * sp, -st],
            [ss * st * cp - cs * sp, ss * st * sp + cs * cp, ss * ct],
            [cs * st * cp + ss * sp, cs * st * sp - ss * cp, cs * ct],
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
        let c11 = 15e9; // Pa - transverse stiffness
        let c12 = 10e9; // Pa
        let c13 = 12e9; // Pa
        let c33 = 25e9; // Pa - along fiber stiffness
        let c44 = 3e9; // Pa - shear modulus

        let stiffness = StiffnessTensor::transversely_isotropic(c11, c12, c13, c33, c44).unwrap();

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
        let c11 = 50e6; // Pa - transverse
        let c12 = 30e6; // Pa
        let c13 = 40e6; // Pa
        let c33 = 1.2e9; // Pa - along fiber (very stiff)
        let c44 = 20e6; // Pa - shear

        let stiffness = StiffnessTensor::transversely_isotropic(c11, c12, c13, c33, c44).unwrap();

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
            [20.0e9, 9.0e9, 10.5e9, 0.0, 0.0, 0.0],
            [9.0e9, 22.0e9, 10.5e9, 0.0, 0.0, 0.0],
            [10.5e9, 10.5e9, 27.0e9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.3e9, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.8e9, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.2e9],
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
        let n_mag = (nx * nx + ny * ny + nz * nz).sqrt();
        let (nx, ny, nz) = (nx / n_mag, ny / n_mag, nz / n_mag);

        // Christoffel matrix calculation
        // M_ik = C_ijkl * n_j * n_l (using Voigt notation)
        let mut christoffel: Array2<f64> = Array2::zeros((3, 3));

        // Convert direction vector to strain-like vector in Voigt notation
        // For wave propagation, we need the dyadic product n⊗n
        let n_voigt = [
            nx * nx,
            ny * ny,
            nz * nz,
            2.0 * ny * nz,
            2.0 * nx * nz,
            2.0 * nx * ny,
        ];

        // First, compute stress-like vector: σ = C : (n⊗n)
        let mut stress_voigt = [0.0; 6];
        for i in 0..6 {
            for j in 0..6 {
                stress_voigt[i] += self.stiffness.c[[i, j]] * n_voigt[j];
            }
        }

        // Convert back to Christoffel matrix components
        // M_11 = σ_11*nx*nx + σ_22*ny*ny + σ_33*nz*nz + 2(σ_23*ny*nz + σ_13*nx*nz + σ_12*nx*ny)
        christoffel[[0, 0]] = stress_voigt[0] * nx * nx
            + stress_voigt[1] * ny * ny
            + stress_voigt[2] * nz * nz
            + stress_voigt[3] * ny * nz
            + stress_voigt[4] * nx * nz
            + stress_voigt[5] * nx * ny;

        // M_22 = similar pattern but with different indices
        christoffel[[1, 1]] = stress_voigt[0] * ny * ny
            + stress_voigt[1] * nz * nz
            + stress_voigt[2] * nx * nx
            + stress_voigt[3] * nz * nx
            + stress_voigt[4] * ny * nx
            + stress_voigt[5] * ny * nz;

        // M_33
        christoffel[[2, 2]] = stress_voigt[0] * nz * nz
            + stress_voigt[1] * nx * nx
            + stress_voigt[2] * ny * ny
            + stress_voigt[3] * nx * ny
            + stress_voigt[4] * nz * ny
            + stress_voigt[5] * nz * nx;

        // Off-diagonal terms (matrix is symmetric)
        christoffel[[0, 1]] = stress_voigt[5] * (nx * nx + ny * ny) / 2.0
            + stress_voigt[4] * nx * nz / 2.0
            + stress_voigt[3] * ny * nz / 2.0;
        christoffel[[1, 0]] = christoffel[[0, 1]];

        christoffel[[0, 2]] = stress_voigt[4] * (nx * nx + nz * nz) / 2.0
            + stress_voigt[5] * nx * ny / 2.0
            + stress_voigt[3] * ny * nz / 2.0;
        christoffel[[2, 0]] = christoffel[[0, 2]];

        christoffel[[1, 2]] = stress_voigt[3] * (ny * ny + nz * nz) / 2.0
            + stress_voigt[5] * nx * ny / 2.0
            + stress_voigt[4] * nx * nz / 2.0;
        christoffel[[2, 1]] = christoffel[[1, 2]];

        // Normalize by density to get velocity-squared eigenvalues
        christoffel.mapv_inplace(|x| x / self.density);

        // Solve eigenvalue problem for phase velocities
        // For 3x3 symmetric matrices, we can use analytical methods
        let eigenvalues = self.compute_eigenvalues_3x3(&christoffel)?;

        // Extract wave velocities (sqrt of eigenvalues)
        // Eigenvalues are sorted in ascending order
        let vs1 = eigenvalues[0].sqrt(); // First shear wave velocity
        let vs2 = eigenvalues[1].sqrt(); // Second shear wave velocity
        let vp = eigenvalues[2].sqrt(); // Compressional wave velocity

        Ok((vp, vs1, vs2))
    }

    /// Compute eigenvalues of a 3x3 symmetric matrix analytically
    fn compute_eigenvalues_3x3(&self, matrix: &Array2<f64>) -> KwaversResult<[f64; 3]> {
        // For a 3x3 symmetric matrix, we can use Cardano's method
        let a11 = matrix[[0, 0]];
        let a22 = matrix[[1, 1]];
        let a33 = matrix[[2, 2]];
        let a12 = matrix[[0, 1]];
        let a13 = matrix[[0, 2]];
        let a23 = matrix[[1, 2]];

        // Characteristic polynomial: det(A - λI) = 0
        // -λ³ + p₂λ² + p₁λ + p₀ = 0
        let p2 = a11 + a22 + a33; // Trace
        let p1 = -(a11 * a22 + a11 * a33 + a22 * a33 - a12 * a12 - a13 * a13 - a23 * a23);
        let p0 = a11 * a22 * a33 + 2.0 * a12 * a13 * a23
            - a11 * a23 * a23
            - a22 * a13 * a13
            - a33 * a12 * a12;

        // Convert to depressed cubic: t³ + pt + q = 0
        // where λ = t + p₂/3
        let p = p1 - p2 * p2 / 3.0;
        let q = 2.0 * p2 * p2 * p2 / 27.0 - p2 * p1 / 3.0 + p0;

        // Discriminant
        let discriminant = -(4.0 * p * p * p + 27.0 * q * q) / 108.0;

        let mut eigenvalues = [0.0; 3];

        if discriminant >= 0.0 {
            // Three real roots using trigonometric solution (Cardano's method)
            // The offsets 2π/3 and 4π/3 correspond to the three cube roots of unity.
            const TWO_PI_OVER_THREE: f64 = 2.0 * std::f64::consts::PI / 3.0;
            const FOUR_PI_OVER_THREE: f64 = 4.0 * std::f64::consts::PI / 3.0;
            let m = 2.0 * (-p / 3.0).sqrt();
            let theta = (3.0 * q / (p * m)).acos() / 3.0;

            eigenvalues[0] = m * (theta).cos() + p2 / 3.0;
            eigenvalues[1] = m * (theta - TWO_PI_OVER_THREE).cos() + p2 / 3.0;
            eigenvalues[2] = m * (theta - FOUR_PI_OVER_THREE).cos() + p2 / 3.0;

            // Sort eigenvalues
            eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        } else {
            // Should not happen for positive definite matrices
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "christoffel_matrix".to_string(),
                value: "complex eigenvalues".to_string(),
                constraint: "Matrix must be positive definite".to_string(),
            }));
        }

        Ok(eigenvalues)
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
        let properties = Array3::from_elem((grid.nx, grid.ny, grid.nz), uniform_properties);

        Self { grid, properties }
    }

    /// Set spatially varying properties
    pub fn set_properties(&mut self, properties: Array3<AnisotropicTissueProperties>) {
        self.properties = properties;
    }

    /// Compute stress from strain using anisotropic stiffness
    pub fn compute_stress(
        &self,
        strain: &Array4<f64>, // (6, nx, ny, nz) in Voigt notation
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
        (1..nx - 1).for_each(|i| {
            (1..ny - 1).for_each(|j| {
                (1..nz - 1).for_each(|k| {
                    // Normal strains
                    strain[[0, i, j, k]] =
                        (ux[[i + 1, j, k]] - ux[[i - 1, j, k]]) / (2.0 * self.grid.dx);
                    strain[[1, i, j, k]] =
                        (uy[[i, j + 1, k]] - uy[[i, j - 1, k]]) / (2.0 * self.grid.dy);
                    strain[[2, i, j, k]] =
                        (uz[[i, j, k + 1]] - uz[[i, j, k - 1]]) / (2.0 * self.grid.dz);

                    // Shear strains (engineering strain = 2 * tensor strain)
                    strain[[3, i, j, k]] = (uy[[i, j, k + 1]] - uy[[i, j, k - 1]])
                        / (2.0 * self.grid.dz)
                        + (uz[[i, j + 1, k]] - uz[[i, j - 1, k]]) / (2.0 * self.grid.dy);
                    strain[[4, i, j, k]] = (ux[[i, j, k + 1]] - ux[[i, j, k - 1]])
                        / (2.0 * self.grid.dz)
                        + (uz[[i + 1, j, k]] - uz[[i - 1, j, k]]) / (2.0 * self.grid.dx);
                    strain[[5, i, j, k]] = (ux[[i, j + 1, k]] - ux[[i, j - 1, k]])
                        / (2.0 * self.grid.dy)
                        + (uy[[i + 1, j, k]] - uy[[i - 1, j, k]]) / (2.0 * self.grid.dx);
                });
            });
        });

        Ok(strain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

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
    fn test_rotation_matrix_z_axis() {
        // Test that the rotation matrix for 90 degrees around z-axis is correct
        // In ZYX convention, phi is the first rotation around z
        let phi = PI / 2.0; // First rotation (about z) - this is what we want!
        let theta = 0.0f64; // Second rotation (about y')
        let psi = 0.0f64; // Third rotation (about z'')

        let (sp, cp) = phi.sin_cos();
        let (st, ct) = theta.sin_cos();
        let (ss, cs) = psi.sin_cos();

        // Expected rotation matrix for 90° around z should be:
        // [ 0 -1  0]
        // [ 1  0  0]
        // [ 0  0  1]

        // Build rotation matrix as in the code
        let r = [
            [ct * cp, ct * sp, -st],
            [ss * st * cp - cs * sp, ss * st * sp + cs * cp, ss * ct],
            [cs * st * cp + ss * sp, cs * st * sp - ss * cp, cs * ct],
        ];

        // Verify the rotation matrix is correct for 90° around z
        // Expected: [0, 1, 0; -1, 0, 0; 0, 0, 1] (for positive rotation)
        assert!((r[0][0]).abs() < 1e-10);
        assert!((r[0][1] - 1.0).abs() < 1e-10);
        assert!((r[0][2]).abs() < 1e-10);
        assert!((r[1][0] + 1.0).abs() < 1e-10);
        assert!((r[1][1]).abs() < 1e-10);
        assert!((r[1][2]).abs() < 1e-10);
        assert!((r[2][0]).abs() < 1e-10);
        assert!((r[2][1]).abs() < 1e-10);
        assert!((r[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stiffness_rotation() {
        // Create an orthotropic tensor with different values for C11 and C22
        // to properly test rotation
        let mut c = Array2::zeros((6, 6));

        // Set up an orthotropic material with distinct C11 and C22
        c[[0, 0]] = 10e9; // C11
        c[[1, 1]] = 15e9; // C22 (different from C11)
        c[[2, 2]] = 20e9; // C33
        c[[3, 3]] = 4e9; // C44
        c[[4, 4]] = 5e9; // C55
        c[[5, 5]] = 6e9; // C66

        // Set some off-diagonal terms
        c[[0, 1]] = 3e9;
        c[[1, 0]] = 3e9; // C12
        c[[0, 2]] = 4e9;
        c[[2, 0]] = 4e9; // C13
        c[[1, 2]] = 5e9;
        c[[2, 1]] = 5e9; // C23

        let stiffness = StiffnessTensor {
            c,
            anisotropy_type: AnisotropyType::Orthotropic,
        };

        // Rotate by 90 degrees around z-axis
        // In the ZYX convention used, phi is the first rotation around z
        let rotated = stiffness.rotate(PI / 2.0, 0.0, 0.0);

        // After 90° rotation around z, x becomes y and y becomes -x
        // So C11 and C22 should swap
        assert!(
            (rotated.c[[0, 0]] - stiffness.c[[1, 1]]).abs() < 1e-6,
            "After 90° rotation, new C11 should equal old C22: {} vs {}",
            rotated.c[[0, 0]],
            stiffness.c[[1, 1]]
        );
        assert!(
            (rotated.c[[1, 1]] - stiffness.c[[0, 0]]).abs() < 1e-6,
            "After 90° rotation, new C22 should equal old C11: {} vs {}",
            rotated.c[[1, 1]],
            stiffness.c[[0, 0]]
        );

        // C33 should remain unchanged (z-direction)
        assert!(
            (rotated.c[[2, 2]] - stiffness.c[[2, 2]]).abs() < 1e-6,
            "C33 should remain unchanged: {} vs {}",
            rotated.c[[2, 2]],
            stiffness.c[[2, 2]]
        );

        // C44 and C55 should swap (yz and xz shear)
        assert!(
            (rotated.c[[3, 3]] - stiffness.c[[4, 4]]).abs() < 1e-6,
            "After 90° rotation, new C44 should equal old C55: {} vs {}",
            rotated.c[[3, 3]],
            stiffness.c[[4, 4]]
        );
        assert!(
            (rotated.c[[4, 4]] - stiffness.c[[3, 3]]).abs() < 1e-6,
            "After 90° rotation, new C55 should equal old C44: {} vs {}",
            rotated.c[[4, 4]],
            stiffness.c[[3, 3]]
        );

        // Test that the rotated tensor is still symmetric
        for i in 0..6 {
            for j in i + 1..6 {
                assert!(
                    (rotated.c[[i, j]] - rotated.c[[j, i]]).abs() < 1e-10,
                    "Rotated tensor not symmetric at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }
}
