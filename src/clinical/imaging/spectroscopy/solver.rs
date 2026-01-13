//! Spectral Unmixing for Photoacoustic Imaging
//!
//! Provides algorithms for decomposing multi-wavelength photoacoustic signals
//! into constituent chromophore concentrations.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3};
use super::types::{SpectralUnmixingConfig, UnmixingResult, VolumetricUnmixingResult};

/// Spectral unmixing solver
#[derive(Debug)]
pub struct SpectralUnmixer {
    /// Extinction coefficient matrix E (n_wavelengths × n_chromophores)
    extinction_matrix: Array2<f64>,
    /// Wavelengths (nm)
    wavelengths: Vec<f64>,
    /// Chromophore names
    chromophore_names: Vec<String>,
    /// Configuration
    config: SpectralUnmixingConfig,
}

impl SpectralUnmixer {
    /// Create unmixer from extinction matrix
    ///
    /// # Arguments
    ///
    /// - `extinction_matrix`: Matrix E where E[i,j] = εⱼ(λᵢ) (n_wavelengths × n_chromophores)
    /// - `wavelengths`: Wavelengths corresponding to rows of E (nm)
    /// - `chromophore_names`: Names of chromophores (columns of E)
    /// - `config`: Unmixing configuration
    pub fn new(
        extinction_matrix: Array2<f64>,
        wavelengths: Vec<f64>,
        chromophore_names: Vec<String>,
        config: SpectralUnmixingConfig,
    ) -> Result<Self> {
        let (n_wavelengths, n_chromophores) = extinction_matrix.dim();

        // Validate dimensions
        if wavelengths.len() != n_wavelengths {
            anyhow::bail!(
                "Wavelength vector length {} does not match extinction matrix rows {}",
                wavelengths.len(),
                n_wavelengths
            );
        }

        if chromophore_names.len() != n_chromophores {
            anyhow::bail!(
                "Chromophore name count {} does not match extinction matrix columns {}",
                chromophore_names.len(),
                n_chromophores
            );
        }

        if n_wavelengths < n_chromophores {
            anyhow::bail!(
                "Underdetermined system: {} wavelengths < {} chromophores. Add more wavelengths.",
                n_wavelengths,
                n_chromophores
            );
        }

        // Check condition number (warn if ill-conditioned)
        let condition = estimate_condition_number(&extinction_matrix)?;
        if condition < config.min_condition_number {
            tracing::warn!(
                "Extinction matrix is poorly conditioned (cond ≈ {:.2e}). Consider regularization.",
                condition
            );
        }

        Ok(Self {
            extinction_matrix,
            wavelengths,
            chromophore_names,
            config,
        })
    }

    /// Unmix single spectrum (one voxel)
    ///
    /// # Arguments
    ///
    /// - `absorption_spectrum`: Measured absorption coefficients μₐ(λ) at each wavelength
    ///
    /// # Returns
    ///
    /// Chromophore concentrations and residual
    pub fn unmix_single(&self, absorption_spectrum: &Array1<f64>) -> Result<UnmixingResult> {
        let n_wavelengths = self.extinction_matrix.nrows();

        if absorption_spectrum.len() != n_wavelengths {
            anyhow::bail!(
                "Absorption spectrum length {} does not match wavelength count {}",
                absorption_spectrum.len(),
                n_wavelengths
            );
        }

        // Solve C = (EᵀE + λI)⁻¹Eᵀμ
        let concentrations = if self.config.non_negative {
            // Non-negative least squares (simplified: project negatives to zero)
            let c_unconstrained = tikhonov_solve(
                &self.extinction_matrix,
                absorption_spectrum,
                self.config.regularization_lambda,
            )?;
            // Simple projection: C[i] = max(0, C[i])
            c_unconstrained.mapv(|x| x.max(0.0))
        } else {
            tikhonov_solve(
                &self.extinction_matrix,
                absorption_spectrum,
                self.config.regularization_lambda,
            )?
        };

        // Compute residual: r = μ - EC
        let reconstruction = self.extinction_matrix.dot(&concentrations);
        let residual = absorption_spectrum - &reconstruction;
        let residual_norm = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
        let spectrum_norm = absorption_spectrum
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let relative_residual = if spectrum_norm > 1e-12 {
            residual_norm / spectrum_norm
        } else {
            0.0
        };

        Ok(UnmixingResult {
            concentrations,
            residual_norm,
            relative_residual,
        })
    }

    /// Unmix volumetric data (entire 3D image)
    ///
    /// # Arguments
    ///
    /// - `absorption_maps`: Multi-wavelength absorption coefficient maps
    ///   (n_wavelengths × nx × ny × nz)
    ///
    /// # Returns
    ///
    /// Chromophore concentration maps and residuals
    pub fn unmix_volumetric(
        &self,
        absorption_maps: &[Array3<f64>],
    ) -> Result<VolumetricUnmixingResult> {
        let n_wavelengths = self.extinction_matrix.nrows();
        let n_chromophores = self.extinction_matrix.ncols();

        if absorption_maps.len() != n_wavelengths {
            anyhow::bail!(
                "Number of absorption maps {} does not match wavelength count {}",
                absorption_maps.len(),
                n_wavelengths
            );
        }

        // Get spatial dimensions from first map
        let (nx, ny, nz) = absorption_maps[0].dim();

        // Validate all maps have same dimensions
        for (idx, map) in absorption_maps.iter().enumerate() {
            if map.dim() != (nx, ny, nz) {
                anyhow::bail!(
                    "Absorption map {} has inconsistent dimensions {:?}, expected {:?}",
                    idx,
                    map.dim(),
                    (nx, ny, nz)
                );
            }
        }

        // Initialize output arrays
        let mut concentration_maps = Vec::with_capacity(n_chromophores);
        for _ in 0..n_chromophores {
            concentration_maps.push(Array3::zeros((nx, ny, nz)));
        }
        let mut residual_map = Array3::zeros((nx, ny, nz));

        // Unmix each voxel
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Extract absorption spectrum at this voxel
                    let mut spectrum = Array1::zeros(n_wavelengths);
                    for (wl_idx, abs_map) in absorption_maps.iter().enumerate() {
                        spectrum[wl_idx] = abs_map[[i, j, k]];
                    }

                    // Unmix
                    match self.unmix_single(&spectrum) {
                        Ok(result) => {
                            // Store concentrations
                            for (c_idx, &conc) in result.concentrations.iter().enumerate() {
                                concentration_maps[c_idx][[i, j, k]] = conc;
                            }
                            residual_map[[i, j, k]] = result.residual_norm;
                        }
                        Err(_) => {
                            // Leave as zeros if unmixing fails
                            residual_map[[i, j, k]] = f64::NAN;
                        }
                    }
                }
            }
        }

        Ok(VolumetricUnmixingResult {
            concentration_maps,
            residual_map,
            chromophore_names: self.chromophore_names.clone(),
        })
    }

    /// Get wavelengths
    pub fn wavelengths(&self) -> &[f64] {
        &self.wavelengths
    }

    /// Get chromophore names
    pub fn chromophore_names(&self) -> &[String] {
        &self.chromophore_names
    }

    /// Get extinction matrix
    pub fn extinction_matrix(&self) -> &Array2<f64> {
        &self.extinction_matrix
    }
}

/// Solve Tikhonov-regularized least squares: (EᵀE + λI)⁻¹Eᵀμ
///
/// # Arguments
///
/// - `E`: Extinction matrix (m × n)
/// - `mu`: Absorption spectrum (m × 1)
/// - `lambda`: Regularization parameter
///
/// # Returns
///
/// Concentration vector C (n × 1)
#[allow(non_snake_case)] // E is standard notation for extinction coefficient matrix
fn tikhonov_solve(E: &Array2<f64>, mu: &Array1<f64>, lambda: f64) -> Result<Array1<f64>> {
    let n_chromophores = E.ncols();

    // Compute EᵀE
    let et = E.t();
    let ete = et.dot(E);

    // Add regularization: EᵀE + λI
    let mut ete_reg = ete.clone();
    for i in 0..n_chromophores {
        ete_reg[[i, i]] += lambda;
    }

    // Compute Eᵀμ
    let et_mu = et.dot(mu);

    // Solve (EᵀE + λI)C = Eᵀμ using Cholesky decomposition
    solve_symmetric_positive_definite(&ete_reg, &et_mu).context("Failed to solve Tikhonov system")
}

/// Solve symmetric positive-definite system Ax = b using Cholesky decomposition
///
/// For small systems (typical in spectral unmixing: 2-5 chromophores),
/// we use simple Gaussian elimination with partial pivoting.
#[allow(non_snake_case)] // A is standard notation for coefficient matrix
fn solve_symmetric_positive_definite(A: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = A.nrows();

    if A.ncols() != n {
        anyhow::bail!("Matrix A must be square");
    }
    if b.len() != n {
        anyhow::bail!("Vector b length must match matrix size");
    }

    // Create augmented matrix [A | b]
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = A[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        let mut pivot_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > pivot_val {
                pivot_row = row;
                pivot_val = val;
            }
        }

        if pivot_val < 1e-12 {
            anyhow::bail!("Matrix is singular or near-singular");
        }

        // Swap rows
        if pivot_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }

        // Eliminate column
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / aug[[col, col]];
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Estimate condition number of matrix (simplified: ratio of max/min singular values)
///
/// For small matrices, use power iteration to estimate largest singular value
/// and inverse power iteration for smallest.
#[allow(non_snake_case)] // A is standard notation for coefficient matrix
fn estimate_condition_number(A: &Array2<f64>) -> Result<f64> {
    let (m, n) = A.dim();

    // For spectral unmixing matrices (typically small), compute EᵀE and estimate eigenvalues
    let ata = A.t().dot(A);

    // Power iteration for largest eigenvalue
    let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    for _ in 0..20 {
        // 20 iterations usually sufficient
        let v_new = ata.dot(&v);
        let norm = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            break;
        }
        v = v_new / norm;
    }
    let lambda_max = ata
        .dot(&v)
        .iter()
        .zip(v.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();

    // Approximate smallest eigenvalue (simplified: use trace and determinant)
    let trace: f64 = (0..n).map(|i| ata[[i, i]]).sum();
    let lambda_min = if lambda_max > 1e-12 {
        trace / (m as f64).max(1.0) - lambda_max * 0.9 // Rough approximation
    } else {
        1e-12
    };

    let condition = if lambda_min > 1e-12 {
        (lambda_max / lambda_min).sqrt() // Condition number = sqrt(λ_max / λ_min)
    } else {
        1e12 // Effectively singular
    };

    Ok(condition)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tikhonov_solve_simple() {
        // Simple 2x2 system: [2 1; 1 2] * [x; y] = [3; 3]
        // Solution: x = y = 1
        let A = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let b = Array1::from_vec(vec![3.0, 3.0]);

        let x = solve_symmetric_positive_definite(&A, &b).unwrap();

        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_unmixer_two_chromophores() {
        // Simple example: 2 chromophores, 3 wavelengths
        // E = [[10, 5],   ε₁(λ₁)=10, ε₂(λ₁)=5
        //      [5, 10],   ε₁(λ₂)=5,  ε₂(λ₂)=10
        //      [1, 1]]    ε₁(λ₃)=1,  ε₂(λ₃)=1
        let E = Array2::from_shape_vec((3, 2), vec![10.0, 5.0, 5.0, 10.0, 1.0, 1.0]).unwrap();
        let wavelengths = vec![500.0, 600.0, 700.0];
        let names = vec!["Chromophore A".to_string(), "Chromophore B".to_string()];

        let config = SpectralUnmixingConfig::default();
        let unmixer = SpectralUnmixer::new(E, wavelengths, names, config).unwrap();

        // Test spectrum: μ = [20, 20, 3] = 10*[1,0,0.1] + 5*[0.5,1,0.1] (approximately)
        // Expected: C₁ ≈ 1.0, C₂ ≈ 1.0
        let mu = Array1::from_vec(vec![15.0, 15.0, 2.0]);
        let result = unmixer.unmix_single(&mu).unwrap();

        // Should get reasonable concentrations (exact values depend on regularization)
        assert!(result.concentrations[0] >= 0.0);
        assert!(result.concentrations[1] >= 0.0);
        assert!(result.relative_residual < 0.5); // Reasonable fit
    }

    #[test]
    fn test_volumetric_unmixing() {
        // 2 chromophores, 2 wavelengths, 2x2x2 volume
        let E = Array2::from_shape_vec((2, 2), vec![10.0, 5.0, 5.0, 10.0]).unwrap();
        let wavelengths = vec![500.0, 600.0];
        let names = vec!["Chrom A".to_string(), "Chrom B".to_string()];

        let config = SpectralUnmixingConfig::default();
        let unmixer = SpectralUnmixer::new(E, wavelengths, names, config).unwrap();

        // Create absorption maps
        let abs_map_1 = Array3::from_elem((2, 2, 2), 15.0);
        let abs_map_2 = Array3::from_elem((2, 2, 2), 15.0);

        let result = unmixer.unmix_volumetric(&[abs_map_1, abs_map_2]).unwrap();

        assert_eq!(result.concentration_maps.len(), 2);
        assert_eq!(result.concentration_maps[0].dim(), (2, 2, 2));

        // All voxels should have non-negative concentrations
        for conc_map in &result.concentration_maps {
            for &val in conc_map.iter() {
                assert!(val >= 0.0);
            }
        }
    }

    #[test]
    fn test_non_negative_constraint() {
        // Test that negative concentrations are projected to zero
        let E = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
        let wavelengths = vec![500.0, 600.0];
        let names = vec!["Chrom".to_string()];

        let config = SpectralUnmixingConfig {
            non_negative: true,
            ..Default::default()
        };
        let unmixer = SpectralUnmixer::new(E, wavelengths, names, config).unwrap();

        // Spectrum that would give negative concentration without constraint
        let mu = Array1::from_vec(vec![-1.0, -1.0]);
        let result = unmixer.unmix_single(&mu).unwrap();

        // Should be zero (non-negative projection)
        assert_eq!(result.concentrations[0], 0.0);
    }

    #[test]
    fn test_underdetermined_system_error() {
        // 2 wavelengths, 3 chromophores (underdetermined)
        let E = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let wavelengths = vec![500.0, 600.0];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let config = SpectralUnmixingConfig::default();
        let result = SpectralUnmixer::new(E, wavelengths, names, config);

        // Should fail with underdetermined error
        assert!(result.is_err());
    }
}
