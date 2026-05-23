use super::models::HyperelasticModel;
use std::f64::consts::PI;

/// Compute strain invariants from deformation gradient
#[must_use]
pub fn compute_invariants(_model: &HyperelasticModel, f: &[[f64; 3]; 3]) -> (f64, f64, f64) {
    // Compute right Cauchy-Green tensor C = F^T · F
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for row in f.iter().take(3) {
                c[i][j] += row[i] * row[j];
            }
        }
    }

    // Eigenvalues of C are λ²
    let lambda_sq = matrix_eigenvalues(&c);
    let lambda = lambda_sq.map(|x| x.sqrt());

    // Strain invariants
    let i1 = lambda_sq.iter().sum::<f64>();
    let i2 = lambda_sq[2].mul_add(
        lambda_sq[0],
        lambda_sq[0].mul_add(lambda_sq[1], lambda_sq[1] * lambda_sq[2]),
    );
    let j = lambda.iter().product::<f64>();

    (i1, i2, j)
}

/// Compute left Cauchy-Green tensor B = F · F^T
pub fn left_cauchy_green(_model: &HyperelasticModel, f: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for (v1, v2) in f[i].iter().zip(f[j].iter()) {
                b[i][j] += v1 * v2;
            }
        }
    }
    b
}

/// Compute principal stretches from deformation gradient
///
/// # Theorem Reference
/// Principal stretches λᵢ are the square roots of eigenvalues of the right Cauchy-Green tensor C = F^T * F.
/// For Ogden hyperelastic materials, these are essential for computing the strain energy density.
///
/// # Arguments
/// * `f` - Deformation gradient tensor F (3x3 matrix)
///
/// # Returns
/// Principal stretches [λ₁, λ₂, λ₃] sorted in ascending order
pub fn principal_stretches(_model: &HyperelasticModel, f: &[[f64; 3]; 3]) -> [f64; 3] {
    // Compute right Cauchy-Green tensor C = F^T * F
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for row in f.iter().take(3) {
                c[i][j] += row[i] * row[j];
            }
        }
    }

    // Compute eigenvalues of C and take square roots for principal stretches
    let lambda_sq = matrix_eigenvalues(&c);
    lambda_sq.map(|x| x.sqrt())
}

/// Compute eigenvalues of 3x3 symmetric matrix using Jacobi eigenvalue algorithm
///
/// # Theorem Reference
/// Implements the Jacobi method for symmetric matrix diagonalization.
/// Golub & Van Loan (1996): "Matrix Computations", Algorithm 8.4.2
/// Converges quadratically for well-conditioned symmetric matrices.
///
/// Convergence criterion: ||A||_F * ε where ε is machine precision
/// Handles indefinite matrices and ensures numerical stability.
///
/// # Arguments
/// * `m` - 3x3 symmetric matrix
///
/// # Returns
/// Array of eigenvalues [λ₁, λ₂, λ₃] sorted in ascending order
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub fn matrix_eigenvalues(m: &[[f64; 3]; 3]) -> [f64; 3] {
    // Jacobi eigenvalue algorithm for 3x3 symmetric matrices
    // Based on Golub & Van Loan, Matrix Computations (3rd ed.), Algorithm 8.4.2

    let mut a = *m; // Copy matrix to avoid modifying input
    let mut eigenvalues = [0.0; 3];

    // Compute Frobenius norm for convergence criterion
    let mut frobenius_norm = 0.0;
    for row in &a {
        for val in row {
            frobenius_norm += val * val;
        }
    }
    frobenius_norm = frobenius_norm.sqrt();

    // Convergence tolerance: relative to matrix norm
    let tolerance = frobenius_norm * f64::EPSILON.sqrt(); // ~1e-8 for typical matrices

    // Jacobi iterations (typically converges in 5-15 iterations for 3x3)
    for iteration in 0..100 {
        // Maximum iterations with safety limit
        // Find largest off-diagonal element in absolute value
        let mut max_off_diag = 0.0;
        let mut p = 0;
        let mut q = 1;

        for (i, row) in a.iter().enumerate() {
            for (j, &val) in row.iter().enumerate().skip(i + 1) {
                let val_abs = val.abs();
                if val_abs > max_off_diag {
                    max_off_diag = val_abs;
                    p = i;
                    q = j;
                }
            }
        }

        // Check convergence: all off-diagonal elements small relative to matrix norm
        if max_off_diag < tolerance || iteration >= 99 {
            break;
        }

        // Compute rotation parameters
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        let theta = if app == aqq {
            PI / 4.0
        } else {
            0.5 * ((2.0 * apq) / (app - aqq)).atan()
        };

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // Apply Jacobi rotation to the remaining rows/columns
        // There is exactly one index k != p and k != q in {0, 1, 2}
        let k = 3 - p - q;

        let akp = a[k][p];
        let akq = a[k][q];
        a[k][p] = akp.mul_add(cos_theta, -(akq * sin_theta));
        a[k][q] = akp.mul_add(sin_theta, akq * cos_theta);
        a[p][k] = a[k][p];
        a[q][k] = a[k][q];

        // Update diagonal and off-diagonal elements
        a[p][p] = (2.0 * apq * sin_theta).mul_add(
            -cos_theta,
            (app * cos_theta).mul_add(cos_theta, aqq * sin_theta * sin_theta),
        );
        a[q][q] = (2.0 * apq * sin_theta).mul_add(
            cos_theta,
            (app * sin_theta).mul_add(sin_theta, aqq * cos_theta * cos_theta),
        );
        a[p][q] = 0.0;
        a[q][p] = 0.0;
    }

    // Extract eigenvalues from diagonal
    eigenvalues[0] = a[0][0];
    eigenvalues[1] = a[1][1];
    eigenvalues[2] = a[2][2];

    // Sort eigenvalues in ascending order
    eigenvalues.sort_by(|a, b| a.total_cmp(b));

    eigenvalues
}
