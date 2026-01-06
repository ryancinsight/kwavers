use crate::KwaversResult;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisType {
    Legendre,
    Chebyshev,
    Fourier,
}

/// Build Vandermonde matrix for given nodes and basis
/// V_ij = phi_j(xi_i)
pub fn build_vandermonde(
    nodes: &Array1<f64>,
    poly_order: usize,
    basis_type: BasisType,
) -> KwaversResult<Array2<f64>> {
    let n_nodes = nodes.len();
    let n_modes = poly_order + 1;

    if n_nodes != n_modes {
        // For collocation/nodal DG, usually n_nodes = n_modes
        // But we can build V for oversampling too.
    }

    let mut v = Array2::zeros((n_nodes, n_modes));

    match basis_type {
        BasisType::Legendre => {
            for i in 0..n_nodes {
                let xi = nodes[i];
                for j in 0..n_modes {
                    // Use normalized Legendre polynomials
                    // P_j(x) normalized so that integral_{-1}^1 P_j^2 dx = 1
                    // Standard Legendre: integral = 2/(2j+1)
                    // So multiply by sqrt((2j+1)/2)
                    let p_val = legendre_poly(j, xi);
                    let norm_factor = ((2 * j + 1) as f64 / 2.0).sqrt();
                    v[[i, j]] = p_val * norm_factor;
                }
            }
        }
        _ => {
            return Err(crate::error::KwaversError::NotImplemented(format!(
                "Basis type {:?} not implemented",
                basis_type
            )))
        }
    }

    Ok(v)
}

fn legendre_poly(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    let mut l_prev = 1.0;
    let mut l_curr = x;

    for i in 1..n {
        let l_next = ((2 * i + 1) as f64 * x * l_curr - i as f64 * l_prev) / ((i + 1) as f64);
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}
