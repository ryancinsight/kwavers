/// CPU reference dispatcher for the FDTD pressure update.
///
/// Mirrors the CPU [`kwavers_solver::forward::fdtd::dispatch::FdtdStencilDispatcher`]
/// interface. This path computes the 6-point Laplacian stencil in Rust over
/// `leto::Array3` storage and serves as the value-semantic reference for
/// provider-specific GPU implementations.
///
/// # Algorithm — scalar wave equation (Yee 1966)
///
/// ```text
/// p^{n+1}[i,j,k] = 2·p^n[i,j,k] − p^{n-1}[i,j,k]
///                + coeff · ∇²p^n[i,j,k]
/// ```
///
/// where `coeff = (c·dt/dx)²` and `∇²` is the 6-point central-difference
/// Laplacian on an isotropic grid (dx = dy = dz).
///
/// Interior cells only (`1 ≤ i,j,k ≤ N−2`); boundary cells are set to zero
/// (Dirichlet condition), matching the `kspace_shift_apply` entry point in
/// `pstd_gpu/shaders/pstd.wgsl`.
///
/// # CPU Reference Contract
///
/// This dispatcher uses `f64` arithmetic throughout. Provider-specific GPU
/// kernels with different scalar precision must prove equivalence against this
/// path with a tolerance derived from their scalar contract and evaluation
/// order.
///
/// # References
///
/// - Yee KS (1966). IEEE Trans Antennas Propag 14(3):302–307.
/// - Moczo P et al. (2014). The Finite-Difference Modelling of Earthquake
///   Motions. Cambridge Univ. Press. (6-point Laplacian, §3.1)
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3 as LetoArray3;

#[derive(Debug)]
pub struct FdtdCpuReferenceDispatcher {
    nx: usize,
    ny: usize,
    nz: usize,
}

impl FdtdCpuReferenceDispatcher {
    /// Create dispatcher for a grid of size `nx × ny × nz`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` when any dimension is < 3 (stencil requires at
    /// least one interior layer).
    pub fn new(nx: usize, ny: usize, nz: usize) -> KwaversResult<Self> {
        if nx < 3 || ny < 3 || nz < 3 {
            return Err(KwaversError::InvalidInput(
                "FdtdCpuReferenceDispatcher: grid dimensions must be >= 3".to_string(),
            ));
        }
        Ok(Self { nx, ny, nz })
    }

    /// Compute `p^{n+1}` and write into `output`.
    ///
    /// ## Parameters
    /// - `p_curr`  — `p^n`  (current pressure field)
    /// - `p_prev`  — `p^{n-1}` (previous pressure field)
    /// - `coeff`   — `(c·dt/dx)²` (dimensionless CFL coefficient²)
    /// - `output`  — written in-place; every element is overwritten
    ///
    /// ## Errors
    ///
    /// Returns `InvalidInput` if any field shape does not match `(nx, ny, nz)`.
    pub fn update_pressure_into(
        &mut self,
        p_curr: &LetoArray3<f64>,
        p_prev: &LetoArray3<f64>,
        coeff: f64,
        output: &mut LetoArray3<f64>,
    ) -> KwaversResult<()> {
        let expected = [self.nx, self.ny, self.nz];
        if p_curr.shape() != expected || p_prev.shape() != expected || output.shape() != expected {
            return Err(KwaversError::InvalidInput(format!(
                "FdtdCpuReferenceDispatcher: field shape mismatch (expected {:?})",
                expected
            )));
        }

        Self::zero_boundary_faces(output);

        for k in 1..self.nz - 1 {
            for j in 1..self.ny - 1 {
                for i in 1..self.nx - 1 {
                    let laplacian = p_curr[[i - 1, j, k]]
                        + p_curr[[i + 1, j, k]]
                        + p_curr[[i, j - 1, k]]
                        + p_curr[[i, j + 1, k]]
                        + p_curr[[i, j, k - 1]]
                        + p_curr[[i, j, k + 1]]
                        - 6.0 * p_curr[[i, j, k]];

                    output[[i, j, k]] =
                        2.0 * p_curr[[i, j, k]] - p_prev[[i, j, k]] + coeff * laplacian;
                }
            }
        }

        Ok(())
    }

    /// Convenience wrapper — allocates and returns the updated pressure field.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn update_pressure(
        &mut self,
        p_curr: &LetoArray3<f64>,
        p_prev: &LetoArray3<f64>,
        coeff: f64,
    ) -> KwaversResult<LetoArray3<f64>> {
        let mut result = LetoArray3::zeros([self.nx, self.ny, self.nz]);
        self.update_pressure_into(p_curr, p_prev, coeff, &mut result)?;
        Ok(result)
    }

    /// Grid dimensions `(nx, ny, nz)`
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Zero the six boundary faces of a 3D volume.
    fn zero_boundary_faces(output: &mut LetoArray3<f64>) {
        let [nx, ny, nz] = output.shape();

        for j in 0..ny {
            for k in 0..nz {
                output[[0, j, k]] = 0.0;
                output[[nx - 1, j, k]] = 0.0;
            }
        }

        for i in 0..nx {
            for k in 0..nz {
                output[[i, 0, k]] = 0.0;
                output[[i, ny - 1, k]] = 0.0;
            }
        }

        for i in 0..nx {
            for j in 0..ny {
                output[[i, j, 0]] = 0.0;
                output[[i, j, nz - 1]] = 0.0;
            }
        }
    }
}