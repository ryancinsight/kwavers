use super::SourceHandler;
use crate::domain::grid::Grid;
use crate::domain::source::SourceMode;
use ndarray::Array3;

impl SourceHandler {
    /// Precompute per-voxel source_kappa for velocity source injection.
    ///
    /// ## Theorem (K-Wave ifftshift Convention)
    /// K-Wave's source_kappa is stored as `ifftshift(cos(c_ref·|k|·dt/2))` so that
    /// element `[i,j,k]` in physical space equals `cos(c_ref·|k_standard[(i+Nx/2)%Nx,`
    /// `(j+Ny/2)%Ny, (k+Nz/2)%Nz]|·dt/2)`. The ifftshift maps DC (`k=0`) to the
    /// grid corner `[0,0,0]`, matching the C++ k-Wave binary's direct-indexing convention.
    ///
    /// `source_kappa_fft`: the `(Nx,Ny,Nz)` k-space correction array in **standard FFT
    /// order** (`kappa_fft[0,0,0]` = 1.0 at DC). The ifftshift mapping is applied here.
    pub fn set_velocity_source_kappa(
        &mut self,
        source_kappa_fft: &Array3<f64>,
        grid_nx: usize,
        grid_ny: usize,
        grid_nz: usize,
    ) {
        self.u_kappa = self
            .u_indices
            .iter()
            .map(|&(i, j, k, _)| {
                let ki = (i + grid_nx / 2) % grid_nx;
                let kj = (j + grid_ny / 2) % grid_ny;
                let kk = (k + grid_nz / 2) % grid_nz;
                source_kappa_fft[[ki, kj, kk]]
            })
            .collect();
    }

    /// Precompute k-Wave-compatible per-source-point velocity-source scale
    /// factors `2·c₀·Δt/Δα` for each spatial axis α.
    ///
    /// **Theorem (Cox, Beard, Treeby — IEEE IUS 2018, k-Wave additive source
    /// scaling).** For an additive velocity source on a leapfrog grid the
    /// per-step injection that reproduces the canonical analytical mass-flow
    /// boundary condition is
    ///
    /// ```text
    ///   u_α(x_s, t_n) ← u_α(x_s, t_n) + (2·c₀(x_s)·Δt/Δα) · κ(x_s) · u_α^src(t_n)
    /// ```
    ///
    /// where `c₀(x_s)` is the per-source-point sound speed, `Δα` is the grid
    /// spacing along the injection axis, and `κ(x_s)` is the optional source
    /// kappa correction stored in `Self::u_kappa`. This matches
    /// `kwave/solvers/kspace_solver.py:533`
    /// (`scale = 2 * c0_src * self.dt / di`) and the C++ k-Wave binary's
    /// `velocity_source_input` term.
    ///
    /// `Dirichlet` mode is normalised differently (replacement, not addition),
    /// so its scale is fixed at 1.0 and the user-supplied signal is treated
    /// as the velocity value itself.
    ///
    /// On a degenerate axis (`grid.dα == 0` is impossible here, but
    /// `grid.nα == 1` means there is no propagation along α), the per-axis
    /// scale is still computed and applied; the velocity field along that
    /// axis is simply zero everywhere so the multiplication is a no-op.
    pub fn prepare_velocity_source_scaling(
        &mut self,
        grid: &Grid,
        c0: &Array3<f64>,
        dt: f64,
    ) {
        if self.u_indices.is_empty() {
            self.u_scale_x.clear();
            self.u_scale_y.clear();
            self.u_scale_z.clear();
            return;
        }

        let dx = grid.dx;
        let dy = grid.dy;
        let dz = grid.dz;
        let n = self.u_indices.len();
        self.u_scale_x = Vec::with_capacity(n);
        self.u_scale_y = Vec::with_capacity(n);
        self.u_scale_z = Vec::with_capacity(n);

        for &(i, j, k, _weight) in &self.u_indices {
            let c0_val = c0[[i, j, k]];
            let (sx, sy, sz) = match self.source.u_mode {
                SourceMode::Dirichlet => (1.0_f64, 1.0_f64, 1.0_f64),
                SourceMode::Additive | SourceMode::AdditiveNoCorrection => (
                    2.0 * c0_val * dt / dx,
                    2.0 * c0_val * dt / dy,
                    2.0 * c0_val * dt / dz,
                ),
            };
            self.u_scale_x.push(sx);
            self.u_scale_y.push(sy);
            self.u_scale_z.push(sz);
        }
    }

    /// Precompute k-Wave compatible pressure source scaling for mass (rho) and pressure updates.
    ///
    /// This mirrors `scale_source_terms` in k-Wave for additive/dirichlet modes on a uniform grid.
    pub fn prepare_pressure_source_scaling(&mut self, grid: &Grid, c0: &Array3<f64>, dt: f64) {
        if self.p_indices.is_empty() {
            self.p_scale_rho.clear();
            self.p_scale_p.clear();
            return;
        }

        let mut dim_count = 0usize;
        if grid.nx > 1 {
            dim_count += 1;
        }
        if grid.ny > 1 {
            dim_count += 1;
        }
        if grid.nz > 1 {
            dim_count += 1;
        }
        let n_dim = if dim_count == 0 {
            1.0
        } else {
            dim_count as f64
        };

        let dx = grid.dx;

        self.p_scale_rho = Vec::with_capacity(self.p_indices.len());
        self.p_scale_p = Vec::with_capacity(self.p_indices.len());

        for &(i, j, k, _weight) in &self.p_indices {
            let c0_val = c0[[i, j, k]];
            let (scale_rho, scale_p) = match self.source.p_mode {
                SourceMode::Dirichlet => {
                    let rho_scale = 1.0 / (n_dim * c0_val * c0_val);
                    (rho_scale, 1.0)
                }
                SourceMode::Additive | SourceMode::AdditiveNoCorrection => {
                    // K-Wave Python pre-scales source.p (Pa) → density (kg/m³) via:
                    //   source_p *= 2*dt / (N * c0 * dx)
                    // rho_scale = 2*dt / (n_dim * c0 * dx)
                    // p_scale   = 2·dt·c₀ / (N_prop · dx)
                    let rho_scale = 2.0 * dt / (n_dim * c0_val * dx);
                    let n_prop = self.source_propagation_dim.max(1.0);
                    let p_scale = (2.0 * dt * c0_val) / (n_prop * dx);
                    (rho_scale, p_scale)
                }
            };

            self.p_scale_rho.push(scale_rho);
            self.p_scale_p.push(scale_p);
        }
    }
}
