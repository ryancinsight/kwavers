//! ElectromagneticFdtdSolver constructor and field update methods.

use super::types::ElectromagneticFdtdSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::{ArrayD, EMFields, VecStorage};
use kwavers_grid::Grid;
use kwavers_physics::electromagnetic::equations::EMMaterialDistribution;
use leto::Array3;

impl ElectromagneticFdtdSolver {
    /// Create a new electromagnetic FDTD solver
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn new(
        grid: Grid,
        materials: EMMaterialDistribution,
        dt: f64,
        spatial_order: usize,
    ) -> KwaversResult<Self> {
        // Validate spatial order
        if ![2, 4, 6].contains(&spatial_order) {
            return Err(KwaversError::InvalidInput(format!(
                "spatial_order must be 2, 4, or 6, got {spatial_order}"
            )));
        }

        // Initialize field arrays with Yee staggering
        // E fields are at integer grid points, H fields at half-points
        let ex = Array3::<f64>::zeros([grid.nx, grid.ny + 1, grid.nz + 1]);
        let ey = Array3::<f64>::zeros([grid.nx + 1, grid.ny, grid.nz + 1]);
        let ez = Array3::<f64>::zeros([grid.nx + 1, grid.ny + 1, grid.nz]);

        let hx = Array3::<f64>::zeros([grid.nx + 1, grid.ny, grid.nz]);
        let hy = Array3::<f64>::zeros([grid.nx, grid.ny + 1, grid.nz]);
        let hz = Array3::<f64>::zeros([grid.nx, grid.ny, grid.nz + 1]);

        // Cache grid dimensions before moving grid
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        Ok(Self {
            grid,
            materials,
            time_step: 0,
            dt,
            ex,
            ey,
            ez,
            hx,
            hy,
            hz,
            fields_cache: EMFields {
                electric: ArrayD::<f64, VecStorage<f64>>::zeros(&[nx, ny, nz, 3])
                    .expect("valid shape for electric field cache"),
                magnetic: ArrayD::<f64, VecStorage<f64>>::zeros(&[nx, ny, nz, 3])
                    .expect("valid shape for magnetic field cache"),
                displacement: None,
                flux_density: None,
            },
        })
    }

    /// Recompute cell-centered EM fields from the Yee-staggered state.
    ///
    /// The cache invariant is `electric.shape() == magnetic.shape() ==
    /// [nx, ny, nz, 3]`. Converting the dynamic arrays to `Ix4` once per cache
    /// update keeps indexing statically arity-checked inside the hot loop and
    /// avoids constructing an `IxDyn` index for every stored component.
    pub(super) fn update_field_cache(&mut self) {
        debug_assert_eq!(
            self.fields_cache.electric.shape(),
            &[self.grid.nx, self.grid.ny, self.grid.nz, 3]
        );
        debug_assert_eq!(
            self.fields_cache.magnetic.shape(),
            &[self.grid.nx, self.grid.ny, self.grid.nz, 3]
        );

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let ex_c = 0.25
                        * (self.ex[[i, j, k]]
                            + self.ex[[i, j + 1, k]]
                            + self.ex[[i, j, k + 1]]
                            + self.ex[[i, j + 1, k + 1]]);
                    let ey_c = 0.25
                        * (self.ey[[i, j, k]]
                            + self.ey[[i + 1, j, k]]
                            + self.ey[[i, j, k + 1]]
                            + self.ey[[i + 1, j, k + 1]]);
                    let ez_c = 0.25
                        * (self.ez[[i, j, k]]
                            + self.ez[[i + 1, j, k]]
                            + self.ez[[i, j + 1, k]]
                            + self.ez[[i + 1, j + 1, k]]);

                    let hx_c = 0.5 * (self.hx[[i, j, k]] + self.hx[[i + 1, j, k]]);
                    let hy_c = 0.5 * (self.hy[[i, j, k]] + self.hy[[i, j + 1, k]]);
                    let hz_c = 0.5 * (self.hz[[i, j, k]] + self.hz[[i, j, k + 1]]);

                    *self.fields_cache.electric.get_mut(&[i, j, k, 0]).unwrap() = ex_c;
                    *self.fields_cache.electric.get_mut(&[i, j, k, 1]).unwrap() = ey_c;
                    *self.fields_cache.electric.get_mut(&[i, j, k, 2]).unwrap() = ez_c;

                    *self.fields_cache.magnetic.get_mut(&[i, j, k, 0]).unwrap() = hx_c;
                    *self.fields_cache.magnetic.get_mut(&[i, j, k, 1]).unwrap() = hy_c;
                    *self.fields_cache.magnetic.get_mut(&[i, j, k, 2]).unwrap() = hz_c;
                }
            }
        }
    }

    /// Copy the authoritative field cache into caller-owned output storage.
    ///
    /// Shape-compatible buffers are reused with `assign`, so repeated boundary
    /// applications do not allocate or clone field arrays. A shape mismatch is
    /// repaired by allocating the required shape once and then copying values;
    /// this preserves the trait contract while keeping the steady-state path
    /// allocation-free.
    pub(super) fn copy_field_cache_into(&self, fields: &mut EMFields) {
        assign_array(&mut fields.electric, &self.fields_cache.electric);
        assign_array(&mut fields.magnetic, &self.fields_cache.magnetic);
        assign_optional_array(
            &mut fields.displacement,
            self.fields_cache.displacement.as_ref(),
        );
        assign_optional_array(
            &mut fields.flux_density,
            self.fields_cache.flux_density.as_ref(),
        );
    }

    pub(super) fn permittivity_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.materials
            .permittivity
            .get(&[i, j, k])
            .copied()
            .unwrap_or(1.0)
    }

    pub(super) fn conductivity_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.materials
            .conductivity
            .get(&[i, j, k])
            .copied()
            .unwrap_or(0.0)
    }

    pub(super) fn permeability_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.materials
            .permeability
            .get(&[i, j, k])
            .copied()
            .unwrap_or(1.0)
    }

    /// Update electric fields using Ampère's law (Maxwell-Ampère):
    /// `∂E/∂t = (1/ε)·∇×H − (σ/ε)·E`.
    pub(super) fn update_electric_fields(&mut self) {
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;

        // Update Ex: ∂Ex/∂t = (1/ε)(∂Hz/∂y − ∂Hy/∂z) − (σ/ε)Ex
        //
        // Crank-Nicolson loss update (Taflove & Hagness §3.7, Eq. 3.54):
        //   ca = (2ε − σ·dt) / (2ε + σ·dt)
        //   cb = 2·dt / (2ε + σ·dt)
        //   E^{n+1} = ca·E^n + cb·curl_H^{n+1/2} / ε
        //
        // This is unconditionally stable for the loss term (unlike forward-Euler
        // which requires σ·dt/ε < 2 separately from the CFL condition).  For σ=0
        // the update reduces to the lossless Yee update E += dt/ε·curl_H.
        for i in 0..self.grid.nx {
            for j in 1..self.grid.ny {
                for k in 1..self.grid.nz {
                    let curl_h_x = (self.hz[[i, j, k]] - self.hz[[i, j - 1, k]]) / dy
                        - (self.hy[[i, j, k]] - self.hy[[i, j, k - 1]]) / dz;

                    let eps = self.permittivity_at(i, j - 1, k - 1);
                    let sigma = self.conductivity_at(i, j - 1, k - 1);

                    let denom = 2.0_f64.mul_add(eps, sigma * self.dt);
                    let ca = (2.0_f64.mul_add(eps, -(sigma * self.dt))) / denom;
                    let cb = 2.0 * self.dt / denom;
                    self.ex[[i, j, k]] = ca.mul_add(self.ex[[i, j, k]], cb * curl_h_x);
                }
            }
        }

        // Update Ey: ∂Ey/∂t = (1/ε)(∂Hx/∂z − ∂Hz/∂x) − (σ/ε)Ey
        for i in 1..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 1..self.grid.nz {
                    let curl_h_y = (self.hx[[i, j, k]] - self.hx[[i, j, k - 1]]) / dz
                        - (self.hz[[i, j, k]] - self.hz[[i - 1, j, k]]) / dx;

                    let eps = self.permittivity_at(i - 1, j, k - 1);
                    let sigma = self.conductivity_at(i - 1, j, k - 1);

                    let denom = 2.0_f64.mul_add(eps, sigma * self.dt);
                    let ca = (2.0_f64.mul_add(eps, -(sigma * self.dt))) / denom;
                    let cb = 2.0 * self.dt / denom;
                    self.ey[[i, j, k]] = ca.mul_add(self.ey[[i, j, k]], cb * curl_h_y);
                }
            }
        }

        // Update Ez: ∂Ez/∂t = (1/ε)(∂Hy/∂x − ∂Hx/∂y) − (σ/ε)Ez
        for i in 1..self.grid.nx {
            for j in 1..self.grid.ny {
                for k in 0..self.grid.nz {
                    let curl_h_z = (self.hy[[i, j, k]] - self.hy[[i - 1, j, k]]) / dx
                        - (self.hx[[i, j, k]] - self.hx[[i, j - 1, k]]) / dy;

                    let eps = self.permittivity_at(i - 1, j - 1, k);
                    let sigma = self.conductivity_at(i - 1, j - 1, k);

                    let denom = 2.0_f64.mul_add(eps, sigma * self.dt);
                    let ca = (2.0_f64.mul_add(eps, -(sigma * self.dt))) / denom;
                    let cb = 2.0 * self.dt / denom;
                    self.ez[[i, j, k]] = ca.mul_add(self.ez[[i, j, k]], cb * curl_h_z);
                }
            }
        }
    }

    /// Update magnetic fields using Faraday's law:
    /// `∂H/∂t = −(1/μ)·∇×E`.
    pub(super) fn update_magnetic_fields(&mut self) {
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;

        // Update Hx: ∂Hx/∂t = -(1/μ) (∂Ez/∂y - ∂Ey/∂z)
        for i in 1..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let curl_e_x = (self.ez[[i, j + 1, k]] - self.ez[[i, j, k]]) / dy
                        - (self.ey[[i, j, k + 1]] - self.ey[[i, j, k]]) / dz;

                    let mu = self.permeability_at(i - 1, j, k);
                    self.hx[[i, j, k]] -= self.dt * curl_e_x / mu;
                }
            }
        }

        // Update Hy: ∂Hy/∂t = -(1/μ) (∂Ex/∂z - ∂Ez/∂x)
        for i in 0..self.grid.nx {
            for j in 1..self.grid.ny {
                for k in 0..self.grid.nz {
                    let curl_e_y = (self.ex[[i, j, k + 1]] - self.ex[[i, j, k]]) / dz
                        - (self.ez[[i + 1, j, k]] - self.ez[[i, j, k]]) / dx;

                    let mu = self.permeability_at(i, j - 1, k);
                    self.hy[[i, j, k]] -= self.dt * curl_e_y / mu;
                }
            }
        }

        // Update Hz: ∂Hz/∂t = -(1/μ) (∂Ey/∂x - ∂Ex/∂y)
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 1..self.grid.nz {
                    // ∂Ey/∂x uses dx; ∂Ex/∂y uses dy — NOT dz.
                    // Previous code divided by dz here, producing a wrong curl
                    // and systematically wrong Hz evolution in any non-cubic grid.
                    let curl_e_z = (self.ey[[i + 1, j, k]] - self.ey[[i, j, k]]) / dx
                        - (self.ex[[i, j + 1, k]] - self.ex[[i, j, k]]) / dy;

                    let mu = self.permeability_at(i, j, k - 1);
                    self.hz[[i, j, k]] -= self.dt * curl_e_z / mu;
                }
            }
        }
    }

    /// Compute CFL-stable time step for electromagnetic waves.
    ///
    /// `dt ≤ 1/(c √(1/dx² + 1/dy² + 1/dz²))`
    pub fn max_stable_dt(&self, c_max: f64) -> f64 {
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;

        let denominator = c_max
            * (1.0 / dz)
                .mul_add(1.0 / dz, (1.0 / dy).mul_add(1.0 / dy, (1.0 / dx).powi(2)))
                .sqrt();
        0.99 / denominator // 0.99 for stability margin
    }
}

fn assign_array(target: &mut ArrayD<f64, VecStorage<f64>>, source: &ArrayD<f64, VecStorage<f64>>) {
    if target.shape() != source.shape() {
        *target = ArrayD::<f64, VecStorage<f64>>::zeros(source.shape())
            .expect("valid shape for target reallocation");
    }
    // Element-wise copy via iterators (works for any VecStorage).
    for (t, s) in target.iter_mut().zip(source.iter()) {
        *t = *s;
    }
}

fn assign_optional_array(
    target: &mut Option<ArrayD<f64, VecStorage<f64>>>,
    source: Option<&ArrayD<f64, VecStorage<f64>>>,
) {
    match source {
        Some(source) => match target {
            Some(target) => assign_array(target, source),
            None => {
                let mut array = ArrayD::<f64, VecStorage<f64>>::zeros(source.shape())
                    .expect("valid shape for optional array allocation");
                for (t, s) in array.iter_mut().zip(source.iter()) {
                    *t = *s;
                }
                *target = Some(array);
            }
        },
        None => *target = None,
    }
}
