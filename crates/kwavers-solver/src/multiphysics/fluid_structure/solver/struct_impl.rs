use moirai_parallel::ParallelSliceMut;
use ndarray::{Array1, Array3};

use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::coefficients::ReflectionTransmissionCoefficients;
use super::super::interface::FsiInterface;

/// Compute primary axis-aligned step direction from a general normal vector.
///
/// Returns the unit integer step (di, dj, dk) along the dominant axis.
/// Used for ghost layer traversal along the interface normal.
fn primary_normal_step(n0: f64, n1: f64, n2: f64) -> (i64, i64, i64) {
    let (a0, a1, a2) = (n0.abs(), n1.abs(), n2.abs());
    if a0 >= a1 && a0 >= a2 {
        (n0.signum() as i64, 0, 0)
    } else if a1 >= a2 {
        (0, n1.signum() as i64, 0)
    } else {
        (0, 0, n2.signum() as i64)
    }
}

///
/// Implements partitioned coupling with ghost cell exchange following
/// Farhat & Roux (1991) Dirichlet-Neumann iteration.
///
/// ## Ghost Cell Layout
///
/// For each interface cell (i,j,k) with outward normal n̂:
/// - `p_fluid_ghost[i,j,k]` = σ_nn (solid normal stress → fluid pressure BC)
/// - `t_solid_ghost[d][i,j,k]` = −p · n̂[d] (fluid pressure → solid traction BC)
///
/// Reference: Farhat, C. & Roux, F.X. (1991). Int J Numer Methods Eng 32(6), 1205–1227.
/// DOI: 10.1002/nme.1620320604
#[derive(Debug)]
pub struct FluidStructureSolver {
    /// Interface parameters
    pub interface: FsiInterface,
    /// Ghost cell layer thickness
    ghost_layers: usize,
    /// Relaxation parameter for convergence
    relaxation: f64,
    /// Convergence tolerance
    tolerance: f64,
    /// Ghost pressure values (fluid domain): p_ghost = σ_nn at interface + extrapolated layers
    pub p_fluid_ghost: Array3<f64>,
    /// Ghost traction values (solid domain): t_ghost[d] = −p · n̂[d]
    pub t_solid_ghost: [Array3<f64>; 3],
    /// Previous ghost pressure workspace for relaxation.
    ///
    /// This buffer is solver-owned so each Dirichlet-Neumann iteration can
    /// preserve the prior ghost state with `assign` instead of allocating a
    /// full-volume clone.  It has the same shape as `p_fluid_ghost`.
    p_fluid_ghost_prev: Array3<f64>,
    /// Previous ghost traction workspace for relaxation.
    ///
    /// Each component shares the exact dimensions of `t_solid_ghost[d]`.
    /// The fixed array keeps the three Cartesian components statically routed
    /// without heap-dispatched trait objects or per-iteration `Vec` allocation.
    t_solid_ghost_prev: [Array3<f64>; 3],
}

impl FluidStructureSolver {
    /// Create new FSI solver
    pub fn new(interface: FsiInterface) -> Self {
        let shape = interface.interface_mask.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
        Self {
            ghost_layers: 2,
            relaxation: 0.8,
            tolerance: 1e-6,
            p_fluid_ghost: Array3::zeros((nx, ny, nz)),
            t_solid_ghost: [
                Array3::zeros((nx, ny, nz)),
                Array3::zeros((nx, ny, nz)),
                Array3::zeros((nx, ny, nz)),
            ],
            p_fluid_ghost_prev: Array3::zeros((nx, ny, nz)),
            t_solid_ghost_prev: [
                Array3::zeros((nx, ny, nz)),
                Array3::zeros((nx, ny, nz)),
                Array3::zeros((nx, ny, nz)),
            ],
            interface,
        }
    }

    /// Apply interface conditions with coupling.
    /// # Errors
    /// - Returns [`KwaversError::InternalError`] if the precondition for a InternalError-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_interface_conditions(
        &mut self,
        fluid_pressure: &mut Array3<f64>,
        fluid_velocity: &mut [Array3<f64>; 3],
        _solid_displacement: &mut [Array3<f64>; 3],
        solid_velocity: &mut [Array3<f64>; 3],
        solid_stress: &mut [Array3<f64>; 6],
        _dt: f64,
    ) -> KwaversResult<()> {
        let max_iterations = 100;
        let mut iteration = 0;
        let mut converged = false;

        while !converged && iteration < max_iterations {
            self.p_fluid_ghost_prev.assign(&self.p_fluid_ghost);
            for (prev, current) in self
                .t_solid_ghost_prev
                .iter_mut()
                .zip(self.t_solid_ghost.iter())
            {
                prev.assign(current);
            }

            self.exchange_ghost_cells(fluid_pressure, solid_stress)?;

            if iteration > 0 {
                let omega = self.relaxation;
                {
                    let new_slice = self.p_fluid_ghost.as_slice_mut()
                        .expect("p_fluid_ghost is contiguous (default Array3 layout)");
                    let prev_slice = self.p_fluid_ghost_prev.as_slice()
                        .expect("p_fluid_ghost_prev is contiguous (default Array3 layout)");
                    new_slice.par_mut().enumerate(|idx, new_val: &mut f64| {
                        let prev = prev_slice[idx];
                        *new_val = omega * *new_val + (1.0 - omega) * prev;
                    });
                }
                for (tsg, tgp) in self
                    .t_solid_ghost
                    .iter_mut()
                    .zip(self.t_solid_ghost_prev.iter())
                    .take(3)
                {
                    {
                        let new_slice = tsg.as_slice_mut()
                            .expect("t_solid_ghost component is contiguous (1D view)");
                        let prev_slice = tgp.as_slice()
                            .expect("t_solid_ghost_prev component is contiguous (1D view)");
                        new_slice.par_mut().enumerate(|idx, new_val: &mut f64| {
                            let prev = prev_slice[idx];
                            *new_val = omega * *new_val + (1.0 - omega) * prev;
                        });
                    }
                }
            }

            self.compute_interface_traction(fluid_pressure, solid_stress)?;
            converged = self.check_convergence(fluid_velocity, solid_velocity)?;
            iteration += 1;
        }

        if !converged {
            return Err(KwaversError::InternalError(format!(
                "FSI coupling did not converge within {} iterations",
                max_iterations
            )));
        }

        Ok(())
    }

    /// Exchange ghost cell data across the fluid-structure interface.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn exchange_ghost_cells(
        &mut self,
        fluid_pressure: &Array3<f64>,
        solid_stress: &[Array3<f64>; 6],
    ) -> KwaversResult<()> {
        let [n0, n1, n2] = self.interface.normal;
        let shape = fluid_pressure.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
        let (sx, sy, sz) = (nx as i64, ny as i64, nz as i64);

        self.p_fluid_ghost.fill(0.0);
        for traction_component in self.t_solid_ghost.iter_mut() {
            traction_component.fill(0.0);
        }

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if !self.interface.interface_mask[(i, j, k)] {
                        continue;
                    }

                    let p = fluid_pressure[(i, j, k)];
                    let sxx = solid_stress[0][(i, j, k)];
                    let syy = solid_stress[1][(i, j, k)];
                    let szz = solid_stress[2][(i, j, k)];
                    let sxy = solid_stress[3][(i, j, k)];
                    let sxz = solid_stress[4][(i, j, k)];
                    let syz = solid_stress[5][(i, j, k)];

                    let sigma_nn = n0 * n0 * sxx
                        + n1 * n1 * syy
                        + n2 * n2 * szz
                        + 2.0 * n0 * n1 * sxy
                        + 2.0 * n0 * n2 * sxz
                        + 2.0 * n1 * n2 * syz;

                    self.p_fluid_ghost[(i, j, k)] = sigma_nn;
                    self.t_solid_ghost[0][(i, j, k)] = -p * n0;
                    self.t_solid_ghost[1][(i, j, k)] = -p * n1;
                    self.t_solid_ghost[2][(i, j, k)] = -p * n2;
                }
            }
        }

        let (di, dj, dk) = primary_normal_step(n0, n1, n2);

        if di != 0 || dj != 0 || dk != 0 {
            for g in 1..self.ghost_layers as i64 {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if !self.interface.interface_mask[(i, j, k)] {
                                continue;
                            }

                            let (ig, jg, kg) =
                                (i as i64 + di * g, j as i64 + dj * g, k as i64 + dk * g);
                            let (ii, ji, ki) =
                                (i as i64 - di * g, j as i64 - dj * g, k as i64 - dk * g);

                            let ghost_valid =
                                ig >= 0 && ig < sx && jg >= 0 && jg < sy && kg >= 0 && kg < sz;
                            if !ghost_valid {
                                continue;
                            }

                            let phi_interface = self.p_fluid_ghost[(i, j, k)];
                            let phi_interior =
                                if ii >= 0 && ii < sx && ji >= 0 && ji < sy && ki >= 0 && ki < sz {
                                    fluid_pressure[[ii as usize, ji as usize, ki as usize]]
                                } else {
                                    phi_interface
                                };
                            self.p_fluid_ghost[[ig as usize, jg as usize, kg as usize]] =
                                2.0 * phi_interface - phi_interior;

                            for traction_component in self.t_solid_ghost.iter_mut().take(3) {
                                let t_iface = traction_component[(i, j, k)];
                                traction_component[[ig as usize, jg as usize, kg as usize]] =
                                    t_iface;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute traction vector at interface.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_interface_traction(
        &self,
        fluid_pressure: &Array3<f64>,
        solid_stress: &[Array3<f64>; 6],
    ) -> KwaversResult<Array1<f64>> {
        let [nx, ny, nz] = self.interface.normal;
        let mut traction = Array1::zeros(3);

        ndarray::Zip::indexed(&self.interface.interface_mask)
            .and(fluid_pressure.view())
            .for_each(|(i, j, k), mask, &p| {
                if *mask {
                    let s_xx = solid_stress[0][[i, j, k]];
                    let s_yy = solid_stress[1][[i, j, k]];
                    let s_zz = solid_stress[2][[i, j, k]];
                    let s_xy = solid_stress[3][[i, j, k]];
                    let s_xz = solid_stress[4][[i, j, k]];
                    let s_yz = solid_stress[5][[i, j, k]];

                    let t_x = -p * nx + s_xx * nx + s_xy * ny + s_xz * nz;
                    let t_y = -p * ny + s_xy * nx + s_yy * ny + s_yz * nz;
                    let t_z = -p * nz + s_xz * nx + s_yz * ny + s_zz * nz;

                    traction[0] += t_x;
                    traction[1] += t_y;
                    traction[2] += t_z;
                }
            });

        Ok(traction)
    }

    /// Check convergence of coupling iterations.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn check_convergence(
        &self,
        fluid_velocity: &[Array3<f64>; 3],
        solid_velocity: &[Array3<f64>; 3],
    ) -> KwaversResult<bool> {
        let mut max_error = 0.0_f64;

        [
            self.interface.normal[0],
            self.interface.normal[1],
            self.interface.normal[2],
        ]
        .iter()
        .enumerate()
        .for_each(|(dim, &n)| {
            ndarray::Zip::indexed(&self.interface.interface_mask)
                .and(fluid_velocity[dim].view())
                .and(solid_velocity[dim].view())
                .for_each(|(_i, _j, _k), mask, &v_f, &v_s| {
                    if *mask {
                        let error = (v_f * n - v_s * n).abs();
                        max_error = max_error.max(error);
                    }
                });
        });

        Ok(max_error < self.tolerance)
    }

    /// Get interface transmission and reflection coefficients.
    pub fn get_coefficients(&self, incidence_angle: f64) -> ReflectionTransmissionCoefficients {
        if incidence_angle.abs() < 1e-10 {
            ReflectionTransmissionCoefficients::normal_incidence(&self.interface)
        } else {
            ReflectionTransmissionCoefficients::oblique_incidence(&self.interface, incidence_angle)
        }
    }
}
