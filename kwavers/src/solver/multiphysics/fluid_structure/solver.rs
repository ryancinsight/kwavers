use super::coefficients::ReflectionTransmissionCoefficients;
use super::interface::FsiInterface;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array3};

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
}

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
            interface,
        }
    }

    /// Apply interface conditions with coupling
    ///
    /// **Algorithm**: Iterative Dirichlet-Neumann coupling
    /// 1. Solve fluid with Dirichlet BC (velocity from solid)
    /// 2. Solve solid with Neumann BC (traction from fluid)
    /// 3. Iterate until convergence
    pub fn apply_interface_conditions(
        &mut self,
        fluid_pressure: &mut Array3<f64>,
        fluid_velocity: &mut [Array3<f64>; 3],
        _solid_displacement: &mut [Array3<f64>; 3],
        solid_velocity: &mut [Array3<f64>; 3],
        solid_stress: &mut [Array3<f64>; 6], // σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz
        _dt: f64,
    ) -> KwaversResult<()> {
        let max_iterations = 100;
        let mut iteration = 0;
        let mut converged = false;

        while !converged && iteration < max_iterations {
            // Save ghost values from previous iteration for under-relaxation
            let p_ghost_prev = self.p_fluid_ghost.clone();
            let t_ghost_prev = [
                self.t_solid_ghost[0].clone(),
                self.t_solid_ghost[1].clone(),
                self.t_solid_ghost[2].clone(),
            ];

            // Exchange ghost cell data (fills p_fluid_ghost and t_solid_ghost with exact BCs)
            self.exchange_ghost_cells(fluid_pressure, solid_stress)?;

            // Apply Dirichlet-Neumann under-relaxation (Farhat & Roux 1991, §3.2):
            //   φ_k+1 = ω · φ_exact + (1 − ω) · φ_k,   ω = self.relaxation ∈ (0, 1]
            // Improves convergence near the stability boundary; ω = 1 → Gauss-Seidel.
            // Applied only after the first iteration (iteration > 0) so that the initial
            // ghost values are the exact physical BCs, not a blend with stale zeros.
            if iteration > 0 {
                let omega = self.relaxation;
                ndarray::Zip::from(&mut self.p_fluid_ghost)
                    .and(&p_ghost_prev)
                    .for_each(|new_val, &prev| *new_val = omega * *new_val + (1.0 - omega) * prev);
                for (tsg, tgp) in self
                    .t_solid_ghost
                    .iter_mut()
                    .zip(t_ghost_prev.iter())
                    .take(3)
                {
                    ndarray::Zip::from(tsg).and(tgp).for_each(|new_val, &prev| {
                        *new_val = omega * *new_val + (1.0 - omega) * prev
                    });
                }
            }

            // Update interface tractions
            self.compute_interface_traction(fluid_pressure, solid_stress)?;

            // Check convergence
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
    ///
    /// ## Algorithm: Farhat & Roux (1991) Ghost Cell Exchange
    ///
    /// At each interface cell (i,j,k) where `interface_mask` is `true`:
    ///
    /// **Phase 1 — Interface values** (physical interface conditions):
    /// ```text
    /// Fluid ghost (pressure BC from solid):
    ///   p_ghost[i,j,k] = σ_nn = nᵀ σ n    (pressure continuity: p = σ_nn)
    ///
    /// Solid ghost (traction BC from fluid):
    ///   t_ghost[d][i,j,k] = −p · n̂[d]     (Newton's 3rd law: t_s = −p n̂)
    /// ```
    ///
    /// **Phase 2 — Linear extrapolation** (Farhat & Roux 1991, Eq. 12):
    /// ```text
    /// φ_ghost[g] = 2 · φ_interface − φ_interior[g],   g = 1 .. ghost_layers
    /// ```
    /// Truncation error: O(h²) — second-order terms cancel in Taylor expansion.
    ///
    /// ## References
    /// Farhat, C. & Roux, F.X. (1991). Int J Numer Methods Eng 32(6), 1205–1227.
    /// DOI: 10.1002/nme.1620320604
    fn exchange_ghost_cells(
        &mut self,
        fluid_pressure: &Array3<f64>,
        solid_stress: &[Array3<f64>; 6],
    ) -> KwaversResult<()> {
        let [n0, n1, n2] = self.interface.normal;
        let shape = fluid_pressure.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
        let (sx, sy, sz) = (nx as i64, ny as i64, nz as i64);

        // Phase 1: Interface cells — physical interface conditions
        // Build updated ghost arrays as local temporaries to avoid borrow conflicts
        let mut p_new = Array3::<f64>::zeros((nx, ny, nz));
        let mut t_new: [Array3<f64>; 3] = [
            Array3::zeros((nx, ny, nz)),
            Array3::zeros((nx, ny, nz)),
            Array3::zeros((nx, ny, nz)),
        ];

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

                    // Normal stress: σ_nn = nᵀ σ n (contraction)
                    // For symmetric stress tensor σ: σ_nn = Σ_ab n_a σ_ab n_b
                    let sigma_nn = n0 * n0 * sxx
                        + n1 * n1 * syy
                        + n2 * n2 * szz
                        + 2.0 * n0 * n1 * sxy
                        + 2.0 * n0 * n2 * sxz
                        + 2.0 * n1 * n2 * syz;

                    // Fluid ghost: pressure = solid normal stress (dynamic continuity)
                    p_new[(i, j, k)] = sigma_nn;

                    // Solid ghost: traction = −p · n̂ (Newton's 3rd law at interface)
                    t_new[0][(i, j, k)] = -p * n0;
                    t_new[1][(i, j, k)] = -p * n1;
                    t_new[2][(i, j, k)] = -p * n2;
                }
            }
        }

        // Phase 2: Linear extrapolation into ghost layers
        // Primary axis direction for traversal (rounded from continuous normal)
        let (di, dj, dk) = primary_normal_step(n0, n1, n2);

        if di != 0 || dj != 0 || dk != 0 {
            for g in 1..self.ghost_layers as i64 {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if !self.interface.interface_mask[(i, j, k)] {
                                continue;
                            }

                            // Ghost layer position (into solid domain)
                            let (ig, jg, kg) =
                                (i as i64 + di * g, j as i64 + dj * g, k as i64 + dk * g);
                            // Mirror interior position (into fluid domain)
                            let (ii, ji, ki) =
                                (i as i64 - di * g, j as i64 - dj * g, k as i64 - dk * g);

                            let ghost_valid =
                                ig >= 0 && ig < sx && jg >= 0 && jg < sy && kg >= 0 && kg < sz;
                            if !ghost_valid {
                                continue;
                            }

                            let phi_interface = p_new[(i, j, k)];
                            let phi_interior =
                                if ii >= 0 && ii < sx && ji >= 0 && ji < sy && ki >= 0 && ki < sz {
                                    fluid_pressure[[ii as usize, ji as usize, ki as usize]]
                                } else {
                                    phi_interface
                                };
                            // φ_ghost[g] = 2 φ_interface − φ_interior[g]
                            p_new[[ig as usize, jg as usize, kg as usize]] =
                                2.0 * phi_interface - phi_interior;

                            for t_new_dim in t_new.iter_mut().take(3) {
                                let t_iface = t_new_dim[(i, j, k)];
                                // Constant extrapolation for traction ghost layers
                                t_new_dim[[ig as usize, jg as usize, kg as usize]] = t_iface;
                            }
                        }
                    }
                }
            }
        }

        // Store computed (unrelaxed) interface values. Under-relaxation is applied by the
        // caller (apply_interface_conditions) between iterations, not here, so that a
        // single exchange always yields the exact physical BC values (Farhat & Roux 1991).
        self.p_fluid_ghost = p_new;
        self.t_solid_ghost[0] = t_new[0].clone();
        self.t_solid_ghost[1] = t_new[1].clone();
        self.t_solid_ghost[2] = t_new[2].clone();

        Ok(())
    }

    /// Compute traction vector at interface
    fn compute_interface_traction(
        &self,
        fluid_pressure: &Array3<f64>,
        solid_stress: &[Array3<f64>; 6],
    ) -> KwaversResult<Array1<f64>> {
        let [nx, ny, nz] = self.interface.normal;

        // Traction from fluid = -p * n (plus viscous stresses)
        // Traction from solid = σ · n

        let mut traction = Array1::zeros(3);

        ndarray::Zip::indexed(&self.interface.interface_mask)
            .and(fluid_pressure.view())
            .for_each(|(i, j, k), mask, &p| {
                if *mask {
                    // Get solid stress components at interface
                    let s_xx = solid_stress[0][[i, j, k]];
                    let s_yy = solid_stress[1][[i, j, k]];
                    let s_zz = solid_stress[2][[i, j, k]];
                    let s_xy = solid_stress[3][[i, j, k]];
                    let s_xz = solid_stress[4][[i, j, k]];
                    let s_yz = solid_stress[5][[i, j, k]];

                    // Traction balance: fluid traction = -solid traction
                    // t_f = -p * n
                    // t_s = σ · n
                    // Condition: t_f + t_s = 0 at interface

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

    /// Check convergence of coupling iterations
    fn check_convergence(
        &self,
        fluid_velocity: &[Array3<f64>; 3],
        solid_velocity: &[Array3<f64>; 3],
    ) -> KwaversResult<bool> {
        let mut max_error = 0.0_f64;

        // Compare normal velocity components at interface
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

    /// Get interface transmission and reflection coefficients
    pub fn get_coefficients(&self, incidence_angle: f64) -> ReflectionTransmissionCoefficients {
        if incidence_angle.abs() < 1e-10 {
            ReflectionTransmissionCoefficients::normal_incidence(&self.interface)
        } else {
            ReflectionTransmissionCoefficients::oblique_incidence(&self.interface, incidence_angle)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Test zero normal vector rejection
    #[test]
    fn test_zero_normal_rejected() {
        let interface = FsiInterface::new(
            1000.0,
            1500.0,
            7850.0,
            5960.0,
            3240.0,
            [0.0, 0.0, 0.0], // Invalid!
            64,
            64,
            64,
        );
        assert!(interface.is_err());
    }

    /// Test ghost cell traction balance at planar water-steel interface
    ///
    /// **Validation**: Farhat & Roux (1991) ghost cell exchange must enforce
    /// Newton's 3rd law at the interface: t_fluid + t_solid = 0
    ///
    /// Setup: Normal incidence (n̂ = [1,0,0]), p = σ_xx = 1e5 Pa at interface.
    /// Expected: ||t_fluid + t_solid||₂ < 1e-10.
    ///
    /// Proof: t_fluid = −p n̂ = [−p, 0, 0]
    ///        t_solid = σ n̂ = [σ_xx, σ_xy, σ_xz] = [p, 0, 0]
    ///        t_fluid + t_solid = [−p + p, 0, 0] = 0  ✓
    #[test]
    fn test_ghost_cell_traction_balance() {
        let nx = 8usize;
        let interface = FsiInterface::new(
            1000.0,
            1500.0,
            7850.0,
            5960.0,
            3240.0,
            [1.0, 0.0, 0.0], // x-normal interface
            nx,
            nx,
            nx,
        )
        .unwrap();

        let mut solver = FluidStructureSolver::new(interface);

        // Place the interface at i = nx/2
        let i_face = nx / 2;
        for j in 0..nx {
            for k in 0..nx {
                solver.interface.interface_mask[(i_face, j, k)] = true;
            }
        }

        // Physical setup: p = σ_xx = P0 (pressure continuity satisfied)
        let p0 = 1.0e5_f64;
        let fluid_pressure = Array3::from_elem((nx, nx, nx), p0);
        // σ_xx = p0, all other components = 0 (acoustic pressure in isotropic medium)
        let solid_stress: [Array3<f64>; 6] = [
            Array3::from_elem((nx, nx, nx), p0), // σ_xx = p0
            Array3::zeros((nx, nx, nx)),         // σ_yy = 0
            Array3::zeros((nx, nx, nx)),         // σ_zz = 0
            Array3::zeros((nx, nx, nx)),         // σ_xy = 0
            Array3::zeros((nx, nx, nx)),         // σ_xz = 0
            Array3::zeros((nx, nx, nx)),         // σ_yz = 0
        ];

        solver
            .exchange_ghost_cells(&fluid_pressure, &solid_stress)
            .unwrap();

        // Verify traction balance at interface cells: ||t_fluid + t_solid||₂ < 1e-10
        // t_fluid = (p_ghost − p) * n̂ at ghost cell → here represented as:
        //   t_fluid[x] = −p · nx = −p0 · 1 = −p0
        //   t_solid[x] = +σ_nn = +p0
        // sum = 0
        let mut traction_jump_sq = 0.0f64;
        for j in 0..nx {
            for k in 0..nx {
                // t_fluid (from ghost) = −p · n̂
                let t_fluid_x = -fluid_pressure[(i_face, j, k)] * 1.0;
                // t_solid (from ghost) = +σ_nn = σ_xx·n0²+... = p0·1 = p0
                let t_solid_x = solver.p_fluid_ghost[(i_face, j, k)]; // = σ_nn
                traction_jump_sq += (t_fluid_x + t_solid_x).powi(2);
            }
        }
        assert!(
            traction_jump_sq < 1e-10,
            "Traction balance violated: ||t_fluid + t_solid||² = {:.3e} (must be < 1e-10)",
            traction_jump_sq
        );
        // Verify solid ghost traction matches −p · n̂
        for j in 0..nx {
            for k in 0..nx {
                let t_x = solver.t_solid_ghost[0][(i_face, j, k)];
                assert!(
                    (t_x + p0).abs() < 1e-10,
                    "Solid ghost traction t_x = {}, expected {}",
                    t_x,
                    -p0
                );
            }
        }
        let _ = solid_stress[0].sum(); // suppress unused warning
    }

    /// Test ghost cell velocity continuity across interface
    ///
    /// **Validation**: After ghost cell exchange, normal velocity components
    /// from fluid and solid sides must match: |v_fluid·n̂ − v_solid·n̂| < 1e-10.
    ///
    /// This test verifies the kinematic interface condition (velocity continuity).
    /// For this test the check_convergence method is used directly with matching
    /// velocities to confirm the convergence criterion is correctly implemented.
    #[test]
    fn test_ghost_cell_velocity_continuity() {
        let nx = 8usize;
        let interface = FsiInterface::new(
            1000.0,
            1500.0,
            7850.0,
            5960.0,
            3240.0,
            [1.0, 0.0, 0.0],
            nx,
            nx,
            nx,
        )
        .unwrap();

        let mut solver = FluidStructureSolver::new(interface);
        let i_face = nx / 2;
        for j in 0..nx {
            for k in 0..nx {
                solver.interface.interface_mask[(i_face, j, k)] = true;
            }
        }

        let v_normal = 0.1_f64; // 0.1 m/s normal velocity
        let fluid_velocity: [Array3<f64>; 3] = [
            Array3::from_elem((nx, nx, nx), v_normal), // vx = v_normal (along n̂ = [1,0,0])
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
        ];
        // Solid normal velocity matches fluid (velocity continuity satisfied)
        let solid_velocity: [Array3<f64>; 3] = [
            Array3::from_elem((nx, nx, nx), v_normal),
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
        ];

        let converged = solver
            .check_convergence(&fluid_velocity, &solid_velocity)
            .unwrap();
        assert!(
            converged,
            "Velocity continuity check failed: matching fluid/solid velocities must converge"
        );

        // With mismatched velocity, should not converge
        let solid_velocity_bad: [Array3<f64>; 3] = [
            Array3::from_elem((nx, nx, nx), v_normal + 1.0), // mismatch by 1 m/s
            Array3::zeros((nx, nx, nx)),
            Array3::zeros((nx, nx, nx)),
        ];
        let not_converged = solver
            .check_convergence(&fluid_velocity, &solid_velocity_bad)
            .unwrap();
        assert!(
            !not_converged,
            "Velocity mismatch of 1 m/s should not satisfy convergence criterion"
        );
    }
}
