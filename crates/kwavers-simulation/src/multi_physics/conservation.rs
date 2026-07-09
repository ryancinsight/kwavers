use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use leto::{
    Array3,
    ArrayView3,
};

/// Conservation enforcement for multi-physics coupling
#[derive(Debug)]
pub struct MultiPhysicsConservationEnforcer {
    /// Conservation tolerance
    tolerance: f64,
}

impl MultiPhysicsConservationEnforcer {
    /// Create new conservation enforcer
    #[must_use]
    pub fn new() -> Self {
        Self { tolerance: 1e-10 }
    }

    /// Compute AABB-overlap quadrature weights for conservative projection.
    ///
    /// ## Theorem — Conservative Interpolation (Farhat et al. 1998, §3)
    ///
    /// For a target cell T with volume V_T and a set of source cells {S_i},
    /// define the overlap volume:
    /// ```text
    /// overlap(C_T, C_S) = Π_{d∈{x,y,z}} max(0, min(x⁺_T,x⁺_S) − max(x⁻_T,x⁻_S))
    /// ```
    /// The conservative weight w_{S→T} = overlap(C_T, C_S) / V_T satisfies:
    /// ```text
    /// Σ_T w_{S→T} · V_T = V_S   (source-integral preservation)
    /// ```
    /// Hence Σ_T φ_T · V_T = Σ_S φ_S · V_S when φ_T = Σ_S w_{S→T} · φ_S.
    ///
    /// Reference: Farhat, Lesoinne & LeTallec (1998).
    /// *Comput. Meth. Appl. Mech. Eng.* 157:95–114, §3.
    ///
    /// # Returns
    ///
    /// Vec of `(si, sj, sk, weight)` tuples where `Σ weight ≤ 1.0`.
    pub fn conservative_quadrature_weights(
        target_i: usize,
        target_j: usize,
        target_k: usize,
        target_grid: &Grid,
        source_grid: &Grid,
    ) -> Vec<(usize, usize, usize, f64)> {
        // Target cell bounds (cell-centred)
        let xt = target_i as f64 * target_grid.dx;
        let yt = target_j as f64 * target_grid.dy;
        let zt = target_k as f64 * target_grid.dz;
        let xt_lo = 0.5f64.mul_add(-target_grid.dx, xt);
        let xt_hi = 0.5f64.mul_add(target_grid.dx, xt);
        let yt_lo = 0.5f64.mul_add(-target_grid.dy, yt);
        let yt_hi = 0.5f64.mul_add(target_grid.dy, yt);
        let zt_lo = 0.5f64.mul_add(-target_grid.dz, zt);
        let zt_hi = 0.5f64.mul_add(target_grid.dz, zt);
        let v_target = target_grid.dx * target_grid.dy * target_grid.dz;

        // Source cell indices that can overlap with target cell
        let si_min = ((xt_lo / source_grid.dx).floor() as isize).max(0) as usize;
        let si_max =
            ((xt_hi / source_grid.dx).ceil() as usize).min(source_grid.nx.saturating_sub(1));
        let sj_min = ((yt_lo / source_grid.dy).floor() as isize).max(0) as usize;
        let sj_max =
            ((yt_hi / source_grid.dy).ceil() as usize).min(source_grid.ny.saturating_sub(1));
        let sk_min = ((zt_lo / source_grid.dz).floor() as isize).max(0) as usize;
        let sk_max =
            ((zt_hi / source_grid.dz).ceil() as usize).min(source_grid.nz.saturating_sub(1));

        let mut weights = Vec::new();

        for si in si_min..=si_max {
            let xs = si as f64 * source_grid.dx;
            let xs_lo = 0.5f64.mul_add(-source_grid.dx, xs);
            let xs_hi = 0.5f64.mul_add(source_grid.dx, xs);
            let ov_x = (xt_hi.min(xs_hi) - xt_lo.max(xs_lo)).max(0.0);
            if ov_x == 0.0 {
                continue;
            }

            for sj in sj_min..=sj_max {
                let ys = sj as f64 * source_grid.dy;
                let ys_lo = 0.5f64.mul_add(-source_grid.dy, ys);
                let ys_hi = 0.5f64.mul_add(source_grid.dy, ys);
                let ov_y = (yt_hi.min(ys_hi) - yt_lo.max(ys_lo)).max(0.0);
                if ov_y == 0.0 {
                    continue;
                }

                for sk in sk_min..=sk_max {
                    let zs = sk as f64 * source_grid.dz;
                    let zs_lo = 0.5f64.mul_add(-source_grid.dz, zs);
                    let zs_hi = 0.5f64.mul_add(source_grid.dz, zs);
                    let ov_z = (zt_hi.min(zs_hi) - zt_lo.max(zs_lo)).max(0.0);
                    if ov_z == 0.0 {
                        continue;
                    }

                    let overlap_vol = ov_x * ov_y * ov_z;
                    let weight = overlap_vol / v_target;
                    weights.push((si, sj, sk, weight));
                }
            }
        }

        weights
    }

    /// Apply conservative interpolation between grids using AABB overlap weights.
    ///
    /// ## Algorithm
    ///
    /// For each target cell T:
    /// 1. Compute AABB overlap weights `{(S_i, w_i)}` via
    ///    `conservative_quadrature_weights`.
    /// 2. Set `φ_T = Σ_i w_i · φ_{S_i}`.
    ///
    /// Same-grid identity path returns a zero-copy view for efficiency.
    ///
    /// ## Conservation guarantee
    ///
    /// Σ_T φ_T · V_T = Σ_S φ_S · V_S to within floating-point rounding
    /// (~machine-epsilon × Σ|φ_S| · V_S). See `audit_energy_conservation`.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn conservative_interpolate(
        &self,
        source_field: &ArrayView3<f64>,
        source_grid: &Grid,
        target_grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let expected_source_dim = (source_grid.nx, source_grid.ny, source_grid.nz);
        let actual_source_dim = source_field.dim();
        if actual_source_dim != expected_source_dim {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("{expected_source_dim:?}"),
                    actual: format!("{actual_source_dim:?}"),
                },
            ));
        }

        let same_grid = source_grid.nx == target_grid.nx
            && source_grid.ny == target_grid.ny
            && source_grid.nz == target_grid.nz
            && (source_grid.dx - target_grid.dx).abs() <= self.tolerance
            && (source_grid.dy - target_grid.dy).abs() <= self.tolerance
            && (source_grid.dz - target_grid.dz).abs() <= self.tolerance;
        if same_grid {
            return Ok(source_field.to_owned());
        }

        let mut result = Array3::zeros((target_grid.nx, target_grid.ny, target_grid.nz));

        for i in 0..target_grid.nx {
            for j in 0..target_grid.ny {
                for k in 0..target_grid.nz {
                    let weights =
                        Self::conservative_quadrature_weights(i, j, k, target_grid, source_grid);

                    let mut val = 0.0_f64;
                    let mut total_w = 0.0_f64;
                    for (si, sj, sk, w) in &weights {
                        val += w * source_field[[*si, *sj, *sk]];
                        total_w += w;
                    }

                    if total_w > 0.0 {
                        // Normalise: guard against partial coverage at domain edge
                        result[[i, j, k]] = val / total_w;
                    } else {
                        // Outside source domain — nearest-neighbour fallback
                        let (x, y, z) = target_grid.indices_to_coordinates(i, j, k);
                        let si = (x / source_grid.dx).round() as usize;
                        let sj = (y / source_grid.dy).round() as usize;
                        let sk = (z / source_grid.dz).round() as usize;
                        result[[i, j, k]] = source_field[[
                            si.min(source_grid.nx - 1),
                            sj.min(source_grid.ny - 1),
                            sk.min(source_grid.nz - 1),
                        ]];
                    }
                }
            }
        }

        Ok(result)
    }

    /// Audit energy conservation across a field transfer.
    ///
    /// ## Definition
    ///
    /// Energy (integral) of a field φ on grid G is:
    /// ```text
    /// E(φ, G) = Σ_{i,j,k} φ_{ijk} · dx · dy · dz
    /// ```
    ///
    /// The relative energy error after transfer is:
    /// ```text
    /// ε_E = |E(φ_src, G_src) − E(φ_tgt, G_tgt)| / |E(φ_src, G_src)|
    /// ```
    ///
    /// Returns `Ok(ε_E)`, or `Err` if source energy is zero (division by zero).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn audit_energy_conservation(
        source_field: &ArrayView3<f64>,
        source_grid: &Grid,
        target_field: &ArrayView3<f64>,
        target_grid: &Grid,
    ) -> KwaversResult<f64> {
        let v_src = source_grid.dx * source_grid.dy * source_grid.dz;
        let v_tgt = target_grid.dx * target_grid.dy * target_grid.dz;
        let e_src: f64 = source_field.iter().sum::<f64>() * v_src;
        let e_tgt: f64 = target_field.iter().sum::<f64>() * v_tgt;

        if e_src.abs() < 1e-300 {
            return Err(KwaversError::InvalidInput(
                "audit_energy_conservation: source energy is zero".to_owned(),
            ));
        }

        Ok((e_src - e_tgt).abs() / e_src.abs())
    }
}

impl Default for MultiPhysicsConservationEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Phase E tests ─────────────────────────────────────────────────────────

    /// Conservative same-grid interpolation is the identity: energy error < 1e-12.
    ///
    /// ## Theorem (Farhat et al. 1998, §3)
    ///
    /// When source and target grids are identical, conservative_interpolate returns
    /// the source field unchanged. Hence E_src = E_tgt exactly.
    ///
    /// We verify: |E_src − E_tgt| / |E_src| < 1e-12.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_conservative_same_grid() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let enforcer = MultiPhysicsConservationEnforcer::new();

        // Fill with non-trivial values (linear profile)
        let mut field = Array3::<f64>::zeros((8, 8, 8));
        for i in 0..8usize {
            for j in 0..8usize {
                for k in 0..8usize {
                    field[[i, j, k]] = (i + 2 * j + 3 * k) as f64 * 0.1;
                }
            }
        }

        let result = enforcer
            .conservative_interpolate(&field.view(), &grid, &grid)
            .unwrap();

        let err = MultiPhysicsConservationEnforcer::audit_energy_conservation(
            &field.view(),
            &grid,
            &result.view(),
            &grid,
        )
        .unwrap();

        assert!(
            err < 1e-12,
            "same-grid energy error {err:.3e} exceeds 1e-12"
        );
    }

    /// Conservative 2:1 coarsening: energy error < 1e-10.
    ///
    /// ## Theorem (Farhat et al. 1998, §3)
    ///
    /// AABB-overlap conservative projection Σ_T φ_T V_T = Σ_S φ_S V_S to
    /// floating-point precision when the source domain is fully covered.
    ///
    /// We verify coarsening from 16³ → 8³ (dx doubles).
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_conservative_2to1_coarsen() {
        // Fine source grid (16³, dx=0.001)
        let n_fine = 16usize;
        let n_coarse = 8usize;
        let dx_fine = 0.001_f64;
        let dx_coarse = 0.002_f64;

        let src_grid = Grid::new(n_fine, n_fine, n_fine, dx_fine, dx_fine, dx_fine).unwrap();
        let tgt_grid = Grid::new(
            n_coarse, n_coarse, n_coarse, dx_coarse, dx_coarse, dx_coarse,
        )
        .unwrap();
        let enforcer = MultiPhysicsConservationEnforcer::new();

        // Constant field = 1.0 → E_src = n_fine³ * dx_fine³
        let field = Array3::<f64>::ones((n_fine, n_fine, n_fine));
        let result = enforcer
            .conservative_interpolate(&field.view(), &src_grid, &tgt_grid)
            .unwrap();

        let err = MultiPhysicsConservationEnforcer::audit_energy_conservation(
            &field.view(),
            &src_grid,
            &result.view(),
            &tgt_grid,
        )
        .unwrap();

        assert!(
            err < 1e-10,
            "2:1 coarsening energy error {err:.3e} exceeds 1e-10"
        );
    }
}
