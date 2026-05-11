use crate::domain::field::UnifiedFieldType;
use ndarray::{s, Array3};
use std::collections::HashMap;

/// Sorted field keys for deterministic flatten/unflatten ordering
pub(super) fn sorted_field_keys(
    fields: &HashMap<UnifiedFieldType, Array3<f64>>,
) -> Vec<UnifiedFieldType> {
    let mut keys: Vec<UnifiedFieldType> = fields.keys().copied().collect();
    keys.sort_by_key(|k| k.index());
    keys
}

/// Flatten field map to a single `Array3<f64>` by stacking along axis 0.
///
/// Fields are stacked in the order given by `field_order` (sorted by
/// `UnifiedFieldType::index()`).  Each field of shape `(nx, ny, nz)` becomes
/// rows `[i*nx .. (i+1)*nx]` in the output of shape `(n_fields*nx, ny, nz)`.
pub(super) fn flatten_fields(
    fields: &HashMap<UnifiedFieldType, Array3<f64>>,
    field_order: &[UnifiedFieldType],
) -> Array3<f64> {
    if field_order.is_empty() {
        return Array3::zeros((1, 1, 1));
    }

    let first = &fields[&field_order[0]];
    let (nx, ny, nz) = first.dim();
    let n_fields = field_order.len();

    let mut stacked = Array3::zeros((n_fields * nx, ny, nz));
    for (i, ft) in field_order.iter().enumerate() {
        let src = &fields[ft];
        stacked
            .slice_mut(s![i * nx..(i + 1) * nx, .., ..])
            .assign(src);
    }
    stacked
}

/// Unflatten solution vector back to field map.
///
/// Inverse of [`flatten_fields`]: splits the stacked array along axis 0
/// and writes each block back into the corresponding field.
pub(super) fn unflatten_fields(
    u: &Array3<f64>,
    fields: &mut HashMap<UnifiedFieldType, Array3<f64>>,
    field_order: &[UnifiedFieldType],
) {
    if field_order.is_empty() {
        return;
    }

    let total_rows = u.dim().0;
    let nx = total_rows / field_order.len();

    for (i, ft) in field_order.iter().enumerate() {
        if let Some(field) = fields.get_mut(ft) {
            field.assign(&u.slice(s![i * nx..(i + 1) * nx, .., ..]));
        }
    }
}

/// Compute L2 norm
pub(super) fn norm(a: &Array3<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ============================================================================
// Free-standing helper: 3-D Laplacian via central finite differences
// ============================================================================

/// Compute the 3-D Laplacian ∇²f using second-order central differences.
///
/// ## Algorithm (second-order central finite differences)
///
/// ```text
/// ∇²f[i,j,k] ≈ (f[i+1,j,k] - 2f[i,j,k] + f[i-1,j,k]) / dx²
///             + (f[i,j+1,k] - 2f[i,j,k] + f[i,j-1,k]) / dy²
///             + (f[i,j,k+1] - 2f[i,j,k] + f[i,j,k-1]) / dz²
/// ```
///
/// Truncation error: O(dx², dy², dz²).
/// Boundary nodes use zero-gradient (homogeneous Neumann) ghost-cell conditions.
///
/// ## Parameters
/// - `dx`, `dy`, `dz`: physical cell spacings (m). Must be > 0.
///
/// ## Reference
/// - LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary and Partial
///   Differential Equations*. SIAM, Philadelphia. §1.3, Eq. (1.3).
pub(super) fn laplacian_3d(
    field: &Array3<f64>,
    grid_dims: (usize, usize, usize),
    dx: f64,
    dy: f64,
    dz: f64,
) -> Array3<f64> {
    let (nx, ny, nz) = field.dim();
    let _ = grid_dims; // passed for call-site documentation; field.dim() is authoritative

    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);
    let inv_dz2 = 1.0 / (dz * dz);

    let mut lap = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // x-direction
                let d2x = if nx > 2 {
                    let im = if i == 0 { 0 } else { i - 1 };
                    let ip = if i == nx - 1 { nx - 1 } else { i + 1 };
                    (2.0f64.mul_add(-field[[i, j, k]], field[[ip, j, k]]) + field[[im, j, k]]) * inv_dx2
                } else {
                    0.0
                };
                // y-direction
                let d2y = if ny > 2 {
                    let jm = if j == 0 { 0 } else { j - 1 };
                    let jp = if j == ny - 1 { ny - 1 } else { j + 1 };
                    (2.0f64.mul_add(-field[[i, j, k]], field[[i, jp, k]]) + field[[i, jm, k]]) * inv_dy2
                } else {
                    0.0
                };
                // z-direction
                let d2z = if nz > 2 {
                    let km = if k == 0 { 0 } else { k - 1 };
                    let kp = if k == nz - 1 { nz - 1 } else { k + 1 };
                    (2.0f64.mul_add(-field[[i, j, k]], field[[i, j, kp]]) + field[[i, j, km]]) * inv_dz2
                } else {
                    0.0
                };

                lap[[i, j, k]] = d2x + d2y + d2z;
            }
        }
    }

    lap
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_flatten_unflatten_round_trip() {
        let nx = 4;
        let ny = 3;
        let nz = 2;
        let mut fields = HashMap::new();
        let mut pressure = Array3::zeros((nx, ny, nz));
        pressure[[1, 1, 1]] = 42.0;
        let mut temp = Array3::zeros((nx, ny, nz));
        temp[[2, 0, 0]] = 7.0;

        fields.insert(UnifiedFieldType::Pressure, pressure.clone());
        fields.insert(UnifiedFieldType::Temperature, temp.clone());

        let order = sorted_field_keys(&fields);
        let flat = flatten_fields(&fields, &order);
        assert_eq!(flat.dim(), (2 * nx, ny, nz));

        // Unflatten back
        let mut out_fields = HashMap::new();
        out_fields.insert(UnifiedFieldType::Pressure, Array3::zeros((nx, ny, nz)));
        out_fields.insert(UnifiedFieldType::Temperature, Array3::zeros((nx, ny, nz)));
        unflatten_fields(&flat, &mut out_fields, &order);

        assert!((out_fields[&UnifiedFieldType::Pressure][[1, 1, 1]] - 42.0).abs() < 1e-15);
        assert!((out_fields[&UnifiedFieldType::Temperature][[2, 0, 0]] - 7.0).abs() < 1e-15);
    }

    #[test]
    fn test_laplacian_uniform_field() {
        // Laplacian of a constant field should be zero (interior)
        let field = Array3::from_elem((8, 8, 8), 5.0);
        let dx = 1e-3;
        let lap = laplacian_3d(&field, (8, 8, 8), dx, dx, dx);

        // Interior points should be exactly 0
        for i in 1..7 {
            for j in 1..7 {
                for k in 1..7 {
                    assert!(
                        lap[[i, j, k]].abs() < 1e-15,
                        "Laplacian of constant should be 0 at interior [{i},{j},{k}], got {}",
                        lap[[i, j, k]]
                    );
                }
            }
        }
    }

    /// Spacing of 1 m vs 1 mm on identical fields → Laplacian differs by 10⁶.
    ///
    /// ∇²f at interior node is proportional to 1/dx², so changing dx from 1.0
    /// to 1e-3 scales the output by (1/1e-3)² / (1/1.0)² = 1e6.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_laplacian_unit_vs_nonunit_spacing() {
        // f = 1 everywhere except a single interior spike to produce non-zero Laplacian
        let mut field = Array3::from_elem((5, 5, 5), 1.0);
        field[[2, 2, 2]] = 2.0; // spike: Laplacian at (2,2,2) = (1-2·2+1)/dx² × 3 = -6/dx²

        let lap_1m = laplacian_3d(&field, (5, 5, 5), 1.0, 1.0, 1.0);
        let lap_1mm = laplacian_3d(&field, (5, 5, 5), 1e-3, 1e-3, 1e-3);

        let ratio = lap_1mm[[2, 2, 2]] / lap_1m[[2, 2, 2]];
        assert!(
            (ratio - 1e6).abs() < 1.0,
            "1 mm vs 1 m Laplacian ratio should be 1e6, got {ratio}"
        );
    }

    /// ∇²(x²) = 2 exactly for any uniform dx (second-order central difference is exact
    /// for polynomials of degree ≤ 2).
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_laplacian_quadratic_field_exact() {
        let n = 10;
        let dx = 0.5e-3; // 0.5 mm
                         // Build field f[i,j,k] = (i·dx)² so that ∇²f = d²/dx² (x²) = 2
        let field = Array3::from_shape_fn((n, n, n), |(i, _j, _k)| {
            let x = i as f64 * dx;
            x * x
        });
        let lap = laplacian_3d(&field, (n, n, n), dx, dx, dx);

        // Interior x-nodes (not boundary); y,z contributions are 0 since field
        // doesn't vary in j,k. Check a mid-plane interior node.
        let i = n / 2;
        let j = n / 2;
        let k = n / 2;
        assert!(
            (lap[[i, j, k]] - 2.0).abs() < 1e-8,
            "∇²(x²) should equal 2.0, got {}",
            lap[[i, j, k]]
        );
    }

    /// All-zero field → Laplacian is zero everywhere for any spacing.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_laplacian_zero_field() {
        let field = Array3::zeros((6, 6, 6));
        for &dx in &[1.0, 1e-3, 1e-6] {
            let lap = laplacian_3d(&field, (6, 6, 6), dx, dx, dx);
            assert!(
                lap.iter().all(|&v| v.abs() < 1e-15),
                "Laplacian of zero field must be zero for dx={dx}"
            );
        }
    }
}
