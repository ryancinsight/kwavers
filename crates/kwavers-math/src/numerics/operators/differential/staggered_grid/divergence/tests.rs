//! The adjointness identity, asserted directly.
//!
//! `D = −Gᵀ` is the property the Yee leapfrog's energy conservation rests on,
//! so it is tested as an identity on arbitrary fields rather than inferred from
//! a simulation staying bounded.

use super::*;
use leto::Array3;

fn operator() -> StaggeredGridOperator {
    // Deliberately anisotropic spacings: a divergence that dropped or swapped a
    // spacing would still pass an isotropic test.
    StaggeredGridOperator::new(1.5e-4, 3.0e-4, 7.0e-4).expect("valid spacings")
}

/// Deterministic, non-symmetric field values — a symmetric or linear field can
/// hide an asymmetric boundary error.
fn seeded(shape: [usize; 3], salt: f64) -> Array3<f64> {
    let mut field = Array3::<f64>::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let t = i as f64 * 0.7 + j as f64 * 1.3 + k as f64 * 2.1 + salt;
                field[[i, j, k]] = t.sin() * 1.7 + t.cos() * 0.4 + 0.3 * (i as f64 - j as f64);
            }
        }
    }
    field
}

/// **`⟨Gp, u⟩ = −⟨p, Du⟩`** along X, for arbitrary `p` and `u` whose far face
/// vanishes — the exact identity the leapfrog needs.
///
/// `G` is the velocity update's forward difference: `(p[i+1] − p[i])/Δx` for
/// `i < nx−1`, and the far face is zeroed, which is what the FDTD velocity
/// update does.
#[test]
fn divergence_is_the_negative_adjoint_of_the_forward_difference_x() {
    let op = operator();
    let shape = [7usize, 4, 3];
    let pressure = seeded(shape, 0.0);
    let mut velocity = seeded(shape, 4.2);

    // The two closures differ by exactly `p[0]·(u[1] − 2u[0])/Δx`, so a test
    // field with a vanishing low-face pressure would satisfy the identity under
    // *both* and prove nothing. Guard the discriminating power explicitly: an
    // earlier draft of this check used a field with `p[0] = 0` and reported a
    // zero residual for the defective closure.
    for j in 0..shape[1] {
        for k in 0..shape[2] {
            assert!(
                pressure[[0, j, k]].abs() > 1e-3,
                "low-face pressure must be non-zero or this test is vacuous"
            );
        }
    }
    // The far face carries no velocity, matching the solver's Dirichlet edge.
    for j in 0..shape[1] {
        for k in 0..shape[2] {
            velocity[[shape[0] - 1, j, k]] = 0.0;
        }
    }

    // ⟨Gp, u⟩ over the interior faces.
    let mut gradient_inner = 0.0;
    for i in 0..shape[0] - 1 {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let gradient = (pressure[[i + 1, j, k]] - pressure[[i, j, k]]) / op.dx;
                gradient_inner += gradient * velocity[[i, j, k]];
            }
        }
    }

    let mut divergence = Array3::<f64>::zeros(shape);
    op.apply_divergence_x_into(velocity.view(), &mut divergence)
        .expect("divergence applies");
    let mut divergence_inner = 0.0;
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                divergence_inner += pressure[[i, j, k]] * divergence[[i, j, k]];
            }
        }
    }

    let scale = gradient_inner.abs().max(divergence_inner.abs()).max(1.0);
    assert!(
        (gradient_inner + divergence_inner).abs() < 1e-9 * scale,
        "adjointness violated: <Gp,u> = {gradient_inner:.12e}, <p,Du> = {divergence_inner:.12e}"
    );
}

/// The same identity along Y and Z, so a per-axis slip cannot hide.
#[test]
fn divergence_is_the_negative_adjoint_along_every_axis() {
    let op = operator();
    let shape = [4usize, 6, 5];

    for axis in 0..3 {
        let pressure = seeded(shape, axis as f64);
        let mut velocity = seeded(shape, 9.1 + axis as f64);
        let spacing = [op.dx, op.dy, op.dz][axis];

        // Zero the far face along this axis.
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let index = [i, j, k];
                    if index[axis] == shape[axis] - 1 {
                        velocity[index] = 0.0;
                    }
                }
            }
        }

        let mut gradient_inner = 0.0;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let index = [i, j, k];
                    if index[axis] + 1 >= shape[axis] {
                        continue;
                    }
                    let mut next = index;
                    next[axis] += 1;
                    let gradient = (pressure[next] - pressure[index]) / spacing;
                    gradient_inner += gradient * velocity[index];
                }
            }
        }

        let mut divergence = Array3::<f64>::zeros(shape);
        match axis {
            0 => op.apply_divergence_x_into(velocity.view(), &mut divergence),
            1 => op.apply_divergence_y_into(velocity.view(), &mut divergence),
            _ => op.apply_divergence_z_into(velocity.view(), &mut divergence),
        }
        .expect("divergence applies");

        let mut divergence_inner = 0.0;
        for value in divergence.iter().zip(pressure.iter()) {
            divergence_inner += value.0 * value.1;
        }

        let scale = gradient_inner.abs().max(divergence_inner.abs()).max(1.0);
        assert!(
            (gradient_inner + divergence_inner).abs() < 1e-9 * scale,
            "axis {axis}: adjointness violated, {gradient_inner:.9e} vs {divergence_inner:.9e}"
        );
    }
}

/// The interior is unchanged from the backward difference; only the low face
/// differs. This pins that the fix is a boundary closure and not a change to
/// the stencil.
#[test]
fn interior_matches_the_backward_difference() {
    let op = operator();
    let shape = [6usize, 3, 2];
    let field = seeded(shape, 1.1);

    let mut backward = Array3::<f64>::zeros(shape);
    op.apply_backward_x_into(field.view(), &mut backward)
        .expect("backward applies");
    let mut divergence = Array3::<f64>::zeros(shape);
    op.apply_divergence_x_into(field.view(), &mut divergence)
        .expect("divergence applies");

    for i in 1..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                assert_eq!(
                    backward[[i, j, k]],
                    divergence[[i, j, k]],
                    "interior cell {i},{j},{k} must be untouched"
                );
            }
        }
    }
    // And the low face is the zero-flux value, not the one-sided difference.
    for j in 0..shape[1] {
        for k in 0..shape[2] {
            assert_eq!(divergence[[0, j, k]], field[[0, j, k]] / op.dx);
            assert_ne!(divergence[[0, j, k]], backward[[0, j, k]]);
        }
    }
}

/// A velocity field that vanishes at the low face has zero flux there, so both
/// closures agree — the physically consistent case.
#[test]
fn closures_agree_when_the_low_face_velocity_vanishes() {
    let op = operator();
    let shape = [5usize, 2, 2];
    let mut field = seeded(shape, 2.5);
    for j in 0..shape[1] {
        for k in 0..shape[2] {
            field[[0, j, k]] = 0.0;
        }
    }

    let mut divergence = Array3::<f64>::zeros(shape);
    op.apply_divergence_x_into(field.view(), &mut divergence)
        .expect("divergence applies");
    for j in 0..shape[1] {
        for k in 0..shape[2] {
            assert_eq!(divergence[[0, j, k]], 0.0);
        }
    }
}
