//! Local frequency estimation (LFE) inversion.
//!
//! Estimates the local spatial wavenumber of a harmonic shear-wave displacement
//! field directly, then maps it to shear-wave speed and modulus. Unlike the
//! pointwise `direct` method (which divides the Laplacian `∇²u` by `u` and is
//! therefore sensitive to noise and to the zero-crossings of `u`), LFE forms the
//! ratio of *windowed* energies of the first derivative and the field itself.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::ElasticityMap;
use kwavers_physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;
use leto::Array3 as LetoArray3;
use ndarray::Array3;

use super::super::algorithms::{fill_boundaries, spatial_smoothing};
use super::super::types::elasticity_map_from_speed;

/// Number of box-smoothing passes used to build the local averaging window.
///
/// For a locally-planar harmonic field `u = A sin(k·x + φ)`, the pointwise ratio
/// `|∇u|² / u²` oscillates (it is `|k|² cot²(k·x + φ)`). Averaging numerator and
/// denominator over a window wide compared with one zero-crossing replaces the
/// oscillating factors by their means (`⟨cos²⟩ = ⟨sin²⟩ = 1/2`), leaving the
/// phase-independent estimate `⟨|∇u|²⟩ / ⟨u²⟩ → |k|²`.
const LFE_SMOOTHING_PASSES: usize = 3;

/// Local frequency estimation inversion.
///
/// # Theory
///
/// For a time-harmonic shear wave at angular frequency `ω`, the local wavenumber
/// magnitude `|k|` satisfies, in the windowed mean,
///
/// ```text
/// |k|²(x) ≈ ⟨|∇u|²⟩_W(x) / ⟨u²⟩_W(x)
/// ```
///
/// where `⟨·⟩_W` is a local spatial average. The shear-wave speed and modulus
/// follow from `c_s = ω / |k|` and `μ = ρ c_s²` (Young's modulus `E ≈ 3μ` for
/// nearly-incompressible tissue, handled by [`elasticity_map_from_speed`]).
///
/// # References
///
/// - Knutsson, H., Westin, C.-F., Granlund, G. (1994). "Local multiscale frequency
///   and bandwidth estimation." *Proc. IEEE ICIP*.
/// - Oliphant, T. E., Manduca, A., Ehman, R. L., Greenleaf, J. F. (2001).
///   "Complex-valued stiffness reconstruction for MR elastography by algebraic
///   inversion of the differential equation." *Magn. Reson. Med.*, 45(2), 299-310.
///
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
pub(super) fn local_frequency_estimation_inversion(
    displacement: &DisplacementField,
    grid: &Grid,
    density: f64,
    frequency: f64,
) -> KwaversResult<ElasticityMap> {
    let u = &displacement.uz;
    let (nx, ny, nz) = u.dim();

    // 1. Gradient-energy and signal-energy fields.
    //    Central differences on the interior; edges keep the zero initialiser
    //    and are repaired by `fill_boundaries` after the speed map is formed.
    let mut grad_energy = Array3::zeros((nx, ny, nz));
    let mut signal_energy = Array3::zeros((nx, ny, nz));

    let inv_2dx = if grid.dx > 0.0 {
        1.0 / (2.0 * grid.dx)
    } else {
        0.0
    };
    let inv_2dy = if grid.dy > 0.0 {
        1.0 / (2.0 * grid.dy)
    } else {
        0.0
    };
    let inv_2dz = if grid.dz > 0.0 {
        1.0 / (2.0 * grid.dz)
    } else {
        0.0
    };

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                signal_energy[[i, j, k]] = u[[i, j, k]] * u[[i, j, k]];
            }
        }
    }

    // A central z-difference needs an interior along z (nz ≥ 3). For a 2-D
    // plane-strain field (nz = 1, or a thin nz = 2 slab) there is no z-variation,
    // so ∂u/∂z = 0 and every z-layer carries the in-plane wavenumber — this is the
    // conventional clinical SWE case (a 2-D imaging plane). Without this guard the
    // 3-D interior loop `1..nz-1` is empty for nz = 1 and the estimate collapses
    // to the speed clamp (silent garbage on 2-D input).
    let has_z_interior = nz >= 3;
    let (k_lo, k_hi) = if has_z_interior { (1, nz - 1) } else { (0, nz) };
    for i in 1..nx.saturating_sub(1) {
        for j in 1..ny.saturating_sub(1) {
            for k in k_lo..k_hi {
                let du_dx = (u[[i + 1, j, k]] - u[[i - 1, j, k]]) * inv_2dx;
                let du_dy = (u[[i, j + 1, k]] - u[[i, j - 1, k]]) * inv_2dy;
                let du_dz = if has_z_interior {
                    (u[[i, j, k + 1]] - u[[i, j, k - 1]]) * inv_2dz
                } else {
                    0.0
                };
                grad_energy[[i, j, k]] = du_dx.mul_add(du_dx, du_dy.mul_add(du_dy, du_dz * du_dz));
            }
        }
    }

    // 2. Local averaging window (box smoothing, repeated).
    for _ in 0..LFE_SMOOTHING_PASSES {
        spatial_smoothing(&mut grad_energy);
        spatial_smoothing(&mut signal_energy);
    }

    // 3. Wavenumber → speed, with the same physiological clamp as `direct`.
    let omega = TWO_PI * frequency;
    let w2 = omega * omega;
    let k2_max = w2 / (0.5 * 0.5); // c_s = 0.5 m/s
    let k2_min = w2 / (20.0 * 20.0); // c_s = 20 m/s
    let eps = grad_energy.iter().cloned().fold(0.0_f64, f64::max) * 1e-9 + f64::MIN_POSITIVE;

    let mut shear_wave_speed = LetoArray3::zeros([nx, ny, nz]);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let k_squared = grad_energy[[i, j, k]] / (signal_energy[[i, j, k]] + eps);
                let valid_k2 = k_squared.clamp(k2_min, k2_max);
                shear_wave_speed[[i, j, k]] = omega / valid_k2.sqrt();
            }
        }
    }

    spatial_smoothing(&mut shear_wave_speed);
    fill_boundaries(&mut shear_wave_speed);

    Ok(elasticity_map_from_speed(shear_wave_speed, density))
}
