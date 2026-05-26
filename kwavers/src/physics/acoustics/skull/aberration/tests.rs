
use crate::core::constants::fundamental::DENSITY_WATER;
use crate::core::constants::numerical::MHZ_TO_HZ;
use crate::domain::grid::Grid;
use crate::physics::acoustics::skull::HeterogeneousSkull;
use ndarray::Array3;

use super::constants::C_WATER_DEFAULT;
use super::AberrationCorrection;
use crate::core::constants::numerical::{TWO_PI};

fn make_test_skull(
    nx: usize,
    ny: usize,
    nz: usize,
    bone_start: usize,
    bone_end: usize,
    c_bone: f64,
) -> HeterogeneousSkull {
    let mut sound_speed = Array3::from_elem((nx, ny, nz), C_WATER_DEFAULT);
    let density = Array3::from_elem((nx, ny, nz), DENSITY_WATER);
    let attenuation = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in bone_start..bone_end.min(nz) {
                sound_speed[[i, j, k]] = c_bone;
            }
        }
    }
    HeterogeneousSkull {
        sound_speed,
        density,
        attenuation,
    }
}

/// Phase in a pure-water grid must be zero everywhere.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_zero_phase_in_water() {
    let grid = Grid::new(8, 8, 16, 1e-3, 1e-3, 1e-3).unwrap();
    let skull = make_test_skull(8, 8, 16, 0, 0, C_WATER_DEFAULT);
    let ac = AberrationCorrection::new(&grid, &skull);
    let phases = ac.compute_time_reversal_phases(500e3).unwrap();
    for v in &phases {
        assert!(
            v.abs() < 1e-12,
            "Phase in pure water must be zero, got {v:.3e}"
        );
    }
}

/// A uniform bone slab must produce `Delta k * thickness`.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_uniform_bone_slab_phase() {
    let f = 500e3_f64;
    let c_bone = 3000.0_f64;
    let dz = 1e-3_f64;
    let n_bone = 4_usize;
    let nz = 16_usize;

    let grid = Grid::new(4, 4, nz, 1e-3, 1e-3, dz).unwrap();
    let skull = make_test_skull(4, 4, nz, 4, 4 + n_bone, c_bone);
    let ac = AberrationCorrection::new(&grid, &skull).with_water_speed(C_WATER_DEFAULT);
    let phases = ac.compute_time_reversal_phases(f).unwrap();

    let k_water = TWO_PI * f / C_WATER_DEFAULT;
    let k_bone = TWO_PI * f / c_bone;
    let expected = (k_bone - k_water) * (n_bone as f64 * dz);
    let phi_computed = phases[[0, 0, nz - 1]];
    let rel_err = (phi_computed - expected).abs() / expected.abs().max(1e-12);
    assert!(
        rel_err < 1e-10,
        "Phase integral: expected {expected:.6} rad, got {phi_computed:.6} rad"
    );
}

/// Correction phases must be the exact negation of aberration phases.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_correction_is_negation_of_aberration() {
    let grid = Grid::new(6, 6, 12, 1e-3, 1e-3, 1e-3).unwrap();
    let skull = make_test_skull(6, 6, 12, 3, 7, 2800.0);
    let ac = AberrationCorrection::new(&grid, &skull);
    let aberr = ac.compute_time_reversal_phases(MHZ_TO_HZ).unwrap();
    let corr = ac.compute_correction_phases(MHZ_TO_HZ).unwrap();
    for (a, c) in aberr.iter().zip(corr.iter()) {
        assert!(
            (a + c).abs() < 1e-14,
            "Phi_aberr + Phi_corr must equal zero: {:.3e}",
            a + c
        );
    }
}

/// Element corrections must equal the negation of the aperture phase map.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_element_corrections_match_aperture_map() {
    let grid = Grid::new(8, 8, 12, 1e-3, 1e-3, 1e-3).unwrap();
    let skull = make_test_skull(8, 8, 12, 2, 6, 3100.0);
    let ac = AberrationCorrection::new(&grid, &skull);
    let f = 700e3_f64;
    let phases = ac.compute_time_reversal_phases(f).unwrap();
    let aperture = ac.aperture_phase_map(f).unwrap();
    let x_pos: Vec<f64> = (0..8).map(|ii| ii as f64 * 1e-3).collect();
    let y_pos: Vec<f64> = vec![2e-3; 8];

    let corr = ac.compute_element_corrections(f, &x_pos, &y_pos).unwrap();
    let corr_from_map = ac.element_corrections_from_map(&phases, &x_pos, &y_pos);
    let j_grid = (2e-3_f64 / 1e-3).round() as usize;
    for k in 0..8 {
        let expected_corr = -aperture[[k, j_grid]];
        assert!((corr[k] - corr_from_map[k]).abs() < 1e-12);
        assert!((corr[k] - expected_corr).abs() / expected_corr.abs().max(1e-10) < 1e-10);
    }
}

/// Mismatched element coordinate arrays must return a dimension error.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_element_corrections_reject_mismatched_lengths() {
    let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
    let skull = make_test_skull(4, 4, 4, 1, 2, 2800.0);
    let ac = AberrationCorrection::new(&grid, &skull);
    let err = ac.compute_element_corrections(500e3, &[0.0, 1e-3], &[0.0]);
    assert!(err.is_err());
}

/// Phase changes monotonically through a faster-than-water bone slab, then stays flat.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_phase_monotone_through_bone_then_flat() {
    let grid = Grid::new(4, 4, 20, 1e-3, 1e-3, 1e-3).unwrap();
    let skull = make_test_skull(4, 4, 20, 6, 10, 3000.0);
    let ac = AberrationCorrection::new(&grid, &skull);
    let phases = ac.compute_time_reversal_phases(500e3).unwrap();

    for k in 0..6 {
        assert!(phases[[0, 0, k]].abs() < 1e-12);
    }
    for k in 6..9 {
        assert!(phases[[0, 0, k + 1]] < phases[[0, 0, k]]);
    }
    let phi_after = phases[[0, 0, 10]];
    for k in 10..20 {
        assert!((phases[[0, 0, k]] - phi_after).abs() < 1e-12);
    }
}

/// Uniform skull properties produce a spatially uniform aperture map.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_phase_spatially_uniform_for_uniform_skull() {
    let grid = Grid::new(6, 6, 10, 1e-3, 1e-3, 1e-3).unwrap();
    let skull = make_test_skull(6, 6, 10, 2, 6, 2800.0);
    let ac = AberrationCorrection::new(&grid, &skull);
    let aperture = ac.aperture_phase_map(500e3).unwrap();
    let ref_phi = aperture[[0, 0]];
    for i in 0..6 {
        for j in 0..6 {
            assert!((aperture[[i, j]] - ref_phi).abs() < 1e-12);
        }
    }
}

/// Phase correction scales linearly with frequency.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_phase_scales_linearly_with_frequency() {
    let grid = Grid::new(4, 4, 12, 1e-3, 1e-3, 1e-3).unwrap();
    let skull = make_test_skull(4, 4, 12, 3, 7, 2800.0);
    let ac = AberrationCorrection::new(&grid, &skull);
    let p1 = ac.compute_time_reversal_phases(500e3).unwrap();
    let p2 = ac.compute_time_reversal_phases(MHZ_TO_HZ).unwrap();
    let ratio = p2[[0, 0, 11]] / p1[[0, 0, 11]];
    assert!(
        (ratio - 2.0).abs() < 1e-10,
        "Phase must scale linearly with frequency: ratio={ratio:.6}"
    );
}
