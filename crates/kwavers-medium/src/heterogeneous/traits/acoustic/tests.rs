//! Tests for heterogeneous medium acoustic property implementations.
//!
//! All thresholds are analytically derived from the Szabo (1994) power-law:
//!   α(f) = α₀ · (f / f_ref)^y   [α₀ stored in dB/(cm·MHz^y)]
//! surfaced in Np/m via × DB_TO_NP × 100 [cm/m].
//! with exact f64 arithmetic — no tolerances tighter than `f64::EPSILON` are
//! used, and no tolerances are artificially widened.

use crate::acoustic::AcousticProperties;
use crate::heterogeneous::core::HeterogeneousMedium;
use kwavers_core::constants::acoustic_parameters::DB_TO_NP;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

/// dB/cm → Np/m conversion for the stored α₀ prefactor.
const DB_CM_TO_NP_M: f64 = DB_TO_NP * 100.0;
use kwavers_grid::Grid;
use leto::Array3;

/// Construct a minimal 4×4×4 grid for point-query tests.
/// # Panics
/// - Panics if `grid construction must succeed`.
///
fn small_grid() -> Grid {
    Grid::new(4, 4, 4, 1.0, 1.0, 1.0).expect("grid construction must succeed")
}

/// Construct a uniform HeterogeneousMedium with scalar α₀, y, and f_ref.
fn uniform_medium(alpha0: f64, alpha_power: f64, f_ref: f64) -> HeterogeneousMedium {
    let mut m = HeterogeneousMedium::new(4, 4, 4, false);
    m.absorption = Array3::from_elem([4, 4, 4], alpha0);
    m.alpha_power = Array3::from_elem([4, 4, 4], alpha_power);
    m.reference_frequency = f_ref;
    m
}

/// **Theorem (Szabo 1994):**  α(f_ref) = α₀ · 1^y = α₀ dB/cm for any y,
/// surfaced as α₀ · DB_TO_NP · 100 in Np/m.
///
/// At the reference frequency the frequency ratio is exactly 1, so the
/// power-law factor is 1 regardless of the exponent.  The computed value
/// must equal α₀ · DB_CM_TO_NP_M to machine precision.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_absorption_at_reference_frequency_equals_alpha0() {
    let f_ref = MHZ_TO_HZ; // 1 MHz
    let alpha0 = 0.75_f64;
    let medium = uniform_medium(alpha0, 1.5, f_ref);
    let grid = small_grid();

    let result = medium
        .absorption_coefficient(1.5, 1.5, 1.5, &grid, f_ref)
        .into_base();
    let expected = alpha0 * DB_CM_TO_NP_M;
    assert!(
        (result - expected).abs() < f64::EPSILON * expected,
        "at f=f_ref: expected {expected} Np/m, got {result}"
    );
}

/// **Theorem (Szabo 1994):**  α(f) = α₀ · (f/f_ref)^y.
///
/// With α₀=1 dB/cm, y=2, f=2·f_ref the result must be exactly
/// 4 · DB_CM_TO_NP_M Np/m.
/// Proof: (2·f_ref / f_ref)^2 = 2² = 4.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_power_law_exponent_y2_doubles_frequency() {
    let f_ref = 500.0e3; // 500 kHz
    let medium = uniform_medium(1.0, 2.0, f_ref);
    let grid = small_grid();

    let result = medium
        .absorption_coefficient(1.5, 1.5, 1.5, &grid, 2.0 * f_ref)
        .into_base();
    let expected = 4.0_f64 * DB_CM_TO_NP_M; // 1.0 * 2^2 dB/cm, in Np/m
    assert!(
        (result - expected).abs() < 1e-12,
        "y=2, f=2f_ref: expected {expected}, got {result}"
    );
}

/// **Theorem (Szabo 1994):** y=1.5 is the canonical tissue exponent
/// (Szabo 1994, Table I).  At f=4·f_ref:
///   α = α₀ · 4^1.5 = α₀ · 8 dB/cm, surfaced in Np/m.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_power_law_tissue_exponent_y1_5() {
    let f_ref = MHZ_TO_HZ;
    let alpha0 = 0.5_f64;
    let medium = uniform_medium(alpha0, 1.5, f_ref);
    let grid = small_grid();

    let result = medium
        .absorption_coefficient(1.5, 1.5, 1.5, &grid, 4.0 * f_ref)
        .into_base();
    let expected = alpha0 * 8.0 * DB_CM_TO_NP_M; // 0.5 * 4^1.5 = 4 dB/cm, in Np/m
    assert!(
        (result - expected).abs() < 1e-10,
        "y=1.5, f=4f_ref: expected {expected}, got {result}"
    );
}

/// **Theorem:** y=1 gives linear frequency dependence.
/// At f=3·f_ref: α = α₀ · 3 dB/cm, surfaced in Np/m.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_power_law_linear_exponent_y1() {
    let f_ref = 200.0e3;
    let alpha0 = 2.0_f64;
    let medium = uniform_medium(alpha0, 1.0, f_ref);
    let grid = small_grid();

    let result = medium
        .absorption_coefficient(1.5, 1.5, 1.5, &grid, 3.0 * f_ref)
        .into_base();
    let expected = alpha0 * 3.0 * DB_CM_TO_NP_M;
    assert!(
        (result - expected).abs() < 1e-12,
        "y=1, f=3f_ref: expected {expected}, got {result}"
    );
}

/// **Theorem:** `alpha_power()` returns the stored exponent at continuous
/// coordinates (nearest-neighbor, no interpolation).
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_alpha_power_query_returns_stored_exponent() {
    let mut m = HeterogeneousMedium::new(4, 4, 4, false);
    m.alpha_power[[1, 1, 1]] = 2.3;
    let grid = small_grid();

    // Nearest-neighbor: coordinates (1.0, 1.0, 1.0) map to voxel [1,1,1].
    let result = m.alpha_power(1.0, 1.0, 1.0, &grid);
    assert!(
        (result - 2.3).abs() < f64::EPSILON * 2.3,
        "alpha_power at [1,1,1]: expected 2.3, got {result}"
    );
}

/// **Theorem:** spatially varying exponent maps correct y per voxel.
///
/// Voxel [0,0,0] has y=1.0 (linear), voxel [3,3,3] has y=2.0 (quadratic).
/// At f=2·f_ref:
///   α[0,0,0] = α₀ · 2^1 = 2·α₀ dB/cm
///   α[3,3,3] = α₀ · 2^2 = 4·α₀ dB/cm, both surfaced in Np/m.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_spatially_varying_alpha_power() {
    let f_ref = MHZ_TO_HZ;
    let alpha0 = 1.0_f64;
    let mut m = HeterogeneousMedium::new(4, 4, 4, false);
    m.absorption = Array3::from_elem([4, 4, 4], alpha0);
    m.alpha_power = Array3::from_elem([4, 4, 4], 1.5);
    m.alpha_power[[0, 0, 0]] = 1.0;
    m.alpha_power[[3, 3, 3]] = 2.0;
    m.reference_frequency = f_ref;
    let grid = small_grid();

    let r0 = m
        .absorption_coefficient(0.0, 0.0, 0.0, &grid, 2.0 * f_ref)
        .into_base();
    let r3 = m
        .absorption_coefficient(3.0, 3.0, 3.0, &grid, 2.0 * f_ref)
        .into_base();
    assert!(
        (r0 - 2.0 * DB_CM_TO_NP_M).abs() < 1e-10,
        "[0,0,0] y=1: expected {} Np/m, got {r0}",
        2.0 * DB_CM_TO_NP_M
    );
    assert!(
        (r3 - 4.0 * DB_CM_TO_NP_M).abs() < 1e-10,
        "[3,3,3] y=2: expected {} Np/m, got {r3}",
        4.0 * DB_CM_TO_NP_M
    );
}

/// **Theorem:** default `alpha_power` in `HeterogeneousMedium::new()` is 1.0
/// (linear frequency dependence), per the struct contract.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_default_alpha_power_is_one() {
    let m = HeterogeneousMedium::new(4, 4, 4, false);
    // Every voxel must be exactly 1.0.
    for v in m.alpha_power.iter() {
        assert!(
            (*v - 1.0).abs() < f64::EPSILON,
            "default alpha_power voxel is {v}, expected 1.0"
        );
    }
}

/// **Theorem:** `TissueFactory` sets alpha_power=1.5 per Szabo (1994) Table I.
/// # Panics
/// - Panics if `grid must be valid`.
///
#[test]
fn test_tissue_factory_alpha_power_is_1_5() {
    use crate::heterogeneous::factory::tissue::TissueFactory;
    let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).expect("grid must be valid");
    let medium = TissueFactory::create_tissue_medium(&grid);
    for v in medium.alpha_power.iter() {
        assert!(
            (*v - 1.5).abs() < f64::EPSILON,
            "tissue alpha_power voxel is {v}, expected 1.5"
        );
    }
}
