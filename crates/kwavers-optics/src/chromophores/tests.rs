//! Value-semantic tests for chromophore spectral physics.
//!
//! Verified properties:
//! - `ExtinctionSpectrum`: exact tabulated lookup, linear interpolation,
//!   boundary extrapolation, and empty-database rejection.
//! - `HemoglobinDatabase`: Beer-Lambert absorption formula, SO₂ computation,
//!   isosbestic-point extinction ordering, and domain rejections.

use super::{ExtinctionSpectrum, HemoglobinDatabase};

// ─── ExtinctionSpectrum ──────────────────────────────────────────────────────

/// Exact-wavelength lookup must return the stored coefficient without rounding error.
///
/// Tabulated entry: 800 nm → 3264.0 (M⁻¹·cm⁻¹) for HbO₂ (Prahl 1999 per-heme
/// ε = 816 × 4 for the tetramer).
#[test]
fn extinction_spectrum_exact_wavelength_returns_tabulated_value() {
    let db = HemoglobinDatabase::standard();
    let eps = db.hbo2_extinction(800.0).unwrap();
    assert!(
        (eps - 3264.0).abs() < 1e-9,
        "HbO₂ at 800 nm: expected 3264.0, got {eps}"
    );
}

/// Linear interpolation at a wavelength between two tabulated entries.
///
/// For HbO₂:
/// - 775 nm → 2708.8
/// - 800 nm → 3264.0
///
/// At 787 nm: t = (787 - 775) / (800 - 775) = 12/25 = 0.48
/// ε = 2708.8 + 0.48 × (3264.0 − 2708.8) = 2708.8 + 0.48 × 555.2 = 2975.296
#[test]
fn extinction_spectrum_linear_interpolation_between_tabulated_points() {
    let db = HemoglobinDatabase::standard();
    let eps = db.hbo2_extinction(787.0).unwrap();
    let expected = 2708.8 + 0.48 * (3264.0 - 2708.8); // 2975.296
    assert!(
        (eps - expected).abs() < 1e-9,
        "HbO₂ interpolated at 787 nm: expected {expected:.4}, got {eps:.4}"
    );
}

/// Below-range query extrapolates to the minimum-wavelength entry.
///
/// The HbO₂ table starts at 450 nm (106112 M⁻¹·cm⁻¹).  A query at 400 nm
/// (below the table) must return the 450 nm value, not an error.
#[test]
fn extinction_spectrum_below_range_extrapolates_to_lower_bound() {
    let db = HemoglobinDatabase::standard();
    let eps_below = db.hbo2_extinction(400.0).unwrap();
    let eps_min = db.hbo2_extinction(450.0).unwrap();
    assert!(
        (eps_below - eps_min).abs() < 1e-9,
        "Below-range query: expected {eps_min:.1}, got {eps_below:.1}"
    );
}

/// Above-range query extrapolates to the maximum-wavelength entry.
///
/// The table ends at 1000 nm.  A query at 1100 nm must return the 1000 nm value.
#[test]
fn extinction_spectrum_above_range_extrapolates_to_upper_bound() {
    let db = HemoglobinDatabase::standard();
    let eps_above = db.hbo2_extinction(1100.0).unwrap();
    let eps_max = db.hbo2_extinction(1000.0).unwrap();
    assert!(
        (eps_above - eps_max).abs() < 1e-9,
        "Above-range query: expected {eps_max:.1}, got {eps_above:.1}"
    );
}

/// An empty `ExtinctionSpectrum` returns an error on any wavelength query.
#[test]
fn extinction_spectrum_empty_database_returns_error() {
    let empty = ExtinctionSpectrum::new("empty", vec![]);
    assert!(
        empty.at_wavelength(700.0).is_err(),
        "empty spectrum must return Err on any query"
    );
}

/// `wavelength_range` returns `None` for an empty spectrum and the correct
/// bounds for a populated one.
#[test]
fn extinction_spectrum_wavelength_range_contract() {
    let empty = ExtinctionSpectrum::new("empty", vec![]);
    assert!(empty.wavelength_range().is_none());

    let db = HemoglobinDatabase::standard();
    let (lo, hi) = db.hbo2_spectrum().wavelength_range().unwrap();
    assert_eq!(lo, 450, "HbO₂ range lower bound");
    assert_eq!(hi, 1000, "HbO₂ range upper bound");
}

// ─── HemoglobinDatabase ──────────────────────────────────────────────────────

/// Beer-Lambert absorption coefficient formula:
/// μ_a = 2.303 × (ε_HbO₂·[HbO₂] + ε_Hb·[Hb]) × 100  (cm⁻¹)
///
/// At 800 nm: ε_HbO₂ = 3264, ε_Hb = 3046.88 (M⁻¹·cm⁻¹, per-tetramer).
/// With [HbO₂] = [Hb] = 1e-3 M:
/// μ_a = 2.303 × (3.264 + 3.04688) × 100 = 2.303 × 630.688 = 1452.474 m⁻¹
#[test]
fn hemoglobin_beer_lambert_absorption_coefficient_at_800_nm() {
    let db = HemoglobinDatabase::standard();
    let mu_a = db.absorption_coefficient(800.0, 1e-3, 1e-3).unwrap();
    let expected = 2.303 * (3264.0 * 1e-3 + 3046.88 * 1e-3) * 100.0;
    assert!(
        (mu_a - expected).abs() < 1e-6,
        "μ_a at 800 nm: expected {expected:.6}, got {mu_a:.6}"
    );
}

/// Oxygen saturation formula: SO₂ = [HbO₂] / ([HbO₂] + [Hb]).
///
/// With [HbO₂] = 0.98·c_total and [Hb] = 0.02·c_total,
/// SO₂ = 0.98 (arterial saturation).
#[test]
fn hemoglobin_oxygen_saturation_formula() {
    let db = HemoglobinDatabase::standard();
    let so2 = db.oxygen_saturation(0.98, 0.02).unwrap();
    assert!(
        (so2 - 0.98).abs() < 1e-14,
        "SO₂ = 0.98/(0.98+0.02) = 0.98; got {so2}"
    );
}

/// Zero total hemoglobin concentration must return an error (division by zero
/// in SO₂ formula).
#[test]
fn hemoglobin_zero_total_concentration_is_rejected() {
    let db = HemoglobinDatabase::standard();
    assert!(
        db.oxygen_saturation(0.0, 0.0).is_err(),
        "SO₂ with zero total must return Err"
    );
}

/// Isosbestic consistency: at a true isosbestic wavelength the HbO₂ and Hb
/// extinctions are (nearly) equal. The Prahl data has exact isosbestic points at
/// ~500 nm and ~797 nm; the tabulated 500 nm entry and the (775,800)-bracketed
/// ~797 nm crossing must agree to within a few percent.
#[test]
fn hemoglobin_isosbestic_extinctions_are_nearly_equal() {
    let db = HemoglobinDatabase::standard();
    // 500 nm is an exact tabulated isosbestic point (83731.2 vs 83448).
    let (o500, h500) = db.extinction_pair(500.0).unwrap();
    assert!(
        (o500 - h500).abs() / o500.max(h500) < 0.01,
        "500 nm isosbestic: ε_HbO₂={o500}, ε_Hb={h500} should match within 1%"
    );
    // ~797 nm isosbestic, sampled at the 800 nm entry (3264 vs 3046.88).
    let (o800, h800) = db.extinction_pair(800.0).unwrap();
    assert!(
        (o800 - h800).abs() / o800.max(h800) < 0.10,
        "≈800 nm isosbestic: ε_HbO₂={o800}, ε_Hb={h800} should match within 10%"
    );
}

/// Spectral ordering that drives pulse oximetry / sO₂ unmixing and is the basis
/// of the recently-corrected data: in the red, deoxy-Hb absorbs far more than
/// HbO₂; in the near-infrared the ordering inverts (HbO₂ > Hb). The curves must
/// therefore cross between 650 nm and 950 nm.
#[test]
fn hemoglobin_red_and_nir_ordering_brackets_isosbestic() {
    let db = HemoglobinDatabase::standard();
    // Red (650 nm): ε_Hb ≫ ε_HbO₂ (≈10×) — why oxygenated blood looks bright red.
    let (o650, h650) = db.extinction_pair(650.0).unwrap();
    assert!(
        h650 > 5.0 * o650,
        "650 nm: ε_Hb({h650}) must greatly exceed ε_HbO₂({o650})"
    );
    // NIR (950 nm): ε_HbO₂ > ε_Hb — the oximetry/NIRS window.
    let (o950, h950) = db.extinction_pair(950.0).unwrap();
    assert!(
        o950 > h950,
        "950 nm: ε_HbO₂({o950}) must exceed ε_Hb({h950})"
    );
    // Sign of (HbO₂ − Hb) flips between red and NIR ⇒ an isosbestic crossing.
    assert!(
        (o650 - h650) < 0.0 && (o950 - h950) > 0.0,
        "HbO₂−Hb must change sign between 650 nm and 950 nm"
    );
}

/// The isosbestic-point list must contain 800 nm (standard reference).
#[test]
fn hemoglobin_isosbestic_points_include_800() {
    let pts = HemoglobinDatabase::isosbestic_points();
    assert!(
        pts.contains(&797),
        "797 nm must be in the isosbestic-point list: {pts:?}"
    );
}

/// `typical_blood_parameters` returns a tuple with physically plausible values:
/// total Hb ~2.3 mM, arterial SO₂ ≥ 0.95, venous SO₂ ≥ 0.6.
#[test]
fn hemoglobin_typical_blood_parameters_are_plausible() {
    let (total_mm, arterial_so2, venous_so2) = HemoglobinDatabase::typical_blood_parameters();
    assert!(
        total_mm > 1e-3 && total_mm < 10e-3,
        "total Hb {total_mm} M outside physiological range"
    );
    assert!(
        (0.95..=1.0).contains(&arterial_so2),
        "arterial SO₂ {arterial_so2} outside 0.95–1.0"
    );
    assert!(
        (0.6..=0.95).contains(&venous_so2),
        "venous SO₂ {venous_so2} outside 0.6–0.95"
    );
}
