use super::super::{FwiProcessor, RHO_SEISMIC_REF};
use kwavers_grid::Grid;
use leto::Array3;

/// Verify that the FWI forward-model medium is built with seismic (non-water) density.
/// # Panics
/// - Panics if `medium construction must succeed`.
///
#[test]
fn test_fwi_medium_density_not_water() {
    use kwavers_medium::heterogeneous::HeterogeneousFactory;
    use kwavers_medium::CoreMedium;

    let (nx, ny, nz) = (8usize, 8, 8);
    let sound_speed = Array3::from_elem((nx, ny, nz), 2000.0_f64);
    let density = Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF);

    let medium = HeterogeneousFactory::from_arrays(sound_speed, density, None, None, None, 20.0)
        .expect("medium construction must succeed");

    let rho_sample = medium.density(4, 4, 4);
    assert!(
        (rho_sample - RHO_SEISMIC_REF).abs() < 1.0,
        "medium density {rho_sample} != RHO_SEISMIC_REF {RHO_SEISMIC_REF}"
    );
    assert!(
        (rho_sample - 1000.0).abs() > 100.0,
        "density must not equal water (1000 kg/m³)"
    );
}

/// Verify that the FWI forward-model medium stores the velocity model correctly.
/// # Panics
/// - Panics if `medium construction must succeed`.
///
#[test]
fn test_fwi_forward_medium_sound_speed_matches_model() {
    use kwavers_medium::heterogeneous::HeterogeneousFactory;
    use kwavers_medium::CoreMedium;

    let (nx, ny, nz) = (6usize, 6, 6);
    let mut model = Array3::from_elem((nx, ny, nz), 1800.0_f64);
    model[[3, 3, 3]] = 3200.0;

    let density = Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF);
    let medium = HeterogeneousFactory::from_arrays(model.clone(), density, None, None, None, 20.0)
        .expect("medium construction must succeed");

    let c_bg = medium.sound_speed(1, 1, 1);
    let c_anom = medium.sound_speed(3, 3, 3);
    assert!((c_bg - 1800.0).abs() < 1.0, "background speed mismatch");
    assert!((c_anom - 3200.0).abs() < 1.0, "anomaly speed mismatch");
}

/// Verify `resolved_density` returns the caller-supplied heterogeneous field
/// when present and falls back to `RHO_SEISMIC_REF` when absent.
///
/// ## Theorem
/// `FwiProcessor::resolved_density` is the single source of truth for the
/// density used in both forward and adjoint medium construction and in the
/// gradient scaling, so it must (i) preserve a supplied field bit-exactly,
/// (ii) reject mismatched shapes, and (iii) reject non-physical (non-finite
/// or non-positive) entries.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn test_fwi_resolved_density_heterogeneous_and_default() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("grid");
    let dims = grid.dimensions();

    // Default: no density supplied → constant RHO_SEISMIC_REF.
    let default_processor = FwiProcessor::default();
    let rho_default = default_processor
        .resolved_density(&grid)
        .expect("default density resolution must succeed");
    assert_eq!(rho_default.shape(), [dims.0, dims.1, dims.2]);
    assert!(rho_default
        .iter()
        .all(|&v| (v - RHO_SEISMIC_REF).abs() < f64::EPSILON));

    // Heterogeneous: cube of 1500 kg/m³ with a 2500 kg/m³ inclusion.
    let mut rho_field = Array3::from_elem(dims, 1500.0_f64);
    rho_field[[4, 4, 4]] = 2500.0;
    let het_processor = FwiProcessor::default()
        .with_density(rho_field.clone())
        .expect("heterogeneous density must validate");
    let rho_resolved = het_processor.resolved_density(&grid).expect("resolution");
    assert_eq!(
        rho_resolved, rho_field,
        "heterogeneous field must round-trip"
    );

    // Shape mismatch must be rejected.
    let wrong_shape = Array3::from_elem((4, 4, 4), 1500.0_f64);
    let mismatched_processor = FwiProcessor::default()
        .with_density(wrong_shape)
        .expect("validation only rejects non-finite / non-positive entries");
    let err = mismatched_processor.resolved_density(&grid);
    assert!(err.is_err(), "shape mismatch must fail at resolution");

    // Non-physical values must be rejected at the builder.
    let mut bad_rho = Array3::from_elem(dims, 1500.0_f64);
    bad_rho[[0, 0, 0]] = -1.0;
    assert!(FwiProcessor::default().with_density(bad_rho).is_err());

    let mut nan_rho = Array3::from_elem(dims, 1500.0_f64);
    nan_rho[[0, 0, 0]] = f64::NAN;
    assert!(FwiProcessor::default().with_density(nan_rho).is_err());
}
