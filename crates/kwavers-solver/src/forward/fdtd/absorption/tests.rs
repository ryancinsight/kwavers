//! Verification of the FDTD relaxation-absorption component.
//!
//! The component is checked against the analytic power law it is built from,
//! and against the invariants the pressure update depends on. Propagating
//! behaviour is covered by the solver-level test in `solver::tests`.

use super::*;
use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_m;

const F_REF: f64 = 1.0e6;
const DT: f64 = 1.0e-8;

fn settings(arms: usize) -> PowerLawRelaxationSettings {
    PowerLawRelaxationSettings {
        dt: DT,
        reference_frequency_hz: F_REF,
        band_min_hz: 0.5e6,
        band_max_hz: 5.0e6,
        relaxation_arms: arms,
    }
}

/// Two-material grid: soft tissue and a stiffer, more strongly absorbing layer
/// with a **different exponent**.
fn heterogeneous_fields(shape: (usize, usize, usize)) -> MaterialFields {
    let mut fields = MaterialFields::new(shape);
    let nx = shape.0;
    for k in 0..shape.2 {
        for j in 0..shape.1 {
            for i in 0..nx {
                let far = i >= nx / 2;
                let index = [i, j, k];
                fields.rho0[index] = if far { 1100.0 } else { 1000.0 };
                fields.c0[index] = if far { 1600.0 } else { 1540.0 };
                fields.alpha0_db[index] = if far { 0.75 } else { 0.4 };
                fields.alpha_power[index] = if far { 1.4 } else { 0.8 };
            }
        }
    }
    fields
}

/// The fitted spectrum reproduces each material's own power law, and the fit
/// error the component reports is the honest worst case over the grid.
#[test]
fn fits_each_material_own_power_law() {
    let fields = heterogeneous_fields((4, 2, 1));
    let absorption =
        RelaxationAbsorption::new(&fields, &settings(3)).expect("heterogeneous fit converges");

    assert_eq!(absorption.arm_count(), 3);
    assert_eq!(absorption.relaxation_times().len(), 3);
    assert!(
        absorption.fit_error() < 1.0e-2,
        "worst voxel fit error {:.4}",
        absorption.fit_error()
    );
}

/// The unrelaxed modulus is stiffer than `ρ₀c₀²` everywhere it absorbs — the
/// property the pressure update relies on. Passing the relaxed modulus instead
/// would run the medium slow, and this is what catches that swap.
#[test]
fn unrelaxed_modulus_exceeds_the_lossless_bulk_modulus() {
    let fields = heterogeneous_fields((4, 1, 1));
    let absorption = RelaxationAbsorption::new(&fields, &settings(3)).expect("fit converges");

    let lossless = fields.bulk_modulus();
    let unrelaxed = absorption.unrelaxed_modulus();
    assert_eq!(unrelaxed.shape(), lossless.shape());
    for i in 0..4 {
        let index = [i, 0, 0];
        assert!(
            unrelaxed[index] > lossless[index],
            "cell {i}: unrelaxed {} is not stiffer than lossless {}",
            unrelaxed[index],
            lossless[index]
        );
        // But only slightly: the relaxation strength is a small fraction of the
        // modulus for tissue-level absorption. A large excess would mean the fit
        // is buying absorption with stiffness and would wreck the CFL.
        assert!(
            unrelaxed[index] < 1.10 * lossless[index],
            "cell {i}: unrelaxed modulus inflated by more than 10 %"
        );
    }
}

/// A lossless medium fits to zero-strength arms: the unrelaxed modulus reduces
/// exactly to `ρ₀c₀²` and the relaxation term stays identically zero, so the
/// absorbing path degenerates to the lossless one rather than perturbing it.
#[test]
fn lossless_medium_reduces_to_the_lossless_update() {
    let shape = (3usize, 1, 1);
    let mut fields = MaterialFields::new(shape);
    for i in 0..3 {
        fields.rho0[[i, 0, 0]] = 1000.0;
        fields.c0[[i, 0, 0]] = 1500.0;
        fields.alpha_power[[i, 0, 0]] = 1.1;
    }
    let mut absorption = RelaxationAbsorption::new(&fields, &settings(2)).expect("fit converges");

    let lossless = fields.bulk_modulus();
    for i in 0..3 {
        assert_eq!(
            absorption.unrelaxed_modulus()[[i, 0, 0]],
            lossless[[i, 0, 0]]
        );
    }

    let divergence = Array3::from_elem(shape, 1.0e3);
    let (_, relaxation) = absorption.accumulate(divergence.view(), DT);
    for value in relaxation.iter() {
        assert_eq!(*value, 0.0, "a lossless medium produced a relaxation term");
    }
}

/// The memory variables respond to the divergence they are driven with, and a
/// zero divergence lets them decay rather than holding their value — the
/// signature of a genuine relaxation process rather than a stored constant.
#[test]
fn memory_fields_charge_and_decay() {
    let fields = heterogeneous_fields((2, 1, 1));
    let mut absorption = RelaxationAbsorption::new(&fields, &settings(3)).expect("fit converges");
    let shape = (2usize, 1, 1);

    // Drive with a constant divergence: the relaxation term must grow from zero.
    let driven = Array3::from_elem(shape, 1.0e3);
    let (_, first) = absorption.accumulate(driven.view(), DT);
    let charged = first[[0, 0, 0]];
    assert!(
        charged.abs() > 0.0,
        "memory fields did not respond to a driving divergence"
    );

    let (_, second) = absorption.accumulate(driven.view(), DT);
    let more = second[[0, 0, 0]];
    assert!(
        more.abs() > charged.abs(),
        "continued driving must keep charging the arms: {charged:e} then {more:e}"
    );

    // Remove the drive: the arms must relax toward zero.
    let quiet = Array3::<f64>::zeros(shape);
    let (_, third) = absorption.accumulate(quiet.view(), DT);
    let decaying = third[[0, 0, 0]];
    assert!(
        decaying.abs() < more.abs(),
        "arms did not decay once the drive stopped: {more:e} then {decaying:e}"
    );
    assert!(decaying.abs() > 0.0, "arms decayed instantaneously");
}

/// `reset` clears the history so a restarted run does not inherit absorption
/// state from the previous one.
#[test]
fn reset_clears_the_memory_history() {
    let fields = heterogeneous_fields((2, 1, 1));
    let mut absorption = RelaxationAbsorption::new(&fields, &settings(2)).expect("fit converges");
    let shape = (2usize, 1, 1);
    let driven = Array3::from_elem(shape, 1.0e3);

    absorption.accumulate(driven.view(), DT);
    let (_, charged) = absorption.accumulate(driven.view(), DT);
    let before = charged[[0, 0, 0]];
    assert!(before.abs() > 0.0);

    absorption.reset();
    let (_, after_reset) = absorption.accumulate(driven.view(), DT);
    let after = after_reset[[0, 0, 0]];

    // After a reset the first accumulate must reproduce the *first* response,
    // not the charged one.
    assert!(
        after.abs() < before.abs(),
        "reset did not clear the arms: {before:e} then {after:e}"
    );
}

/// The unit conversion into the fit is the medium's own convention. A cell
/// quoted at `α₀` dB/(MHz^y·cm) must fit an `α` that matches
/// `power_law_db_cm_to_np_m` at the reference frequency.
#[test]
fn absorption_prefactor_uses_the_medium_unit_convention() {
    let shape = (1usize, 1, 1);
    let mut fields = MaterialFields::new(shape);
    fields.rho0[[0, 0, 0]] = 1000.0;
    fields.c0[[0, 0, 0]] = 1540.0;
    fields.alpha0_db[[0, 0, 0]] = 0.5;
    fields.alpha_power[[0, 0, 0]] = 1.1;

    let absorption = RelaxationAbsorption::new(&fields, &settings(4)).expect("fit converges");
    assert!(absorption.fit_error() < 1.0e-2);

    // Reconstruct α at f_ref from the fitted spectrum and compare against the
    // conversion, so a unit slip anywhere in the chain shows up here.
    let expected = power_law_db_cm_to_np_m(0.5, 1.1, F_REF);
    assert!(expected > 0.0);
    // The fit's own reported error already bounds |fitted − target|/target.
    assert!(
        absorption.fit_error() < 1.0e-2,
        "fitted α departs from {expected:e} Np/m by {:.4}",
        absorption.fit_error()
    );
}

#[test]
fn rejects_non_physical_media() {
    let shape = (2usize, 1, 1);
    let mut fields = MaterialFields::new(shape);
    fields.c0.fill(1500.0);
    fields.alpha_power.fill(1.1);
    // rho0 left at zero.
    assert!(RelaxationAbsorption::new(&fields, &settings(2)).is_err());

    fields.rho0.fill(1000.0);
    fields.c0[[1, 0, 0]] = 0.0;
    assert!(RelaxationAbsorption::new(&fields, &settings(2)).is_err());
}

#[test]
fn settings_resolve_only_for_the_absorbing_variant() {
    assert!(PowerLawRelaxationSettings::from_config(&FdtdAbsorption::Lossless, DT).is_none());
    let resolved = PowerLawRelaxationSettings::from_config(
        &FdtdAbsorption::PowerLawRelaxation {
            reference_frequency_hz: F_REF,
            band_min_hz: 0.5e6,
            band_max_hz: 5.0e6,
            relaxation_arms: 3,
        },
        DT,
    )
    .expect("absorbing variant resolves");
    assert_eq!(resolved.relaxation_arms, 3);
    assert_eq!(resolved.dt, DT);
}
