use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_diagnostics::reconstruction::breast_ust_fwi::{
    generate_breast_ust_pstd_frequency_dataset, BreastUstPstdDatasetConfig,
};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use kwavers_solver::inverse::fwi::frequency_domain::{
    simulate_pstd_finite_window_born_observation,
    simulate_pstd_finite_window_born_second_order_observation, PstdFiniteWindowBornConfig,
};
use leto::Array3;

#[test]
fn finite_window_born_is_linear_in_slowness_squared_contrast() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = test_config();
    let base = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let small = contrast_volume(0.01);
    let double = contrast_volume(0.02);

    let base_data = simulate_pstd_finite_window_born_observation(
        &base.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("base data");
    let small_data = simulate_pstd_finite_window_born_observation(
        &small.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("small contrast");
    let double_data = simulate_pstd_finite_window_born_observation(
        &double.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("double contrast");

    let mut max_reference: f64 = 0.0;
    let mut max_error: f64 = 0.0;
    for ((&base_value, &small_value), &double_value) in base_data
        .iter()
        .zip(small_data.iter())
        .zip(double_data.iter())
    {
        let expected = (small_value - base_value) * 2.0;
        let got = double_value - base_value;
        max_reference = max_reference.max(expected.norm());
        max_error = max_error.max((got - expected).norm());
    }

    assert!(
        max_reference > f64::EPSILON.sqrt(),
        "contrast increment must be nonzero"
    );
    assert!(
        max_error <= f64::EPSILON.sqrt() * max_reference.max(1.0),
        "finite-window Born increment must be linear in chi: max_error={max_error}, max_reference={max_reference}"
    );
}

#[test]
fn finite_window_born_rejects_off_grid_ring_geometry() {
    let array = MultiRowRingArray::new(4, 1, 0.012, 0.0).expect("ring array");
    let model = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);

    let error = simulate_pstd_finite_window_born_observation(
        &model.clone().into(),
        &array,
        200_000.0,
        test_config(),
        4,
    )
    .expect_err("off-grid geometry must fail");

    assert!(
        error.to_string().contains("not on the centered grid axis"),
        "{error}"
    );
}

/// The isolated second-order Born correction `ps2 = (p0+ps1+ps2) − (p0+ps1)`
/// is quadratic in the slowness-squared contrast chi.  Doubling chi must
/// quadruple the second-order-only contribution to receiver pressure.
#[test]
fn second_order_correction_is_quadratic_in_contrast() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = test_config();
    let small = contrast_volume(0.01);
    let double = contrast_volume(0.02);

    // First-order predictions at each contrast.
    let first_small = simulate_pstd_finite_window_born_observation(
        &small.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("first-order small");
    let first_double = simulate_pstd_finite_window_born_observation(
        &double.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("first-order double");

    // Second-order predictions at each contrast.
    let second_small = simulate_pstd_finite_window_born_second_order_observation(
        &small.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("second-order small");
    let second_double = simulate_pstd_finite_window_born_second_order_observation(
        &double.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("second-order double");

    // Isolate the second-order-only contribution: ps2 = second_order − first_order
    // For chi → 2·chi, ps2 should quadruple: ps2(2χ) ≈ 4·ps2(χ)
    let mut max_reference: f64 = 0.0;
    let mut max_error: f64 = 0.0;
    for (((&first_s, &second_s), &first_d), &second_d) in first_small
        .iter()
        .zip(second_small.iter())
        .zip(first_double.iter())
        .zip(second_double.iter())
    {
        let ps2_small = second_s - first_s;
        let ps2_double = second_d - first_d;
        let expected = ps2_small * 4.0;
        let got = ps2_double;
        max_reference = max_reference.max(expected.norm());
        max_error = max_error.max((got - expected).norm());
    }

    assert!(
        max_reference > f64::EPSILON.sqrt(),
        "second-order correction must be nonzero"
    );
    assert!(
        max_error <= 1.0e-6 * max_reference.max(1.0),
        "second-order correction must be quadratic in chi: max_error={max_error:.6e}, max_reference={max_reference:.6e}"
    );
}

/// The second-order Born-series correction must add a nonzero contribution
/// beyond the first-order model for a heterogeneous medium.  For a
/// homogeneous reference the second-order term vanishes (ps1 = 0 ⇒ ps2 = 0),
/// so we verify that first- and second-order predictions agree on the
/// homogeneous baseline but differ on a heterogeneous volume.
#[test]
fn second_order_differs_from_first_order_on_heterogeneous() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let config = test_config();
    let homogeneous = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let heterogeneous = contrast_volume(0.05);

    let first_homog = simulate_pstd_finite_window_born_observation(
        &homogeneous.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("first order homogeneous");
    let second_homog = simulate_pstd_finite_window_born_second_order_observation(
        &homogeneous.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("second order homogeneous");
    let first_hetero = simulate_pstd_finite_window_born_observation(
        &heterogeneous.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("first order heterogeneous");
    let second_hetero = simulate_pstd_finite_window_born_second_order_observation(
        &heterogeneous.clone().into(),
        &array,
        200_000.0,
        config,
        4,
    )
    .expect("second order heterogeneous");

    // Homogeneous: first and second order must agree (ps2 = 0 when chi = 0).
    let mut homog_diff_norm = 0.0_f64;
    for (&first, &second) in first_homog.iter().zip(second_homog.iter()) {
        homog_diff_norm += (first - second).norm_sqr();
    }
    assert!(
        homog_diff_norm.sqrt() < f64::EPSILON.sqrt(),
        "first- and second-order must agree on homogeneous medium, diff={}",
        homog_diff_norm.sqrt()
    );

    // Heterogeneous: second-order must differ from first-order.
    let mut hetero_diff_norm = 0.0_f64;
    let mut hetero_ref_norm = 0.0_f64;
    for (&first, &second) in first_hetero.iter().zip(second_hetero.iter()) {
        hetero_diff_norm += (first - second).norm_sqr();
        hetero_ref_norm += first.norm_sqr();
    }
    assert!(
        hetero_diff_norm.sqrt() / hetero_ref_norm.sqrt() > f64::EPSILON.sqrt(),
        "second-order must differ from first-order on heterogeneous medium, relative_diff={}",
        hetero_diff_norm.sqrt() / hetero_ref_norm.sqrt()
    );
}

/// The second-order correction must improve the match to actual PSTD data
/// for a finite-contrast volume.  We generate a small (3×3×1) PSTD dataset
/// with CPML disabled, then verify the second-order residual is not worse
/// than the first-order residual.
#[test]
fn second_order_does_not_worsen_pstd_match() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let frequency_hz = 200_000.0;
    let heterogeneous = contrast_volume(0.03);
    let acquisition = BreastUstPstdDatasetConfig {
        spacing_m: 0.005,
        time_step_s: 1.0e-7,
        cycles_per_frequency: 3,
        frequency_bin_cycles: 1,
        source_amplitude_pa: 1.0e3,
        cpml_thickness_cells: 0,
        ..BreastUstPstdDatasetConfig::default()
    };
    let born_config = PstdFiniteWindowBornConfig {
        reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
        spacing_m: acquisition.spacing_m,
        time_step_s: acquisition.time_step_s,
        source_amplitude_pa: acquisition.source_amplitude_pa,
        cycles_per_frequency: acquisition.cycles_per_frequency,
        frequency_bin_cycles: acquisition.frequency_bin_cycles,
    };

    let pstd_data = generate_breast_ust_pstd_frequency_dataset(
        &heterogeneous,
        &array,
        &[frequency_hz],
        acquisition,
    )
    .expect("PSTD data");
    let first_order = simulate_pstd_finite_window_born_observation(
        &heterogeneous.clone().into(),
        &array,
        frequency_hz,
        born_config,
        4,
    )
    .expect("first order");
    let second_order = simulate_pstd_finite_window_born_second_order_observation(
        &heterogeneous.clone().into(),
        &array,
        frequency_hz,
        born_config,
        4,
    )
    .expect("second order");

    let mut pstd_norm_sq = 0.0_f64;
    let mut first_residual_sq = 0.0_f64;
    let mut second_residual_sq = 0.0_f64;
    let pstd_row = pstd_data
        .observed_pressure
        .index_axis::<2>(0, 0)
        .expect("index_axis");
    for ((&pstd, &first), &second) in pstd_row
        .iter()
        .zip(first_order.iter())
        .zip(second_order.iter())
    {
        pstd_norm_sq += pstd.norm_sqr();
        first_residual_sq += (first - pstd).norm_sqr();
        second_residual_sq += (second - pstd).norm_sqr();
    }
    assert!(pstd_norm_sq > 0.0, "PSTD data must be nonzero");

    // Second-order residual must not exceed first-order by more than
    // floating-point noise (allow a small tolerance for the extra FFT pass).
    let first_residual = first_residual_sq.sqrt();
    let second_residual = second_residual_sq.sqrt();
    let tolerance = first_residual * 1.0e-6 + f64::EPSILON.sqrt();
    assert!(
        second_residual <= first_residual + tolerance,
        "second-order PSTD residual {second_residual:.6e} must not exceed first-order {first_residual:.6e} + {tolerance:.6e}"
    );
}

/// The finite-window Born source term `-chi * (p0[n+1] - 2p0[n] + p0[n-1])`
/// is the Fréchet derivative of the production PSTD acquisition map at the
/// homogeneous reference.  This test compares the Born increment against a
/// small-contrast PSTD finite difference.
#[test]
fn source_phasing_is_frechet_derivative() {
    let array = MultiRowRingArray::new(4, 1, 0.01, 0.0).expect("ring array");
    let frequency_hz = 200_000.0;
    let reference = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let acquisition = BreastUstPstdDatasetConfig {
        spacing_m: 0.005,
        time_step_s: 1.0e-7,
        cycles_per_frequency: 3,
        frequency_bin_cycles: 1,
        source_amplitude_pa: 1.0e3,
        cpml_thickness_cells: 0,
        ..BreastUstPstdDatasetConfig::default()
    };
    let born_config = PstdFiniteWindowBornConfig {
        reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
        spacing_m: acquisition.spacing_m,
        time_step_s: acquisition.time_step_s,
        source_amplitude_pa: acquisition.source_amplitude_pa,
        cycles_per_frequency: acquisition.cycles_per_frequency,
        frequency_bin_cycles: acquisition.frequency_bin_cycles,
    };
    let full_delta = 1.0e-4;
    let half_delta = 0.5 * full_delta;

    let full = finite_window_first_variation_residual(
        &reference,
        &contrast_volume(full_delta),
        &array,
        frequency_hz,
        acquisition,
        born_config,
    );
    let half = finite_window_first_variation_residual(
        &reference,
        &contrast_volume(half_delta),
        &array,
        frequency_hz,
        acquisition,
        born_config,
    );

    assert!(
        full.increment_norm > f64::EPSILON.sqrt(),
        "PSTD first variation increment must be nonzero"
    );
    assert!(
        half.normalized_residual < full.normalized_residual,
        "finite-window Born residual must converge under contrast refinement: full={:.6e}, half={:.6e}",
        full.normalized_residual,
        half.normalized_residual
    );
    assert!(
        half.residual_norm < full.residual_norm,
        "absolute first-variation residual must decrease under contrast refinement: full={:.6e}, half={:.6e}",
        full.residual_norm,
        half.residual_norm
    );
}

struct FirstVariationResidual {
    increment_norm: f64,
    residual_norm: f64,
    normalized_residual: f64,
}

fn finite_window_first_variation_residual(
    reference: &Array3<f64>,
    perturbed: &Array3<f64>,
    array: &MultiRowRingArray,
    frequency_hz: f64,
    acquisition: BreastUstPstdDatasetConfig,
    born_config: PstdFiniteWindowBornConfig,
) -> FirstVariationResidual {
    let reference_data =
        generate_breast_ust_pstd_frequency_dataset(reference, array, &[frequency_hz], acquisition)
            .expect("reference PSTD data");
    let perturbed_data =
        generate_breast_ust_pstd_frequency_dataset(perturbed, array, &[frequency_hz], acquisition)
            .expect("perturbed PSTD data");
    let born_reference = simulate_pstd_finite_window_born_observation(
        &reference.clone().into(),
        array,
        frequency_hz,
        born_config,
        array.circumferential_elements(),
    )
    .expect("born reference");
    let born_perturbed = simulate_pstd_finite_window_born_observation(
        &perturbed.clone().into(),
        array,
        frequency_hz,
        born_config,
        array.circumferential_elements(),
    )
    .expect("born perturbed");

    let mut increment_norm_sq = 0.0_f64;
    let mut residual_norm_sq = 0.0_f64;
    for (((&pstd_perturbed, &pstd_reference), &born_perturbed), &born_reference) in perturbed_data
        .observed_pressure
        .index_axis::<2>(0, 0)
        .expect("index_axis")
        .iter()
        .zip(
            reference_data
                .observed_pressure
                .index_axis::<2>(0, 0)
                .expect("index_axis")
                .iter(),
        )
        .zip(born_perturbed.iter())
        .zip(born_reference.iter())
    {
        let pstd_increment = pstd_perturbed - pstd_reference;
        let born_increment = born_perturbed - born_reference;
        increment_norm_sq += pstd_increment.norm_sqr();
        residual_norm_sq += (born_increment - pstd_increment).norm_sqr();
    }

    let increment_norm = increment_norm_sq.sqrt();
    let residual_norm = residual_norm_sq.sqrt();
    FirstVariationResidual {
        increment_norm,
        residual_norm,
        normalized_residual: residual_norm / increment_norm,
    }
}

fn test_config() -> PstdFiniteWindowBornConfig {
    PstdFiniteWindowBornConfig {
        reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
        spacing_m: 0.005,
        time_step_s: 1.0e-7,
        source_amplitude_pa: 1.0e3,
        cycles_per_frequency: 3,
        frequency_bin_cycles: 1,
    }
}

fn contrast_volume(normalized_slowness_squared_delta: f64) -> Array3<f64> {
    let mut sound_speed = Array3::from_elem((3, 3, 1), SOUND_SPEED_WATER_SIM);
    let reference_slowness = 1.0 / SOUND_SPEED_WATER_SIM;
    let target_slowness_squared =
        reference_slowness * reference_slowness * (1.0 + normalized_slowness_squared_delta);
    sound_speed[[1, 1, 0]] = 1.0 / target_slowness_squared.sqrt();
    sound_speed
}
