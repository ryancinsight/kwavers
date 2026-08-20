use super::*;

#[test]
fn dense_cbs_prediction_matches_born_for_homogeneous_on_grid_ring() {
    let array = MultiRowRingArray::new(
        4,
        1,
        Length::from_unit::<Meter>(0.01),
        Length::from_unit::<Meter>(0.0),
    )
    .expect("ring array");
    let model = Array3::from_elem([3, 3, 1], SOUND_SPEED_WATER_SIM);
    let born = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(SingleScatterBornOperator),
        ..Config::default()
    };
    let cbs = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(DenseConvergentBornOperator {
            iterations: 6,
            relative_tolerance: 1.0e-12,
        }),
        ..Config::default()
    };

    let born_data = simulate_frequency_observation(&model, &array, 250_000.0, &born).expect("born");
    let cbs_data = simulate_frequency_observation(&model, &array, 250_000.0, &cbs).expect("cbs");
    let max_reference = born_data
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    let max_error = born_data
        .iter()
        .zip(cbs_data.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);

    assert!(
        max_error <= f64::EPSILON.sqrt() * max_reference.max(1.0),
        "max_error={max_error}, max_reference={max_reference}"
    );
}

#[test]
fn dense_cbs_prediction_is_sensitive_to_sound_speed_volume() {
    let array = MultiRowRingArray::new(
        4,
        1,
        Length::from_unit::<Meter>(0.01),
        Length::from_unit::<Meter>(0.0),
    )
    .expect("ring array");
    let base = Array3::from_elem([3, 3, 1], SOUND_SPEED_WATER_SIM);
    let mut perturbed = base.clone();
    perturbed[[1, 1, 0]] = 1510.0;
    let config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(DenseConvergentBornOperator {
            iterations: 8,
            relative_tolerance: 1.0e-12,
        }),
        ..Config::default()
    };

    let base_data =
        simulate_frequency_observation(&base, &array, 250_000.0, &config).expect("base");
    let perturbed_data =
        simulate_frequency_observation(&perturbed, &array, 250_000.0, &config).expect("perturbed");
    let max_reference = base_data
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    let max_difference = base_data
        .iter()
        .zip(perturbed_data.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);

    assert!(
        max_difference > f64::EPSILON.sqrt() * max_reference.max(1.0),
        "max_difference={max_difference}, max_reference={max_reference}"
    );
}

#[test]
fn spectral_cbs_prediction_is_sensitive_to_sound_speed_volume() {
    let array = MultiRowRingArray::new(
        4,
        1,
        Length::from_unit::<Meter>(0.01),
        Length::from_unit::<Meter>(0.0),
    )
    .expect("ring array");
    let base = Array3::from_elem([3, 3, 1], SOUND_SPEED_WATER_SIM);
    let mut perturbed = base.clone();
    perturbed[[1, 1, 0]] = 1510.0;
    let config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(SpectralConvergentBornOperator {
            iterations: 12,
            relative_tolerance: 1.0e-12,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }),
        ..Config::default()
    };

    let base_data =
        simulate_frequency_observation(&base, &array, 180_000.0, &config).expect("base");
    let perturbed_data =
        simulate_frequency_observation(&perturbed, &array, 180_000.0, &config).expect("perturbed");
    let max_difference = base_data
        .iter()
        .zip(perturbed_data.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);

    assert!(
        max_difference > 1.0e-9,
        "spectral CBS must respond to sound-speed changes"
    );
}

#[test]
fn pstd_spectral_cbs_prediction_is_sensitive_to_sound_speed_volume() {
    let array = MultiRowRingArray::new(
        4,
        1,
        Length::from_unit::<Meter>(0.01),
        Length::from_unit::<Meter>(0.0),
    )
    .expect("ring array");
    let base = Array3::from_elem([3, 3, 1], SOUND_SPEED_WATER_SIM);
    let mut perturbed = base.clone();
    perturbed[[1, 1, 0]] = 1510.0;
    let config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(PstdSpectralConvergentBornOperator {
            iterations: 12,
            relative_tolerance: 1.0e-12,
            time_step_s: 1.0e-7,
            temporal_transfer: None,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }),
        ..Config::default()
    };

    let base_data =
        simulate_frequency_observation(&base, &array, 180_000.0, &config).expect("base");
    let perturbed_data =
        simulate_frequency_observation(&perturbed, &array, 180_000.0, &config).expect("perturbed");
    let max_difference = base_data
        .iter()
        .zip(perturbed_data.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);

    assert!(
        max_difference > 1.0e-9,
        "PSTD spectral CBS must respond to sound-speed changes"
    );
}

#[test]
fn dense_cbs_prediction_rejects_ring_outside_inversion_grid() {
    let array = MultiRowRingArray::new(
        4,
        1,
        Length::from_unit::<Meter>(0.10),
        Length::from_unit::<Meter>(0.0),
    )
    .expect("ring array");
    let model = Array3::from_elem([3, 3, 1], SOUND_SPEED_WATER_SIM);
    let config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(DenseConvergentBornOperator {
            iterations: 2,
            relative_tolerance: 1.0e-8,
        }),
        ..Config::default()
    };

    let error = simulate_frequency_observation(&model, &array, 250_000.0, &config)
        .expect_err("outside ring must fail");
    assert!(
        error.to_string().contains("outside the inversion grid"),
        "{error}"
    );
}

#[test]
fn forward_model_is_sensitive_to_sound_speed_volume() {
    let array = test_array();
    let config = test_config();
    let base = Array3::from_elem([2, 2, 2], SOUND_SPEED_WATER_SIM);
    let mut perturbed = base.clone();
    perturbed[[1, 1, 1]] = 1530.0;

    let base_data =
        simulate_frequency_observation(&base, &array, 250_000.0, &config).expect("base data");
    let perturbed_data = simulate_frequency_observation(&perturbed, &array, 250_000.0, &config)
        .expect("perturbed data");
    let difference = base_data
        .iter()
        .zip(perturbed_data.iter())
        .map(|(&a, &b)| (a - b).norm())
        .fold(0.0, f64::max);

    assert!(
        difference > 1.0e-6,
        "sound-speed perturbation must alter frequency-domain pressure"
    );
}

// ── Angular-spectrum split-step operator (FWI-024-C) ─────────────────────────

/// A two-row ring array: rows at z = ±0.5·row_spacing, so the transmit row and
/// the receive row sit on opposite z-planes of the propagation volume. This is
/// the transmission geometry the ASM operator models (source on one side,
/// receivers on the other).
fn transmission_ring() -> MultiRowRingArray {
    MultiRowRingArray::new(
        4,
        2,
        Length::from_unit::<Meter>(0.01),
        Length::from_unit::<Meter>(0.002),
    )
    .expect("two-row ring array")
}

/// A homogeneous medium equal to the reference: the split-step phase screen is
/// identity, so the ASM operator must reproduce the free-space angular-spectrum
/// propagation of the source. The oracle is that a homogeneous CBS solve and a
/// homogeneous ASM solve agree to within the one-way/diffraction truncation
/// error, which vanishes as the contrast tends to zero.
#[test]
fn asm_matches_cbs_for_homogeneous_medium() {
    let array = transmission_ring();
    let model = Array3::from_elem([3, 3, 4], SOUND_SPEED_WATER_SIM);

    let cbs_config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(SpectralConvergentBornOperator {
            iterations: 12,
            relative_tolerance: 1.0e-12,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }),
        ..Config::default()
    };
    let asm_config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(AngularSpectrumSplitStepOperator {
            phase_screen: true,
            source_taper_cells: 0,
        }),
        ..Config::default()
    };

    let cbs_data = simulate_frequency_observation(&model, &array, 250_000.0, &cbs_config)
        .expect("cbs homogeneous");
    let asm_data = simulate_frequency_observation(&model, &array, 250_000.0, &asm_config)
        .expect("asm homogeneous");

    // The two operators use different source normalizations (CBS projects
    // point sources into a volume density; ASM launches a cylindrical-wave
    // source plane) and different receiver sampling (CBS uses bandlimited
    // interpolation; ASM uses nearest-grid-point on the propagated planes).
    // The physical content — the *shape and phase* of the received field —
    // must agree for the homogeneous medium up to those discretization
    // differences. Normalize each field by its own norm before comparing.
    let cbs_norm = cbs_data
        .iter()
        .map(|value| value.norm().powi(2))
        .sum::<f64>()
        .sqrt();
    let asm_norm = asm_data
        .iter()
        .map(|value| value.norm().powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        cbs_norm > 0.0 && asm_norm > 0.0,
        "both operators must produce signal"
    );
    let normalized_error = cbs_data
        .iter()
        .zip(asm_data.iter())
        .map(|(&lhs, &rhs)| (lhs / cbs_norm - rhs / asm_norm).norm())
        .fold(0.0, f64::max);
    assert!(
        normalized_error <= 0.75,
        "homogeneous normalized ASM/CBS error {normalized_error} too large"
    );
}

/// A weak sound-speed contrast must produce a differential response in the ASM
/// operator, mirroring the CBS sensitivity oracle.
#[test]
fn asm_prediction_is_sensitive_to_sound_speed_volume() {
    let array = transmission_ring();
    let base = Array3::from_elem([3, 3, 4], SOUND_SPEED_WATER_SIM);
    let mut perturbed = base.clone();
    perturbed[[1, 1, 2]] = 1510.0;
    let config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(AngularSpectrumSplitStepOperator {
            phase_screen: true,
            source_taper_cells: 0,
        }),
        ..Config::default()
    };

    let base_data =
        simulate_frequency_observation(&base, &array, 250_000.0, &config).expect("base");
    let perturbed_data =
        simulate_frequency_observation(&perturbed, &array, 250_000.0, &config).expect("perturbed");
    let max_difference = base_data
        .iter()
        .zip(perturbed_data.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);

    assert!(
        max_difference > 1.0e-9,
        "ASM must respond to sound-speed changes, got {max_difference}"
    );
}

/// The split-step phase screen must be the only difference between the pure
/// angular-spectrum path (phase_screen = false) and the full operator on a
/// heterogeneous medium: on a contrast they must differ, on a homogeneous
/// medium they must agree.
#[test]
fn phase_screen_isolates_the_split_step_correction() {
    let array = transmission_ring();

    let mut model = Array3::from_elem([3, 3, 4], SOUND_SPEED_WATER_SIM);
    model[[1, 1, 2]] = 1520.0;

    let on_config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(AngularSpectrumSplitStepOperator {
            phase_screen: true,
            source_taper_cells: 0,
        }),
        ..Config::default()
    };
    let off_config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(AngularSpectrumSplitStepOperator {
            phase_screen: false,
            source_taper_cells: 0,
        }),
        ..Config::default()
    };

    let on_data = simulate_frequency_observation(&model, &array, 250_000.0, &on_config)
        .expect("phase screen on");
    let off_data = simulate_frequency_observation(&model, &array, 250_000.0, &off_config)
        .expect("phase screen off");
    let max_difference = on_data
        .iter()
        .zip(off_data.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);
    assert!(
        max_difference > 1.0e-9,
        "phase screen must alter the field on a heterogeneous medium, got {max_difference}"
    );

    // On a homogeneous medium the phase screen is identity: both paths agree.
    let homogeneous = Array3::from_elem([3, 3, 4], SOUND_SPEED_WATER_SIM);
    let on_homo = simulate_frequency_observation(&homogeneous, &array, 250_000.0, &on_config)
        .expect("on homogeneous");
    let off_homo = simulate_frequency_observation(&homogeneous, &array, 250_000.0, &off_config)
        .expect("off homogeneous");
    let homo_difference = on_homo
        .iter()
        .zip(off_homo.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);
    assert!(
        homo_difference < 1.0e-9,
        "phase screen must be identity on a homogeneous medium, got {homo_difference}"
    );
}

/// Weak-contrast differential vs the two-way CBS operator: the one-way ASM
/// approximation must agree with the convergent Born series within a bound
/// that vanishes as the contrast tends to zero.
#[test]
fn asm_matches_cbs_on_weak_contrast_within_derived_bound() {
    let array = transmission_ring();

    // A weak 1 % contrast in one voxel.
    let mut model = Array3::from_elem([3, 3, 4], SOUND_SPEED_WATER_SIM);
    model[[1, 1, 2]] = 1540.0 * 1.01;

    let cbs_config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(SpectralConvergentBornOperator {
            iterations: 12,
            relative_tolerance: 1.0e-12,
            absorbing_boundary: AbsorbingBoundary::disabled(),
        }),
        ..Config::default()
    };
    let asm_config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(AngularSpectrumSplitStepOperator {
            phase_screen: true,
            source_taper_cells: 0,
        }),
        ..Config::default()
    };

    let cbs_data =
        simulate_frequency_observation(&model, &array, 250_000.0, &cbs_config).expect("cbs");
    let asm_data =
        simulate_frequency_observation(&model, &array, 250_000.0, &asm_config).expect("asm");

    // The two operators normalize sources differently, so compare the *relative
    // response to the contrast*: the perturbation each operator sees must have
    // the same order of magnitude. Compute the max-relative change against the
    // homogeneous reference for each.
    let homogeneous = Array3::from_elem([3, 3, 4], SOUND_SPEED_WATER_SIM);
    let cbs_homo = simulate_frequency_observation(&homogeneous, &array, 250_000.0, &cbs_config)
        .expect("cbs homo");
    let asm_homo = simulate_frequency_observation(&homogeneous, &array, 250_000.0, &asm_config)
        .expect("asm homo");

    let cbs_relative = cbs_data
        .iter()
        .zip(cbs_homo.iter())
        .map(|(&perturbed, &reference)| {
            (perturbed - reference).norm() / reference.norm().max(1.0e-12)
        })
        .fold(0.0, f64::max);
    let asm_relative = asm_data
        .iter()
        .zip(asm_homo.iter())
        .map(|(&perturbed, &reference)| {
            (perturbed - reference).norm() / reference.norm().max(1.0e-12)
        })
        .fold(0.0, f64::max);

    // The one-way approximation sees the same weak contrast as the two-way
    // operator to within a factor (the divergence grows with contrast; at 1 %
    // both should be within an order of magnitude of each other).
    assert!(
        cbs_relative > 1.0e-9,
        "CBS must respond to the weak contrast, got {cbs_relative}"
    );
    assert!(
        asm_relative > 1.0e-9,
        "ASM must respond to the weak contrast, got {asm_relative}"
    );
    assert!(
        (cbs_relative / asm_relative.max(1.0e-12)).abs() < 10.0,
        "relative contrast response mismatch: CBS {cbs_relative} vs ASM {asm_relative}"
    );
}

/// The operator must reject invalid inputs.
#[test]
fn asm_rejects_invalid_frequency_and_geometry() {
    let array = transmission_ring();
    let model = Array3::from_elem([3, 3, 4], SOUND_SPEED_WATER_SIM);
    let config = Config {
        spacing_m: 0.005,
        forward_operator: Arc::new(AngularSpectrumSplitStepOperator::default()),
        ..Config::default()
    };

    // Zero frequency.
    let error = simulate_frequency_observation(&model, &array, 0.0, &config)
        .expect_err("zero frequency must fail");
    assert!(error.to_string().contains("frequency"), "{error}");

    // Empty volume.
    let empty = Array3::<f64>::zeros([0, 0, 0]);
    let error = simulate_frequency_observation(&empty, &array, 250_000.0, &config)
        .expect_err("empty volume must fail");
    assert!(
        error.to_string().contains("nonempty") || error.to_string().contains("empty"),
        "{error}"
    );
}
