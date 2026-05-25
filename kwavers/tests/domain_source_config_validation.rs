use kwavers::domain::source::{DomainSourceParameters, FocusedBowlAperture};

#[test]
fn source_parameters_reject_nonfinite_physics_values() {
    let invalid_cases = [
        (
            "amplitude",
            source_with(|source| source.amplitude = f64::NAN),
        ),
        (
            "frequency",
            source_with(|source| source.frequency = f64::INFINITY),
        ),
        ("radius", source_with(|source| source.radius = f64::NAN)),
        ("phase", source_with(|source| source.phase = f64::INFINITY)),
        (
            "delay",
            source_with(|source| source.delay = f64::NEG_INFINITY),
        ),
        (
            "position[1]",
            source_with(|source| source.position[1] = f64::NAN),
        ),
        (
            "focus[2]",
            source_with(|source| source.focus = Some([0.0, 0.0, f64::INFINITY])),
        ),
        (
            "pulse.cycles",
            source_with(|source| source.pulse.cycles = f64::NAN),
        ),
    ];

    for (expected_parameter, config) in invalid_cases {
        let error = config.validate().unwrap_err();
        assert!(
            error.to_string().contains(expected_parameter),
            "expected parameter {expected_parameter}, got {error}"
        );
    }
}

#[test]
fn source_parameters_reject_nonpositive_counts_and_cycles() {
    let zero_elements = source_with(|source| source.num_elements = Some(0));
    let zero_cycles = source_with(|source| source.pulse.cycles = 0.0);

    assert!(zero_elements
        .validate()
        .unwrap_err()
        .to_string()
        .contains("num_elements"));
    assert!(zero_cycles
        .validate()
        .unwrap_err()
        .to_string()
        .contains("pulse.cycles"));
}

#[test]
fn focused_bowl_aperture_validation_matches_source_domain_bounds() {
    let invalid_apertures = [
        FocusedBowlAperture::PolarSpan { theta_max_rad: 0.0 },
        FocusedBowlAperture::PolarBounds {
            theta_min_rad: 0.7,
            theta_max_rad: 0.6,
        },
        FocusedBowlAperture::AxisProjectionBounds {
            axis_projection_min: -1.2,
            axis_projection_max: 0.8,
        },
        FocusedBowlAperture::AxisReferencePolarBounds {
            radius_of_curvature_m: 0.0,
            theta_min_rad: 0.0,
            theta_max_rad: 0.9,
        },
        FocusedBowlAperture::AxisReferenceHemisphere {
            radius_of_curvature_m: f64::INFINITY,
        },
    ];

    for aperture in invalid_apertures {
        let config = source_with(|source| source.focused_bowl_aperture = aperture);
        assert!(
            config.validate().is_err(),
            "invalid aperture {aperture:?} must fail validation"
        );
    }

    let valid = source_with(|source| {
        source.focused_bowl_aperture = FocusedBowlAperture::AxisReferencePolarBounds {
            radius_of_curvature_m: 0.16,
            theta_min_rad: 0.2,
            theta_max_rad: 1.1,
        };
    });
    valid.validate().unwrap();
}

fn source_with(mut edit: impl FnMut(&mut DomainSourceParameters)) -> DomainSourceParameters {
    let mut source = DomainSourceParameters::default();
    edit(&mut source);
    source
}
