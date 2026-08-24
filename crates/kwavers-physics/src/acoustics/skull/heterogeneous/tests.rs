use super::*;
use crate::acoustics::skull::AcousticSkullProperties;
use aequitas::systems::si::quantities::{Length, MassDensity, ReciprocalLength, Velocity};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::KwaversError;
use leto::Array3;

// The Hill oracle and implementation each use fewer than 16 rounded elementary
// operations; 32 epsilons covers both evaluation paths.
const HILL_ROUNDOFF_FACTOR: f64 = 32.0;

fn bone_properties(c_bone: f64, rho_bone: f64, alpha_bone: f64) -> AcousticSkullProperties {
    AcousticSkullProperties::new(
        Velocity::from_base(c_bone),
        MassDensity::from_base(rho_bone),
        ReciprocalLength::from_base(alpha_bone),
        Length::from_base(0.007),
        None,
    )
    .expect("finite positive bone properties")
}

#[test]
fn test_bvf_water_is_zero() {
    assert_eq!(HeterogeneousSkull::bone_volume_fraction(0.0), 0.0);
}

#[test]
fn test_bvf_cortical_is_one() {
    assert_eq!(HeterogeneousSkull::bone_volume_fraction(1000.0), 1.0);
}

#[test]
fn test_bvf_diploe_midpoint() {
    let phi = HeterogeneousSkull::bone_volume_fraction(500.0);
    assert_eq!(phi, 0.5);
}

#[test]
fn test_bvf_negative_hu_clamped_to_zero() {
    assert_eq!(HeterogeneousSkull::bone_volume_fraction(-100.0), 0.0);
}

#[test]
fn test_bvf_high_hu_clamped_to_one() {
    assert_eq!(HeterogeneousSkull::bone_volume_fraction(2000.0), 1.0);
}

#[test]
fn test_classify_water_is_soft_tissue() {
    assert_eq!(
        HeterogeneousSkull::classify_layer(0.0),
        SkullLayer::SoftTissue
    );
}

#[test]
fn test_classify_diploe() {
    assert_eq!(
        HeterogeneousSkull::classify_layer(400.0),
        SkullLayer::Diploe
    );
}

#[test]
fn test_classify_cortical() {
    assert_eq!(
        HeterogeneousSkull::classify_layer(900.0),
        SkullLayer::Cortical
    );
}

#[test]
fn test_hill_water_limit_gives_c_water() {
    let ct = Array3::from_elem((4, 4, 4), 0.0_f64);
    let skull = HeterogeneousSkull::from_ct_hill(&ct, &bone_properties(3100.0, 2100.0, 20.0))
        .expect("finite test CT");
    for &c in skull.sound_speed.iter() {
        let bound = HILL_ROUNDOFF_FACTOR * f64::EPSILON * SOUND_SPEED_WATER_SIM;
        assert!(
            (c - SOUND_SPEED_WATER_SIM).abs() <= bound,
            "water voxel speed {c:.1} should equal SOUND_SPEED_WATER_SIM={SOUND_SPEED_WATER_SIM}"
        );
    }
}

#[test]
fn test_hill_bone_limit_gives_c_bone() {
    let c_bone = 3100.0_f64;
    let ct = Array3::from_elem((4, 4, 4), 1000.0_f64);
    let skull = HeterogeneousSkull::from_ct_hill(&ct, &bone_properties(c_bone, 2100.0, 20.0))
        .expect("finite test CT");
    for &c in skull.sound_speed.iter() {
        let bound = HILL_ROUNDOFF_FACTOR * f64::EPSILON * c_bone;
        assert!(
            (c - c_bone).abs() <= bound,
            "bone voxel speed {c:.1} should equal c_bone={c_bone}"
        );
    }
}

#[test]
fn test_hill_diploe_speed_between_water_and_bone() {
    let c_bone = 3100.0_f64;
    let ct = Array3::from_elem((4, 4, 4), 500.0_f64);
    let skull = HeterogeneousSkull::from_ct_hill(&ct, &bone_properties(c_bone, 2100.0, 20.0))
        .expect("finite test CT");
    for &c in skull.sound_speed.iter() {
        assert!(
            c > SOUND_SPEED_WATER_SIM && c < c_bone,
            "diploe speed {c:.1} must be strictly between {SOUND_SPEED_WATER_SIM} and {c_bone}"
        );
    }
}

#[test]
fn hill_midpoint_matches_closed_form_modulus() {
    let c_bone = 3100.0_f64;
    let rho_bone = 2100.0_f64;
    let phi = 0.5_f64;
    let ct = Array3::from_elem((1, 1, 1), phi * HU_CORTICAL);
    let skull = HeterogeneousSkull::from_ct_hill(&ct, &bone_properties(c_bone, rho_bone, 20.0))
        .expect("finite test CT");

    let k_bone = rho_bone * c_bone * c_bone;
    let k_water = DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM * SOUND_SPEED_WATER_SIM;
    let density = phi * rho_bone + (1.0 - phi) * DENSITY_WATER_NOMINAL;
    let k_voigt = phi * k_bone + (1.0 - phi) * k_water;
    let k_reuss = 1.0 / (phi / k_bone + (1.0 - phi) / k_water);
    let expected = (0.5 * (k_voigt + k_reuss) / density).sqrt();
    let observed = skull.sound_speed[[0, 0, 0]];
    let bound = HILL_ROUNDOFF_FACTOR * f64::EPSILON * expected;
    assert!((observed - expected).abs() <= bound);
}

#[test]
fn test_hill_density_voigt_rule() {
    let rho_bone = 2100.0_f64;
    let phi = 0.5_f64;
    let hu = phi * HU_CORTICAL;
    let ct = Array3::from_elem((2, 2, 2), hu);
    let skull = HeterogeneousSkull::from_ct_hill(&ct, &bone_properties(3100.0, rho_bone, 20.0))
        .expect("finite test CT");
    let expected_rho = phi * rho_bone + (1.0 - phi) * DENSITY_WATER_NOMINAL;
    for &rho in skull.density.iter() {
        assert_eq!(rho, expected_rho);
    }
}

#[test]
fn test_hill_attenuation_linear_interpolation() {
    let alpha_bone = 20.0_f64;
    let phi = 0.6_f64;
    let hu = phi * HU_CORTICAL;
    let ct = Array3::from_elem((2, 2, 2), hu);
    let skull = HeterogeneousSkull::from_ct_hill(&ct, &bone_properties(3100.0, 2100.0, alpha_bone))
        .expect("finite test CT");
    let expected = phi * alpha_bone + (1.0 - phi) * ALPHA_WATER;
    for &a in skull.attenuation.iter() {
        assert_eq!(a, expected);
    }
}

#[test]
fn test_hill_speed_does_not_exceed_voigt_modulus_speed() {
    let c_bone = 3100.0_f64;
    let rho_bone = 2100.0_f64;
    let k_bone = rho_bone * c_bone * c_bone;
    let k_water = DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM * SOUND_SPEED_WATER_SIM;
    for hu_int in 1_u32..10 {
        let hu = hu_int as f64 * 100.0;
        let phi = HeterogeneousSkull::bone_volume_fraction(hu);
        let rho_eff = phi * rho_bone + (1.0 - phi) * DENSITY_WATER_NOMINAL;
        let k_voigt = phi * k_bone + (1.0 - phi) * k_water;
        let voigt_modulus_speed = (k_voigt / rho_eff).sqrt();
        let ct = Array3::from_elem((1, 1, 1), hu);
        let skull = HeterogeneousSkull::from_ct_hill(&ct, &bone_properties(c_bone, rho_bone, 20.0))
            .expect("finite test CT");
        let hill_speed = skull.sound_speed[[0, 0, 0]];
        let bound = HILL_ROUNDOFF_FACTOR * f64::EPSILON * voigt_modulus_speed;
        assert!(
            hill_speed <= voigt_modulus_speed + bound,
            "Hill speed {hill_speed:.2} exceeds Voigt-modulus speed {voigt_modulus_speed:.2} at HU={hu}"
        );
    }
}

#[test]
fn bone_properties_reject_non_physical_values() {
    let invalid = [
        AcousticSkullProperties::new(
            Velocity::from_base(0.0),
            MassDensity::from_base(1900.0),
            ReciprocalLength::from_base(60.0),
            Length::from_base(0.007),
            None,
        ),
        AcousticSkullProperties::new(
            Velocity::from_base(3100.0),
            MassDensity::from_base(f64::NAN),
            ReciprocalLength::from_base(60.0),
            Length::from_base(0.007),
            None,
        ),
        AcousticSkullProperties::new(
            Velocity::from_base(3100.0),
            MassDensity::from_base(1900.0),
            ReciprocalLength::from_base(-1.0),
            Length::from_base(0.007),
            None,
        ),
    ];

    for result in invalid {
        assert!(matches!(result, Err(KwaversError::InvalidInput(_))));
    }
}

#[test]
fn hill_model_rejects_non_finite_ct_voxels() {
    let ct =
        Array3::from_shape_vec((1, 1, 2), vec![0.0, f64::INFINITY]).expect("shape matches values");
    let error = HeterogeneousSkull::from_ct_hill(&ct, &AcousticSkullProperties::cortical())
        .expect_err("non-finite HU must be rejected");

    assert!(matches!(error, KwaversError::InvalidInput(message) if message.contains("voxel 1")));
}

#[test]
fn hill_model_rejects_empty_ct_volume() {
    let ct = Array3::zeros((0, 1, 1));
    let error = HeterogeneousSkull::from_ct_hill(&ct, &AcousticSkullProperties::cortical())
        .expect_err("empty CT must be rejected");

    assert!(
        matches!(error, KwaversError::InvalidInput(message) if message.contains("at least one voxel"))
    );
}
