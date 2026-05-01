use super::*;
use ndarray::Array3;

#[test]
fn test_bvf_water_is_zero() {
    assert_eq!(HeterogeneousSkull::bone_volume_fraction(0.0), 0.0);
}

#[test]
fn test_bvf_cortical_is_one() {
    assert!((HeterogeneousSkull::bone_volume_fraction(1000.0) - 1.0).abs() < 1e-12);
}

#[test]
fn test_bvf_diploe_midpoint() {
    let phi = HeterogeneousSkull::bone_volume_fraction(500.0);
    assert!(
        (phi - 0.5).abs() < 1e-12,
        "BVF at HU=500 should be 0.5; got {phi:.6}"
    );
}

#[test]
fn test_bvf_negative_hu_clamped_to_zero() {
    assert_eq!(HeterogeneousSkull::bone_volume_fraction(-100.0), 0.0);
}

#[test]
fn test_bvf_high_hu_clamped_to_one() {
    assert!((HeterogeneousSkull::bone_volume_fraction(2000.0) - 1.0).abs() < 1e-12);
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
    let skull = HeterogeneousSkull::from_ct_hill(&ct, 3100.0, 2100.0, 20.0).unwrap();
    for &c in skull.sound_speed.iter() {
        assert!(
            (c - C_WATER).abs() < 1.0,
            "water voxel speed {c:.1} should equal C_WATER={C_WATER}"
        );
    }
}

#[test]
fn test_hill_bone_limit_gives_c_bone() {
    let c_bone = 3100.0_f64;
    let ct = Array3::from_elem((4, 4, 4), 1000.0_f64);
    let skull = HeterogeneousSkull::from_ct_hill(&ct, c_bone, 2100.0, 20.0).unwrap();
    for &c in skull.sound_speed.iter() {
        assert!(
            (c - c_bone).abs() < 1.0,
            "bone voxel speed {c:.1} should equal c_bone={c_bone}"
        );
    }
}

#[test]
fn test_hill_diploe_speed_between_water_and_bone() {
    let c_bone = 3100.0_f64;
    let ct = Array3::from_elem((4, 4, 4), 500.0_f64);
    let skull = HeterogeneousSkull::from_ct_hill(&ct, c_bone, 2100.0, 20.0).unwrap();
    for &c in skull.sound_speed.iter() {
        assert!(
            c > C_WATER && c < c_bone,
            "diploe speed {c:.1} must be strictly between {C_WATER} and {c_bone}"
        );
    }
}

#[test]
fn test_hill_density_voigt_rule() {
    let rho_bone = 2100.0_f64;
    let phi = 0.5_f64;
    let hu = phi * HU_CORTICAL;
    let ct = Array3::from_elem((2, 2, 2), hu);
    let skull = HeterogeneousSkull::from_ct_hill(&ct, 3100.0, rho_bone, 20.0).unwrap();
    let expected_rho = phi * rho_bone + (1.0 - phi) * RHO_WATER;
    for &rho in skull.density.iter() {
        assert!(
            (rho - expected_rho).abs() < 1e-6,
            "density {rho:.3} != Voigt {expected_rho:.3}"
        );
    }
}

#[test]
fn test_hill_attenuation_linear_interpolation() {
    let alpha_bone = 20.0_f64;
    let phi = 0.6_f64;
    let hu = phi * HU_CORTICAL;
    let ct = Array3::from_elem((2, 2, 2), hu);
    let skull = HeterogeneousSkull::from_ct_hill(&ct, 3100.0, 2100.0, alpha_bone).unwrap();
    let expected = phi * alpha_bone + (1.0 - phi) * ALPHA_WATER;
    for &a in skull.attenuation.iter() {
        assert!(
            (a - expected).abs() < 1e-9,
            "attenuation {a:.6} != linear BVF {expected:.6}"
        );
    }
}

#[test]
fn test_hill_speed_does_not_exceed_voigt_modulus_speed() {
    let c_bone = 3100.0_f64;
    let rho_bone = 2100.0_f64;
    let k_bone = rho_bone * c_bone * c_bone;
    let k_water = RHO_WATER * C_WATER * C_WATER;
    for hu_int in 1_u32..10 {
        let hu = hu_int as f64 * 100.0;
        let phi = HeterogeneousSkull::bone_volume_fraction(hu);
        let rho_eff = phi * rho_bone + (1.0 - phi) * RHO_WATER;
        let k_voigt = phi * k_bone + (1.0 - phi) * k_water;
        let voigt_modulus_speed = (k_voigt / rho_eff).sqrt();
        let ct = Array3::from_elem((1, 1, 1), hu);
        let skull = HeterogeneousSkull::from_ct_hill(&ct, c_bone, rho_bone, 20.0).unwrap();
        let hill_speed = skull.sound_speed[[0, 0, 0]];
        assert!(
            hill_speed <= voigt_modulus_speed + 1e-6,
            "Hill speed {hill_speed:.2} exceeds Voigt-modulus speed {voigt_modulus_speed:.2} at HU={hu}"
        );
    }
}
