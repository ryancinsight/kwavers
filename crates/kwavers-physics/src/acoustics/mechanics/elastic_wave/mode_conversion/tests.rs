use super::*;

#[test]
fn test_stiffness_tensor_isotropic() {
    let lambda = 1e10;
    let mu = 5e9;
    let density = 2700.0;

    let tensor = ElasticModeStiffnessTensor::isotropic(lambda, mu, density).unwrap();
    assert_eq!(tensor.symmetry, MaterialSymmetry::Isotropic);
    assert_eq!(tensor.c[[0, 0]], lambda + 2.0 * mu);
    assert_eq!(tensor.c[[3, 3]], mu);
    assert_eq!(tensor.c[[0, 1]], lambda);
}

#[test]
fn test_mode_conversion_config() {
    let config = ModeConversionConfig::default();
    assert!(config.enable_p_to_s);
    assert!(config.enable_s_to_p);
    assert_eq!(config.critical_angle, std::f64::consts::PI / 4.0);
}

#[test]
fn test_viscoelastic_config() {
    let config = ViscoelasticConfig::default();
    assert_eq!(config.q_p, 100.0);
    assert_eq!(config.q_s, 50.0);
    assert!(config.frequency_dependent);
}

/// `ElasticModeStiffnessTensor::hexagonal` encodes the transversely-isotropic stiffness
/// matrix in Voigt notation. Analytical values for a representative
/// geological medium (VTI shale, after Thomsen 1986):
///   c11=38.0, c33=22.0, c12=10.0, c13=8.0, c44=6.0 GPa, ρ=2200 kg/m³.
///
/// The hexagonal symmetry requires c66 = (c11 − c12)/2 = 14 GPa.
#[test]
fn stiffness_tensor_hexagonal_encodes_vti_values_correctly() {
    let (c11, c33, c12, c13, c44) = (38.0e9, 22.0e9, 10.0e9, 8.0e9, 6.0e9);
    let density = 2200.0_f64;
    let t = ElasticModeStiffnessTensor::hexagonal(c11, c33, c12, c13, c44, density).unwrap();

    assert_eq!(t.symmetry, MaterialSymmetry::Hexagonal);
    assert!((t.c[[0, 0]] - c11).abs() < 1.0, "c11");
    assert!((t.c[[1, 1]] - c11).abs() < 1.0, "c22 = c11 for hexagonal");
    assert!((t.c[[2, 2]] - c33).abs() < 1.0, "c33");
    assert!((t.c[[3, 3]] - c44).abs() < 1.0, "c44");
    assert!((t.c[[4, 4]] - c44).abs() < 1.0, "c55 = c44 for hexagonal");
    let c66 = (c11 - c12) / 2.0;
    assert!((t.c[[5, 5]] - c66).abs() < 1.0, "c66");
    assert!((t.c[[0, 1]] - c12).abs() < 1.0, "c12");
    assert!((t.c[[0, 2]] - c13).abs() < 1.0, "c13");
    assert!((t.c[[1, 2]] - c13).abs() < 1.0, "c23");
    for i in 0..6 {
        for j in i + 1..6 {
            assert!(
                (t.c[[i, j]] - t.c[[j, i]]).abs() < 1.0,
                "symmetry c[{i}][{j}]"
            );
        }
    }
}

/// `ElasticModeStiffnessTensor::validate` accepts a physically valid isotropic tensor.
#[test]
fn stiffness_tensor_validate_accepts_valid_isotropic_tensor() {
    let t = ElasticModeStiffnessTensor::isotropic(1e10, 5e9, 2700.0).unwrap();
    assert!(
        t.validate().is_ok(),
        "valid isotropic tensor must pass validate"
    );
}

/// `ElasticModeStiffnessTensor::validate` rejects an asymmetric matrix.
#[test]
fn stiffness_tensor_validate_rejects_asymmetric_matrix() {
    let mut t = ElasticModeStiffnessTensor::isotropic(1e10, 5e9, 2700.0).unwrap();
    t.c[[0, 1]] += 1.0;
    assert!(
        t.validate().is_err(),
        "asymmetric stiffness matrix must be rejected"
    );
}

/// `ElasticModeStiffnessTensor::isotropic` rejects non-positive density or shear modulus μ.
#[test]
fn stiffness_tensor_isotropic_rejects_invalid_parameters() {
    assert!(
        ElasticModeStiffnessTensor::isotropic(1e10, 5e9, 0.0).is_err(),
        "zero density"
    );
    assert!(
        ElasticModeStiffnessTensor::isotropic(1e10, 5e9, -1.0).is_err(),
        "negative density"
    );
    assert!(
        ElasticModeStiffnessTensor::isotropic(1e10, 0.0, 2700.0).is_err(),
        "zero mu"
    );
    assert!(
        ElasticModeStiffnessTensor::isotropic(1e10, -1.0, 2700.0).is_err(),
        "negative mu"
    );
}
