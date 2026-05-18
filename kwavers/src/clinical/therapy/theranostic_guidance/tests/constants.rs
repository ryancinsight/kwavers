use super::super::{
    AnatomyKind, TheranosticInverseConfig, THERANOSTIC_ELASTIC_SHEAR_MODEL,
    THERANOSTIC_FULL_WAVE_INVERSION, THERANOSTIC_HYBRID_PSTD_FDTD_EXPOSURE_READY,
    THERANOSTIC_INVERSE_MODEL_FAMILY, THERANOSTIC_NONLINEAR_WAVE_PROPAGATION,
    THERANOSTIC_OPERATOR_MODEL, THERANOSTIC_WAVEFORM_MODEL, THERANOSTIC_WAVE_EXPOSURE_BACKEND,
    THERANOSTIC_WAVE_EXPOSURE_MODEL,
};

#[test]
fn theranostic_operator_model_names_graph_laplacian_pcg_contract() {
    assert_eq!(
        THERANOSTIC_OPERATOR_MODEL,
        "finite_frequency_same_aperture_graph_laplacian_pcg"
    );
    assert_eq!(
        THERANOSTIC_WAVEFORM_MODEL,
        "source_encoded_time_domain_acoustic_adjoint_rtm"
    );
    assert_eq!(
        THERANOSTIC_WAVE_EXPOSURE_MODEL,
        "source_encoded_time_domain_acoustic_peak_pressure_exposure"
    );
    assert_eq!(THERANOSTIC_WAVE_EXPOSURE_BACKEND, "reference_fdtd_cpml_2d");
    assert!(!THERANOSTIC_HYBRID_PSTD_FDTD_EXPOSURE_READY);
    assert_eq!(
        THERANOSTIC_INVERSE_MODEL_FAMILY,
        "reduced_born_normal_equation_plus_linear_acoustic_rtm_plus_iterative_nonlinear_elastic_fwi"
    );
    assert_eq!(
        THERANOSTIC_ELASTIC_SHEAR_MODEL,
        "iterative_nonlinear_elastic_pstd_fwi_residual_migration"
    );
    assert!(THERANOSTIC_FULL_WAVE_INVERSION);
    assert!(!THERANOSTIC_NONLINEAR_WAVE_PROPAGATION);
    assert_eq!(
        TheranosticInverseConfig::new(AnatomyKind::Kidney).elastic_frequencies_hz,
        vec![250.0, 500.0, 750.0]
    );
    assert_eq!(
        TheranosticInverseConfig::new(AnatomyKind::Kidney).elastic_shear_speed_m_s,
        2.5
    );
    assert_eq!(
        TheranosticInverseConfig::new(AnatomyKind::Kidney).elastic_fwi_iterations,
        3
    );
    assert_eq!(
        TheranosticInverseConfig::new(AnatomyKind::Kidney)
            .waveform_misfit
            .label(),
        "charbonnier"
    );
}
