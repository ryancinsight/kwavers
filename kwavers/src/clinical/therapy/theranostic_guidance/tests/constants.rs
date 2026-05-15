use super::super::{
    AnatomyKind, TheranosticInverseConfig, THERANOSTIC_FULL_WAVE_INVERSION,
    THERANOSTIC_INVERSE_MODEL_FAMILY, THERANOSTIC_NONLINEAR_WAVE_PROPAGATION,
    THERANOSTIC_OPERATOR_MODEL, THERANOSTIC_WAVEFORM_MODEL,
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
        THERANOSTIC_INVERSE_MODEL_FAMILY,
        "reduced_born_normal_equation_plus_linear_acoustic_rtm"
    );
    assert!(!THERANOSTIC_FULL_WAVE_INVERSION);
    assert!(!THERANOSTIC_NONLINEAR_WAVE_PROPAGATION);
    assert_eq!(
        TheranosticInverseConfig::new(AnatomyKind::Kidney)
            .waveform_misfit
            .label(),
        "charbonnier"
    );
}
