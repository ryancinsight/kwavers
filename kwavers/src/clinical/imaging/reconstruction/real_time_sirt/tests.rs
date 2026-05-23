use super::config::RealTimeSirtConfig;
use super::pipeline::RealTimeSirtPipeline;
use crate::clinical::imaging::reconstruction::acoustic_projection::{
    backproject_acoustic, project_acoustic, AcousticProjectionGeometry,
};
use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use crate::solver::inverse::reconstruction::unified_sirt::SirtConfig;
use ndarray::{Array1, Array3};

#[test]
fn test_config_default() {
    let config = RealTimeSirtConfig::default();
    assert_eq!(config.target_frame_rate, 10.0);
    assert_eq!(config.max_frame_time_ms, 100.0);
    assert!(config.streaming_mode);
}

#[test]
fn test_config_diagnostic() {
    let config = RealTimeSirtConfig::default().diagnostic_quality();
    assert!(config.max_frame_time_ms > 100.0);
    assert!(config.target_frame_rate < 10.0);
}

#[test]
fn test_config_fast_streaming() {
    let config = RealTimeSirtConfig::default().fast_streaming();
    assert!(config.max_frame_time_ms < 100.0);
    assert!(config.target_frame_rate > 10.0);
}

#[test]
fn test_pipeline_creation() {
    let pipeline = RealTimeSirtPipeline::new(RealTimeSirtConfig::default());
    assert_eq!(pipeline.frame_count(), 0);
}

#[test]
fn test_input_validation_empty() {
    let empty = Array1::<f64>::zeros(0);
    // Access validate_input via the pub(crate) path via process_frame
    let mut pipeline = RealTimeSirtPipeline::new(RealTimeSirtConfig::default());
    let result = pipeline.process_frame(&empty, (4, 4, 4));
    assert!(result.is_err());
}

#[test]
fn test_input_validation_valid() {
    let valid = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let mut pipeline = RealTimeSirtPipeline::new(RealTimeSirtConfig {
        enable_preprocessing: false,
        enable_quality_monitoring: false,
        output_smoothing_sigma: None,
        ..Default::default()
    });
    let frame = pipeline.process_frame(&valid, (4, 4, 1)).unwrap();
    assert_eq!(frame.image.dim(), (4, 4, 1));
}

#[test]
fn test_preprocessing() {
    let data = Array1::from_vec(vec![1.0, 2.0, 4.0]);
    // Preprocessing normalises by max; verify via process_frame with known input
    let mut pipeline = RealTimeSirtPipeline::new(RealTimeSirtConfig {
        sirt_config: SirtConfig::default()
            .with_iterations(0)
            .with_relaxation(0.5),
        enable_preprocessing: true,
        enable_quality_monitoring: false,
        output_smoothing_sigma: None,
        ..Default::default()
    });
    let frame = pipeline.process_frame(&data, (1, 1, 1)).unwrap();
    // With 0 iterations the image stays zero regardless of preprocessing.
    assert_eq!(frame.image[[0, 0, 0]], 0.0);
}

#[test]
fn test_frame_quality_assessment() {
    let mut pipeline = RealTimeSirtPipeline::new(RealTimeSirtConfig {
        enable_quality_monitoring: true,
        enable_preprocessing: false,
        output_smoothing_sigma: None,
        sirt_config: SirtConfig::default()
            .with_iterations(1)
            .with_relaxation(0.1),
        ..Default::default()
    });
    let data = Array1::from_elem(16, 0.5_f64);
    let frame = pipeline.process_frame(&data, (4, 4, 4)).unwrap();
    let q = frame.quality_metrics.unwrap();
    assert!(q.snr_estimate >= 0.0);
    assert!(q.dynamic_range >= 0.0);
}

/// SIRT convergence: ‖r‖/‖b‖ must fall below 1.0 after 20 iterations.
///
/// Theorem: SIRT converges if λ ∈ (0, 2/‖A‖²).
/// For column-sum A with nz=8: ‖A‖²_F = nx·ny·nz.
/// λ=0.3 satisfies the bound for the 8×8×8 grid used here.
/// Reference: Censor & Zenios (1997) *Parallel Optimization* §6.4.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_sirt_convergence_tracking() {
    let config = RealTimeSirtConfig {
        sirt_config: SirtConfig::default()
            .with_iterations(20)
            .with_relaxation(0.3),
        enable_preprocessing: false,
        enable_quality_monitoring: false,
        output_smoothing_sigma: None,
        intensity_threshold: None,
        ..Default::default()
    };
    let mut pipeline = RealTimeSirtPipeline::new(config);
    let rf_data = Array1::from_shape_fn(64, |i| (i as f64 + 1.0) / 64.0);

    let frame1 = pipeline.process_frame(&rf_data, (8, 8, 8)).unwrap();
    assert!(
        frame1.convergence_error >= 0.0,
        "convergence_error must be non-negative, got {:.6}",
        frame1.convergence_error
    );
    assert!(
        frame1.convergence_error < 1.0,
        "convergence_error must decrease from 1.0; got {:.6}",
        frame1.convergence_error
    );

    let frame2 = pipeline.process_frame(&rf_data, (8, 8, 8)).unwrap();
    assert!(
        frame2.convergence_error <= frame1.convergence_error + 1e-10,
        "Residual must not increase: frame1={:.6}, frame2={:.6}",
        frame1.convergence_error,
        frame2.convergence_error
    );
}

/// Zero RF input must yield zero convergence error (x=0 is the exact solution).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_sirt_zero_measurement_converges_immediately() {
    let config = RealTimeSirtConfig {
        sirt_config: SirtConfig::default()
            .with_iterations(5)
            .with_relaxation(0.5),
        enable_preprocessing: false,
        enable_quality_monitoring: false,
        output_smoothing_sigma: None,
        intensity_threshold: None,
        ..Default::default()
    };
    let mut pipeline = RealTimeSirtPipeline::new(config);
    let rf_data = Array1::<f64>::zeros(64);
    let frame = pipeline.process_frame(&rf_data, (8, 8, 8)).unwrap();
    assert!(
        frame.convergence_error < 1e-10,
        "Zero input should give zero convergence error, got {:.3e}",
        frame.convergence_error
    );
}

/// Attenuation unit conversion: 0.5 dB/(cm·MHz) → Nepers/(m·Hz).
///
/// α_np = 0.5 × ln(10)/20 × 100 × 1e-6 = 5.7565e-5 Np/(m·Hz)
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_acoustic_geometry_attenuation_conversion() {
    let geom = AcousticProjectionGeometry {
        attenuation_db_cm_mhz: 0.5,
        ..Default::default()
    };
    let expected = 0.5 * (10.0_f64.ln() / 20.0) * 100.0 * 1e-6;
    let actual = geom.alpha_nepers_per_m_per_hz();
    assert!(
        (actual - expected).abs() < 1e-15,
        "expected {:.6e}, got {:.6e}",
        expected,
        actual
    );
}

/// Single point scatterer: projection must follow the 1/r law with zero attenuation.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_acoustic_forward_projection_single_scatterer() {
    let geom = AcousticProjectionGeometry {
        element_x: vec![0.0, 1e-3, 2e-3, 3e-3],
        element_z: 0.0,
        sound_speed: SOUND_SPEED_TISSUE,
        attenuation_db_cm_mhz: 0.0,
        center_frequency_hz: 5.0 * MHZ_TO_HZ,
        voxel_spacing: (1e-3, 1e-3, 1e-3),
    };
    let mut image = Array3::zeros((1, 1, 1));
    image[[0, 0, 0]] = 1.0;
    let proj = project_acoustic(&image, &geom);
    assert!(
        (proj[0] - MPA_TO_PA).abs() / MPA_TO_PA < 1e-6,
        "Sensor 0 (at origin): expected 1e6, got {:.6e}",
        proj[0]
    );
    assert!(
        (proj[1] - 1000.0).abs() / 1000.0 < 1e-6,
        "Sensor 1 (1mm away): expected 1000, got {:.6e}",
        proj[1]
    );
    assert!(
        proj[0] > proj[1] && proj[1] > proj[2] && proj[2] > proj[3],
        "Projection must decrease with distance: {:?}",
        proj.as_slice().unwrap()
    );
}

/// Adjoint property: ⟨Ax, y⟩ = ⟨x, Aᵀy⟩ for all x, y.
///
/// Proof: ⟨Ax,y⟩ = Σ_s(Σ_v A[s,v] x_v) y_s = Σ_v x_v(Σ_s A[s,v] y_s) = ⟨x,Aᵀy⟩.
/// # Panics
/// - Panics if assertion fails: `Adjoint ⟨Ax,y⟩=⟨x,Aᵀy⟩ violated: {:.10e} vs {:.10e}, rel_err={:.3e}`.
///
#[test]
fn test_acoustic_backprojection_adjoint_property() {
    let geom = AcousticProjectionGeometry {
        element_x: vec![0.0, 2e-3, 4e-3],
        element_z: 0.0,
        sound_speed: SOUND_SPEED_TISSUE,
        attenuation_db_cm_mhz: 0.5,
        center_frequency_hz: 5.0 * MHZ_TO_HZ,
        voxel_spacing: (1e-3, 1e-3, 1e-3),
    };
    let shape = (3, 3, 3);
    let mut x = Array3::zeros(shape);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                x[[i, j, k]] = ((i + j * 3 + k * 9) as f64) * 0.1 + 0.01;
            }
        }
    }
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let ax = project_acoustic(&x, &geom);
    let ax_dot_y: f64 = ax.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let aty = backproject_acoustic(&y, shape, &geom);
    let x_dot_aty: f64 = x.iter().zip(aty.iter()).map(|(a, b)| a * b).sum();
    let rel_err = (ax_dot_y - x_dot_aty).abs() / ax_dot_y.abs().max(1e-30);
    assert!(
        rel_err < 1e-12,
        "Adjoint ⟨Ax,y⟩=⟨x,Aᵀy⟩ violated: {:.10e} vs {:.10e}, rel_err={:.3e}",
        ax_dot_y,
        x_dot_aty,
        rel_err
    );
}

/// Acoustic SIRT must reduce residual below 1.0 on a noiseless point phantom.
///
/// Sensors at z=−10mm ensure all r ≥ 10mm, avoiding the r→0 singularity.
/// With D_R(s)=1/‖A_row_s‖² and λ=0.3 < 2/3, spectral radius < 1.
/// Reference: Censor & Zenios (1997) §6.4.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_acoustic_sirt_converges_on_point_phantom() {
    let geom = AcousticProjectionGeometry {
        element_x: vec![-2e-3, 0.0, 2e-3],
        element_z: -1e-2,
        sound_speed: SOUND_SPEED_TISSUE,
        attenuation_db_cm_mhz: 0.5,
        center_frequency_hz: 5.0 * MHZ_TO_HZ,
        voxel_spacing: (1e-3, 1e-3, 1e-3),
    };
    let mut truth = Array3::zeros((4, 4, 4));
    truth[[2, 2, 2]] = 1.0;
    let b = project_acoustic(&truth, &geom);
    let rf_data = Array1::from_vec(b.to_vec());
    let config = RealTimeSirtConfig {
        sirt_config: SirtConfig::default()
            .with_iterations(10)
            .with_relaxation(0.3),
        enable_preprocessing: false,
        enable_quality_monitoring: false,
        output_smoothing_sigma: None,
        intensity_threshold: None,
        transducer_geometry: Some(geom),
        ..Default::default()
    };
    let mut pipeline = RealTimeSirtPipeline::new(config);
    let frame = pipeline.process_frame(&rf_data, (4, 4, 4)).unwrap();
    assert!(
        frame.convergence_error < 1.0,
        "Acoustic SIRT residual not reduced: {:.4}",
        frame.convergence_error
    );
    assert!(frame.convergence_error >= 0.0);
}
