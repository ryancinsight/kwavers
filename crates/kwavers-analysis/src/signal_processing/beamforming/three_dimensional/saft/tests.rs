use leto::Array4;

use super::super::{Beamforming3dApodizationWindow, BeamformingAlgorithm3D};
use super::config::SaftConfig;
use super::processor::{distance3, SaftProcessor};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

fn make_processor(nx: usize, ny: usize, nz: usize, ntx: usize, nrx: usize) -> SaftProcessor {
    use super::super::config::BeamformingConfig3D;
    let cfg = BeamformingConfig3D {
        volume_dims: (nx, ny, nz),
        num_elements_3d: (ntx, nrx, 1),
        ..Default::default()
    };
    SaftProcessor::new(SaftConfig::default(), cfg)
}

#[test]
fn test_saft_config_default() {
    let config = SaftConfig::default();
    assert_eq!(config.virtual_sources, 100);
    assert!(matches!(
        config.apodization,
        Beamforming3dApodizationWindow::Hamming
    ));
    assert!(config.coherence_factor_enabled);
    assert_eq!(config.f_number, 1.5);
}

#[test]
fn test_saft_processor_creation() {
    let processor = make_processor(32, 32, 32, 8, 8);
    assert_eq!(processor.config.virtual_sources, 100);
}

#[test]
fn test_saft_from_algorithm() {
    use super::super::config::BeamformingConfig3D;
    let algorithm = BeamformingAlgorithm3D::SAFT3D {
        virtual_sources: 50,
    };
    let processor =
        SaftProcessor::from_algorithm(&algorithm, BeamformingConfig3D::default()).unwrap();
    assert_eq!(processor.config.virtual_sources, 50);
}

#[test]
fn test_time_of_flight_computation() {
    let processor = make_processor(32, 32, 32, 8, 8);
    let tof = processor.compute_time_of_flight(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.001, 0.0, 0.0],
        SOUND_SPEED_TISSUE,
    );
    let expected = (0.001 + 0.001) / SOUND_SPEED_TISSUE;
    assert!((tof - expected).abs() < 1e-12);
}

#[test]
fn test_distance_computation() {
    assert!((distance3([0.0, 0.0, 0.0], [0.001, 0.0, 0.0]) - 0.001).abs() < 1e-12);
}

#[test]
fn test_apodization_weight() {
    let processor = make_processor(32, 32, 32, 8, 8);
    // Hamming center: w(0) = 0.54 + 0.46 = 1.0
    let weight = processor.compute_apodization_weight(50, 50, 100);
    assert!(weight > 0.9 && weight <= 1.0 + 1e-10);
}

#[test]
fn test_coherence_factor() {
    use crate::signal_processing::beamforming::time_domain::coherence::amplitude_coherence_from_sums;
    // Mallart & Fink 1994: CF = |Σx|² / (N · Σx²).
    //
    // Derivation note: the previous assertion `compute_coherence_factor(10,5,10)==0.4`
    // encoded `coherent²/(N·incoherent²)`, where the call site passed
    // `incoherent = Σ|x|` — i.e. it squared the sum of magnitudes instead of
    // using the sum of energies Σ|x|². That formula caps a *perfectly coherent*
    // aperture at CF = 1/N (here 0.1), not 1, over-suppressing every voxel by N.
    // The canonical helper takes `sum_of_squares = Σx²` and is now SSOT.

    // Perfectly coherent aperture of N=10 unit elements: coherent = Σx = 10,
    // sum_of_squares = Σx² = 10 ⇒ CF = 100/(10·10) = 1.
    let coherent = amplitude_coherence_from_sums(10.0, 10.0, 10);
    assert!(
        (coherent - 1.0).abs() < 1e-12,
        "coherent aperture CF should be 1, got {coherent}"
    );

    // Partially coherent: coherent = 5, sum_of_squares = 10, N = 10
    // ⇒ CF = 25/(10·10) = 0.25; bounded in [0,1].
    let partial = amplitude_coherence_from_sums(5.0, 10.0, 10);
    assert!(
        (partial - 0.25).abs() < 1e-12,
        "expected 0.25, got {partial}"
    );
}

#[test]
fn test_demodulate_phase_correctness() {
    let f0 = 2.5 * MHZ_TO_HZ;
    let s = 1.0_f64;

    let (i0, q0) = SaftProcessor::demodulate(s, f0, 0.0);
    assert!((i0 - s).abs() < 1e-12, "I at τ=0: {i0}");
    assert!(q0.abs() < 1e-12, "Q at τ=0: {q0}");

    let tau_half = 0.5 / f0;
    let (ih, qh) = SaftProcessor::demodulate(s, f0, tau_half);
    assert!((ih + s).abs() < 1e-12, "I at τ=T/2: {ih}");
    assert!(qh.abs() < 1e-12, "Q at τ=T/2: {qh}");
}

#[test]
fn test_reconstruct_volume_basic() {
    use super::super::config::BeamformingConfig3D;
    let beamforming_config = BeamformingConfig3D {
        volume_dims: (16, 16, 16),
        num_elements_3d: (4, 4, 1),
        ..Default::default()
    };
    let processor = SaftProcessor::new(SaftConfig::default(), beamforming_config);

    let num_elements = 4 * 4;
    let mut rf_data = Array4::<f32>::zeros((1, num_elements, 512, 1));
    for elem in 0..num_elements {
        rf_data[[0, elem, 256, 0]] = 1.0;
    }

    let vol = processor.reconstruct_volume(&rf_data).unwrap();
    assert_eq!(vol.shape(), [16, 16, 16]);
    let max_val = vol.iter().cloned().fold(0.0_f32, f32::max);
    assert!(
        max_val > 0.0,
        "SAFT output must be non-zero for point target"
    );
}

#[test]
fn test_demodulate_envelope_invariant() {
    let f0 = 2.5 * MHZ_TO_HZ;
    let s = 0.7_f64;
    for tau_ns in [0u64, 25, 50, 75, 100] {
        let tau = tau_ns as f64 * 1e-9;
        let (i, q) = SaftProcessor::demodulate(s, f0, tau);
        let envelope = (i * i + q * q).sqrt();
        assert!(
            (envelope - s.abs()).abs() < 1e-12,
            "Envelope invariant failed at τ={tau_ns} ns: {envelope:.6} ≠ {s}"
        );
    }
}

#[test]
fn test_input_validation() {
    let processor = make_processor(32, 32, 32, 8, 8);
    let empty = Array4::<f32>::zeros((0, 0, 0, 0));
    assert!(processor.validate_input(&empty).is_err());
    let valid = Array4::<f32>::zeros((1, 64, 512, 1));
    processor.validate_input(&valid).unwrap();
}
