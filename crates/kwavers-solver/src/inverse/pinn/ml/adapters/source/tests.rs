//! Tests for [`PinnAcousticSource`] adapter.

use super::*;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_signal::waveform::SineWave;
use kwavers_source::PointSource;

#[test]
fn test_point_source_adapter() {
    let signal = Arc::new(SineWave::new(MHZ_TO_HZ, 1.0, 0.0));
    let position = (0.01, 0.02, 0.03);
    let domain_source = PointSource::new(position, signal);

    let pinn_source = PinnAcousticSource::from_domain_source(&domain_source, 0.0)
        .expect("Should adapt successfully");

    assert_eq!(pinn_source.position, position);
    assert_eq!(pinn_source.source_class, PinnSourceClass::Monopole);
    assert!((pinn_source.frequency - MHZ_TO_HZ).abs() < 1e-6);
    assert!((pinn_source.amplitude - 1.0).abs() < 1e-6);
}

#[test]
fn test_source_term_coefficient() {
    let pinn_source = PinnAcousticSource {
        position: (0.0, 0.0, 0.0),
        source_class: PinnSourceClass::Monopole,
        frequency: MHZ_TO_HZ, // 1 MHz
        amplitude: 100.0,
        phase: 0.0,
        focal_properties: None,
    };

    let coeff_t0 = pinn_source.source_term_coefficient(0.0);
    assert!((coeff_t0 - 100.0).abs() < 1e-6);

    let t_quarter = 0.25 / MHZ_TO_HZ; // quarter period at 1 MHz
    let coeff_quarter = pinn_source.source_term_coefficient(t_quarter);
    assert!(coeff_quarter.abs() < 1e-6);
}

#[test]
fn test_is_near_position() {
    let pinn_source = PinnAcousticSource {
        position: (0.0, 0.0, 0.0),
        source_class: PinnSourceClass::Monopole,
        frequency: MHZ_TO_HZ, // 1 MHz
        amplitude: 1.0,
        phase: 0.0,
        focal_properties: None,
    };

    assert!(pinn_source.is_near_position(0.0, 0.0, 0.0, 1e-3));
    assert!(pinn_source.is_near_position(0.0005, 0.0, 0.0, 1e-3));
    assert!(!pinn_source.is_near_position(0.002, 0.0, 0.0, 1e-3));
}

#[test]
fn test_adapt_multiple_sources() {
    let signal1 = Arc::new(SineWave::new(MHZ_TO_HZ, 1.0, 0.0));
    let signal2 = Arc::new(SineWave::new(2.0 * MHZ_TO_HZ, 2.0, 0.0));

    let source1: Arc<dyn Source> = Arc::new(PointSource::new((0.0, 0.0, 0.0), signal1));
    let source2: Arc<dyn Source> = Arc::new(PointSource::new((0.01, 0.0, 0.0), signal2));

    let sources = vec![source1, source2];
    let pinn_sources = adapt_sources(&sources, 0.0).expect("Should adapt all sources");

    assert_eq!(pinn_sources.len(), 2);
    assert!((pinn_sources[0].frequency - MHZ_TO_HZ).abs() < 1e-6);
    assert!((pinn_sources[1].frequency - 2.0 * MHZ_TO_HZ).abs() < 1e-6);
}

#[test]
fn test_focal_properties_extraction() {
    use kwavers_source::wavefront::gaussian::{GaussianConfig, GaussianSource};

    let signal = Arc::new(SineWave::new(MHZ_TO_HZ, 1.0, 0.0));
    let config = GaussianConfig {
        focal_point: (0.0, 0.0, 0.05),
        waist_radius: 1e-3,
        wavelength: 1.5e-3,
        direction: (0.0, 0.0, 1.0),
        ..Default::default()
    };
    let gaussian_source = GaussianSource::new(config, signal);

    let pinn_source = PinnAcousticSource::from_domain_source(&gaussian_source, 0.0)
        .expect("Should adapt Gaussian source");

    assert!(
        pinn_source.focal_properties.is_some(),
        "Gaussian source should have focal properties"
    );

    let focal_props = pinn_source.focal_properties.unwrap();

    assert!(
        (focal_props.focal_length - 0.05).abs() < 1e-3,
        "Focal length should be ~5cm, got {}",
        focal_props.focal_length
    );
    assert!(
        (focal_props.spot_size - 1e-3).abs() < 1e-6,
        "Spot size should be 1mm, got {}",
        focal_props.spot_size
    );
    assert!(
        focal_props.f_number.is_some(),
        "F-number should be available"
    );
    assert!(
        focal_props.focal_gain.is_some(),
        "Focal gain should be available"
    );
}

#[test]
fn test_unfocused_source_no_focal_properties() {
    let signal = Arc::new(SineWave::new(MHZ_TO_HZ, 1.0, 0.0));
    let point_source = PointSource::new((0.0, 0.0, 0.0), signal);

    let pinn_source = PinnAcousticSource::from_domain_source(&point_source, 0.0)
        .expect("Should adapt point source");

    assert!(
        pinn_source.focal_properties.is_none(),
        "Point source should not have focal properties"
    );
}
