use super::search::BeamformSearch;
use super::types::{LocalizationBeamformSearchConfig, LocalizationBeamformingMethod};
use crate::signal_processing::beamforming::time_domain::DelayReference;
use kwavers_domain::sensor::beamforming::processor::BeamformingProcessor;
use kwavers_domain::sensor::beamforming::BeamformingCoreConfig;

fn make_processor(n: usize) -> BeamformingProcessor {
    let cfg = BeamformingCoreConfig::default();
    let positions = (0..n).map(|i| [i as f64 * 0.01, 0.0, 0.0]).collect();
    BeamformingProcessor::new(cfg, positions)
}

#[test]
fn centered_cube_generates_points() {
    let processor = make_processor(4);

    let cfg = LocalizationBeamformSearchConfig {
        method: LocalizationBeamformingMethod::SrpDasTimeDomain {
            delay_reference: DelayReference::recommended_default(),
        },
        ..Default::default()
    };

    let search = BeamformSearch::new(processor, cfg).expect("construct search");

    let pts = search
        .generate_points([0.0, 0.0, 0.0])
        .expect("generate points");
    assert!(!pts.is_empty());
}
