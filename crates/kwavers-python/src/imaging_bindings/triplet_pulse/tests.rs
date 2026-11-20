use super::{build_cloud, TripletPulseConfig};
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;

#[test]
fn custom_frequency_keeps_the_named_burst_contract() {
    let _subscriber = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let (frequency, preset) =
        info_span!("arrange").in_scope(|| (2.0e6_f64, TripletPulseConfig::kuji_2024()));

    let cloud = info_span!("act").in_scope(|| build_cloud(frequency, 0.26, false));

    info_span!("assert").in_scope(|| {
        assert_eq!(cloud.config.frequency_hz.to_bits(), frequency.to_bits());
        assert_eq!(cloud.drive.frequency_hz.to_bits(), frequency.to_bits());
        assert_eq!(cloud.config.burst_cycles, preset.burst_cycles);
        assert_eq!(
            cloud.config.phases.phase_180_deg.to_bits(),
            180.0_f64.to_bits()
        );
    });
}
