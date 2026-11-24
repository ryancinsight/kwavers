use super::support::{near, trace};
use kwavers_driver::{
    run_experiment, DriverManifest, EnergyBudgetInputs, ResistorPackage, TileStimulationProfile,
};

#[test]
fn four_tile_manifest_to_selected_simulator() {
    let _trace = trace("96 lanes to acoustic report");
    let mut manifest = tracing::info_span!("arrange").in_scope(|| DriverManifest {
        hv_board: "qualification.kicad_pcb".into(),
        tx_connector: "J2".into(),
        tx_nets: (0..96).map(|i| format!("TX_{i}")).collect(),
        programming: "four 24-lane tiles".into(),
        aperture_m: 4.3e-3,
        frequency_hz: 500_000.0,
        sound_speed_m_s: 1540.0,
        focal_m: 0.010,
        timing_step_s: 20e-9,
        stimulation: None,
        tile_profiles: (0..4)
            .map(|i| {
                TileStimulationProfile::from_article_with(
                    1000.0 + 50.0 * f64::from(i),
                    0.25 * f64::from(i),
                    90.0 * f64::from(i),
                    25e-6 + 5e-6 * f64::from(i),
                )
            })
            .collect(),
    });
    let inputs = || EnergyBudgetInputs {
        c_load_f: 50e-12,
        r_on_ohm: 15.0,
        r_series_ohm: 56.0,
        ampacity_headroom_a: 20.0,
        damping_footprint: ResistorPackage::Smd2512,
    };
    let decoded = DriverManifest::from_text(&manifest.to_text()).unwrap();
    assert_eq!(decoded.tx_nets, manifest.tx_nets);
    assert_eq!(decoded.tile_profiles.len(), 4);
    assert_eq!(decoded.frequency_hz, 500_000.0);
    let budget = decoded.validate_v2_energy_budget(inputs()).unwrap();
    #[cfg(feature = "kwavers")]
    let backend = kwavers_driver::KwaversSim;
    #[cfg(not(feature = "kwavers"))]
    let backend = kwavers_driver::InCrateAcousticSim;
    let report = tracing::info_span!("act")
        .in_scope(|| run_experiment(&decoded, &budget, &backend, 40.0, 30.0).unwrap());
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(budget.lanes, 96);
        let validated = kwavers_driver::validate_against_budget(&decoded, &budget).unwrap();
        let expected = report.record.metrics.focal_pressure_pa.powi(2) / (2.0 * 1.48e6 * 1e4);
        tracing::info!(
            validation_isppa = validated.isppa_w_cm2,
            experiment_isppa = report.record.metrics.isppa_w_cm2,
            reference = expected
        );
        near(
            validated.isppa_w_cm2,
            validated.focal_pressure_pa.powi(2) / (2.0 * 1.48e6 * 1e4),
        );
        let metrics = &report.record.metrics;
        assert!(
            metrics.lateral_extent_mm > 0.0
                && metrics.axial_extent_mm > 0.0
                && metrics.focal_pressure_pa > 0.0
        );
        // Definitions applied to the actually returned pressure, independent of the field model.
        near(
            metrics.mechanical_index,
            metrics.focal_pressure_pa * 1e-6 / 0.5_f64.sqrt(),
        );
        near(
            metrics.isppa_w_cm2,
            metrics.focal_pressure_pa.powi(2) / (2.0 * 1.48e6 * 1e4),
        );
        assert_eq!(report.thermal.per_tile_rise_k.len(), 4);
        for (&rise, &watts) in report
            .thermal
            .per_tile_rise_k
            .iter()
            .zip(&budget.per_tile_device_total_w)
        {
            near(rise, watts * 40.0);
        }
        tracing::info!(
            pressure_pa = metrics.focal_pressure_pa,
            lateral_mm = metrics.lateral_extent_mm,
            axial_mm = metrics.axial_extent_mm,
            peak_rise_k = metrics.peak_thermal_rise_k
        );
        manifest.tx_nets.pop();
        assert!(manifest
            .validate_v2_energy_budget(inputs())
            .unwrap_err()
            .contains("TX lanes"));
    });
}

#[test]
fn legacy_manifest_survives_public_read_write_read() {
    let _trace = trace("existing v1 design file");
    let original=tracing::info_span!("arrange").in_scope(|| DriverManifest::from_text(
        "format=kicad-routing-driver-manifest-v1\nhv_board=tile.kicad_pcb\ntx_connector=J2\ntx_nets=TX_0,TX_1\nprogramming=JTAG\naperture_m=0.0043\nfrequency_hz=500000\nsound_speed_m_s=1540\nfocal_m=0.01\ntiming_step_s=2e-8\n"
    ).unwrap());
    let encoded = tracing::info_span!("act").in_scope(|| original.to_text());
    tracing::info_span!("assert")
        .in_scope(|| assert_eq!(DriverManifest::from_text(&encoded).unwrap(), original));
}
