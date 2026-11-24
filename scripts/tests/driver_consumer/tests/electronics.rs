use super::support::{near, trace};
use kwavers_driver::{
    available_pulsers, compare_pulsers, driver_efficiency, junction_temperature_k,
    pulser_dissipation, thermal_time_constant_s, transient_rise_k, PulserOp,
};

#[test]
fn pulser_candidates_for_a_96_channel_tile_stack() {
    let _trace = trace("96-channel selection");
    let catalog = tracing::info_span!("arrange").in_scope(available_pulsers);
    let options =
        tracing::info_span!("act").in_scope(|| compare_pulsers(96, 150.0, 2.0, 0.125, 50.0));
    tracing::info_span!("assert").in_scope(|| {
        let mut parts: Vec<_> = options.iter().map(|c| c.part_number).collect();
        parts.sort_unstable();
        assert_eq!(parts, ["HV7355K6-G", "MAX14815", "MD1715"]);
        for c in &options {
            let ic = catalog
                .iter()
                .find(|ic| ic.part_number == c.part_number)
                .unwrap();
            assert_eq!(c.ics_required * ic.channels, 96);
            assert_eq!(c.total_channels, 96);
        }
        assert!(options.windows(2).all(|w| w[0].score <= w[1].score));
        assert!(compare_pulsers(96, 201.0, 2.0, 0.125, 50.0).is_empty());
    });
}

#[test]
fn pulsed_capacitive_load_and_junction_transient() {
    let _trace = trace("burst power to temperature");
    let op = tracing::info_span!("arrange").in_scope(|| PulserOp {
        drive_hz: 2e6,
        duty: 0.125,
        c_load_f: 50e-12,
        v_pp: 80.0,
        r_on_ohm: 8.0,
        r_series_ohm: 56.0,
        q_g_c: 4e-9,
        v_gate: 5.0,
        q_rr_c: 1e-9,
    });
    for (duty, scale) in [(op.duty, 1.0), (op.duty / 2.0, 0.5), (0.0, 0.0)] {
        let heat = tracing::info_span!("act", duty)
            .in_scope(|| pulser_dissipation(&PulserOp { duty, ..op }));
        tracing::info_span!("assert").in_scope(|| {
            // 160 nJ per edge twice at 250 kcycles/s; R shares 1:7.
            // Gate and recovery add 20 nJ and 80 nJ per active cycle.
            near(heat.dynamic_total, 0.080 * scale);
            near(heat.dynamic_device, 0.010 * scale);
            near(heat.dynamic_series_r, 0.070 * scale);
            near(heat.gate, 0.005 * scale);
            near(heat.recovery, 0.020 * scale);
            near(heat.device_total, 0.035 * scale);
            near(
                driver_efficiency(0.105 * scale, heat.device_total + heat.dynamic_series_r),
                if duty == 0.0 { 0.0 } else { 0.5 },
            );
            // Eight identical active channels in one package share theta=40 K/W.
            let power = 8.0 * heat.device_total;
            let tj = junction_temperature_k(298.0, 2.0, 40.0, power);
            near(tj, 300.0 + 11.2 * scale);
            let tau = thermal_time_constant_s(40.0, 0.25);
            near(tau, 10.0);
            near(transient_rise_k(power * 40.0, tau, 0.0), 0.0);
            // At t=tau ln(2), the first-order response is exactly half the steady rise.
            near(
                transient_rise_k(power * 40.0, tau, 10.0 * std::f64::consts::LN_2),
                5.6 * scale,
            );
            tracing::info!(
                duty,
                device_w = heat.device_total,
                resistor_w = heat.dynamic_series_r,
                junction_k = tj
            );
        });
    }
}

#[test]
fn signal_delay_and_acoustic_wavelength() {
    let _trace = trace("clock distribution and element spacing");
    // In a unit-permittivity medium the microstrip effective epsilon is one.
    let (width, height, er) = tracing::info_span!("arrange").in_scope(|| (0.15, 0.20, 1.0));
    let delay = tracing::info_span!("act")
        .in_scope(|| kwavers_driver::microstrip_delay_s_per_m(width, height, er));
    tracing::info_span!("assert").in_scope(|| {
        near(delay, 1.0 / 299_800_000.0);
        assert!(kwavers_driver::within_skew(0.006, 0.0063, delay, 2e-12));
        assert!(!kwavers_driver::within_skew(0.006, 0.002, delay, 2e-12));
        near(kwavers_driver::wavelength_m(1540.0, 2e6), 0.00077);
    });
}
