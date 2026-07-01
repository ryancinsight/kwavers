use super::*;

#[test]
fn clearance_clips_by_largest_ellipsoid_axis() {
    let r = clipped_lateral_radius_for_clearance(2.0e-3, 3.0e-3, 5.0);
    assert!((r - 0.6e-3).abs() < 1e-12);
    let unconstrained = clipped_lateral_radius_for_clearance(2.0e-3, 30.0e-3, 5.0);
    assert!((unconstrained - 2.0e-3).abs() < 1e-12);
}

#[test]
fn backscatter_drops_monotonically_with_fractionation() {
    // Intact tissue keeps full backscatter; fully fractionated → liquefied
    // floor; the lesion is hypoechoic relative to its surround.
    let f = [0.0, 0.25, 0.5, 0.75, 1.0];
    let si = 1.0;
    let sl = 0.05;
    let s = fractionation_backscatter_coefficient(&f, si, sl, 2.0);
    assert!((s[0] - si).abs() < 1e-12, "intact backscatter preserved");
    assert!((s[4] - sl).abs() < 1e-12, "liquefied floor reached");
    assert!(
        s.windows(2).all(|w| w[1] <= w[0] + 1e-15),
        "monotone non-increasing"
    );
    // γ = 2 quadratic: at f = 0.5, σ = sl + (si-sl)*0.25.
    assert!((s[2] - (sl + (si - sl) * 0.25)).abs() < 1e-12);
    // Hypoechoic contrast: completed lesion backscatter ≪ intact.
    assert!(s[4] < 0.1 * s[0]);
}

#[test]
fn impedance_mixes_linearly_for_rim_echo() {
    let f = [0.0, 0.5, 1.0];
    let zi = 1.65e6; // intact liver ≈ 1.65 MRayl
    let zl = 1.50e6; // liquefied homogenate ≈ water-like
    let z = fractionation_acoustic_impedance(&f, zi, zl);
    assert!((z[0] - zi).abs() < 1e-6);
    assert!((z[1] - 0.5 * (zi + zl)).abs() < 1e-6);
    assert!((z[2] - zl).abs() < 1e-6);
    // A gradient exists across a partial→full boundary (drives the rim echo).
    assert!((z[0] - z[2]).abs() > 1e5);
}

#[test]
fn ellipsoid_rejects_false_allowed_voxel() {
    let nx = 7;
    let ny = 7;
    let nz = 7;
    let mut mask = vec![true; nx * ny * nz];
    let idx = (3 * ny + 3) * nz + 4;
    mask[idx] = false;
    assert!(!ellipsoid_respects_allowed_mask(
        &mask, nx, ny, nz, 3, 3, 3, 1.1, 1.1, 1.0,
    ));
}

#[test]
fn measured_spectrum_scaling_preserves_value_semantics() {
    let out = scale_measured_emission_spectrum(&[1.0, -2.0, 3.0], 0.5, 2.0);
    assert_eq!(out, vec![1.0, 0.0, 3.0]);
}

#[test]
fn delivered_progress_is_monotone() {
    let k = delivered_histotripsy_progress(&[0.0, 1.0, 2.0], 1.0, 2.0);
    assert_eq!(k[0], 0.0);
    assert!(k[1] > k[0]);
    assert!(k[2] > k[1]);
}

#[test]
fn per_spot_cavitation_dose_grid_preserves_row_major_contract() {
    let lateral = [0.0, 2.0e-3];
    let axial = [0.0, 1.0e-3];
    let pressures = [0.0, 1.0e6, 2.0e6];
    let power = [0.0, 2.0, 8.0];
    let grid = per_spot_cavitation_dose_grid(PerSpotCavitationDoseInput {
        lateral_offsets_m: &lateral,
        axial_offsets_m: &axial,
        p_target_pa: 1.0e6,
        f0_hz: 1.0e6,
        c_m_s: 1540.0,
        pressures_pa: &pressures,
        cavitation_power: &power,
        n_pulses_per_spot: 3,
        goal_pressure_pa: 1.5e6,
        attenuation_np_m: 8.0,
        apodized: true,
    })
    .expect("finite monotone pressure sweep is valid");

    assert_eq!(grid.axial_count, 2);
    assert_eq!(grid.lateral_count, 2);
    assert_eq!(grid.dose.len(), 4);
    assert_eq!(grid.efficiency.len(), 4);
    assert_eq!(grid.p_spot_pa.len(), 4);
    assert!((grid.efficiency[0] - 1.0).abs() < 1.0e-12);
    assert!((grid.p_spot_pa[0] - 1.0e6).abs() < 1.0e-6);
    assert!((grid.dose[0] - 6.0).abs() < 1.0e-12);
    assert!((grid.goal_dose - 15.0).abs() < 1.0e-12);

    let expected_p = 1.0e6
        * crate::analytical::transducer::electronic_steering_efficiency(
            2.0e-3, 1.0e-3, 1.0e6, 1540.0, true,
        )
        * (-8.0_f64 * 1.0e-3).exp();
    let expected_power = 2.0 * expected_p / 1.0e6;
    let idx = 3;
    assert!((grid.p_spot_pa[idx] - expected_p).abs() < 1.0e-6);
    assert!((grid.dose[idx] - 3.0 * expected_power).abs() < 1.0e-12);
}

#[test]
fn per_spot_cavitation_dose_grid_clamps_pressure_sweep_endpoints() {
    let grid = per_spot_cavitation_dose_grid(PerSpotCavitationDoseInput {
        lateral_offsets_m: &[0.0],
        axial_offsets_m: &[0.0],
        p_target_pa: 3.0e6,
        f0_hz: 1.0e6,
        c_m_s: 1540.0,
        pressures_pa: &[1.0e6, 2.0e6],
        cavitation_power: &[4.0, 7.0],
        n_pulses_per_spot: 2,
        goal_pressure_pa: 0.5e6,
        attenuation_np_m: 0.0,
        apodized: true,
    })
    .expect("endpoint-clamped sweep is valid");

    assert_eq!(grid.dose, vec![14.0]);
    assert_eq!(grid.goal_dose, 8.0);
}

#[test]
fn per_spot_cavitation_dose_grid_rejects_invalid_pressure_axis() {
    assert_eq!(
        per_spot_cavitation_dose_grid(PerSpotCavitationDoseInput {
            lateral_offsets_m: &[0.0],
            axial_offsets_m: &[0.0],
            p_target_pa: 1.0e6,
            f0_hz: 1.0e6,
            c_m_s: 1540.0,
            pressures_pa: &[0.0, 0.0],
            cavitation_power: &[0.0, 1.0],
            n_pulses_per_spot: 1,
            goal_pressure_pa: 1.0e6,
            attenuation_np_m: 0.0,
            apodized: true,
        }),
        None
    );
}

#[test]
fn cavitation_monitor_trace_integrates_constant_curve_without_jitter() {
    let trace = cavitation_monitor_trace(CavitationMonitorTraceInput {
        pressures_pa: &[0.0, 100.0, 200.0],
        cavitation_power: &[0.0, 10.0, 40.0],
        n_pulses: 3,
        prf_hz: 1.0,
        p_start_pa: 100.0,
        target_signal: 0.0,
        inertial_cap: 1000.0,
        gain: 0.0,
        jitter_sigma: 0.0,
        goal_fraction: 0.5,
        seed: 7,
    })
    .expect("finite monotone pressure sweep is valid");

    assert_eq!(trace.time_s, vec![0.0, 1.0, 2.0]);
    assert_eq!(trace.cavitation_signal, vec![10.0, 10.0, 10.0]);
    assert_eq!(trace.power_percent, vec![25.0, 25.0, 25.0]);
    assert_eq!(trace.cumulative_dose, vec![0.0, 10.0, 20.0]);
    assert_eq!(trace.goal_dose, 10.0);
}

#[test]
fn cavitation_monitor_trace_recruits_pressure_when_under_target() {
    let trace = cavitation_monitor_trace(CavitationMonitorTraceInput {
        pressures_pa: &[0.0, 100.0, 200.0],
        cavitation_power: &[0.0, 10.0, 40.0],
        n_pulses: 2,
        prf_hz: 2.0,
        p_start_pa: 100.0,
        target_signal: 15.0,
        inertial_cap: 1000.0,
        gain: 0.5,
        jitter_sigma: 0.0,
        goal_fraction: 1.0,
        seed: 7,
    })
    .expect("finite monotone pressure sweep is valid");

    assert_eq!(trace.cavitation_signal, vec![10.0, 25.0]);
    assert_eq!(trace.power_percent, vec![25.0, 56.25]);
    assert_eq!(trace.cumulative_dose, vec![0.0, 8.75]);
    assert_eq!(trace.goal_dose, 8.75);
}

#[test]
fn cavitation_monitor_trace_rejects_invalid_inputs() {
    assert_eq!(
        cavitation_monitor_trace(CavitationMonitorTraceInput {
            pressures_pa: &[0.0, 0.0],
            cavitation_power: &[0.0, 1.0],
            n_pulses: 1,
            prf_hz: 1.0,
            p_start_pa: 1.0,
            target_signal: 0.0,
            inertial_cap: 1.0,
            gain: 0.0,
            jitter_sigma: 0.0,
            goal_fraction: 1.0,
            seed: 0,
        }),
        None
    );
}

#[test]
fn simulated_population_monitor_trace_integrates_population_emissions() {
    let trace = simulated_population_monitor_trace(SimulatedPopulationMonitorInput {
        f0_hz: 1.0e6,
        medium: PopulationMedium {
            p0_pa: 101_325.0,
            rho: 998.0,
            c_liquid: 1481.0,
            mu: 1.0e-3,
            sigma: 0.0725,
            pv: 2330.0,
            gamma: 1.4,
        },
        n_bubbles: 2,
        n_pulses: 2,
        prf_hz: 1.0,
        p_start_pa: 80.0e3,
        p_min_pa: 70.0e3,
        p_max_pa: 90.0e3,
        target_signal: 0.0,
        inertial_cap: f64::MAX,
        gain: 0.0,
        goal_fraction: 0.5,
        seed: 11,
        r0_median_m: 2.0e-6,
        r0_sigma_ln: 0.05,
        n_cycles: 2.0,
        n_out: 96,
        r_obs_m: 5.0e-2,
        rel_halfwidth: 0.12,
        noise_floor: 0.0,
        thermal_effects: false,
        shell: PopulationShell {
            coated: true,
            chi: 0.5,
            shell_viscosity: 0.5,
            shell_thickness: 3.0e-9,
            sigma_initial: 0.04,
            steps_per_cycle: 160,
        },
    })
    .expect("finite simulated population monitor input is valid");

    assert_eq!(trace.time_s, vec![0.0, 1.0]);
    assert_eq!(trace.cavitation_signal.len(), 2);
    assert_eq!(trace.stable_signal.len(), 2);
    assert_eq!(trace.broadband_signal.len(), 2);
    assert!(trace
        .cavitation_signal
        .iter()
        .chain(trace.stable_signal.iter())
        .chain(trace.broadband_signal.iter())
        .all(|value| value.is_finite() && *value >= 0.0));
    for ((&total, &stable), &broadband) in trace
        .cavitation_signal
        .iter()
        .zip(trace.stable_signal.iter())
        .zip(trace.broadband_signal.iter())
    {
        assert!((total - (stable + broadband)).abs() <= f64::EPSILON * total.max(1.0));
    }
    let expected_power = (80.0_f64 / 90.0).powi(2) * 100.0;
    for &power in &trace.power_percent {
        assert!((power - expected_power).abs() < 1.0e-12);
    }
    let expected_dose = 0.5 * (trace.cavitation_signal[0] + trace.cavitation_signal[1]);
    assert_eq!(trace.cumulative_dose[0], 0.0);
    assert!((trace.cumulative_dose[1] - expected_dose).abs() <= 1.0e-12 * expected_dose.max(1.0));
    assert!((trace.goal_dose - 0.5 * expected_dose).abs() <= 1.0e-12 * expected_dose.max(1.0));
}

#[test]
fn simulated_population_monitor_trace_rejects_invalid_pressure_bounds() {
    assert_eq!(
        simulated_population_monitor_trace(SimulatedPopulationMonitorInput {
            f0_hz: 1.0e6,
            medium: PopulationMedium {
                p0_pa: 101_325.0,
                rho: 998.0,
                c_liquid: 1481.0,
                mu: 1.0e-3,
                sigma: 0.0725,
                pv: 2330.0,
                gamma: 1.4,
            },
            n_bubbles: 1,
            n_pulses: 1,
            prf_hz: 1.0,
            p_start_pa: 80.0e3,
            p_min_pa: 90.0e3,
            p_max_pa: 70.0e3,
            target_signal: 0.0,
            inertial_cap: 1.0,
            gain: 0.0,
            goal_fraction: 1.0,
            seed: 0,
            r0_median_m: 2.0e-6,
            r0_sigma_ln: 0.05,
            n_cycles: 2.0,
            n_out: 96,
            r_obs_m: 5.0e-2,
            rel_halfwidth: 0.12,
            noise_floor: 0.0,
            thermal_effects: false,
            shell: PopulationShell {
                coated: false,
                chi: 0.5,
                shell_viscosity: 0.5,
                shell_thickness: 3.0e-9,
                sigma_initial: 0.04,
                steps_per_cycle: 160,
            },
        }),
        None
    );
}

#[test]
fn closed_loop_cavitation_sonication_recruits_until_target() {
    let trace = closed_loop_cavitation_sonication(ClosedLoopCavitationSonicationInput {
        pressures_pa: &[0.0, 100.0, 200.0],
        stable_power: &[0.0, 10.0, 40.0],
        inertial_power: &[0.0, 1.0, 4.0],
        n_bursts: 3,
        burst_duration_s: 0.5,
        p_start_pa: 100.0,
        stable_target: 20.0,
        inertial_limit: 100.0,
        gain: 0.5,
    })
    .expect("finite monotone pressure sweep is valid");

    assert_eq!(trace.pressure_pa, vec![100.0, 150.0, 150.0]);
    assert_eq!(trace.stable_emission, vec![10.0, 25.0, 25.0]);
    assert_eq!(trace.inertial_emission, vec![1.0, 2.5, 2.5]);
    assert_eq!(trace.stable_dose, vec![0.0, 8.75, 21.25]);
    assert_eq!(trace.inertial_dose, vec![0.0, 0.875, 2.125]);
}

#[test]
fn closed_loop_cavitation_sonication_backs_off_on_inertial() {
    let trace = closed_loop_cavitation_sonication(ClosedLoopCavitationSonicationInput {
        pressures_pa: &[0.0, 100.0, 200.0],
        stable_power: &[0.0, 10.0, 40.0],
        inertial_power: &[0.0, 10.0, 40.0],
        n_bursts: 2,
        burst_duration_s: 1.0,
        p_start_pa: 200.0,
        stable_target: 0.0,
        inertial_limit: 20.0,
        gain: 0.5,
    })
    .expect("finite monotone pressure sweep is valid");

    assert_eq!(trace.pressure_pa, vec![200.0, 100.0]);
    assert_eq!(trace.stable_emission, vec![40.0, 10.0]);
    assert_eq!(trace.inertial_emission, vec![40.0, 10.0]);
    assert_eq!(trace.stable_dose, vec![0.0, 25.0]);
    assert_eq!(trace.inertial_dose, vec![0.0, 25.0]);
}

#[test]
fn closed_loop_cavitation_sonication_rejects_invalid_inputs() {
    assert_eq!(
        closed_loop_cavitation_sonication(ClosedLoopCavitationSonicationInput {
            pressures_pa: &[0.0, 0.0],
            stable_power: &[0.0, 1.0],
            inertial_power: &[0.0, 1.0],
            n_bursts: 1,
            burst_duration_s: 1.0,
            p_start_pa: 1.0,
            stable_target: 0.0,
            inertial_limit: 1.0,
            gain: 0.0,
        }),
        None
    );
}

#[test]
fn raster_cavitation_pulsing_matches_sequential_schedule_contract() {
    let trace = raster_cavitation_pulsing(RasterPulsingInput {
        spot_lateral_m: &[0.0, 0.0],
        spot_axial_m: &[0.0, 0.0],
        p_target_pa: 100.0,
        f0_hz: 1.0e6,
        c_m_s: 1540.0,
        cav_pressures_pa: &[0.0, 100.0],
        cav_dose_per_pulse: &[0.0, 10.0],
        pulses_per_spot: 2,
        prf_hz: 10.0,
        schedule: RasterPulsingSchedule::Sequential,
        interleave_group: 0,
        attenuation_np_m: 0.0,
        apodized: true,
        tau_dissolution_s: 0.0,
        shielding_g: 1.2,
        tau_thermal_s: 1.0,
        thermal_gain_k_per_pulse: 2.0,
        goal_dose: 15.0,
        n_time_samples: 4,
    })
    .expect("finite raster pulsing input is valid");

    for (actual, expected) in trace.time_s.iter().zip([0.0, 0.1, 0.2, 0.3]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    for (actual, expected) in trace.coverage.iter().zip([0.0, 0.5, 0.5, 1.0]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    for (actual, expected) in trace.cumulative_dose.iter().zip([10.0, 20.0, 30.0, 40.0]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    assert_eq!(trace.per_spot_dose, vec![20.0, 20.0]);
    let expected_peak_temp = 2.0 + 2.0 * (-0.1_f64).exp();
    assert!((trace.per_spot_peak_temp_k[0] - expected_peak_temp).abs() < 1.0e-12);
    assert!((trace.per_spot_peak_temp_k[1] - expected_peak_temp).abs() < 1.0e-12);
    assert_eq!(trace.efficacy, 1.0);
    assert_eq!(trace.dt_spot_s, 0.1);
    assert!((trace.treatment_s - 0.3).abs() < 1.0e-12);
    assert_eq!(trace.p_spot_pa, vec![100.0, 100.0]);
}

#[test]
fn raster_cavitation_pulsing_uses_interleaved_interval_for_shielding() {
    let trace = raster_cavitation_pulsing(RasterPulsingInput {
        spot_lateral_m: &[0.0, 0.0],
        spot_axial_m: &[0.0, 0.0],
        p_target_pa: 100.0,
        f0_hz: 1.0e6,
        c_m_s: 1540.0,
        cav_pressures_pa: &[0.0, 100.0],
        cav_dose_per_pulse: &[0.0, 10.0],
        pulses_per_spot: 2,
        prf_hz: 10.0,
        schedule: RasterPulsingSchedule::Interleaved,
        interleave_group: 2,
        attenuation_np_m: 0.0,
        apodized: true,
        tau_dissolution_s: 0.4,
        shielding_g: 1.0,
        tau_thermal_s: 1.0,
        thermal_gain_k_per_pulse: 0.0,
        goal_dose: 0.0,
        n_time_samples: 2,
    })
    .expect("finite raster pulsing input is valid");

    let efficacy = (-((1.0_f64 / 0.2) * 0.4 - 1.0)).exp();
    assert!((trace.efficacy - efficacy).abs() < 1.0e-12);
    assert_eq!(trace.dt_spot_s, 0.2);
    assert_eq!(trace.per_spot_dose, vec![10.0 + 10.0 * efficacy; 2]);
    assert_eq!(trace.coverage, vec![0.0, 0.0]);
}

#[test]
fn raster_cavitation_pulsing_rejects_invalid_axes() {
    assert_eq!(
        raster_cavitation_pulsing(RasterPulsingInput {
            spot_lateral_m: &[0.0],
            spot_axial_m: &[],
            p_target_pa: 100.0,
            f0_hz: 1.0e6,
            c_m_s: 1540.0,
            cav_pressures_pa: &[0.0, 100.0],
            cav_dose_per_pulse: &[0.0, 10.0],
            pulses_per_spot: 1,
            prf_hz: 10.0,
            schedule: RasterPulsingSchedule::Sequential,
            interleave_group: 0,
            attenuation_np_m: 0.0,
            apodized: true,
            tau_dissolution_s: 0.0,
            shielding_g: 0.0,
            tau_thermal_s: 1.0,
            thermal_gain_k_per_pulse: 0.0,
            goal_dose: 0.0,
            n_time_samples: 1,
        }),
        None
    );
}

#[test]
fn cloud_erosion_validation_fits_empirical_scale() {
    let model = [0.0, 1.0, 2.0, 3.0];
    let reference = [0.0, 2.0, 4.0, 6.0];
    let metrics = cloud_erosion_validation_metrics(&reference, &model)
        .expect("nonzero paired erosion curves are valid");
    assert!((metrics.model_scale - 2.0).abs() < 1.0e-12);
    assert!(metrics.rmse < 1.0e-12);
    assert!(metrics.normalized_rmse < 1.0e-12);
    assert!(metrics.max_abs_error < 1.0e-12);
    assert!((metrics.pearson_r - 1.0).abs() < 1.0e-12);
    assert_eq!(metrics.sample_count, 4);
}

#[test]
fn cloud_erosion_validation_rejects_nonfinite_samples() {
    assert_eq!(
        cloud_erosion_validation_metrics(&[0.0, f64::NAN], &[0.0, 1.0]),
        None
    );
    assert_eq!(
        cloud_erosion_validation_metrics(&[0.0, 1.0], &[0.0, 0.0]),
        None
    );
}

#[test]
fn boiling_lesion_from_profile_returns_physical_axes() {
    let r = [0.0, 0.5e-3, 1.0e-3, 1.5e-3];
    let b = [1.0, 0.9, 0.6, 0.2];
    let out = boiling_lesion_from_pressure_profile(
        &r, &b, 80.0e6, 80.0e-3, 1.0e6, 1540.0, 1060.0, 4.0, 5.0, 3600.0, 63.0, 20.0e-3, 4.0,
        10.0e-3, 0.95,
    )
    .expect("profile should boil");
    assert!(out.pulses >= 1);
    assert!(out.lateral_radius_m > 0.0);
    assert!((out.axial_radius_m / out.lateral_radius_m - 4.0).abs() < 1.0e-12);
    assert!(out.pulse_ms > 0.0 && out.pulse_ms <= 20.0);
}

#[test]
fn boiling_time_profile_increases_as_pressure_falls() {
    let b = [1.0, 0.75, 0.5];
    let t = boiling_time_profile_from_pressure(
        &b, 80.0e6, 80.0e-3, 1.0e6, 1540.0, 1060.0, 4.0, 5.0, 3600.0, 63.0,
    );
    assert_eq!(t.len(), b.len());
    assert!(t[0] < t[1]);
    assert!(t[1] < t[2]);
}

#[test]
fn receiver_channel_psd_decreases_with_range() {
    let psd = [2.0, 3.0];
    let recv = [0.01, 0.0, 0.0, 0.02, 0.0, 0.0];
    let ch = receiver_channel_psd_from_source(&psd, [0.0, 0.0, 0.0], &recv, 0.0);
    assert_eq!(ch.len(), 4);
    assert!(ch[0] > ch[2]);
    assert!(ch[1] > ch[3]);
    let sum = integrate_channel_psd(&ch, 2, 2);
    assert_eq!(sum.len(), 2);
    assert!((sum[0] - (ch[0] + ch[2])).abs() < 1.0e-12);
}
