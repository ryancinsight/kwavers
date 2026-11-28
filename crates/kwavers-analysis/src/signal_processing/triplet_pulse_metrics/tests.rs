use super::*;

#[test]
fn hp_response_matches_kuji_shape() {
    let fc = 2.5e6;
    assert_eq!(
        kuji_hp_response(fc, fc, KujiFilterShape::KujiRaisedCosine),
        1.0
    );
    assert_eq!(
        kuji_hp_response(0.85 * fc, fc, KujiFilterShape::KujiRaisedCosine),
        1e-3
    );
    assert_eq!(
        kuji_hp_response(0.5 * fc, fc, KujiFilterShape::KujiRaisedCosine),
        1e-3
    );
    // Transition midpoint u = 0.5: 1e-3 + (1 - 1e-3) * 0.5.
    let mid = kuji_hp_response(0.925 * fc, fc, KujiFilterShape::KujiRaisedCosine);
    assert!((mid - 0.5005).abs() < 1e-12, "mid {mid}");
    assert_eq!(
        kuji_hp_response(0.9 * fc, fc, KujiFilterShape::BrickWall),
        0.0
    );
    assert_eq!(KUJI_FC_GRID_HZ.len(), 15);
    assert_eq!(KUJI_FC_GRID_HZ[0], 2.3e6);
    assert_eq!(KUJI_FC_GRID_HZ[14], 3.7e6);
}

#[test]
fn filter_trace_suppresses_below_stopband_and_passes_passband() {
    // Bin-centred tones: n = 10000 at dt = 20 ns gives df = 5 kHz,
    // so 2 MHz (bin 400) and 3 MHz (bin 600) are exactly periodic
    // in the window — no leakage to confuse the response check.
    let dt = 20e-9;
    let n = 10000;
    let low: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * 2.0e6 * i as f64 * dt).sin())
        .collect();
    let out = kuji_filter_trace(&low, dt, 2.5e6, KujiFilterShape::KujiRaisedCosine, false).unwrap();
    let amp = out.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    assert!(amp < 2e-3, "2 MHz amplitude after HP: {amp}");
    let high: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * 3.0e6 * i as f64 * dt).sin())
        .collect();
    let out =
        kuji_filter_trace(&high, dt, 2.5e6, KujiFilterShape::KujiRaisedCosine, false).unwrap();
    let amp = out.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    assert!((amp - 1.0).abs() < 0.02, "3 MHz amplitude after HP: {amp}");
}

fn impulse_rf(geom: &TripletPulseGeometry, sx: f64, sy: f64, scale: f64) -> Array2<f64> {
    let n_samples = 1800;
    let mut rf = Array2::zeros([geom.n_elem(), n_samples]);
    for (e, &x_el) in geom.element_x_m.iter().enumerate() {
        let d_tx = geom.transmit_delay_s(sx, sy);
        let dx = sx - x_el;
        let dy = sy - geom.element_depth_m;
        let d_rx = (dx * dx + dy * dy).sqrt() / geom.sound_speed_m_s;
        let idx = ((d_tx + d_rx) / geom.dt_s).round() as usize;
        rf[[e, idx]] = scale;
    }
    rf
}

#[test]
fn coherent_das_envelope_localizes_impulse() {
    let g = TripletPulseGeometry::stage1().unwrap();
    let rf = impulse_rf(&g, 19.0e-3, 20.0e-3, 1.0);
    let xs: Vec<f64> = (0..57).map(|i| 5.0e-3 + i as f64 * 0.5e-3).collect();
    let ys: Vec<f64> = (0..49).map(|i| 12.0e-3 + i as f64 * 0.5e-3).collect();
    let img = coherent_das_envelope(rf.view(), &g, &xs, &ys, 8).unwrap();
    // Source pixel: x = 19.0 mm is index 28, y = 20 mm is index 16.
    assert!((img[[16, 28]] - 1.0).abs() < 0.05, "peak {}", img[[16, 28]]);
}

#[test]
fn known_ratio_images_give_known_contrast() {
    // img1 = 4 * img3 before, img1 = 2 * img3 after, uniformly:
    // reductions 4 and 2, contrast = 20*log10(2) = 6.0206 dB.
    let xs = vec![0.0, 1.0e-3];
    let ys = vec![0.0, 1.0e-3];
    let roi = RectRoi {
        x_lo_m: -1.0e-3,
        x_hi_m: 2.0e-3,
        y_lo_m: -1.0e-3,
        y_hi_m: 2.0e-3,
    };
    let b3 = Array2::from_elem([2, 2], 0.5);
    let b1 = Array2::from_elem([2, 2], 2.0);
    let a3 = Array2::from_elem([2, 2], 0.25);
    let a1 = Array2::from_elem([2, 2], 0.5);
    let r = kuji_contrast_from_images(b1.view(), b3.view(), a1.view(), a3.view(), &xs, &ys, &roi)
        .unwrap();
    assert_eq!(r.red_tissue, 4.0);
    assert_eq!(r.red_cavitation, 2.0);
    assert!((r.contrast_db - 20.0 * 2.0_f64.log10()).abs() < 1e-12);
}

#[test]
fn ensemble_averages_reductions_before_contrast() {
    let results = [KujiReductions::new(4.0, 2.0), KujiReductions::new(2.0, 2.0)];
    let ens = kuji_ensemble(&results).unwrap();
    assert_eq!(ens.mean_red_tissue, 3.0);
    assert_eq!(ens.mean_red_cavitation, 2.0);
    assert!((ens.contrast_db - 20.0 * 1.5_f64.log10()).abs() < 1e-12);
    // Per-realisation contrasts 6.0206 and 0 dB: mean 3.0103,
    // sample SD 4.2568.
    assert!((ens.per_realization_mean_db - 10.0 * 2.0_f64.log10()).abs() < 1e-9);
    assert!((ens.per_realization_sd_db - 4.2568).abs() < 1e-3);
    let lower = kuji_ensemble(&[KujiReductions::new(2.0, 2.0)]).unwrap();
    let (fc, c) = optimal_cutoff(&[(2.5e6, ens), (3.0e6, lower)]).unwrap();
    assert_eq!(fc, 2.5e6);
    assert!(c > 0.0);
}

#[test]
fn iwasaki_contrast_known_values() {
    let xs = vec![0.0, 1.0e-3, 2.0e-3];
    let ys = vec![0.0, 1.0e-3, 2.0e-3];
    let bg = RectRoi {
        x_lo_m: -1.0e-3,
        x_hi_m: 3.0e-3,
        y_lo_m: -1.0e-3,
        y_hi_m: 3.0e-3,
    };
    let mut frame = Array2::from_elem([3, 3], 1.0);
    frame[[1, 1]] = 10.0;
    // Background mean includes the peak pixel: (8*1 + 10)/9 = 2.
    let c = iwasaki_contrast(
        std::slice::from_ref(&frame),
        &xs,
        &ys,
        &bg,
        IwasakiBackgroundAveraging::Linear,
    )
    .unwrap();
    assert!((c - 20.0 * 5.0_f64.log10()).abs() < 1e-12, "contrast {c}");
    // Spatiotemporal max: a later frame's larger peak wins.
    let mut frame2 = Array2::from_elem([3, 3], 1.0);
    frame2[[0, 0]] = 20.0;
    let c2 = iwasaki_contrast(
        &[frame.clone(), frame2],
        &xs,
        &ys,
        &bg,
        IwasakiBackgroundAveraging::Linear,
    )
    .unwrap();
    assert!(
        c2 > c,
        "spatiotemporal max must raise contrast: {c2} vs {c}"
    );
}

#[test]
fn noise_sigma_is_parameterised_by_inband_snr_not_trace_max() {
    // WS-C arithmetic: σ = echo_rms · 10^(−SNR/20).
    assert_eq!(inband_noise_sigma(2.0, 0.0), 2.0);
    let s33 = inband_noise_sigma(1.0, KUJI_EXPERIMENT_INBAND_SNR_DB);
    assert!((s33 - 10f64.powf(-33.0 / 20.0)).abs() < 1e-15, "σ33 {s33}");
    // B1's accidental DAS-level floor was ≈ −23 dB: the noise
    // RMS is then ≈ 14× the echo it was meant to perturb.
    let s_neg23 = inband_noise_sigma(1.0, -23.0);
    assert!((s_neg23 - 14.13).abs() < 0.01, "σ(−23 dB) {s_neg23}");
    // Monotone: more headroom, less noise.
    assert!(inband_noise_sigma(1.0, 40.0) < s33);
}

/// WS-C pure-noise asymptote: noise-only channel RF through the
/// full metric pipeline (declared floor → filter → coherent
/// DAS → ROI reduction) reads ≈ 2.40 linear — the value B1's
/// floored headline reductions (2.44–2.50) were pinned to.
/// Tolerance ±0.6: the asymptote is a statistic of the DAS
/// noise field (finite grid/ROI realisation), not an identity.
#[test]
fn noise_only_rf_reads_the_pure_noise_asymptote() {
    let g = TripletPulseGeometry::stage1().unwrap();
    let n_samples = 1800;
    let zeros = Array2::zeros([g.n_elem(), n_samples]);
    let echo_rms = 1.0;
    let mk_phase = |seed: u64| {
        add_declared_noise_to_rf(
            zeros.view(),
            g.dt_s,
            3.0e6,
            KujiFilterShape::KujiRaisedCosine,
            true,
            echo_rms,
            &DeclaredNoiseFloor {
                inband_snr_db: 0.0,
                seed,
            },
        )
        .unwrap()
    };
    let ph0 = mk_phase(21_000);
    let ph120 = mk_phase(21_001);
    let ph240 = mk_phase(21_002);
    let xs: Vec<f64> = (0..9).map(|i| 15.0e-3 + i as f64 * 1.0e-3).collect();
    let ys: Vec<f64> = (0..9).map(|i| 16.0e-3 + i as f64 * 1.0e-3).collect();
    let (img1, img3) = kuji_images(
        PhaseRf {
            ph0: ph0.view(),
            ph120: ph120.view(),
            ph240: ph240.view(),
        },
        &g,
        &xs,
        &ys,
        KujiFilter {
            fc_hz: 3.0e6,
            shape: KujiFilterShape::KujiRaisedCosine,
            probe_band: true,
        },
        8,
    )
    .unwrap();
    let roi = RectRoi {
        x_lo_m: 14.0e-3,
        x_hi_m: 24.0e-3,
        y_lo_m: 15.0e-3,
        y_hi_m: 25.0e-3,
    };
    let red = roi_mean_reduction(img1.view(), img3.view(), &xs, &ys, &roi).unwrap();
    assert!(
        (red - KUJI_PURE_NOISE_REDUCTION_ASYMPTOTE).abs() < 0.6,
        "noise-only reduction {red} vs asymptote {KUJI_PURE_NOISE_REDUCTION_ASYMPTOTE}"
    );
}
