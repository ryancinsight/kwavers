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
