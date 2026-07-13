use eunomia::Complex64;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

use super::*;

#[test]
fn gaussian_beam_size() {
    let x = vec![-1e-3, 0.0, 1e-3];
    let z = vec![0.0, 5e-3, 10e-3];
    let t = Complex64::new(1.0, 0.0);
    let (re, im) = focused_gaussian_beam_2d(
        &x,
        &z,
        0.0,
        5e-3,
        MHZ_TO_HZ,
        SOUND_SPEED_WATER_SIM,
        1e-3,
        t,
        0.0,
        0.1,
    );
    assert_eq!(re.len(), 9);
    assert_eq!(im.len(), 9);
}

#[test]
fn gaussian_beam_peak_at_focus() {
    let x = vec![0.0];
    let z: Vec<f64> = vec![-5e-3, 0.0, 5e-3];
    let t = Complex64::new(1.0, 0.0);
    let (re, _) = focused_gaussian_beam_2d(
        &x,
        &z,
        0.0,
        0.0,
        MHZ_TO_HZ,
        SOUND_SPEED_WATER_SIM,
        1e-3,
        t,
        0.0,
        0.1,
    );
    assert!(re[1].abs() >= re[0].abs().min(re[2].abs()));
}

#[test]
fn backprop_normalisation() {
    let x = vec![0.0];
    let z = vec![0.01, 0.02];
    let (re, _) = backprop_green_function_2d(&x, &z, 0.0, 0.0, 1000.0, 1500.0);
    let mag0 = re[0].abs();
    let mag1 = re[1].abs();
    assert!(mag0 > mag1, "mag0={} mag1={}", mag0, mag1);
}

#[test]
fn rtm_imaging_normalised_max_one() {
    let fwd_r = vec![1.0, 2.0, 3.0];
    let fwd_i = vec![0.0, 0.0, 0.0];
    let bwd_r = vec![1.0, 1.0, 1.0];
    let bwd_i = vec![0.0, 0.0, 0.0];
    let img = rtm_imaging_condition(&fwd_r, &fwd_i, &bwd_r, &bwd_i, 1, 3);
    let max = img.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!((max - 1.0).abs() < 1e-10);
}

#[test]
fn multi_freq_fusion_mean() {
    let a = vec![0.0, 1.0, 2.0];
    let b = vec![2.0, 1.0, 0.0];
    let fused = rtm_multi_frequency_fusion(&[a, b]);
    assert_eq!(fused, vec![1.0, 1.0, 1.0]);
}

#[test]
fn modulation_frequencies_increasing() {
    let f = temporal_modulation_frequencies(MHZ_TO_HZ, 5, SOUND_SPEED_WATER_SIM, 0.1);
    assert_eq!(f.len(), 5);
    for i in 1..f.len() {
        assert!(f[i] > f[i - 1]);
    }
}

#[test]
fn suppression_gain_no_reflection_is_one() {
    assert!((standing_wave_suppression_gain(0.0) - 1.0).abs() < 1e-15);
}

#[test]
fn suppression_gain_increases_with_reflection() {
    let g1 = standing_wave_suppression_gain(0.2);
    let g2 = standing_wave_suppression_gain(0.5);
    assert!(g1 > 1.0 && g2 > g1);
}

#[test]
fn backprop_3d_amplitude_inverse_r() {
    let x = vec![0.0];
    let y = vec![0.0];
    let z = vec![0.01, 0.02];
    let (re, im) = backprop_green_function_3d(&x, &y, &z, 0.0, 0.0, 0.0, 1000.0);
    let mag0 = (re[0] * re[0] + im[0] * im[0]).sqrt();
    let mag1 = (re[1] * re[1] + im[1] * im[1]).sqrt();
    let ratio = mag0 / mag1;
    assert!((ratio - 2.0).abs() < 1e-6, "ratio={}", ratio);
}

#[test]
fn backprop_3d_grid_length() {
    let x = vec![0.0, 0.01];
    let y = vec![0.0, 0.01, 0.02];
    let z = vec![0.0, 0.01, 0.02, 0.03];
    let (re, im) = backprop_green_function_3d(&x, &y, &z, 0.0, 0.0, 0.0, 500.0);
    assert_eq!(re.len(), 2 * 3 * 4);
    assert_eq!(im.len(), 2 * 3 * 4);
}

#[test]
fn source_normalized_max_one() {
    let fwd_r = vec![1.0, 2.0, 3.0];
    let fwd_i = vec![0.0, 0.0, 0.0];
    let bwd_r = vec![0.5, 1.5, 2.0];
    let bwd_i = vec![0.0, 0.0, 0.0];
    let img = rtm_source_normalized_condition(&fwd_r, &fwd_i, &bwd_r, &bwd_i, 1e-4);
    let max = img.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!((max - 1.0).abs() < 1e-9, "max={}", max);
}

#[test]
fn source_normalized_removes_amplitude_bias() {
    let fwd_r = vec![100.0, 1.0];
    let fwd_i = vec![0.0, 0.0];
    let bwd_r = vec![50.0, 1.0];
    let bwd_i = vec![0.0, 0.0];
    let img = rtm_source_normalized_condition(&fwd_r, &fwd_i, &bwd_r, &bwd_i, 1e-6);
    assert!(img[1] > img[0], "img={:?}", img);
}

#[test]
fn aperture_weighted_fusion_equal_weights() {
    let a = vec![0.0, 2.0, 4.0];
    let b = vec![2.0, 4.0, 6.0];
    let fused_w = rtm_aperture_weighted_fusion(&[a.clone(), b.clone()], &[1.0, 1.0]);
    let fused_u = rtm_multi_frequency_fusion(&[a, b]);
    for (w, u) in fused_w.iter().zip(fused_u.iter()) {
        assert!((w - u).abs() < 1e-12, "w={} u={}", w, u);
    }
}

#[test]
fn aperture_weighted_fusion_zero_weights_uniform_fallback() {
    let a = vec![0.0, 1.0];
    let b = vec![2.0, 3.0];
    let fused = rtm_aperture_weighted_fusion(&[a, b], &[0.0, 0.0]);
    assert!((fused[0] - 1.0).abs() < 1e-12, "fused={:?}", fused);
    assert!((fused[1] - 2.0).abs() < 1e-12, "fused={:?}", fused);
}
