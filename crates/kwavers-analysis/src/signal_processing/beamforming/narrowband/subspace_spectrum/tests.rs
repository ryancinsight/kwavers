//! Value-semantic tests for the narrowband subspace spatial spectra.
//!
//! Three tiers, all with analytically-derived thresholds:
//! 1. **Eigenvalue split (Theorem 22.2):** a constructed `R = σ_s² Σ e_k e_kᴴ +
//!    σ_n² I` with orthonormal signal vectors has exactly `K` eigenvalues equal to
//!    `σ_s² + σ_n²` and `N-K` equal to `σ_n²`.
//! 2. **Super-resolution vs DAS:** on a constructed multi-source narrowband model
//!    `R = Σ_k a_k a_kᴴ + σ_n² I`, the Eigenspace-MV and MUSIC maps peak at the true
//!    sources with a strictly higher peak-to-sidelobe ratio than conventional DAS.
//! 3. **End-to-end public API:** the `*_spatial_spectrum_point` functions, fed
//!    synthetic time-domain sensor data for one delayed tone-burst source, score the
//!    true source location above an off-target location.

use super::*;
use crate::signal_processing::beamforming::adaptive::subspace::{EigenspaceMV, MUSIC};
use crate::signal_processing::beamforming::narrowband::steering::NarrowbandSteering;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, TWO_PI};
use kwavers_math::linear_algebra::EigenDecomposition;
use ndarray::{Array1, Array2, Array3};
use eunomia::Complex64;

/// Uniform linear array of `n` elements with pitch `d` along x, centred on origin.
fn linear_array(n: usize, d: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| [(i as f64 - 0.5 * (n as f64 - 1.0)) * d, 0.0, 0.0])
        .collect()
}

/// Steering vector a(p) for the array at frequency f (unit-magnitude phasors).
fn steering(positions: &[[f64; 3]], p: [f64; 3], f: f64, c: f64) -> Array1<Complex64> {
    NarrowbandSteering::new(positions.to_vec(), c)
        .expect("steering geometry")
        .steering_vector_point(p, f)
        .expect("steering vector")
        .into_array()
}

#[test]
fn eigenvalue_split_matches_theorem_22_2() {
    // N=6 sensors, K=2 orthonormal signal directions, σ_s²=4, σ_n²=0.25.
    let n = 6usize;
    let k = 2usize;
    let sigma_s2 = 4.0_f64;
    let sigma_n2 = 0.25_f64;

    // Two orthonormal complex signal vectors via columns of a small DFT-like basis.
    let mut e: Vec<Array1<Complex64>> = Vec::new();
    for col in 0..k {
        let mut v = Array1::<Complex64>::from_elem(n, Complex64::default());
        for (row, value) in v.iter_mut().enumerate() {
            let phase = TWO_PI * (col as f64 + 1.0) * (row as f64) / (n as f64);
            *value = Complex64::new(0.0, phase).exp() / (n as f64).sqrt();
        }
        e.push(v);
    }
    // Orthonormality sanity (distinct DFT columns are orthogonal).
    let cross: Complex64 = e[0]
        .iter()
        .zip(e[1].iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    assert!(cross.norm() < 1e-12, "signal vectors must be orthonormal");

    // R = σ_n² I + σ_s² Σ_k e_k e_kᴴ.
    let mut r = Array2::<Complex64>::from_elem((n, n), Complex64::default());
    for i in 0..n {
        r[(i, i)] += Complex64::new(sigma_n2, 0.0);
    }
    for ev in &e {
        for i in 0..n {
            for j in 0..n {
                r[(i, j)] += Complex64::new(sigma_s2, 0.0) * ev[i] * ev[j].conj();
            }
        }
    }

    let (eigenvalues, _) =
        EigenDecomposition::hermitian_eigendecomposition_complex(&r).expect("eig");
    let mut vals: Vec<f64> = (0..n).map(|i| eigenvalues[i]).collect();
    vals.sort_by(|a, b| b.total_cmp(a));

    // K largest = σ_s² + σ_n²; remaining N-K = σ_n² (Theorem 22.2).
    for &lam in vals.iter().take(k) {
        assert!(
            (lam - (sigma_s2 + sigma_n2)).abs() < 1e-9,
            "signal eigenvalue {lam} != σ_s²+σ_n² = {}",
            sigma_s2 + sigma_n2
        );
    }
    for &lam in vals.iter().skip(k) {
        assert!(
            (lam - sigma_n2).abs() < 1e-9,
            "noise eigenvalue {lam} != σ_n² = {sigma_n2}"
        );
    }
}

/// Construct the narrowband signal-model covariance for `K` equal-power incoherent
/// sources at `source_points`: `R = σ_s² Σ_k a_k a_kᴴ + σ_n² I`.
fn build_source_covariance(
    positions: &[[f64; 3]],
    source_points: &[[f64; 3]],
    f: f64,
    c: f64,
    sigma_s2: f64,
    sigma_n2: f64,
) -> Array2<Complex64> {
    let n = positions.len();
    let mut r = Array2::<Complex64>::from_elem((n, n), Complex64::default());
    for i in 0..n {
        r[(i, i)] += Complex64::new(sigma_n2, 0.0);
    }
    for &sp in source_points {
        let a = steering(positions, sp, f, c);
        for i in 0..n {
            for j in 0..n {
                r[(i, j)] += Complex64::new(sigma_s2, 0.0) * a[i] * a[j].conj();
            }
        }
    }
    r
}

#[test]
fn subspace_beats_das_peak_to_sidelobe() {
    // 12-element λ/2 array, two sources at known lateral positions, depth 40 mm.
    let c = SOUND_SPEED_WATER_SIM;
    let f = MHZ_TO_HZ; // 1 MHz
    let lambda = c / f;
    let positions = linear_array(12, lambda / 2.0);
    let depth = 0.040;
    let sources = [[-0.006, 0.0, depth], [0.006, 0.0, depth]];
    let r = build_source_covariance(&positions, &sources, f, c, 4.0, 0.1);

    // Lateral scan across the focal plane at the source depth.
    let esmv = EigenspaceMV::new(2);
    let music = MUSIC::new(2);
    let n_grid = 81usize;
    let (x0, x1) = (-0.020, 0.020);

    let mut das = Vec::with_capacity(n_grid);
    let mut esmv_map = Vec::with_capacity(n_grid);
    let mut music_map = Vec::with_capacity(n_grid);
    for ig in 0..n_grid {
        let x = x0 + (x1 - x0) * ig as f64 / (n_grid as f64 - 1.0);
        let p = [x, 0.0, depth];
        let a = steering(&positions, p, f, c);

        // Conventional (Bartlett/DAS) power: aᴴ R a.
        let mut das_p = Complex64::new(0.0, 0.0);
        for i in 0..positions.len() {
            let mut ra_i = Complex64::new(0.0, 0.0);
            for j in 0..positions.len() {
                ra_i += r[(i, j)] * a[j];
            }
            das_p += a[i].conj() * ra_i;
        }
        das.push(das_p.re);
        esmv_map.push(esmv.signal_subspace_response(&r, &a).expect("esmv"));
        music_map.push(music.pseudospectrum(&r, &a).expect("music"));
    }

    // Peak-to-sidelobe ratio: peak value / median of the map. Super-resolution
    // methods concentrate energy at the sources, so their PSR strictly exceeds DAS.
    let psr = |m: &[f64]| -> f64 {
        let peak = m.iter().cloned().fold(f64::MIN, f64::max);
        let mut sorted = m.to_vec();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        peak / median.max(1e-30)
    };

    let psr_das = psr(&das);
    let psr_esmv = psr(&esmv_map);
    let psr_music = psr(&music_map);

    assert!(
        psr_esmv > psr_das,
        "Eigenspace-MV PSR {psr_esmv:.3} should exceed DAS PSR {psr_das:.3}"
    );
    assert!(
        psr_music > psr_das,
        "MUSIC PSR {psr_music:.3} should exceed DAS PSR {psr_das:.3}"
    );

    // Both subspace maps must peak near a true source (within one grid step of ±6 mm).
    let grid_step = (x1 - x0) / (n_grid as f64 - 1.0);
    let argmax_x = |m: &[f64]| -> f64 {
        let (i, _) = m
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty");
        x0 + (x1 - x0) * i as f64 / (n_grid as f64 - 1.0)
    };
    for (label, m) in [("esmv", &esmv_map), ("music", &music_map)] {
        let xm = argmax_x(m);
        let near = sources.iter().any(|s| (xm - s[0]).abs() <= 2.0 * grid_step);
        assert!(near, "{label} peak at x={xm:.4} not near a source (±6 mm)");
    }
}

#[test]
fn end_to_end_localizes_single_tone_source() {
    // One narrowband tone-burst source; the public sensor_data → spectrum path must
    // score the true location above an off-target location for both methods.
    let c = SOUND_SPEED_WATER_SIM;
    let f = MHZ_TO_HZ; // 1 MHz
    let fs = 20.0 * MHZ_TO_HZ; // 20 MHz
    let lambda = c / f;
    let positions = linear_array(10, lambda / 2.0);
    let n_samples = 1024usize;
    let source = [0.003, 0.0, 0.035];
    let offset = [0.015, 0.0, 0.035];

    // Synthesize delayed continuous tone at each sensor: x_i(t)=cos(2πf(t-τ_i)).
    let mut data = Array3::<f64>::zeros((positions.len(), 1, n_samples));
    for (i, &pos) in positions.iter().enumerate() {
        let dx = pos[0] - source[0];
        let dy = pos[1] - source[1];
        let dz = pos[2] - source[2];
        let tau = (dx * dx + dy * dy + dz * dz).sqrt() / c;
        for t in 0..n_samples {
            let time = t as f64 / fs;
            data[(i, 0, t)] = (TWO_PI * f * (time - tau)).cos();
        }
    }

    let cfg = SubspaceSpectrumConfig {
        frequency_hz: f,
        sampling_frequency_hz: fs,
        sound_speed: c,
        num_sources: 1,
        diagonal_loading: 1e-3,
    };

    let music_true = music_spatial_spectrum_point(&data, &positions, source, &cfg).expect("m_true");
    let music_off = music_spatial_spectrum_point(&data, &positions, offset, &cfg).expect("m_off");
    assert!(
        music_true > 2.0 * music_off,
        "MUSIC: true {music_true:.4e} should dominate offset {music_off:.4e}"
    );

    let esmv_true =
        eigenspace_mv_spatial_spectrum_point(&data, &positions, source, &cfg).expect("e_true");
    let esmv_off =
        eigenspace_mv_spatial_spectrum_point(&data, &positions, offset, &cfg).expect("e_off");
    assert!(
        esmv_true > 2.0 * esmv_off,
        "Eigenspace-MV: true {esmv_true:.4e} should dominate offset {esmv_off:.4e}"
    );
}

#[test]
fn rejects_too_many_sources() {
    let positions = linear_array(4, 1e-3);
    let data = Array3::<f64>::zeros((4, 1, 64));
    let cfg = SubspaceSpectrumConfig {
        frequency_hz: MHZ_TO_HZ,
        sampling_frequency_hz: 20.0 * MHZ_TO_HZ,
        sound_speed: SOUND_SPEED_WATER_SIM,
        num_sources: 4, // K must be < N = 4
        diagonal_loading: 1e-3,
    };
    let err = music_spatial_spectrum_point(&data, &positions, [0.0, 0.0, 0.03], &cfg)
        .expect_err("K>=N must error");
    assert!(err.to_string().contains("num_sources"));
}
