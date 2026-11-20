//! Stage 2 image parity: Rust envelope DAS + contrast vs the Python prototype.
//!
//! Reference data: `stage1/out/rf_<tag>_ph*.npy` (the prototype's saved RF),
//! located via `KWAVERS_TRIPLET_STAGE1_OUT` or the default
//! `<crate>/../../kwavers-work/stage1/out`. Expected numbers are the
//! prototype's own `stage2/contrast.py` output, re-run 2026-11-19 — NOT the
//! papers' numbers (see the module docs in
//! `kwavers_analysis::signal_processing::triplet_pulse` for why the two
//! statistics disagree about Iwasaki 2018).
//!
//! Tolerance is ±1.0 dB. The Rust pipeline mirrors the prototype operation
//! for operation (FFT brick-wall filter → coherent mean → Hilbert envelope
//! → nearest-sample DAS → ROI mean/max vs the tissue-only baseline), so
//! agreement should be far tighter; the tolerance covers FFT-roundoff and
//! the strict-vs-inclusive band-edge convention only.

use consus_npy::read_npy;
use kwavers_analysis::signal_processing::triplet_pulse::{
    combine_methods, envelope_das, triplet_pulse_contrast, ImageRoi, TripletPulseContrast,
    TripletPulseGeometry,
};
use kwavers_signal::FrequencyFilter;
use leto::Array2;
use std::fs::File;
use std::path::{Path, PathBuf};

const DT_S: f64 = 30.0e-9;
const N_ELEM: usize = 32;
const N_SAMPLES: usize = 1800;
const TOL_DB: f64 = 1.0;

/// Full-band (LP 4 MHz) mean contrasts, 1P / PI / 3P, per bubble tag.
const FULL_MEAN: [(&str, [f64; 3]); 3] = [
    ("a2b2", [23.7, 25.2, 26.6]),
    ("a2b3", [19.4, 19.0, 24.1]),
    ("a2b5", [22.1, 19.0, 24.1]),
];
/// Full-band (LP 4 MHz) max contrasts (Iwasaki's statistic).
const FULL_MAX: [(&str, [f64; 3]); 3] = [
    ("a2b2", [37.8, 39.7, 39.1]),
    ("a2b3", [32.3, 30.8, 34.3]),
    ("a2b5", [36.6, 30.0, 33.7]),
];
/// 1.5f0-band (2.61 ± 0.25 MHz) mean contrasts.
const BAND_MEAN: [(&str, [f64; 3]); 3] = [
    ("a2b2", [0.5, 37.0, 59.6]),
    ("a2b3", [0.3, 21.7, 45.0]),
    ("a2b5", [0.3, 9.7, 28.0]),
];
/// 1.5f0-band max contrasts.
const BAND_MAX: [(&str, [f64; 3]); 3] = [
    ("a2b2", [1.2, 44.4, 66.8]),
    ("a2b3", [0.7, 30.8, 53.9]),
    ("a2b5", [0.7, 12.9, 32.5]),
];
/// Tissue-only run against itself (bubble ROI vs tissue ROI, same image).
const TISSUE_ONLY_FULL_MEAN: [f64; 3] = [1.3, 3.6, 6.8];
const TISSUE_ONLY_BAND_MEAN: [f64; 3] = [0.3, 3.3, 0.2];

/// `numpy.linspace(start, stop, n)`, bit-for-bit: step from
/// `(stop - start) / (n - 1)`, value `start + i * step`, last value pinned
/// to exactly `stop`.
///
/// This matters, not pedantry: the tissue-band ROI boundary at y = 18 mm
/// sits exactly on `|y - 20 mm| = 2 mm`, and the naive `12e-3 + i * 0.5e-3`
/// grid differs from numpy's by 1 ulp there (numpy's step is
/// `0.0004999999999999999`, not the `0.5e-3` literal), which flips the
/// 18 mm row out of the ROI (88 vs 99 pixels) and moves the 3P contrast
/// by ~0.4 dB. The DAS image itself is unaffected (1 ulp cannot flip a
/// nearest-sample delay); only the ROI membership is.
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    let step = (stop - start) / (n - 1) as f64;
    (0..n)
        .map(|i| {
            if i + 1 == n {
                stop
            } else {
                start + i as f64 * step
            }
        })
        .collect()
}

fn reference_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("KWAVERS_TRIPLET_STAGE1_OUT") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../kwavers-work/stage1/out")
}

/// Load one phase RF file; keep the first `N_ELEM` receiver rows
/// (tissue-only files carry 9 extra slab-probe rows after the aperture).
fn load_phase(dir: &Path, tag: &str, phase: &str) -> Array2<f64> {
    let path = dir.join(format!("rf_{tag}_ph{phase}.npy"));
    let file = File::open(&path).unwrap_or_else(|e| {
        panic!(
            "triplet parity: cannot open {}: {e} (set KWAVERS_TRIPLET_STAGE1_OUT)",
            path.display()
        )
    });
    let array = read_npy::<f64>(file).expect("triplet parity: .npy parse failed");
    let shape = array.shape();
    assert_eq!(
        shape[1],
        N_SAMPLES,
        "unexpected sample count in {}",
        path.display()
    );
    assert!(
        shape[0] >= N_ELEM,
        "unexpected row count {} in {}",
        shape[0],
        path.display()
    );
    let values = array.values();
    let rows: Vec<f64> = (0..N_ELEM)
        .flat_map(|r| values[r * N_SAMPLES..(r + 1) * N_SAMPLES].iter().copied())
        .collect();
    Array2::from_shape_vec([N_ELEM, N_SAMPLES], rows).expect("row shape must match")
}

fn filter_rows(rf: &Array2<f64>, mode: &str) -> Array2<f64> {
    let filter = FrequencyFilter::new();
    let [n_rows, n_cols] = rf.shape();
    let mut out = Array2::zeros([n_rows, n_cols]);
    for r in 0..n_rows {
        let row: Vec<f64> = (0..n_cols).map(|c| rf[[r, c]]).collect();
        let filtered = match mode {
            "lp" => filter.lowpass(&row, DT_S, 4.0e6).expect("lowpass"),
            "bp" => filter
                .bandpass(&row, DT_S, 2.36e6, 2.86e6)
                .expect("bandpass"),
            _ => unreachable!(),
        };
        for (c, &v) in filtered.iter().enumerate() {
            out[[r, c]] = v;
        }
    }
    out
}

/// DAS images for the three methods of one tag under one filter mode.
fn tag_images(
    dir: &Path,
    tag: &str,
    mode: &str,
    geometry: &TripletPulseGeometry,
    xs: &[f64],
    ys: &[f64],
) -> [Array2<f64>; 3] {
    let ph0 = filter_rows(&load_phase(dir, tag, "0"), mode);
    let ph120 = filter_rows(&load_phase(dir, tag, "120"), mode);
    let ph240 = filter_rows(&load_phase(dir, tag, "240"), mode);
    let ph180 = filter_rows(&load_phase(dir, tag, "180"), mode);
    let methods =
        combine_methods(ph0.view(), ph120.view(), ph240.view(), ph180.view()).expect("combine");
    [
        envelope_das(methods.one_pulse.view(), geometry, xs, ys).expect("DAS 1P"),
        envelope_das(methods.pulse_inversion.view(), geometry, xs, ys).expect("DAS PI"),
        envelope_das(methods.triplet_pulse.view(), geometry, xs, ys).expect("DAS 3P"),
    ]
}

fn contrasts(
    images: &[Array2<f64>; 3],
    baseline: &[Array2<f64>; 3],
    xs: &[f64],
    ys: &[f64],
) -> [TripletPulseContrast; 3] {
    let bubble_roi = ImageRoi::bubble_disk();
    let tissue_roi = ImageRoi::tissue_band();
    [
        triplet_pulse_contrast(
            images[0].view(),
            baseline[0].view(),
            xs,
            ys,
            &bubble_roi,
            &tissue_roi,
        )
        .expect("contrast 1P"),
        triplet_pulse_contrast(
            images[1].view(),
            baseline[1].view(),
            xs,
            ys,
            &bubble_roi,
            &tissue_roi,
        )
        .expect("contrast PI"),
        triplet_pulse_contrast(
            images[2].view(),
            baseline[2].view(),
            xs,
            ys,
            &bubble_roi,
            &tissue_roi,
        )
        .expect("contrast 3P"),
    ]
}

/// Opt-in table for the parity report (`KWAVERS_TRIPLET_PRINT_PARITY=1`).
/// The `println!` is deliberate test output, not library logging.
#[allow(
    clippy::print_stdout,
    reason = "opt-in parity table printed by the test"
)]
fn check_table(
    label: &str,
    got: &[TripletPulseContrast; 3],
    expected_mean: &[f64; 3],
    expected_max: &[f64; 3],
) {
    if std::env::var_os("KWAVERS_TRIPLET_PRINT_PARITY").is_some() {
        for (i, name) in ["1P", "PI", "3P"].iter().enumerate() {
            println!(
                "{label} {name}: mean {:.2} dB (prototype {:.1}), max {:.2} dB (prototype {:.1})",
                got[i].mean_db, expected_mean[i], got[i].max_db, expected_max[i]
            );
        }
    }
    for (i, name) in ["1P", "PI", "3P"].iter().enumerate() {
        assert!(
            (got[i].mean_db - expected_mean[i]).abs() <= TOL_DB,
            "{label} {name}: mean contrast {:.1} dB, prototype {:.1} dB",
            got[i].mean_db,
            expected_mean[i]
        );
        assert!(
            (got[i].max_db - expected_max[i]).abs() <= TOL_DB,
            "{label} {name}: max contrast {:.1} dB, prototype {:.1} dB",
            got[i].max_db,
            expected_max[i]
        );
    }
}

#[test]
fn stage2_image_parity_vs_prototype() {
    let dir = reference_dir();
    assert!(
        dir.join("rf_a2_ph0.npy").exists(),
        "triplet parity: reference RF not found in {} (set KWAVERS_TRIPLET_STAGE1_OUT)",
        dir.display()
    );
    let geometry = TripletPulseGeometry::stage1().expect("geometry");
    let xs = linspace(5.0e-3, 33.0e-3, 57);
    let ys = linspace(12.0e-3, 36.0e-3, 49);

    for mode in ["lp", "bp"] {
        let baseline = tag_images(&dir, "a2", mode, &geometry, &xs, &ys);
        // Tissue-only run against itself.
        let self_contrast = contrasts(&baseline, &baseline, &xs, &ys);
        let expected_self = if mode == "lp" {
            TISSUE_ONLY_FULL_MEAN
        } else {
            TISSUE_ONLY_BAND_MEAN
        };
        for (i, name) in ["1P", "PI", "3P"].iter().enumerate() {
            assert!(
                (self_contrast[i].mean_db - expected_self[i]).abs() <= TOL_DB,
                "a2 self-contrast {mode} {name}: {:.1} dB, prototype {:.1} dB",
                self_contrast[i].mean_db,
                expected_self[i]
            );
        }
        let (mean_table, max_table) = if mode == "lp" {
            (FULL_MEAN, FULL_MAX)
        } else {
            (BAND_MEAN, BAND_MAX)
        };
        for (tag, expected_mean, expected_max) in
            mean_table.iter().zip(max_table.iter()).map(|(m, x)| {
                assert_eq!(m.0, x.0);
                (m.0, m.1, x.1)
            })
        {
            let images = tag_images(&dir, tag, mode, &geometry, &xs, &ys);
            let got = contrasts(&images, &baseline, &xs, &ys);
            check_table(
                &format!("{tag} {mode}"),
                &got,
                &expected_mean,
                &expected_max,
            );
            // 1.5f0-band mean ordering 1P < PI < 3P at every R0.
            if mode == "bp" {
                assert!(
                    got[0].mean_db < got[1].mean_db && got[1].mean_db < got[2].mean_db,
                    "{tag} 1.5f0 ordering failed: {:.1} / {:.1} / {:.1} dB",
                    got[0].mean_db,
                    got[1].mean_db,
                    got[2].mean_db
                );
            }
        }
    }
}
