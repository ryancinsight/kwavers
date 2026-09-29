//! Synthetic-aperture pulse-echo receive-data synthesis for real-time B-mode.
//!
//! Given a set of point scatterers (an acoustic-reflectivity sampling of the tissue)
//! and an imaging-array geometry, [`simulate_receive_rf`] produces the per-element
//! channel RF that the array records: each scatterer re-radiates a Gaussian-modulated
//! tone burst that reaches element `s` after the one-way time of flight
//! `|r_i − r_s| / c`, with `1/r` spreading and reflectivity weighting; contributions
//! sum coherently (linear acoustics, first-Born single scatter). This is the model a
//! one-way delay-and-sum beamformer inverts (synthetic-transmit-focusing / passive
//! receive), so beamforming the RF reconstructs the reflectivity map with a realistic
//! point-spread function and speckle — a genuine receive-data → image pipeline.

use leto::{Array2, ArrayView1, ArrayView2};
use std::f64::consts::{LN_2, PI};

/// Geometry and acquisition settings for [`simulate_receive_rf`].
///
/// The three arrays are borrowed views, so a caller sweeping `n_samples` or a
/// bandwidth pays no copy; the scalars are named because `c`, `fs`, `f0` and
/// `frac_bw` are four adjacent `f64` arguments in a row, which is where a
/// positional call becomes unreadable.
///
/// `Clone + Copy` rather than also `Debug`/`PartialEq`: `ArrayView` is `Copy`
/// but implements neither of the latter two, so deriving them would not compile.
#[derive(Clone, Copy)]
pub struct SimulateReceiveRfInput<'a> {
    /// Scatterer positions `(n_scat, 3)` in metres.
    pub scat_pos: ArrayView2<'a, f64>,
    /// Reflectivity per scatterer `(n_scat,)`.
    pub scat_amp: ArrayView1<'a, f64>,
    /// Array element positions `(n_elem, 3)` in metres.
    pub elem_pos: ArrayView2<'a, f64>,
    /// Sound speed [m/s].
    pub c: f64,
    /// Sampling frequency [Hz].
    pub fs: f64,
    /// Imaging centre frequency [Hz].
    pub f0: f64,
    /// Fractional −6 dB pulse bandwidth, which sets the pulse length via
    /// `σ_t = √(2 ln2)/(π·frac_bw·f0)`.
    pub frac_bw: f64,
    /// RF record length in samples.
    pub n_samples: usize,
}

/// Synthesize per-element channel RF (shape `(n_elem, n_samples)`) from point
/// scatterers `scat_pos` (`(n_scat, 3)` `m`) with reflectivity `scat_amp` (`(n_scat,)`)
/// recorded by an array at `elem_pos` (`(n_elem, 3)` `m`).
#[must_use]
pub fn simulate_receive_rf(input: SimulateReceiveRfInput<'_>) -> Array2<f64> {
    let SimulateReceiveRfInput {
        scat_pos,
        scat_amp,
        elem_pos,
        c,
        fs,
        f0,
        frac_bw,
        n_samples,
    } = input;
    let n_elem = elem_pos.shape()[0];
    let mut rf = Array2::<f64>::zeros([n_elem, n_samples]);
    if n_samples == 0 || n_elem == 0 || !(c > 0.0 && fs > 0.0 && f0 > 0.0) {
        return rf;
    }
    let bw = (frac_bw.max(1e-3)) * f0;
    let sigma_t = (2.0 * LN_2).sqrt() / (PI * bw); // Gaussian envelope std [s]
    let half = (4.0 * sigma_t * fs).ceil() as isize; // ±4σ pulse support [samples]
    let two_sig2 = 2.0 * sigma_t * sigma_t;
    let w0 = 2.0 * PI * f0;
    let dmin = c / fs; // spreading floor (one sample)
    let n_scat = scat_pos.shape()[0];
    for i in 0..n_scat {
        let a = scat_amp[[i]];
        if a == 0.0 || !a.is_finite() {
            continue;
        }
        let (sx, sy, sz) = (scat_pos[[i, 0]], scat_pos[[i, 1]], scat_pos[[i, 2]]);
        for s in 0..n_elem {
            let dx = sx - elem_pos[[s, 0]];
            let dy = sy - elem_pos[[s, 1]];
            let dz = sz - elem_pos[[s, 2]];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let tof = d / c;
            let amp = a / d.max(dmin); // 1/r spreading
            let center = tof * fs;
            let n0 = center.floor() as isize;
            let lo = (n0 - half).max(0);
            let hi = (n0 + half).min(n_samples as isize - 1);
            let mut n = lo;
            while n <= hi {
                let tau = n as f64 / fs - tof;
                let env = (-(tau * tau) / two_sig2).exp();
                rf[[s, n as usize]] += amp * env * (w0 * tau).cos();
                n += 1;
            }
        }
    }
    rf
}

/// Log-compress an envelope image using a fixed reference amplitude.
///
/// This preserves temporal intensity changes across a B-mode sequence. Per-frame
/// peak normalization is intentionally not performed here because it destroys
/// baseline-relative lesion contrast.
#[must_use]
pub fn bmode_db_fixed_reference(envelope: &[f64], reference: f64, floor_db: f64) -> Vec<f64> {
    let r = reference.max(1.0e-300);
    envelope
        .iter()
        .map(|&v| {
            let db = 20.0 * ((v.max(0.0) / r).max(1.0e-12)).log10();
            db.clamp(floor_db, 0.0)
        })
        .collect()
}

/// Baseline-relative delta B-mode in dB.
///
/// Returns `20 log10((env_t + eps)/(env_0 + eps))` for each pixel. Positive
/// values indicate echogenic brightening; negative values indicate attenuation
/// or hypoechoic lesion development.
#[must_use]
pub fn delta_bmode_db(envelope: &[f64], baseline: &[f64], epsilon: f64) -> Vec<f64> {
    let n = envelope.len().min(baseline.len());
    let eps = epsilon.max(1.0e-300);
    (0..n)
        .map(|i| 20.0 * ((envelope[i].max(0.0) + eps) / (baseline[i].max(0.0) + eps)).log10())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array1;

    fn positions<const N: usize>(rows: [[f64; 3]; N]) -> Array2<f64> {
        Array2::from_shape_vec([N, 3], rows.into_iter().flatten().collect())
            .expect("three-dimensional positions match their declared row count")
    }

    #[test]
    fn echo_peaks_at_one_way_time_of_flight() {
        // One scatterer on the array normal; the nearest element sees the echo at
        // its one-way time of flight |r_i − r_s|/c.
        let c = 1540.0;
        let fs = 40e6;
        let f0 = 3e6;
        let scat = positions([[0.0, 0.02, 0.0]]); // 20 mm depth along +y
        let amp = Array1::from(vec![1.0]);
        let elem = positions([[0.0, 0.0, 0.0]]); // single element at origin
        let rf = simulate_receive_rf(SimulateReceiveRfInput {
            scat_pos: scat.view(),
            scat_amp: amp.view(),
            elem_pos: elem.view(),
            c,
            fs,
            f0,
            frac_bw: 0.6,
            n_samples: 2048,
        });
        // Peak sample index ≈ (d/c)·fs.
        let expected = (0.02 / c * fs).round() as usize;
        let row = rf
            .index_axis::<1>(0, 0)
            .expect("single-element receive data has row zero");
        let peak = (0..row.size())
            .max_by(|&a, &b| row[a].abs().total_cmp(&row[b].abs()))
            .unwrap();
        assert!(
            (peak as isize - expected as isize).abs() <= 1,
            "peak {peak} vs {expected}"
        );
    }

    #[test]
    fn zero_reflectivity_gives_silence() {
        let scat = positions([[0.0, 0.02, 0.0]]);
        let amp = Array1::from(vec![0.0]);
        let elem = positions([[0.0, 0.0, 0.0]]);
        let rf = simulate_receive_rf(SimulateReceiveRfInput {
            scat_pos: scat.view(),
            scat_amp: amp.view(),
            elem_pos: elem.view(),
            c: 1540.0,
            fs: 40e6,
            f0: 3e6,
            frac_bw: 0.6,
            n_samples: 512,
        });
        assert!(rf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn closer_scatterer_arrives_earlier() {
        let (c, fs, f0) = (1540.0, 40e6, 3e6);
        let elem = positions([[0.0, 0.0, 0.0]]);
        let amp = Array1::from(vec![1.0]);
        let near = simulate_receive_rf(SimulateReceiveRfInput {
            scat_pos: positions([[0.0, 0.01, 0.0]]).view(),
            scat_amp: amp.view(),
            elem_pos: elem.view(),
            c,
            fs,
            f0,
            frac_bw: 0.6,
            n_samples: 2048,
        });
        let far = simulate_receive_rf(SimulateReceiveRfInput {
            scat_pos: positions([[0.0, 0.03, 0.0]]).view(),
            scat_amp: amp.view(),
            elem_pos: elem.view(),
            c,
            fs,
            f0,
            frac_bw: 0.6,
            n_samples: 2048,
        });
        let pk = |r: &Array2<f64>| {
            let row = r
                .index_axis::<1>(0, 0)
                .expect("single-element receive data has row zero");
            (0..row.size())
                .max_by(|&a, &b| row[a].abs().total_cmp(&row[b].abs()))
                .unwrap()
        };
        assert!(pk(&near) < pk(&far));
    }

    #[test]
    fn bmode_db_fixed_reference_clamps_to_floor_and_zero_db() {
        let db = bmode_db_fixed_reference(&[1.0, 0.1, 1.0e-3, 0.0], 1.0, -40.0);
        let expected = [0.0, -20.0, -40.0, -40.0];

        assert_eq!(db.len(), expected.len());
        for (actual, expected) in db.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-12,
                "actual={actual} expected={expected}"
            );
        }
    }
}
