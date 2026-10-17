//! Passive acoustic mapping (PAM) channel — time-exposure acoustics.
//!
//! During a therapy burst the cavitation cloud at the focus emits broadband
//! acoustic energy; the array records it passively (no imaging transmit). PAM
//! reconstructs the spatial **emission source map** by, for every candidate
//! pixel, time-aligning the channels to that pixel's time-of-flight and summing —
//! coherent only at the true source, which lights up as the lesion. This is the
//! through-skull modality that needs no inversion and is sensitive to the strong
//! cavitation source (Gyöngy & Coussios 2010; Norton & Won 2000).
//!
//! # Why a bespoke aligner
//!
//! The repository's `BeamformingProcessor::delay_and_sum_with` beamforms to a single
//! focal point and aligns to the **latest** arrival (advance by `max−τ_i`), which
//! is the wrong sign for focusing time-of-flight emission data at a pixel (its
//! own docstring flags this for transient localization). PAM here grid-searches
//! the pixel and aligns by `+(τ_i − τ_min)` — advancing each channel by its own
//! propagation delay so the emission stacks coherently at the true source.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array2, Array3};
use std::borrow::Cow;

/// Geometry of the monitored 2-D coronal (x–z) PAM image slice.
#[derive(Clone, Debug)]
pub struct PamMonitorConfig {
    /// Background sound speed for time-of-flight [m/s].
    pub sound_speed_m_s: f64,
    /// Physical position of pixel `(0, 0)` `[x0, y_slice, z0]` in metres.
    pub origin_m: [f64; 3],
    /// Pixel spacing in metres.
    pub spacing_m: f64,
    /// Image pixels along x.
    pub nx: usize,
    /// Image pixels along z.
    pub nz: usize,
}

/// Strictly positive and non-NaN: a NaN fails `> 0.0`, so it is rejected too.
/// Used for input-validation guards (a NaN sound speed / sample rate is invalid).
#[inline]
fn is_positive(x: f64) -> bool {
    x > 0.0
}

#[inline]
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
}

/// Pixel `(ix, iz)` physical position on the coronal slice.
#[inline]
fn pixel_position(cfg: &PamMonitorConfig, ix: usize, iz: usize) -> [f64; 3] {
    [
        cfg.origin_m[0] + ix as f64 * cfg.spacing_m,
        cfg.origin_m[1],
        cfg.origin_m[2] + iz as f64 * cfg.spacing_m,
    ]
}

/// Add a shifted channel to the per-time coherent sums in sample order.
#[inline]
fn add_channel(coherent: &mut [f64], samples: &[f64]) {
    let (sums, sum_tail) = coherent[..samples.len()].as_chunks_mut::<8>();
    let (chunks, sample_tail) = samples.as_chunks::<8>();
    // One checked chunk borrow amortizes iterator/bounds work across eight
    // independent time samples; each sample's channel order stays unchanged.
    for ([d0, d1, d2, d3, d4, d5, d6, d7], &[s0, s1, s2, s3, s4, s5, s6, s7]) in
        sums.iter_mut().zip(chunks)
    {
        *d0 += s0;
        *d1 += s1;
        *d2 += s2;
        *d3 += s3;
        *d4 += s4;
        *d5 += s5;
        *d6 += s6;
        *d7 += s7;
    }
    for (sum, &sample) in sum_tail.iter_mut().zip(sample_tail) {
        *sum += sample;
    }
}

/// Time-exposure-acoustics passive map over the monitored slice.
///
/// `sensor_data` is `(n_elements, 1, n_samples)` passively-recorded RF; the map
/// value at each pixel is the time-integrated squared coherent sum
/// `Σ_t (Σ_i s_i[t + (τ_i − τ_min)·f_s])²`, peaking at the emission source.
///
/// # Errors
/// Returns `KwaversError::InvalidInput` on element-count/shape mismatch or
/// non-positive sound speed / sample rate.
pub fn passive_acoustic_map(
    sensor_data: &Array3<f64>,
    element_positions: &[[f64; 3]],
    sample_rate: f64,
    cfg: &PamMonitorConfig,
) -> KwaversResult<Array2<f64>> {
    let [n_elements, _channels, n_samples] = sensor_data.shape();
    if n_elements != element_positions.len() {
        return Err(KwaversError::InvalidInput(format!(
            "PAM: sensor_data has {n_elements} elements but {} positions given",
            element_positions.len()
        )));
    }
    if !is_positive(cfg.sound_speed_m_s) || !is_positive(sample_rate) {
        return Err(KwaversError::InvalidInput(
            "PAM: sound speed and sample rate must be positive".to_owned(),
        ));
    }

    let mut map = Array2::<f64>::zeros((cfg.nx, cfg.nz));
    if cfg.nx == 0 || cfg.nz == 0 || n_samples == 0 {
        return Ok(map);
    }
    // Borrow the strided backing once. Array indexing otherwise reacquires
    // the complete storage slice for every sample; the retained view preserves
    // arbitrary layouts and the same bounds-checked element access.
    let sensor_view = sensor_data.view();
    let sensor_rows = (0..n_elements)
        .map(|i| {
            sensor_view
                .slice_with::<1>(&[
                    leto::SliceArg::Index(i as isize),
                    leto::SliceArg::Index(0),
                    leto::SliceArg::All,
                ])
                .map_err(|error| {
                    KwaversError::InvalidInput(format!("PAM: invalid first-channel shape: {error}"))
                })
        })
        .collect::<KwaversResult<Vec<_>>>()?;
    // Retain contiguous rows without copying. Normalize other layouts once
    // in logical sample order, so the same kernel serves both layouts.
    let row_slices = sensor_rows
        .iter()
        .map(|row| match row.as_slice() {
            Some(samples) => Cow::Borrowed(samples),
            None => Cow::Owned(row.iter().copied().collect::<Vec<_>>()),
        })
        .collect::<Vec<_>>();
    let mut tof = vec![0.0_f64; n_elements];
    let mut shifts = vec![0_usize; n_elements];
    let mut coherent = vec![0.0_f64; n_samples];
    for ix in 0..cfg.nx {
        for iz in 0..cfg.nz {
            let p = pixel_position(cfg, ix, iz);
            let mut tof_min = f64::INFINITY;
            for (i, &e) in element_positions.iter().enumerate() {
                let t = distance(e, p) / cfg.sound_speed_m_s;
                tof[i] = t;
                tof_min = tof_min.min(t);
            }
            // Per-channel sample advance to align this pixel's emission to t=0.
            // Geometry and rounded advances are constant across time samples.
            for (shift, &t) in shifts.iter_mut().zip(&tof) {
                *shift = ((t - tof_min) * sample_rate).round() as usize;
            }
            coherent.fill(0.0);
            // Sweep each channel's samples once. Each time sample still adds
            // channels in their original order; the final energy reduction
            // also keeps its original time order. Layout dispatch and delay
            // bounds are resolved once per channel instead of per sample.
            for (samples, &shift) in row_slices.iter().zip(&shifts) {
                if shift < n_samples {
                    add_channel(&mut coherent, &samples[shift..]);
                }
            }
            let energy = coherent.iter().fold(0.0, |sum, &value| sum + value * value);
            map[[ix, iz]] = energy;
        }
    }
    Ok(map)
}

/// Synthesize the passively-recorded channel data for a point emitter at
/// `source_m`: each element receives a Gaussian pulse delayed by its
/// time-of-flight. Models the cavitation cloud's broadband burst for validation
/// and for driving the monitor from the lesion centroid.
///
/// # Errors
/// Returns `KwaversError::InvalidInput` on non-positive sound speed/sample rate.
pub fn synthesize_emission(
    source_m: [f64; 3],
    element_positions: &[[f64; 3]],
    sample_rate: f64,
    sound_speed_m_s: f64,
    n_samples: usize,
    pulse_halfwidth_samples: f64,
) -> KwaversResult<Array3<f64>> {
    if !is_positive(sound_speed_m_s)
        || !is_positive(sample_rate)
        || !is_positive(pulse_halfwidth_samples)
    {
        return Err(KwaversError::InvalidInput(
            "PAM synth: sound speed, sample rate, and pulse width must be positive".to_owned(),
        ));
    }
    let n = element_positions.len();
    let mut data = Array3::<f64>::zeros((n, 1, n_samples));
    let mut data_view = data.view_mut();
    for (i, &e) in element_positions.iter().enumerate() {
        // 1/r geometric spreading + TOF delay.
        let r = distance(e, source_m).max(0.5 / sample_rate * sound_speed_m_s);
        let centre = distance(e, source_m) / sound_speed_m_s * sample_rate;
        let amplitude = 1.0 / r;
        for t in 0..n_samples {
            let u = (t as f64 - centre) / pulse_halfwidth_samples;
            data_view[[i, 0, t]] = amplitude * (-u * u).exp();
        }
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesized_samples_match_delayed_gaussian_with_geometric_spreading() {
        let _capture = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .without_time()
                .with_test_writer()
                .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
                .finish(),
        );
        let _test = tracing::info_span!("analytic_pam_emission").entered();
        let elements = {
            let _arrange = tracing::info_span!("arrange").entered();
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        };
        let data = {
            let _act = tracing::info_span!("act").entered();
            synthesize_emission([0.0; 3], &elements, 1.0, 1.0, 4, 1.0).unwrap()
        };
        let _assert = tracing::info_span!("assert").entered();
        // For c=f_s=width=1, s_i[t]=exp(-(t-r_i)^2)/r_i.
        // Correctly rounded analytic e^-1 and e^-4 references. Rust exp has
        // unspecified precision; Miri deliberately perturbs each call by up
        // to four ULP (rust-lang/miri src/math.rs::host_unary_float_op).
        // One host-libm rounding ULP plus those four gives this five-ULP
        // bound. Halving is exact, and the exp(0) peaks are checked exactly.
        let e1 = 0.367_879_441_171_442_33_f64;
        let e4 = 0.018_315_638_888_734_18_f64;
        assert_eq!(data.shape(), [2, 1, 4]);
        assert_eq!(data[[0, 0, 1]], 1.0);
        assert_eq!(data[[1, 0, 2]], 0.5);
        let expected = [e1, 1.0, e1, e4, 0.5 * e4, 0.5 * e1, 0.5, 0.5 * e1];
        for (&actual, reference) in data.as_slice().unwrap().iter().zip(expected) {
            assert!(actual.to_bits().abs_diff(reference.to_bits()) <= 5);
        }
    }

    #[test]
    fn analytic_relative_delays_preserve_coherent_time_energy() {
        let _capture = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .without_time()
                .with_test_writer()
                .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
                .finish(),
        );
        let _test = tracing::info_span!("analytic_pam_energy").entered();
        let (data, strided, invalid_channels, elements, cfg) = {
            let _arrange = tracing::info_span!("arrange").entered();
            let data = Array3::from_shape_vec(
                (2, 1, 11),
                (1..=11)
                    .map(f64::from)
                    .chain((1..=11).map(|value| 10.0 * f64::from(value)))
                    .collect(),
            )
            .unwrap();
            let strided = Array3::new(
                leto::Layout::try_new([2, 1, 11], [1, 22, 2], 0).unwrap(),
                leto::VecStorage::new(
                    (1..=11)
                        .flat_map(|value| [f64::from(value), 10.0 * f64::from(value)])
                        .collect(),
                ),
            )
            .unwrap();
            let elements = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
            let cfg = PamMonitorConfig {
                sound_speed_m_s: 1.0,
                origin_m: [0.0; 3],
                spacing_m: 1.0,
                nx: 3,
                nz: 1,
            };
            (data, strided, Array3::zeros((2, 0, 11)), elements, cfg)
        };
        let (maps, invalid) = {
            let _act = tracing::info_span!("act").entered();
            (
                [data, strided]
                    .map(|data| passive_acoustic_map(&data, &elements, 1.0, &cfg).unwrap()),
                passive_acoustic_map(&invalid_channels, &elements, 1.0, &cfg),
            )
        };
        let _assert = tracing::info_span!("assert").entered();
        // Eleven samples exercise complete chunks and nonzero remainders.
        // Pixels 0/1 have advances [0,1]: sum_{j=0..9}(21+11j)^2 + 11^2.
        // Pixel 2 has advances [1,0]: sum_{j=0..9}(12+11j)^2 + 110^2.
        // All values and squares are exact binary64 integers, including the
        // final partially overlapping sample; no numerical tolerance enters.
        for map in maps {
            assert_eq!(map.shape(), [3, 1]);
            assert_eq!(map[[0, 0]], 59_806.0);
            assert_eq!(map[[1, 0]], 59_806.0);
            assert_eq!(map[[2, 0]], 59_905.0);
        }
        match invalid.unwrap_err() {
            KwaversError::InvalidInput(message) => {
                assert!(message.starts_with("PAM: invalid first-channel shape:"));
            }
            error => panic!("unexpected invalid-channel error: {error}"),
        }
    }

    /// Ring of `n` elements in the x–z plane at y = `y_slice`, radius `r`,
    /// centred at `(cx, y_slice, cz)`.
    fn ring_xz(n: usize, r: f64, cx: f64, y_slice: f64, cz: f64) -> Vec<[f64; 3]> {
        (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                [cx + r * a.cos(), y_slice, cz + r * a.sin()]
            })
            .collect()
    }

    #[test]
    fn pam_localizes_a_point_emitter() {
        let c = 1500.0;
        let fs = 2.0e6; // 2 MHz
        let spacing = 1.0e-3;
        let (nx, nz) = (24usize, 24usize);
        let y_slice = 0.0;
        // Image grid centred at origin: pixel (0,0) at (-12mm, 0, -12mm).
        let origin = [
            -(nx as f64) / 2.0 * spacing,
            y_slice,
            -(nz as f64) / 2.0 * spacing,
        ];
        let cfg = PamMonitorConfig {
            sound_speed_m_s: c,
            origin_m: origin,
            spacing_m: spacing,
            nx,
            nz,
        };
        // Emitter at pixel (15, 9).
        let (sx, sz) = (15usize, 9usize);
        let source = pixel_position(&cfg, sx, sz);
        let elements = ring_xz(24, 0.03, 0.0, y_slice, 0.0);

        let n_samples = 400;
        let data = synthesize_emission(source, &elements, fs, c, n_samples, 3.0).unwrap();
        let map = passive_acoustic_map(&data, &elements, fs, &cfg).unwrap();

        // Argmax pixel.
        let (mut pk_i, mut pk_k, mut pk_v) = (0usize, 0usize, f64::NEG_INFINITY);
        for ix in 0..nx {
            for iz in 0..nz {
                if map[[ix, iz]] > pk_v {
                    pk_v = map[[ix, iz]];
                    pk_i = ix;
                    pk_k = iz;
                }
            }
        }
        // Localization within one pixel of the true emitter.
        let di = (pk_i as isize - sx as isize).abs();
        let dk = (pk_k as isize - sz as isize).abs();
        assert!(
            di <= 1 && dk <= 1,
            "PAM peak ({pk_i},{pk_k}) must localize the emitter ({sx},{sz})"
        );
        // Peak must dominate the off-source background.
        let mean = map.iter().sum::<f64>() / (nx * nz) as f64;
        assert!(
            pk_v > 3.0 * mean,
            "PAM peak {pk_v} must exceed 3× mean {mean}"
        );
    }

    #[test]
    fn wrong_pixel_is_incoherent() {
        // Energy at the true source must exceed energy at a far pixel — the
        // coherent-vs-incoherent contrast that makes PAM a source map.
        let c = 1500.0;
        let fs = 2.0e6;
        let spacing = 1.0e-3;
        let (nx, nz) = (24usize, 24usize);
        let origin = [
            -(nx as f64) / 2.0 * spacing,
            0.0,
            -(nz as f64) / 2.0 * spacing,
        ];
        let cfg = PamMonitorConfig {
            sound_speed_m_s: c,
            origin_m: origin,
            spacing_m: spacing,
            nx,
            nz,
        };
        let source = pixel_position(&cfg, 12, 12);
        let elements = ring_xz(24, 0.03, 0.0, 0.0, 0.0);
        let data = synthesize_emission(source, &elements, fs, c, 400, 3.0).unwrap();
        let map = passive_acoustic_map(&data, &elements, fs, &cfg).unwrap();
        assert!(
            map[[12, 12]] > map[[2, 2]] * 2.0,
            "source pixel must out-focus a far pixel"
        );
    }

    #[test]
    fn rejects_mismatched_element_count() {
        let cfg = PamMonitorConfig {
            sound_speed_m_s: 1500.0,
            origin_m: [0.0; 3],
            spacing_m: 1e-3,
            nx: 4,
            nz: 4,
        };
        let data = Array3::<f64>::zeros((8, 1, 50));
        let positions = vec![[0.0; 3]; 6];
        assert!(passive_acoustic_map(&data, &positions, 2e6, &cfg).is_err());
    }
}
