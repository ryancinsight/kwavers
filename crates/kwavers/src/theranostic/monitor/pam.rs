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
    let mut tof = vec![0.0_f64; n_elements];
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
            let mut energy = 0.0_f64;
            for o in 0..n_samples {
                let mut coherent = 0.0_f64;
                for (i, &t) in tof.iter().enumerate() {
                    let shift = ((t - tof_min) * sample_rate).round() as usize;
                    let idx = o + shift;
                    if idx < n_samples {
                        coherent += sensor_data[[i, 0, idx]];
                    }
                }
                energy += coherent * coherent;
            }
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
    for (i, &e) in element_positions.iter().enumerate() {
        // 1/r geometric spreading + TOF delay.
        let r = distance(e, source_m).max(0.5 / sample_rate * sound_speed_m_s);
        let centre = distance(e, source_m) / sound_speed_m_s * sample_rate;
        let amplitude = 1.0 / r;
        for t in 0..n_samples {
            let u = (t as f64 - centre) / pulse_halfwidth_samples;
            data[[i, 0, t]] = amplitude * (-u * u).exp();
        }
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

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