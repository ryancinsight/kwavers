//! Triplet-pulse envelope DAS + image contrast (Rust port step 4, KW-IMG-046).
//!
//! This is a faithful port of the Python prototype's Stage 2 image pipeline
//! (`stage2/contrast.py` over `stage1/analyze_stage1.py::das_image`):
//! Hilbert-envelope delay-and-sum over the aperture receivers, then bubble-ROI
//! vs tissue-ROI contrast in dB. It deliberately does NOT reuse the coherent
//! RF DAS in [`crate::signal_processing::beamforming::imaging_das`]:
//! that kernel linearly interpolates raw RF and normalises by the number of
//! in-range sensors, while the prototype takes the envelope first, samples
//! nearest, and always divides by the full element count. The two algorithms
//! give different contrasts, so parity against the prototype decides — and
//! parity is what this module is for.
//!
//! # Prototype algorithm (mirrored exactly)
//!
//! * Envelope: `|hilbert(RF)|` per receiver row (FFT analytic signal, Marple
//!   1999 — [`kwavers_signal::analytic::instantaneous_envelope_2d`]).
//! * Transmit: point source at the array centre (`d_tx = hypot(x - x0, y -
//!   y_el)`), the same delay law the prototype's DAS uses.
//! * Receive: nearest sample, `idx = round((d_tx + d_rx) / c / dt)`;
//!   out-of-range samples contribute 0 but still divide by `n_elem`.
//! * Image: `img[iy, ix] = (1 / n_elem) * sum_e env[e, idx]`.
//! * Methods: 1P = phase 0; PI = mean(0, 180); 3P = mean(0, 120, 240).
//! * Filters (applied to the RF before the coherent sum, as in the
//!   prototype): 4 MHz low-pass (probe bandwidth) for the full band, or a
//!   2.61 ± 0.25 MHz band-pass for the 1.5f0 band. Both are zero-phase FFT
//!   brick-wall filters ([`kwavers_signal::FrequencyFilter`]).
//! * Contrast: bubble ROI = disk r = 2 mm around the cloud centre
//!   (19.2, 20) mm; tissue ROI = same depth band (|y - 20| <= 2 mm),
//!   lateral 28–33 mm. The background is the tissue ROI of the
//!   **tissue-only baseline run**, not of the combined run — the combined
//!   run's own tissue ROI is contaminated by the bubble's DAS sidelobes
//!   (small 10 mm aperture, no apodisation). The tissue mean is floored
//!   at 1e-30, as in the prototype.
//!
//! # Statistics — read this before quoting a number
//!
//! The prototype reports both statistics and they disagree about the
//! paper. [`TripletPulseContrast`] therefore carries both:
//!
//! * `mean_db` — mean bubble-ROI brightness vs mean tissue brightness.
//!   This is the prototype's headline number (full band, R0 = 2 µm:
//!   23.7 / 25.2 / 26.6 dB for 1P / PI / 3P; 1.5f0 band: 0.5 / 37.0 /
//!   59.6 dB). The ordering 1P < PI < 3P holds here, weakly in the full
//!   band (+2.9 dB from 1P to 3P) and strongly in the 1.5f0 band.
//! * `max_db` — **max** bubble-ROI brightness vs mean tissue brightness.
//!   This is Iwasaki 2018's own statistic (paper: 28.9 / 34.8 / 39.4 dB).
//!   With matched statistics the prototype does NOT reproduce the paper:
//!   full band, R0 = 2 µm: 37.8 / 39.7 / 39.1 dB — ordering flat, 1P
//!   +8.9 dB too high, 3P matching by coincidence; R0 = 5 µm the ordering
//!   is inverted (36.6 / 30.0 / 33.7 dB). The mean-based ordering and the
//!   max-based failure are both prototype facts; neither is paper parity.
//!   (Review 2026-11-19 §C2; re-measured from saved `stage1/out/rf_a2b*.npy`.)
//!
//! # Scope limits (fitted vs predicted stays explicit)
//!
//! * Forced cloud, no nucleation: the cloud is assumed to exist at every
//!   PNP, so contrast-vs-PNP is monotonic-falling and Kuji's 0.26 MPa
//!   peak and falling optimal cutoff are NOT expected behaviour here.
//! * Proxy metric, not Kuji's metric: this contrast is a B-mode-style
//!   ROI-mean (or ROI-max) ratio on 3P envelope images. Kuji 2024's
//!   quantity is the contrast between cavitation and tissue regions in
//!   the **3P/1P distribution** (per-pixel 1P ÷ 3P image, ROI 62–68 mm ×
//!   ±3 mm, before/after frames). The 2.45 MHz "peak reproduction" is a
//!   peak in this proxy, not in Kuji's Fig. 14 quantity (review §C3,
//!   §K5). Do not claim Kuji-metric parity from these numbers.
//!   **Superseded for the papers' own quantities (Build 14,
//!   KW-IMG-047):** the corrected Kuji metric (coherent DAS of
//!   high-pass-filtered RF, per-pixel linear 1P/3P reduction,
//!   before/after ROI means, ensemble statistics) and the corrected
//!   Iwasaki statistic live in
//!   [`super::triplet_pulse_metrics`]; use them for any Kuji/Iwasaki
//!   claim. This module remains only for prototype parity.
//! * Transmit inconsistency, inherited from the prototype (review §C4,
//!   D13): the PSTD transmit is a plane wave (32 elements fired
//!   simultaneously, no delay law) while this DAS assumes a point-source
//!   transmit from the array centre. At the centred cloud the two agree;
//!   off-centre they differ by up to ~0.44 µs (~0.77 cycles), smearing
//!   the lateral PSF. The matched diverging-wave pair measures +0.7 dB
//!   on centred-ROI contrast (D13), so the contrast numbers are only
//!   weakly affected — but the "arc" is partly a transmit-model artefact,
//!   not pure aperture physics. This module keeps the prototype's delay
//!   law for parity and documents the inconsistency rather than silently
//!   "fixing" it.
//! * Absolute dB are floor-limited: the prototype tissue is one
//!   realisation of ±4% sound-speed-only speckle with no attenuation,
//!   so the tissue background sits near the numerical floor in 3P. Only
//!   the ordering and the shape are meaningful, not the absolute level
//!   (review §B3; D5 real-speckle background: far-ROI LP 4 MHz
//!   suppression −18.8 ± 1.2 dB vs the paper's −17.8 ± 0.42 dB).
//!
//! # References
//!
//! * Iwasaki et al. 2018, JJAP 57, 07LF12 (contrast definition: max bubble
//!   brightness vs average background brightness).
//! * Kuji et al. 2024, JJAP 63, 04SP10 (3P/1P distribution metric — NOT
//!   computed here; see the proxy-metric note above).
//! * Marple (1999), "Computing the discrete-time analytic signal via FFT",
//!   IEEE Trans. Signal Process. (Hilbert envelope).

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_signal::analytic::instantaneous_envelope_2d;
use leto::{Array2, ArrayView2};

/// Floor applied to the tissue-ROI mean before the dB ratio, matching the
/// prototype (`max(tm, 1e-30)`). Envelope images are non-negative, so a
/// zero tissue mean means an all-zero background, not a negative one.
const TISSUE_MEAN_FLOOR: f64 = 1e-30;

/// Geometry of one triplet-pulse acquisition, in the prototype's coordinates.
///
/// Lateral axis `x`, depth axis `y`; the array sits at depth `y_el` and the
/// transmit origin is the array centre `x0`. Element lateral positions are
/// evenly spaced across `aperture_m`, centred on `x0` — exactly the
/// prototype's `linspace((nx*dx - ap)/2, (nx*dx + ap)/2, n_elem)`.
#[derive(Debug, Clone)]
pub struct TripletPulseGeometry {
    /// Lateral position of each aperture element [m].
    pub element_x_m: Vec<f64>,
    /// Depth of the array face [m] (prototype: 2 mm).
    pub element_depth_m: f64,
    /// Lateral position of the transmit origin / array centre [m].
    pub transmit_origin_x_m: f64,
    /// Sound speed used for the DAS delays [m/s] (prototype: 1540, tissue).
    pub sound_speed_m_s: f64,
    /// RF sampling interval [s] (prototype: 30 ns).
    pub dt_s: f64,
}

impl TripletPulseGeometry {
    /// Build the geometry from the array description.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when `n_elem` is 0, the
    /// aperture or domain width is not finite and positive, the aperture
    /// exceeds the domain width, or the sound speed / sampling interval
    /// is not finite and positive.
    pub fn new(
        n_elem: usize,
        aperture_m: f64,
        domain_width_m: f64,
        element_depth_m: f64,
        sound_speed_m_s: f64,
        dt_s: f64,
    ) -> KwaversResult<Self> {
        if n_elem == 0 {
            return Err(KwaversError::InvalidInput(
                "triplet_pulse: n_elem must be > 0".to_owned(),
            ));
        }
        if !aperture_m.is_finite() || aperture_m <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "triplet_pulse: aperture_m must be finite and > 0; got {aperture_m}"
            )));
        }
        if !domain_width_m.is_finite() || domain_width_m <= 0.0 || aperture_m > domain_width_m {
            return Err(KwaversError::InvalidInput(format!(
                "triplet_pulse: domain_width_m must be finite, > 0 and >= aperture_m; got {domain_width_m} for aperture {aperture_m}"
            )));
        }
        if !sound_speed_m_s.is_finite() || sound_speed_m_s <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "triplet_pulse: sound_speed_m_s must be finite and > 0; got {sound_speed_m_s}"
            )));
        }
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "triplet_pulse: dt_s must be finite and > 0; got {dt_s}"
            )));
        }
        let x0 = domain_width_m / 2.0;
        let lo = (domain_width_m - aperture_m) / 2.0;
        let hi = (domain_width_m + aperture_m) / 2.0;
        let element_x_m = if n_elem == 1 {
            vec![x0]
        } else {
            (0..n_elem)
                .map(|e| lo + (hi - lo) * e as f64 / (n_elem - 1) as f64)
                .collect()
        };
        Ok(Self {
            element_x_m,
            element_depth_m,
            transmit_origin_x_m: x0,
            sound_speed_m_s,
            dt_s,
        })
    }

    /// Stage 1 geometry: 32 elements over a 10 mm aperture on a 38.4 mm
    /// wide domain (`nx * dx` = 256 × 0.15 mm), array face at 2 mm depth,
    /// tissue sound speed 1540 m/s, 30 ns sampling.
    ///
    /// # Errors
    /// Propagates the [`Self::new`] validation errors (unreachable for
    /// these constants).
    pub fn stage1() -> KwaversResult<Self> {
        Self::new(32, 10.0e-3, 38.4e-3, 2.0e-3, 1540.0, 30.0e-9)
    }

    /// Number of aperture elements.
    #[must_use]
    pub fn n_elem(&self) -> usize {
        self.element_x_m.len()
    }

    /// Point-source transmit travel time from the array centre to `(x, y)`.
    #[must_use]
    pub fn transmit_delay_s(&self, x_m: f64, y_m: f64) -> f64 {
        let dx = x_m - self.transmit_origin_x_m;
        let dy = y_m - self.element_depth_m;
        (dx * dx + dy * dy).sqrt() / self.sound_speed_m_s
    }
}

/// 1P / PI / 3P envelope images (or RF) from the four phase transmissions.
#[derive(Debug, Clone)]
pub struct TripletPulseMethods<T> {
    /// Single-pulse (phase 0° only).
    pub one_pulse: T,
    /// Pulse inversion (mean of 0° and 180°).
    pub pulse_inversion: T,
    /// Triplet pulse (mean of 0°, 120° and 240°).
    pub triplet_pulse: T,
}

/// Combine the four phase-transmission RF channel records into 1P / PI / 3P.
///
/// Each input is `(n_elem, n_samples)`; each output is the same shape.
/// 1P = phase 0; PI = mean(0, 180); 3P = mean(0, 120, 240). This mirrors
/// `kwavers_imaging::ultrasound::triplet_pulse::combine_echoes` for
/// single-receiver traces, applied here per channel row; it is implemented
/// locally because this crate does not depend on `kwavers-imaging`.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] when the four inputs differ in
/// shape, are empty, or contain non-finite samples.
pub fn combine_methods(
    ph0: ArrayView2<'_, f64>,
    ph120: ArrayView2<'_, f64>,
    ph240: ArrayView2<'_, f64>,
    ph180: ArrayView2<'_, f64>,
) -> KwaversResult<TripletPulseMethods<Array2<f64>>> {
    let shape = ph0.shape();
    if shape[0] == 0 || shape[1] == 0 {
        return Err(KwaversError::InvalidInput(
            "triplet_pulse: phase RF must be non-empty".to_owned(),
        ));
    }
    for (name, other) in [("ph120", ph120), ("ph240", ph240), ("ph180", ph180)] {
        if other.shape() != shape {
            return Err(KwaversError::InvalidInput(format!(
                "triplet_pulse: {name} shape {:?} does not match ph0 shape {shape:?}",
                other.shape()
            )));
        }
    }
    for (name, view) in [
        ("ph0", ph0),
        ("ph120", ph120),
        ("ph240", ph240),
        ("ph180", ph180),
    ] {
        if !view.iter().all(|v| v.is_finite()) {
            return Err(KwaversError::InvalidInput(format!(
                "triplet_pulse: {name} must contain only finite values"
            )));
        }
    }
    let [n_elem, n_samples] = shape;
    let mut one_pulse = Array2::zeros([n_elem, n_samples]);
    let mut pulse_inversion = Array2::zeros([n_elem, n_samples]);
    let mut triplet_pulse = Array2::zeros([n_elem, n_samples]);
    for e in 0..n_elem {
        for t in 0..n_samples {
            let v0 = ph0[[e, t]];
            let v120 = ph120[[e, t]];
            let v240 = ph240[[e, t]];
            let v180 = ph180[[e, t]];
            one_pulse[[e, t]] = v0;
            pulse_inversion[[e, t]] = (v0 + v180) / 2.0;
            triplet_pulse[[e, t]] = (v0 + v120 + v240) / 3.0;
        }
    }
    Ok(TripletPulseMethods {
        one_pulse,
        pulse_inversion,
        triplet_pulse,
    })
}

/// Prototype-faithful envelope DAS image.
///
/// `rf` is `(n_elem, n_samples)`; `xs_m` / `ys_m` are the image grid
/// (lateral / depth, ascending). Returns the image as `(n_ys, n_xs)`
/// (row = depth, column = lateral), matching the prototype's
/// `img[iy, ix]` layout. The envelope is the Hilbert magnitude per
/// receiver row; each pixel sums the nearest envelope sample over the
/// aperture and divides by the full element count (out-of-range samples
/// contribute 0 but still divide).
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] when the RF row count differs
/// from the geometry's element count, the RF is empty, the grid is
/// empty, or any input is non-finite.
pub fn envelope_das(
    rf: ArrayView2<'_, f64>,
    geometry: &TripletPulseGeometry,
    xs_m: &[f64],
    ys_m: &[f64],
) -> KwaversResult<Array2<f64>> {
    let [n_rows, n_samples] = rf.shape();
    if n_rows == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "triplet_pulse: RF must be non-empty".to_owned(),
        ));
    }
    if n_rows != geometry.n_elem() {
        return Err(KwaversError::InvalidInput(format!(
            "triplet_pulse: RF has {n_rows} rows but the geometry has {} elements",
            geometry.n_elem()
        )));
    }
    if xs_m.is_empty() || ys_m.is_empty() {
        return Err(KwaversError::InvalidInput(
            "triplet_pulse: image grid must be non-empty".to_owned(),
        ));
    }
    if !rf.iter().all(|v| v.is_finite())
        || !xs_m.iter().all(|v| v.is_finite())
        || !ys_m.iter().all(|v| v.is_finite())
    {
        return Err(KwaversError::InvalidInput(
            "triplet_pulse: RF and grid must contain only finite values".to_owned(),
        ));
    }
    let owned =
        Array2::from_shape_vec(rf.shape(), rf.iter().copied().collect()).map_err(|error| {
            KwaversError::InvalidInput(format!("triplet_pulse: RF copy failed: {error}"))
        })?;
    let env = instantaneous_envelope_2d(&owned);
    let n_elem = geometry.n_elem() as f64;
    let mut image = Array2::zeros([ys_m.len(), xs_m.len()]);
    for (iy, &y) in ys_m.iter().enumerate() {
        for (ix, &x) in xs_m.iter().enumerate() {
            let d_tx = geometry.transmit_delay_s(x, y) * geometry.sound_speed_m_s;
            let mut acc = 0.0;
            for (e, &x_el) in geometry.element_x_m.iter().enumerate() {
                let dx = x - x_el;
                let dy = y - geometry.element_depth_m;
                let d_rx = (dx * dx + dy * dy).sqrt();
                let idx = ((d_tx + d_rx) / geometry.sound_speed_m_s / geometry.dt_s).round() as i64;
                if idx >= 0 && (idx as usize) < n_samples {
                    acc += env[[e, idx as usize]];
                }
            }
            image[[iy, ix]] = acc / n_elem;
        }
    }
    Ok(image)
}

/// A region of interest on a DAS image.
#[derive(Debug, Clone, Copy)]
pub enum ImageRoi {
    /// Disk of `radius_m` around (`center_x_m`, `center_y_m`).
    Disk {
        center_x_m: f64,
        center_y_m: f64,
        radius_m: f64,
    },
    /// Lateral band `x_lo_m..=x_hi_m` within `half_depth_m` of
    /// `center_y_m` (the prototype's tissue ROI).
    LateralBand {
        x_lo_m: f64,
        x_hi_m: f64,
        center_y_m: f64,
        half_depth_m: f64,
    },
}

impl ImageRoi {
    /// Bubble ROI of the prototype: disk r = 2 mm around (19.2, 20) mm.
    #[must_use]
    pub const fn bubble_disk() -> Self {
        Self::Disk {
            center_x_m: 19.2e-3,
            center_y_m: 20.0e-3,
            radius_m: 2.0e-3,
        }
    }

    /// Tissue ROI of the prototype: lateral 28–33 mm, |y − 20 mm| ≤ 2 mm.
    #[must_use]
    pub const fn tissue_band() -> Self {
        Self::LateralBand {
            x_lo_m: 28.0e-3,
            x_hi_m: 33.0e-3,
            center_y_m: 20.0e-3,
            half_depth_m: 2.0e-3,
        }
    }

    /// Whether the pixel at `(x_m, y_m)` belongs to this ROI.
    #[must_use]
    pub fn contains(&self, x_m: f64, y_m: f64) -> bool {
        match *self {
            Self::Disk {
                center_x_m,
                center_y_m,
                radius_m,
            } => {
                let dx = x_m - center_x_m;
                let dy = y_m - center_y_m;
                dx * dx + dy * dy <= radius_m * radius_m
            }
            Self::LateralBand {
                x_lo_m,
                x_hi_m,
                center_y_m,
                half_depth_m,
            } => x_m >= x_lo_m && x_m <= x_hi_m && (y_m - center_y_m).abs() <= half_depth_m,
        }
    }
}

/// Mean / max / count of the image pixels inside an ROI.
#[derive(Debug, Clone, Copy)]
pub struct RoiStats {
    /// Mean pixel value.
    pub mean: f64,
    /// Maximum pixel value.
    pub max: f64,
    /// Number of pixels in the ROI.
    pub count: usize,
}

/// Mean and max of the image pixels inside `roi`.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] when the image shape does not
/// match the grid lengths or the ROI contains no pixel.
pub fn roi_stats(
    image: ArrayView2<'_, f64>,
    xs_m: &[f64],
    ys_m: &[f64],
    roi: &ImageRoi,
) -> KwaversResult<RoiStats> {
    let [n_ys, n_xs] = image.shape();
    if n_ys != ys_m.len() || n_xs != xs_m.len() {
        return Err(KwaversError::InvalidInput(format!(
            "triplet_pulse: image shape ({n_ys}, {n_xs}) does not match grid ({} ys, {} xs)",
            ys_m.len(),
            xs_m.len()
        )));
    }
    let mut sum = 0.0;
    let mut max = f64::NEG_INFINITY;
    let mut count = 0_usize;
    for (iy, &y) in ys_m.iter().enumerate() {
        for (ix, &x) in xs_m.iter().enumerate() {
            if roi.contains(x, y) {
                let v = image[[iy, ix]];
                sum += v;
                if v > max {
                    max = v;
                }
                count += 1;
            }
        }
    }
    if count == 0 {
        return Err(KwaversError::InvalidInput(
            "triplet_pulse: ROI contains no pixel".to_owned(),
        ));
    }
    Ok(RoiStats {
        mean: sum / count as f64,
        max,
        count,
    })
}

/// Contrast ratio `20·log10(signal / reference)` in dB, with the
/// prototype's reference floor: `reference.max(1e-30)`.
#[must_use]
pub fn contrast_db(signal: f64, reference: f64) -> f64 {
    20.0 * (signal / reference.max(TISSUE_MEAN_FLOOR)).log10()
}

/// Bubble-vs-tissue image contrast for one method (1P, PI or 3P).
///
/// Both statistics are reported because they disagree about the paper
/// (see the module docs): `mean_db` is the prototype's headline number,
/// `max_db` is Iwasaki 2018's own statistic.
#[derive(Debug, Clone, Copy)]
pub struct TripletPulseContrast {
    /// Mean bubble-ROI brightness.
    pub bubble_mean: f64,
    /// Max bubble-ROI brightness.
    pub bubble_max: f64,
    /// Mean tissue-ROI brightness of the baseline image.
    pub tissue_mean: f64,
    /// `20·log10(bubble_mean / tissue_mean)` [dB].
    pub mean_db: f64,
    /// `20·log10(bubble_max / tissue_mean)` [dB].
    pub max_db: f64,
}

/// Contrast of one method's image against the tissue-only baseline image.
///
/// The bubble ROI is taken from `image` (the combined bubble + tissue
/// run); the tissue ROI is taken from `tissue_baseline_image` (the
/// tissue-only run), because the combined run's own tissue ROI is
/// contaminated by the bubble's DAS sidelobes. Both images share the
/// same grid.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] when either image shape does
/// not match the grid, or either ROI contains no pixel.
pub fn triplet_pulse_contrast(
    image: ArrayView2<'_, f64>,
    tissue_baseline_image: ArrayView2<'_, f64>,
    xs_m: &[f64],
    ys_m: &[f64],
    bubble_roi: &ImageRoi,
    tissue_roi: &ImageRoi,
) -> KwaversResult<TripletPulseContrast> {
    let bubble = roi_stats(image, xs_m, ys_m, bubble_roi)?;
    let tissue = roi_stats(tissue_baseline_image, xs_m, ys_m, tissue_roi)?;
    Ok(TripletPulseContrast {
        bubble_mean: bubble.mean,
        bubble_max: bubble.max,
        tissue_mean: tissue.mean,
        mean_db: contrast_db(bubble.mean, tissue.mean),
        max_db: contrast_db(bubble.max, tissue.mean),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array2;

    /// Prototype grid, built the way numpy linspace builds it (see the
    /// parity test's `linspace` note): the tissue-band boundary at
    /// y = 18 mm sits exactly on `|y - 20 mm| = 2 mm`, so a 1-ulp grid
    /// difference flips that row and moves the band count 88 <-> 99.
    fn stage1_grid() -> (Vec<f64>, Vec<f64>) {
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
        (
            linspace(5.0e-3, 33.0e-3, 57),
            linspace(12.0e-3, 36.0e-3, 49),
        )
    }

    #[test]
    fn geometry_matches_stage1_array() {
        let g = TripletPulseGeometry::stage1().unwrap();
        assert_eq!(g.n_elem(), 32);
        assert!((g.element_x_m[0] - 14.2e-3).abs() < 1e-12);
        assert!((g.element_x_m[31] - 24.2e-3).abs() < 1e-12);
        assert!((g.transmit_origin_x_m - 19.2e-3).abs() < 1e-12);
        // Centre transmit delay: 18 mm at 1540 m/s.
        let d = g.transmit_delay_s(19.2e-3, 20.0e-3);
        assert!((d - 18.0e-3 / 1540.0).abs() < 1e-15, "delay {d}");
    }

    #[test]
    fn geometry_rejects_bad_inputs() {
        assert!(TripletPulseGeometry::new(0, 10.0e-3, 38.4e-3, 2.0e-3, 1540.0, 30.0e-9).is_err());
        assert!(TripletPulseGeometry::new(32, 40.0e-3, 38.4e-3, 2.0e-3, 1540.0, 30.0e-9).is_err());
        assert!(TripletPulseGeometry::new(32, 10.0e-3, 38.4e-3, 2.0e-3, 0.0, 30.0e-9).is_err());
    }

    #[test]
    fn combine_methods_averages_phases() {
        let ph0 = Array2::from_shape_vec((1, 3), vec![3.0, 6.0, 9.0]).unwrap();
        let ph120 = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let ph240 = Array2::from_shape_vec((1, 3), vec![0.0, 3.0, 0.0]).unwrap();
        let ph180 = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let m = combine_methods(ph0.view(), ph120.view(), ph240.view(), ph180.view()).unwrap();
        assert_eq!(m.one_pulse[[0, 1]], 6.0);
        assert_eq!(m.pulse_inversion[[0, 1]], 4.0);
        assert_eq!(m.triplet_pulse[[0, 1]], 3.0);
        // Shape mismatch is an error.
        let wrong = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        assert!(combine_methods(ph0.view(), wrong.view(), ph240.view(), ph180.view()).is_err());
    }

    #[test]
    fn contrast_db_matches_definition() {
        assert!((contrast_db(2.0, 1.0) - 20.0 * 2.0_f64.log10()).abs() < 1e-12);
        // Floor: zero reference does not produce infinity.
        assert!(contrast_db(1.0, 0.0).is_finite());
    }

    #[test]
    fn roi_stats_selects_disk_and_band() {
        let (xs, ys) = stage1_grid();
        // Image value = x in mm, so the disk mean/max are geometric.
        let mut img = Array2::zeros([ys.len(), xs.len()]);
        for (iy, _) in ys.iter().enumerate() {
            for (ix, &x) in xs.iter().enumerate() {
                img[[iy, ix]] = x * 1e3;
            }
        }
        let disk = roi_stats(img.view(), &xs, &ys, &ImageRoi::bubble_disk()).unwrap();
        assert_eq!(disk.count, 50, "disk pixel count {}", disk.count);
        assert!((disk.max - 21.0).abs() < 1e-9, "disk max {}", disk.max);
        let band = roi_stats(img.view(), &xs, &ys, &ImageRoi::tissue_band()).unwrap();
        // 11 lateral columns x 8 depth rows: the y = 18 mm row sits
        // exactly on |y - 20 mm| = 2 mm and numpy's linspace puts it
        // 1 ulp outside the band (see stage1_grid).
        assert_eq!(band.count, 11 * 8, "band pixel count {}", band.count);
        assert!((band.mean - 30.5).abs() < 1e-9, "band mean {}", band.mean);
        // Empty ROI is an error.
        let empty = ImageRoi::Disk {
            center_x_m: 100.0e-3,
            center_y_m: 100.0e-3,
            radius_m: 1.0e-3,
        };
        assert!(roi_stats(img.view(), &xs, &ys, &empty).is_err());
    }

    /// A unit impulse placed at each element's true two-way sample puts
    /// the envelope-DAS peak exactly on the source pixel at full value.
    /// (Source on the image grid: x = 19.0 mm is grid index 28; the
    /// prototype cloud at 19.2 mm falls between grid columns.)
    #[test]
    fn envelope_das_localizes_impulse_source() {
        let g = TripletPulseGeometry::stage1().unwrap();
        let (sx, sy) = (19.0e-3, 20.0e-3);
        let n_samples = 1800;
        let mut rf = Array2::zeros([32, n_samples]);
        for (e, &x_el) in g.element_x_m.iter().enumerate() {
            let d_tx = g.transmit_delay_s(sx, sy);
            let dx = sx - x_el;
            let dy = sy - g.element_depth_m;
            let d_rx = (dx * dx + dy * dy).sqrt() / g.sound_speed_m_s;
            let idx = ((d_tx + d_rx) / g.dt_s).round() as usize;
            rf[[e, idx]] = 1.0;
        }
        let (xs, ys) = stage1_grid();
        let img = envelope_das(rf.view(), &g, &xs, &ys).unwrap();
        // Source pixel (x = 19.2 mm is grid index 28; y = 20 mm is index 16).
        assert!((img[[16, 28]] - 1.0).abs() < 1e-6, "peak {}", img[[16, 28]]);
        let peak = img.iter().fold(0.0_f64, |a, &b| a.max(b));
        assert!((peak - 1.0).abs() < 1e-6, "image max {peak}");
        // Out-of-range pixels are zero: a trace too short for the delay.
        let short = Array2::zeros([32, 10]);
        let img_short = envelope_das(short.view(), &g, &xs, &ys).unwrap();
        assert!(img_short.iter().all(|&v| v == 0.0));
        // Row-count mismatch is an error.
        let wrong = Array2::zeros([31, n_samples]);
        assert!(envelope_das(wrong.view(), &g, &xs, &ys).is_err());
    }
}
