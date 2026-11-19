//! Triplet-pulse / pulse-inversion filter (Kuji, Umemura & Yoshizawa,
//! JJAP 2024).
//!
//! The 3P coherent sum cancels the tissue fundamental, but the envelope
//! component produced by nonlinear propagation leaks through it — and
//! the leak grows with imaging intensity (Kuji 2024). The fix is to
//! high-pass filter the RF *before* the 3P/1P combination: the envelope
//! component is broadband and strongest at low frequencies, while the
//! bubble's 1.5th fractional harmonic sits above the fundamental. The
//! cost is bandwidth, hence depth resolution.
//!
//! Two claims from Kuji 2024 are the acceptance tests for this filter
//! (Stage 3 of the Python prototype, at 1.74 MHz rather than Kuji's
//! 2.45 MHz — at 2.45 MHz the 2nd harmonic would sit on the spatial
//! Nyquist at the prototype's grid spacing):
//! 1. the optimal high-pass cutoff *falls* as the imaging peak negative
//!    pressure rises;
//! 2. contrast peaks at PNP 0.26 MPa (the lowest intensity that still
//!    nucleates bubbles).
//!
//! The second half of the method is the 3P/1P ratio mask: after
//! beamforming, the 3P envelope image is divided pixel-wise by the 1P
//! envelope image. Tissue gives a ratio near zero (its 3P cancels);
//! bubbles keep ~1/3 of their fractional-harmonic amplitude, so the
//! ratio image masks tissue and keeps bubbles. The division runs on
//! envelopes (always non-negative), never on raw RF, and takes a floor
//! so a near-zero 1P denominator cannot blow the ratio up.

use super::frequency_filter::FrequencyFilter;
use kwavers_core::error::KwaversResult;

/// High-pass + 3P/1P ratio filter for triplet-pulse imaging.
#[derive(Debug, Default, Clone, Copy)]
pub struct TripletPulseFilter {
    /// High-pass cutoff (Hz) applied to every phase trace before the
    /// coherent sum. Kuji 2024: the optimum falls as imaging PNP rises.
    pub cutoff_hz: f64,
}

impl TripletPulseFilter {
    /// Create a filter with the given high-pass cutoff (Hz).
    #[must_use]
    pub const fn new(cutoff_hz: f64) -> Self {
        Self { cutoff_hz }
    }

    /// High-pass filter each phase transmission's RF trace.
    ///
    /// Iterates in transmission order (0°/120°/240°/180°). Each item
    /// is the result of applying [`FrequencyFilter::highpass`] to one
    /// trace. The caller can consume each filtered trace before requesting
    /// the next one, which avoids retaining an allocation for every phase.
    pub fn highpass_phases<'a, Trace: AsRef<[f64]> + 'a>(
        &self,
        traces: &'a [Trace],
        dt: f64,
    ) -> impl Iterator<Item = KwaversResult<Vec<f64>>> + use<'a, Trace> {
        let filter = FrequencyFilter::new();
        let cutoff_hz = self.cutoff_hz;
        traces
            .iter()
            .map(move |trace| filter.highpass(trace.as_ref(), dt, cutoff_hz))
    }
}

/// Pixel-wise 3P/1P ratio mask over envelope images.
///
/// `triplet_env` and `one_pulse_env` are the beamformed envelope values
/// at the same pixels (same length). Returns
/// `triplet[i] / max(one_pulse[i], floor)` per pixel: tissue tends to
/// ~0, bubbles to a finite fraction. `floor` must be positive; a good
/// choice is a small multiple of the 1P image's noise level.
/// # Panics
/// Panics if the two images differ in length or `floor` is not positive.
#[must_use]
pub fn ratio_mask(triplet_env: &[f64], one_pulse_env: &[f64], floor: f64) -> Vec<f64> {
    assert_eq!(
        triplet_env.len(),
        one_pulse_env.len(),
        "envelope images must have the same length"
    );
    assert!(floor > 0.0, "ratio floor must be positive");
    triplet_env
        .iter()
        .zip(one_pulse_env.iter())
        .map(|(&t, &o)| t / o.max(floor))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_mask_masks_tissue_keeps_bubbles() {
        // Tissue pixel: 3P ~ 0, 1P strong -> ratio ~ 0.
        // Bubble pixel: 3P ~ 1/3 of the 1P envelope -> ratio ~ 1/3.
        let triplet = vec![0.001, 0.33];
        let one_pulse = vec![1.0, 1.0];
        let ratio = ratio_mask(&triplet, &one_pulse, 1e-6);
        assert!((ratio[0] - 0.001).abs() < 1e-9);
        assert!((ratio[1] - 0.33).abs() < 1e-9);
    }

    #[test]
    fn ratio_mask_floor_bounds_silent_pixels() {
        let triplet = vec![0.5];
        let one_pulse = vec![0.0];
        let ratio = ratio_mask(&triplet, &one_pulse, 0.1);
        assert!((ratio[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn highpass_removes_dc_keeps_high_frequency() {
        let dt = 1.0 / 20.0e6;
        let n = 2000;
        // DC offset + 5 MHz tone: the high-pass at 2 MHz must remove
        // the DC and keep the tone.
        let signal: Vec<f64> = (0..n)
            .map(|i| 1.0 + (2.0 * std::f64::consts::PI * 5.0e6 * i as f64 * dt).sin())
            .collect();
        let filter = TripletPulseFilter::new(2.0e6);
        let phases = [signal.as_slice()];
        let mut outputs = filter.highpass_phases(&phases, dt);
        let out = outputs.next().expect("one phase").expect("valid trace");
        assert!(outputs.next().is_none());
        let mean = out.iter().sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "DC leaked through: mean {mean}");
        let peak = out.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(peak > 0.5, "5 MHz tone attenuated: peak {peak}");
    }

    #[test]
    fn highpass_preserves_phase_order_and_independent_trace_lengths() {
        let filter = TripletPulseFilter::new(0.0);
        let phases: [&[f64]; 3] = [&[2.0, -2.0], &[], &[-7.0]];
        let mut output = filter.highpass_phases(&phases, 0.25);

        for expected in phases {
            assert_eq!(
                output.next().expect("one result per phase").unwrap(),
                expected
            );
        }
        assert!(output.next().is_none());
    }
}
