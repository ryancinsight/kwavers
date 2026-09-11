//! Discrete convolution of signals sampled over their own support.
//!
//! A kernel that is zero outside a window is carried as its onset index plus
//! the samples inside the window ([`SupportSamples`]), so convolving two of
//! them costs the product of their widths rather than of the grid they sit
//! on, and the onset arithmetic — the sum of onsets under convolution — lives
//! in one place. Time-domain impulse-response assembly (spatial impulse
//! responses, transducer arrays) is the consumer; kernels there are tens to
//! hundreds of samples wide, where the direct `O(n·m)` product is the right
//! instrument and an FFT of the grid length is not.
//!
//! Every product carries the grid step `dt`, so `convolve` approximates the
//! continuous convolution integral `∫a(τ)·b(t − τ)dτ` and `area` the integral
//! `∫a dt`; the factorization `∫(a⊛b)dt = ∫a dt·∫b dt` holds exactly for the
//! discrete sums.

/// A signal sampled on a uniform grid over its support only.
///
/// Sample `i` sits at grid index `first_sample + i`; everything before the
/// onset and after the last sample is zero by construction. An empty `samples`
/// is the zero signal, whatever its onset.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SupportSamples {
    /// Grid index of `samples[0]`.
    pub first_sample: usize,
    /// Samples from the onset.
    pub samples: Vec<f64>,
}

impl SupportSamples {
    /// The zero signal.
    #[must_use]
    pub fn zero() -> Self {
        Self::default()
    }

    /// `true` when no sample is carried.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Grid index one past the last sample.
    #[must_use]
    pub fn end(&self) -> usize {
        self.first_sample + self.samples.len()
    }

    /// `Σ samples · dt`, the integral over the support.
    #[must_use]
    pub fn area(&self, dt: f64) -> f64 {
        self.samples.iter().sum::<f64>() * dt
    }

    /// Discrete convolution `(self ⊛ other)·dt`: the onset is the sum of the
    /// onsets, the length `n + m − 1`, and `out[i + j] += a[i]·b[j]·dt`. Either
    /// operand empty gives the zero signal.
    #[must_use]
    pub fn convolve(&self, other: &Self, dt: f64) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::zero();
        }
        let mut samples = vec![0.0_f64; self.samples.len() + other.samples.len() - 1];
        for (i, &a) in self.samples.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let scaled = a * dt;
            for (slot, &b) in samples[i..].iter_mut().zip(&other.samples) {
                *slot += scaled * b;
            }
        }
        Self {
            first_sample: self.first_sample + other.first_sample,
            samples,
        }
    }

    /// `(self ⊛ self)·dt`, the two-way kernel of a one-way response.
    #[must_use]
    pub fn auto_convolve(&self, dt: f64) -> Self {
        self.convolve(self, dt)
    }

    /// Add `weight · other` delayed by `shift` samples, growing the support to
    /// the union of the two.
    pub fn superpose(&mut self, other: &Self, shift: usize, weight: f64) {
        if other.is_empty() {
            return;
        }
        let other_first = other.first_sample + shift;
        let other_end = other_first + other.samples.len();
        if self.is_empty() {
            self.first_sample = other_first;
            self.samples.clear();
            self.samples.resize(other.samples.len(), 0.0);
        } else {
            let first = self.first_sample.min(other_first);
            let end = self.end().max(other_end);
            if first < self.first_sample {
                let lead = self.first_sample - first;
                let mut grown = vec![0.0_f64; lead];
                grown.extend_from_slice(&self.samples);
                self.samples = grown;
                self.first_sample = first;
            }
            if end > self.end() {
                self.samples.resize(end - self.first_sample, 0.0);
            }
        }
        let offset = other_first - self.first_sample;
        for (slot, &value) in self.samples[offset..].iter_mut().zip(&other.samples) {
            *slot += weight * value;
        }
    }

    /// Multiply every sample by `factor`.
    pub fn scale(&mut self, factor: f64) {
        for sample in &mut self.samples {
            *sample *= factor;
        }
    }
}

/// Accumulate `(trace ⊛ pulse)·dt` into `out`, truncated to `out.len()`;
/// trace samples past `out.len()` cannot reach the output and are ignored.
/// Zero trace samples are skipped, so a sparse trace costs its occupied
/// samples times the pulse length.
pub fn convolve_into(out: &mut [f64], trace: &[f64], pulse: &[f64], dt: f64) {
    for (start, &value) in trace.iter().take(out.len()).enumerate() {
        if value == 0.0 {
            continue;
        }
        let scaled = value * dt;
        let span = out.len().saturating_sub(start).min(pulse.len());
        for (slot, &tap) in out[start..start + span].iter_mut().zip(pulse) {
            *slot += scaled * tap;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{convolve_into, SupportSamples};

    fn at(first_sample: usize, samples: &[f64]) -> SupportSamples {
        SupportSamples {
            first_sample,
            samples: samples.to_vec(),
        }
    }

    #[test]
    fn convolution_sums_onsets_and_matches_the_hand_computed_sequence() {
        // (1, 2) at 3 ⊛ (1, 1, 1) at 5, dt = 0.5:
        // out = 0.5·[1, 1+2, 1+2, 2] at onset 8.
        let a = at(3, &[1.0, 2.0]);
        let b = at(5, &[1.0, 1.0, 1.0]);
        let c = a.convolve(&b, 0.5);
        assert_eq!(c.first_sample, 8);
        assert_eq!(c.samples, vec![0.5, 1.5, 1.5, 1.0]);
        // Commutative.
        assert_eq!(b.convolve(&a, 0.5), c);
    }

    #[test]
    fn auto_convolution_doubles_the_onset_and_factorizes_the_area() {
        let h = at(7, &[1.0, 2.0, 0.5]);
        let dt = 0.25;
        let two_way = h.auto_convolve(dt);
        assert_eq!(two_way.first_sample, 14);
        assert_eq!(two_way.samples.len(), 5);
        assert_eq!(two_way.samples, vec![0.25, 1.0, 1.25, 0.5, 0.0625]);
        let area = h.area(dt);
        assert!((two_way.area(dt) - area * area).abs() <= 1e-15);
    }

    #[test]
    fn empty_operands_give_the_zero_signal() {
        let h = at(4, &[1.0, 1.0]);
        assert!(h.convolve(&SupportSamples::zero(), 1.0).is_empty());
        assert!(SupportSamples::zero().auto_convolve(1.0).is_empty());
        let mut acc = SupportSamples::zero();
        acc.superpose(&SupportSamples::zero(), 3, 2.0);
        assert!(acc.is_empty());
    }

    #[test]
    fn superposition_grows_the_support_to_the_union_and_weights_the_addend() {
        let mut acc = at(10, &[1.0, 1.0]);
        // Earlier and shifted: (2, 2, 2) at 6 shifted by 2 → occupies 8..11.
        acc.superpose(&at(6, &[2.0, 2.0, 2.0]), 2, 0.5);
        assert_eq!(acc.first_sample, 8);
        assert_eq!(acc.samples, vec![1.0, 1.0, 2.0, 1.0]);
        // Later, past the end: (3) at 1 shifted by 12 → index 13.
        acc.superpose(&at(1, &[3.0]), 12, 2.0);
        assert_eq!(acc.first_sample, 8);
        assert_eq!(acc.samples, vec![1.0, 1.0, 2.0, 1.0, 0.0, 6.0]);
        // Into an empty accumulator the addend's shifted onset is taken.
        let mut fresh = SupportSamples::zero();
        fresh.superpose(&at(3, &[1.5]), 4, 2.0);
        assert_eq!(fresh, at(7, &[3.0]));
    }

    #[test]
    fn convolve_into_truncates_at_the_output_and_skips_zeros() {
        let trace = [0.0, 2.0, 0.0, 0.0, 1.0, 5.0];
        let pulse = [1.0, -1.0];
        let mut out = [0.0_f64; 5];
        convolve_into(&mut out, &trace, &pulse, 2.0);
        // trace[1]=2 → out[1] += 4, out[2] -= 4; trace[4]=1 → out[4] += 2 (out[5]
        // would be -2 but is past the window); trace[5] is past the window.
        assert_eq!(out, [0.0, 4.0, -4.0, 0.0, 2.0]);
    }
}
