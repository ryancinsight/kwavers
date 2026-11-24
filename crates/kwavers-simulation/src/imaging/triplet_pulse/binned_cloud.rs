//! Binned-distribution cloud sampling for the triplet-pulse cloud
//! path (Build 14 / B3 coupling, KW-IMG-048).
//!
//! The formation model (Builds 1–13 harness) outputs a bubble size
//! distribution on an 80-bin logarithmic radius grid (the
//! `b3_clouds.json` format: `RC_um` bin centres in µm plus
//! normalised counts per bin, physical count in bin *i* =
//! `counts[i] / sum(counts) × N_physical`). Until Build 14 the
//! imaging pipeline could only be fed prescribed monodisperse
//! clouds; this module samples a [`TripletPulseCloud`] from any
//! such binned distribution, closing the formation → imaging
//! coupling in Rust (B3 closed it in Python).
//!
//! # Sampling scheme (matches B3's `b3_couple.py`)
//!
//! * **Size-stratified, largest remainder:** the number of sampled
//!   bubbles drawn from bin *i* is the largest-remainder rounding
//!   of `fraction_i × N_sub` — bin fractions are preserved to
//!   within one bubble at any `N_sub`, deterministically, with no
//!   RNG involved in the counts.
//! * **Positions:** sampled radii are assigned to positions in a
//!   seeded shuffled order (the formation model carries no spatial
//!   size segregation); positions are uniform in a disk of
//!   `disk_radius_m` around the cloud centre (the prototype's 2 mm
//!   cloud disk).
//! * **Determinism:** all randomness comes from a local SplitMix64
//!   stream seeded by the caller; the same seed reproduces the
//!   same cloud byte-for-byte. No crate-external RNG is used.
//!
//! # Validation (B3)
//!
//! B3 imaged the coupled Kuji C = 0.2 formation cloud (count
//! median 2.772 µm; 89.4 % of counts in the median bin) end to end:
//! its corrected-metric contrast was indistinguishable from a
//! monodisperse cloud at its own median radius (|Δ| ≤ 0.5 dB at
//! every PNP) and fell monotonically — polydispersity does not
//! create the 0.26 MPa peak, and the shape is invariant over
//! N = 50 / 250 / 1000. The sampler's contract is therefore the
//! bookkeeping B3 relied on: fractions preserved, moments
//! preserved, deterministic under a fixed seed — each pinned by a
//! test below.

use super::TripletPulseCloud;

/// A bubble size distribution on a binned radius grid, in the
/// formation model's format: bin-centre radii [m] and per-bin
/// counts (any normalisation; fractions are counts / total).
#[derive(Debug, Clone, PartialEq)]
pub struct BinnedSizeDistribution {
    /// Bin-centre radii [m], ascending.
    pub bin_radii_m: Vec<f64>,
    /// Per-bin counts (ensemble weights, same length as radii).
    pub bin_counts: Vec<f64>,
}

impl BinnedSizeDistribution {
    /// Build from the `b3_clouds.json` representation: radii in µm
    /// and normalised counts.
    ///
    /// # Errors
    /// Returns a message when the lengths differ, the grid is
    /// empty, a radius is not finite and positive, a count is
    /// negative or non-finite, or the total count is zero.
    pub fn from_um(bin_radii_um: &[f64], bin_counts: &[f64]) -> Result<Self, String> {
        if bin_radii_um.is_empty() || bin_radii_um.len() != bin_counts.len() {
            return Err("binned distribution: radii and counts must be non-empty and equal length"
                .to_owned());
        }
        if bin_radii_um.iter().any(|r| !r.is_finite() || *r <= 0.0) {
            return Err("binned distribution: radii must be finite and > 0".to_owned());
        }
        if bin_counts.iter().any(|c| !c.is_finite() || *c < 0.0) {
            return Err("binned distribution: counts must be finite and >= 0".to_owned());
        }
        if bin_counts.iter().sum::<f64>() <= 0.0 {
            return Err("binned distribution: total count must be > 0".to_owned());
        }
        Ok(Self {
            bin_radii_m: bin_radii_um.iter().map(|r| r * 1e-6).collect(),
            bin_counts: bin_counts.to_vec(),
        })
    }

    /// Per-bin fractions (counts / total), summing to 1.
    #[must_use]
    pub fn fractions(&self) -> Vec<f64> {
        let total: f64 = self.bin_counts.iter().sum();
        self.bin_counts.iter().map(|c| c / total).collect()
    }

    /// Count-weighted mean radius [m].
    #[must_use]
    pub fn mean_radius_m(&self) -> f64 {
        self.fractions()
            .iter()
            .zip(&self.bin_radii_m)
            .map(|(f, r)| f * r)
            .sum()
    }

    /// Count-median radius [m]: the bin centre at which the
    /// cumulative fraction first reaches 0.5 (the formation
    /// harness's convention on a binned grid).
    #[must_use]
    pub fn median_radius_m(&self) -> f64 {
        let mut acc = 0.0;
        for (f, r) in self.fractions().iter().zip(&self.bin_radii_m) {
            acc += f;
            if acc >= 0.5 {
                return *r;
            }
        }
        *self.bin_radii_m.last().unwrap_or(&0.0)
    }
}

/// Largest-remainder rounding of `fractions × n`: integer counts
/// summing exactly to `n`, each within one of the exact share.
/// Ties in the remainder are broken toward the lower bin index,
/// so the result is deterministic.
#[must_use]
pub fn largest_remainder_counts(fractions: &[f64], n: usize) -> Vec<usize> {
    let exact: Vec<f64> = fractions.iter().map(|f| f * n as f64).collect();
    let mut counts: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
    let mut remaining = n.saturating_sub(counts.iter().sum());
    let mut order: Vec<usize> = (0..fractions.len()).collect();
    order.sort_by(|&a, &b| {
        let ra = exact[a] - exact[a].floor();
        let rb = exact[b] - exact[b].floor();
        rb.total_cmp(&ra).then(a.cmp(&b))
    });
    for &i in &order {
        if remaining == 0 {
            break;
        }
        counts[i] += 1;
        remaining -= 1;
    }
    counts
}

/// SplitMix64 stream — the sampler's only randomness source.
#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform double in [0, 1).
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sample `n_sub` bubbles from a binned distribution: stratified
/// radii (largest remainder) in seeded shuffled order, positions
/// uniform in a disk of `disk_radius_m` around `centre_m`
/// (z = centre z for every bubble — the prototype's planar cloud
/// disk). Returns `(position [m; 3], radius [m])` pairs.
#[must_use]
pub fn sample_binned_cloud(
    distribution: &BinnedSizeDistribution,
    n_sub: usize,
    seed: u64,
    centre_m: [f64; 3],
    disk_radius_m: f64,
) -> Vec<([f64; 3], f64)> {
    let counts = largest_remainder_counts(&distribution.fractions(), n_sub);
    let mut radii: Vec<f64> = Vec::with_capacity(n_sub);
    for (i, &c) in counts.iter().enumerate() {
        radii.extend(std::iter::repeat_n(distribution.bin_radii_m[i], c));
    }
    let mut rng = SplitMix64 { state: seed };
    // Fisher–Yates shuffle: radii assigned to positions in seeded
    // shuffled order (no spatial size segregation in the source
    // model).
    for i in (1..radii.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        radii.swap(i, j);
    }
    radii
        .into_iter()
        .map(|r| {
            let rad = disk_radius_m * rng.next_unit().sqrt();
            let ang = 2.0 * std::f64::consts::PI * rng.next_unit();
            (
                [
                    centre_m[0] + rad * ang.cos(),
                    centre_m[1] + rad * ang.sin(),
                    centre_m[2],
                ],
                r,
            )
        })
        .collect()
}

impl TripletPulseCloud {
    /// Populate this cloud by sampling `n_sub` bubbles from a
    /// binned formation distribution (see the module docs for the
    /// scheme). Returns the number of bubbles added.
    pub fn add_binned_cloud(
        &mut self,
        distribution: &BinnedSizeDistribution,
        n_sub: usize,
        seed: u64,
        centre_m: [f64; 3],
        disk_radius_m: f64,
    ) -> usize {
        let sampled = sample_binned_cloud(distribution, n_sub, seed, centre_m, disk_radius_m);
        let n = sampled.len();
        for (position, radius) in sampled {
            self.add_bubble(position, radius);
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kuji C = 0.2 nofrag imaging distribution, reduced to its
    /// three populated bins (B3 §1: 89.4 % @ 2.772 µm,
    /// 9.9 % @ 2.380 µm, 0.7 % @ 2.043 µm).
    fn kuji_like() -> BinnedSizeDistribution {
        BinnedSizeDistribution::from_um(&[2.043, 2.380, 2.772], &[0.007, 0.099, 0.894]).unwrap()
    }

    #[test]
    fn largest_remainder_preserves_fractions_and_total() {
        let dist = kuji_like();
        for n in [50_usize, 250, 1000] {
            let counts = largest_remainder_counts(&dist.fractions(), n);
            assert_eq!(counts.iter().sum::<usize>(), n);
            for (i, &c) in counts.iter().enumerate() {
                let exact = dist.fractions()[i] * n as f64;
                assert!(
                    (c as f64 - exact).abs() < 1.0,
                    "bin {i}: count {c} vs exact {exact}"
                );
            }
        }
    }

    #[test]
    fn sampling_is_deterministic_under_fixed_seed() {
        let dist = kuji_like();
        let a = sample_binned_cloud(&dist, 250, 11, [0.0, 0.0, 0.0], 1.0e-3);
        let b = sample_binned_cloud(&dist, 250, 11, [0.0, 0.0, 0.0], 1.0e-3);
        assert_eq!(a, b);
        let c = sample_binned_cloud(&dist, 250, 12, [0.0, 0.0, 0.0], 1.0e-3);
        assert_ne!(a, c);
        // The radius multiset is seed-independent (stratification).
        let mut ra: Vec<f64> = a.iter().map(|&(_, r)| r).collect();
        let mut rc: Vec<f64> = c.iter().map(|&(_, r)| r).collect();
        ra.sort_by(f64::total_cmp);
        rc.sort_by(f64::total_cmp);
        assert_eq!(ra, rc);
    }

    #[test]
    fn sampled_moments_match_distribution() {
        let dist = kuji_like();
        assert!((dist.median_radius_m() - 2.772e-6).abs() < 1e-12);
        let sampled = sample_binned_cloud(&dist, 1000, 7, [0.0, 0.0, 0.0], 1.0e-3);
        let mean = sampled.iter().map(|&(_, r)| r).sum::<f64>() / sampled.len() as f64;
        // Largest-remainder error on the mean is < one bin spacing / N.
        assert!(
            (mean - dist.mean_radius_m()).abs() < 5e-9,
            "sampled mean {mean} vs distribution {}",
            dist.mean_radius_m()
        );
        // All positions inside the disk.
        for (pos, _) in &sampled {
            assert!(pos[0].hypot(pos[1]) <= 1.0e-3 + 1e-15);
        }
    }

    #[test]
    fn cloud_constructor_feeds_triplet_cloud() {
        let dist = kuji_like();
        let mut cloud = TripletPulseCloud::iwasaki_2018(
            kwavers_physics::acoustics::bubble_dynamics::acoustic_pressure::PressureAmplitude::from_pascals(
                0.26e6,
            ),
        );
        let added = cloud.add_binned_cloud(&dist, 250, 3, [0.0, 0.0, 0.02], 1.0e-3);
        assert_eq!(added, 250);
        assert_eq!(cloud.bubbles.len(), 250);
        assert!(cloud.bubbles.iter().all(|&(_, r)| r > 0.0));
    }

    #[test]
    fn invalid_distributions_are_rejected() {
        assert!(BinnedSizeDistribution::from_um(&[1.0], &[0.5, 0.5]).is_err());
        assert!(BinnedSizeDistribution::from_um(&[1.0], &[0.0]).is_err());
        assert!(BinnedSizeDistribution::from_um(&[-1.0], &[1.0]).is_err());
    }
}
