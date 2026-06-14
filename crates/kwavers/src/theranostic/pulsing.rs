//! Low-dose pulsing scheme: sparse transmit subsets + therapy/imaging interleave.
//!
//! # Why sparse transmit subsets reduce dose
//!
//! A 1024-element hemispherical array can image the brain by firing every
//! element on every imaging pulse, but that delivers the full array energy each
//! frame. Transmission/echo tomography only needs the *union* of transmit angles
//! over a reconstruction window, not every element on every pulse. Firing a
//! decimated subset per pulse and rotating the subset across pulses delivers the
//! same angular coverage over a window while cutting per-pulse acoustic output —
//! the standard low-dose sparse-acquisition strategy (compressive / sparse-view
//! sampling, e.g. Hauptmann 2018; sparse-view CT, Sidky & Pan 2008).
//!
//! The decimation used here is **modular interleaving**: subset `k` of `K`
//! contains elements `{ i : i mod K == k }`. This guarantees:
//! - every subset is the maximally-spread `1/K` decimation of the aperture
//!   (no clustered holes), and
//! - the union of all `K` subsets is the complete aperture (lossless coverage).
//!
//! # Interleaving therapy and imaging
//!
//! Therapy is delivered as focused bursts; imaging pulses are inserted
//! periodically to reconstruct the monitored slice and watch the lesion grow.
//! [`interleave_schedule`] produces the explicit per-pulse plan (kind, active
//! transmit elements, wall-clock time) that the example loop replays against the
//! forward solver.

/// What a single pulse in the schedule does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PulseKind {
    /// Focused high-amplitude burst that deposits therapeutic energy at the target.
    Therapy,
    /// Low-amplitude diagnostic pulse whose echoes reconstruct the monitored slice.
    Imaging,
}

/// One scheduled pulse.
#[derive(Debug, Clone, PartialEq)]
pub struct PulseFrame {
    /// Sequential pulse index over the whole schedule.
    pub index: usize,
    /// Therapy or imaging.
    pub kind: PulseKind,
    /// Indices of the array elements that transmit on this pulse.
    ///
    /// Therapy frames fire the full active aperture (all elements supplied to the
    /// scheduler); imaging frames fire one rotating sparse subset.
    pub tx_elements: Vec<usize>,
    /// Wall-clock start time of this pulse [s], accumulated from the per-kind
    /// pulse-repetition intervals.
    pub time_s: f64,
}

/// Build `n_subsets` maximally-spread sparse transmit subsets of an
/// `n_elements`-element aperture via modular interleaving.
///
/// Subset `k` is `{ i : 0 ≤ i < n_elements, i mod n_subsets == k }`.
///
/// # Guarantees (verified in tests)
/// - Returns exactly `n_subsets` subsets.
/// - The subsets are disjoint and their union is `0..n_elements` (lossless).
/// - Each subset has `⌈n_elements/n_subsets⌉` or `⌊n_elements/n_subsets⌋`
///   elements (balanced) and is the `1/n_subsets` decimation (maximally spread).
///
/// Returns an empty vector when `n_elements == 0` or `n_subsets == 0`.
#[must_use]
pub fn sparse_transmit_subsets(n_elements: usize, n_subsets: usize) -> Vec<Vec<usize>> {
    if n_elements == 0 || n_subsets == 0 {
        return Vec::new();
    }
    (0..n_subsets)
        .map(|k| (k..n_elements).step_by(n_subsets).collect())
        .collect()
}

/// Interleave focused therapy bursts with sparse-subset imaging pulses into an
/// explicit pulse schedule.
///
/// Each *cycle* emits `therapy_per_cycle` therapy frames (full aperture) followed
/// by `imaging_per_cycle` imaging frames, each imaging frame drawing the next
/// rotating sparse subset (wrapping through `subsets`). Wall-clock time advances
/// by `therapy_pri_s` per therapy frame and `imaging_pri_s` per imaging frame,
/// where PRI is the pulse-repetition interval.
///
/// # Arguments
/// - `full_aperture`  — element indices fired by every therapy frame.
/// - `subsets`        — rotating sparse transmit subsets (see
///   [`sparse_transmit_subsets`]); imaging frames are skipped if empty.
/// - `therapy_per_cycle`, `imaging_per_cycle` — frames of each kind per cycle.
/// - `n_cycles`       — number of therapy/imaging cycles.
/// - `therapy_pri_s`, `imaging_pri_s` — pulse-repetition intervals [s].
///
/// # Guarantees (verified in tests)
/// - Frame count is `n_cycles * (therapy_per_cycle + imaging_per_cycle)` (minus
///   imaging frames when `subsets` is empty).
/// - `time_s` is non-decreasing and starts at 0.
/// - Imaging frames consume subsets in round-robin order.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn interleave_schedule(
    full_aperture: &[usize],
    subsets: &[Vec<usize>],
    therapy_per_cycle: usize,
    imaging_per_cycle: usize,
    n_cycles: usize,
    therapy_pri_s: f64,
    imaging_pri_s: f64,
) -> Vec<PulseFrame> {
    let mut frames = Vec::new();
    let mut t = 0.0_f64;
    let mut index = 0_usize;
    let mut subset_cursor = 0_usize;

    for _cycle in 0..n_cycles {
        for _ in 0..therapy_per_cycle {
            frames.push(PulseFrame {
                index,
                kind: PulseKind::Therapy,
                tx_elements: full_aperture.to_vec(),
                time_s: t,
            });
            index += 1;
            t += therapy_pri_s;
        }
        if subsets.is_empty() {
            continue;
        }
        for _ in 0..imaging_per_cycle {
            let subset = &subsets[subset_cursor % subsets.len()];
            subset_cursor += 1;
            frames.push(PulseFrame {
                index,
                kind: PulseKind::Imaging,
                tx_elements: subset.clone(),
                time_s: t,
            });
            index += 1;
            t += imaging_pri_s;
        }
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn subsets_are_disjoint_and_cover_aperture_losslessly() {
        let n = 1024;
        let k = 8;
        let subsets = sparse_transmit_subsets(n, k);
        assert_eq!(subsets.len(), k);

        // Balanced sizes: ⌈1024/8⌉ = 128 each.
        for s in &subsets {
            assert_eq!(s.len(), 128, "balanced decimation");
        }

        // Disjoint + union == full aperture.
        let mut union = BTreeSet::new();
        for s in &subsets {
            for &e in s {
                assert!(union.insert(e), "subsets must be disjoint");
            }
        }
        let expected: BTreeSet<usize> = (0..n).collect();
        assert_eq!(union, expected, "union must cover the whole aperture");
    }

    #[test]
    fn subsets_are_maximally_spread_decimation() {
        // Subset k must be the strided 1/K decimation: consecutive elements
        // differ by exactly K.
        let subsets = sparse_transmit_subsets(20, 4);
        assert_eq!(subsets[0], vec![0, 4, 8, 12, 16]);
        assert_eq!(subsets[1], vec![1, 5, 9, 13, 17]);
        assert_eq!(subsets[3], vec![3, 7, 11, 15, 19]);
    }

    #[test]
    fn degenerate_inputs_yield_empty() {
        assert!(sparse_transmit_subsets(0, 4).is_empty());
        assert!(sparse_transmit_subsets(64, 0).is_empty());
    }

    #[test]
    fn schedule_counts_kinds_and_orders_time() {
        let aperture: Vec<usize> = (0..16).collect();
        let subsets = sparse_transmit_subsets(16, 4);
        let frames = interleave_schedule(&aperture, &subsets, 3, 1, 5, 1e-3, 5e-4);

        // 5 cycles × (3 therapy + 1 imaging) = 20 frames.
        assert_eq!(frames.len(), 20);
        let n_therapy = frames.iter().filter(|f| f.kind == PulseKind::Therapy).count();
        let n_imaging = frames.iter().filter(|f| f.kind == PulseKind::Imaging).count();
        assert_eq!(n_therapy, 15);
        assert_eq!(n_imaging, 5);

        // Indices are sequential and times non-decreasing, starting at 0.
        assert_eq!(frames[0].time_s, 0.0);
        for w in frames.windows(2) {
            assert_eq!(w[1].index, w[0].index + 1);
            assert!(w[1].time_s >= w[0].time_s);
        }

        // Therapy frames fire the full aperture; imaging frames fire a subset.
        let first_imaging = frames.iter().find(|f| f.kind == PulseKind::Imaging).unwrap();
        assert_eq!(first_imaging.tx_elements.len(), 4);
        assert_eq!(
            frames[0].tx_elements, aperture,
            "therapy fires full aperture"
        );
    }

    #[test]
    fn imaging_frames_rotate_through_subsets_round_robin() {
        let aperture: Vec<usize> = (0..16).collect();
        let subsets = sparse_transmit_subsets(16, 4);
        // 1 therapy + 1 imaging per cycle, 6 cycles → imaging subset cursor wraps.
        let frames = interleave_schedule(&aperture, &subsets, 1, 1, 6, 1e-3, 5e-4);
        let imaging: Vec<&PulseFrame> =
            frames.iter().filter(|f| f.kind == PulseKind::Imaging).collect();
        assert_eq!(imaging.len(), 6);
        // Cursor order: subset 0,1,2,3,0,1.
        assert_eq!(imaging[0].tx_elements, subsets[0]);
        assert_eq!(imaging[3].tx_elements, subsets[3]);
        assert_eq!(imaging[4].tx_elements, subsets[0], "wraps round-robin");
        assert_eq!(imaging[5].tx_elements, subsets[1]);
    }

    #[test]
    fn empty_subsets_skips_imaging_but_keeps_therapy() {
        let aperture: Vec<usize> = (0..8).collect();
        let frames = interleave_schedule(&aperture, &[], 2, 3, 4, 1e-3, 5e-4);
        assert_eq!(frames.len(), 8, "4 cycles × 2 therapy, imaging skipped");
        assert!(frames.iter().all(|f| f.kind == PulseKind::Therapy));
    }
}
