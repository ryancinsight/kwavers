//! Multi-frame Microbubble Tracking via Hungarian Algorithm
//!
//! # Overview
//!
//! Links bubble detections across consecutive frames into continuous tracks using
//! the linear assignment problem (Hungarian algorithm). Each frame-to-frame
//! assignment minimises total squared displacement, subject to a maximum distance
//! gate that prevents physically unreasonable links.
//!
//! # Theorem: Hungarian Algorithm (Kuhn 1955; Munkres 1957)
//!
//! **Problem**: Given a cost matrix C ∈ ℝ^{m×n} (m detections at frame t,
//! n at t+1), find a complete matching that minimises Σᵢ C[i, σ(i)].
//!
//! **Algorithm** (O(n³), augmenting-path formulation):
//! ```text
//! Pad C to square n = max(m, n); fill dummy entries with BIG_M.
//! 1. Row reduce:    C[i,j] ← C[i,j] − min_j C[i,j]
//! 2. Column reduce: C[i,j] ← C[i,j] − min_i C[i,j]
//! 3. Find minimum line cover of all zeros (König's theorem):
//!    a. Mark unmatched rows; alternate-path propagation.
//!    b. Lines = marked cols ∪ unmarked rows.
//! 4. If #lines < n:
//!    δ = min uncovered C[i,j]
//!    subtract δ from uncovered; add δ to doubly-covered; goto 3.
//! 5. Extract assignment from independent zeros.
//! 6. Discard dummy-row or dummy-column assignments.
//! ```
//!
//! **Track lifecycle**:
//! - *Matched*: extend existing track with new detection.
//! - *New detection*: spawn fresh track.
//! - *Lost*: retain track for at most `max_gap` frames before termination.
//!
//! **Localization precision** (Thompson et al. 2002):
//! ```text
//! σ_loc ≈ PSF_FWHM / (2√(2·ln2) · √SNR)
//! ```
//!
//! # References
//!
//! - Kuhn, H. W. (1955). The Hungarian method for the assignment problem.
//!   *Naval Research Logistics Quarterly* 2(1–2):83–97.
//! - Munkres, J. (1957). Algorithms for the assignment and transportation problems.
//!   *J. SIAM* 5(1):32–38.
//! - Crocker, J. C., & Grier, D. G. (1996). Methods of digital video microscopy for
//!   colloidal studies. *J. Colloid Interface Sci.* 179(1):298–310.
//!   DOI: 10.1006/jcis.1996.0217
//! - Thompson, R. E., Larson, D. R., & Webb, W. W. (2002). Precise nanometer
//!   localization analysis for individual fluorescent probes.
//!   *Biophys. J.* 82(5):2775–2783.

use crate::clinical::imaging::functional_ultrasound::ulm::microbubble_detection::BubbleDetection;

// ─── Data types ───────────────────────────────────────────────────────────────

/// A linked trajectory of a single microbubble across multiple frames.
#[derive(Debug, Clone)]
pub struct BubbleTrack {
    /// Unique track identifier
    pub id: usize,
    /// Detections in chronological order
    pub detections: Vec<BubbleDetection>,
    /// Frame index of the last update
    pub last_frame: usize,
    /// Number of consecutive frames with no matching detection
    pub gap: usize,
    /// Whether this track is still active
    pub active: bool,
}

impl BubbleTrack {
    fn new(id: usize, det: BubbleDetection) -> Self {
        let frame = det.frame;
        Self {
            id,
            detections: vec![det],
            last_frame: frame,
            gap: 0,
            active: true,
        }
    }

    /// Instantaneous velocity estimate (pixels/frame) from the last two detections.
    ///
    /// Returns `None` if fewer than two detections are present.
    #[must_use]
    pub fn velocity(&self) -> Option<(f64, f64)> {
        let n = self.detections.len();
        if n < 2 {
            return None;
        }
        let d1 = &self.detections[n - 2];
        let d2 = &self.detections[n - 1];
        let dt = (d2.frame as f64 - d1.frame as f64).max(1.0);
        Some(((d2.x - d1.x) / dt, (d2.z - d1.z) / dt))
    }

    /// Track length in number of detected frames.
    #[must_use]
    pub fn length(&self) -> usize {
        self.detections.len()
    }
}

/// Configuration for Hungarian tracking.
#[derive(Debug, Clone)]
pub struct TrackingConfig {
    /// Maximum allowed displacement between frames \[pixels\].
    /// Detections farther than this gate are not linked.
    pub max_displacement: f64,
    /// Number of consecutive missed frames before a track is terminated.
    pub max_gap: usize,
    /// Minimum track length retained in the final output (shorter tracks are pruned).
    pub min_track_length: usize,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            max_displacement: 10.0,
            max_gap: 3,
            min_track_length: 3,
        }
    }
}

// ─── Hungarian Tracker ────────────────────────────────────────────────────────

/// Multi-frame microbubble tracker using the Hungarian (Munkres) algorithm.
#[derive(Debug)]
pub struct HungarianTracker {
    config: TrackingConfig,
    active_tracks: Vec<BubbleTrack>,
    terminated_tracks: Vec<BubbleTrack>,
    next_id: usize,
}

impl HungarianTracker {
    #[must_use]
    pub fn new(config: TrackingConfig) -> Self {
        Self {
            config,
            active_tracks: Vec::new(),
            terminated_tracks: Vec::new(),
            next_id: 0,
        }
    }

    /// Ingest a new frame of detections and update all tracks.
    ///
    /// # Algorithm
    /// ```text
    /// 1. Build cost matrix C[i,j] = ‖track_i.last − det_j‖²  (∞ if > d_max²)
    /// 2. Run Hungarian on C to get assignment σ
    /// 3. Update matched tracks; spawn new tracks for unmatched detections;
    ///    increment gap for unmatched tracks and terminate if gap > max_gap.
    /// ```
    pub fn update(&mut self, detections: &[BubbleDetection]) {
        let m = self.active_tracks.len();
        let n = detections.len();

        if m == 0 {
            // No existing tracks — spawn one per detection
            for det in detections {
                self.active_tracks
                    .push(BubbleTrack::new(self.next_id, det.clone()));
                self.next_id += 1;
            }
            return;
        }

        if n == 0 {
            // No new detections — increment gaps
            self.increment_gaps_no_detections();
            return;
        }

        // Build squared-distance cost matrix (m × n)
        let dmax2 = self.config.max_displacement * self.config.max_displacement;
        let big_m = 1e12_f64;
        let mut cost: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                let last = self.active_tracks[i].detections.last().unwrap();
                (0..n)
                    .map(|j| {
                        let dx = last.x - detections[j].x;
                        let dz = last.z - detections[j].z;
                        let d2 = dx * dx + dz * dz;
                        if d2 <= dmax2 {
                            d2
                        } else {
                            big_m
                        }
                    })
                    .collect()
            })
            .collect();

        let assignment = hungarian(&cost, m, n, big_m);
        // assignment[i] = Some(j) if track i matched detection j

        let mut matched_dets = vec![false; n];
        let mut matched_tracks = vec![false; m];

        for (i, &j_opt) in assignment.iter().enumerate() {
            if let Some(j) = j_opt {
                matched_tracks[i] = true;
                matched_dets[j] = true;
                let det = detections[j].clone();
                let frame = det.frame;
                self.active_tracks[i].detections.push(det);
                self.active_tracks[i].last_frame = frame;
                self.active_tracks[i].gap = 0;
            }
        }

        // Unmatched tracks — increment gap or terminate
        for i in 0..m {
            if !matched_tracks[i] {
                self.active_tracks[i].gap += 1;
            }
        }

        // Spawn new tracks for unmatched detections
        for (j, &matched) in matched_dets.iter().enumerate() {
            if !matched {
                self.active_tracks
                    .push(BubbleTrack::new(self.next_id, detections[j].clone()));
                self.next_id += 1;
            }
        }

        // Terminate over-gapped tracks
        self.terminate_stale();
    }

    fn increment_gaps_no_detections(&mut self) {
        for t in &mut self.active_tracks {
            t.gap += 1;
        }
        self.terminate_stale();
    }

    fn terminate_stale(&mut self) {
        let max_gap = self.config.max_gap;
        let mut remaining = Vec::new();
        for mut track in self.active_tracks.drain(..) {
            if track.gap > max_gap {
                track.active = false;
                self.terminated_tracks.push(track);
            } else {
                remaining.push(track);
            }
        }
        self.active_tracks = remaining;
    }

    /// Finalise tracking: terminate all remaining active tracks and return all tracks
    /// with length ≥ `min_track_length`.
    pub fn finalize(mut self) -> Vec<BubbleTrack> {
        for mut t in self.active_tracks.drain(..) {
            t.active = false;
            self.terminated_tracks.push(t);
        }
        let min_len = self.config.min_track_length;
        self.terminated_tracks
            .into_iter()
            .filter(|t| t.length() >= min_len)
            .collect()
    }

    /// Number of currently active tracks.
    #[must_use]
    pub fn n_active(&self) -> usize {
        self.active_tracks.len()
    }
}

// ─── Hungarian algorithm ──────────────────────────────────────────────────────

/// Solve the linear assignment problem on a rectangular cost matrix (m tracks × n detections).
///
/// # Theorem (Munkres 1957; Kuhn 1955)
///
/// The minimum-cost perfect matching on an n×n cost matrix can be found in O(n³) time
/// by successive row/column reductions and alternating-path augmentation.
///
/// # Returns
/// `assignment[i]` = `Some(j)` if track `i` was matched to detection `j`,
/// or `None` if the cost was ≥ `big_m` (gated out) or no matching was possible.
fn hungarian(cost: &[Vec<f64>], m: usize, n: usize, big_m: f64) -> Vec<Option<usize>> {
    let sz = m.max(n);
    // Pad to square sz × sz with big_m for dummy entries
    let mut c: Vec<Vec<f64>> = (0..sz)
        .map(|i| {
            let row: Vec<f64> = (0..sz)
                .map(|j| {
                    if i < m && j < n {
                        cost[i][j]
                    } else {
                        big_m
                    }
                })
                .collect();
            row
        })
        .collect();

    // Step 1: row reduce
    for row in &mut c {
        let min_val = row.iter().copied().fold(f64::MAX, f64::min);
        for v in row.iter_mut() {
            *v -= min_val;
        }
    }

    // Step 2: column reduce
    for j in 0..sz {
        let min_val = (0..sz).map(|i| c[i][j]).fold(f64::MAX, f64::min);
        for i in 0..sz {
            c[i][j] -= min_val;
        }
    }

    // Iterative cover / augment until sz lines cover all zeros
    let mut row_cover = vec![false; sz];
    let mut col_cover = vec![false; sz];
    let mut assignment = vec![usize::MAX; sz]; // assignment[row] = col
    let mut col_assignment = vec![usize::MAX; sz]; // col_assignment[col] = row

    loop {
        // Find initial assignment (greedy)
        assignment.fill(usize::MAX);
        col_assignment.fill(usize::MAX);
        for i in 0..sz {
            for j in 0..sz {
                if c[i][j].abs() < 1e-9
                    && assignment[i] == usize::MAX
                    && col_assignment[j] == usize::MAX
                {
                    assignment[i] = j;
                    col_assignment[j] = i;
                }
            }
        }

        // Count assigned rows
        let n_assigned = assignment.iter().filter(|&&a| a != usize::MAX).count();
        if n_assigned == sz {
            break;
        }

        // Minimum line cover (König's theorem via alternating path)
        row_cover.fill(false);
        col_cover.fill(false);

        // Mark unassigned rows
        let mut marked_rows: Vec<bool> = assignment.iter().map(|&a| a == usize::MAX).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..sz {
                if !marked_rows[i] {
                    continue;
                }
                for j in 0..sz {
                    if c[i][j].abs() < 1e-9 && !col_cover[j] {
                        col_cover[j] = true;
                        if col_assignment[j] != usize::MAX && !marked_rows[col_assignment[j]] {
                            marked_rows[col_assignment[j]] = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        // Lines: marked cols + unmarked rows
        for i in 0..sz {
            row_cover[i] = !marked_rows[i];
        }

        let n_lines: usize = row_cover.iter().filter(|&&r| r).count()
            + col_cover.iter().filter(|&&c| c).count();

        if n_lines >= sz {
            break;
        }

        // Find δ = min uncovered value
        let mut delta = f64::MAX;
        for i in 0..sz {
            for j in 0..sz {
                if !row_cover[i] && !col_cover[j] && c[i][j] < delta {
                    delta = c[i][j];
                }
            }
        }

        if delta >= big_m / 2.0 {
            break; // No feasible assignment
        }

        // Subtract δ from uncovered, add δ to doubly covered
        for i in 0..sz {
            for j in 0..sz {
                if !row_cover[i] && !col_cover[j] {
                    c[i][j] -= delta;
                } else if row_cover[i] && col_cover[j] {
                    c[i][j] += delta;
                }
            }
        }
    }

    // Extract results: filter out dummy assignments (j >= n or i >= m or cost ≥ big_m)
    (0..m)
        .map(|i| {
            let j = assignment[i];
            if j < n && cost[i][j] < big_m / 2.0 {
                Some(j)
            } else {
                None
            }
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clinical::imaging::functional_ultrasound::ulm::microbubble_detection::BubbleDetection;

    fn make_det(x: f64, z: f64, frame: usize) -> BubbleDetection {
        BubbleDetection {
            x,
            z,
            amplitude: 1.0,
            sigma: 1.0,
            background: 0.1,
            frame,
        }
    }

    #[test]
    fn test_linear_motion_tracking() {
        // Bubble moving at Δx=1.0, Δz=0.5 per frame over 5 frames should be linked into one track.
        let cfg = TrackingConfig {
            max_displacement: 5.0,
            max_gap: 1,
            min_track_length: 3,
        };
        let mut tracker = HungarianTracker::new(cfg);

        for frame in 0..5 {
            let det = make_det(frame as f64 * 1.0, frame as f64 * 0.5, frame);
            tracker.update(&[det]);
        }

        let tracks = tracker.finalize();
        assert_eq!(tracks.len(), 1, "Should produce exactly one track");
        assert_eq!(tracks[0].length(), 5, "Track should span 5 frames");

        // Check velocity
        let (vx, vz) = tracks[0].velocity().unwrap();
        assert!((vx - 1.0).abs() < 1e-9, "vx={vx}");
        assert!((vz - 0.5).abs() < 1e-9, "vz={vz}");
    }

    #[test]
    fn test_crossing_bubbles_no_identity_swap() {
        // Two bubbles moving in opposite x directions; Hungarian should not swap identities.
        // Bubble A: x decreasing (10→5), Bubble B: x increasing (0→5)
        // At frames 0,1,2 they should remain distinct.
        let cfg = TrackingConfig {
            max_displacement: 10.0,
            max_gap: 1,
            min_track_length: 2,
        };
        let mut tracker = HungarianTracker::new(cfg);

        // Frame 0: A at (10,5), B at (0,5)
        tracker.update(&[make_det(10.0, 5.0, 0), make_det(0.0, 5.0, 0)]);
        // Frame 1: A at (7,5), B at (3,5)
        tracker.update(&[make_det(7.0, 5.0, 1), make_det(3.0, 5.0, 1)]);
        // Frame 2: A at (4,5), B at (6,5)  — they've crossed
        tracker.update(&[make_det(4.0, 5.0, 2), make_det(6.0, 5.0, 2)]);

        let tracks = tracker.finalize();
        assert_eq!(tracks.len(), 2, "Should produce 2 tracks");

        // Each track should have 3 detections
        for t in &tracks {
            assert_eq!(t.length(), 3, "Track {} has {} dets", t.id, t.length());
        }
    }

    #[test]
    fn test_track_termination_after_max_gap() {
        // Bubble disappears after frame 1; track should be terminated after max_gap=2.
        let cfg = TrackingConfig {
            max_displacement: 5.0,
            max_gap: 2,
            min_track_length: 1,
        };
        let mut tracker = HungarianTracker::new(cfg);

        tracker.update(&[make_det(5.0, 5.0, 0)]);
        tracker.update(&[make_det(5.1, 5.0, 1)]);
        // Frames 2,3,4: no detections
        tracker.update(&[]);
        tracker.update(&[]);
        tracker.update(&[]);

        assert_eq!(
            tracker.n_active(),
            0,
            "Track should be terminated after max_gap=2 missed frames"
        );
        let tracks = tracker.finalize();
        assert_eq!(tracks.len(), 1, "Should have one terminated track");
    }

    #[test]
    fn test_hungarian_identity_assignment() {
        // Cost matrix = identity (cheapest is i=j); should return identity assignment.
        let n = 4;
        let mut cost = vec![vec![1e6_f64; n]; n];
        for i in 0..n {
            cost[i][i] = 0.0;
        }
        let assignment = hungarian(&cost, n, n, 1e12);
        for (i, a) in assignment.iter().enumerate() {
            assert_eq!(
                *a,
                Some(i),
                "Identity matrix assignment should map i→i; got {a:?} at i={i}"
            );
        }
    }

    #[test]
    fn test_hungarian_minimum_cost() {
        // Classic 3×3 assignment problem with known solution.
        //  C = [[4, 1, 3],
        //       [2, 0, 5],
        //       [3, 2, 2]]
        // Optimal assignment: row0→col1 (1), row1→col0 (2), row2→col2 (2) → total = 5
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let assignment = hungarian(&cost, 3, 3, 1e12);
        let total: f64 = assignment
            .iter()
            .enumerate()
            .map(|(i, &a)| cost[i][a.unwrap()])
            .sum();
        assert!((total - 5.0).abs() < 1e-9, "Optimal cost should be 5, got {total}");
        // Verify specific assignment
        assert_eq!(assignment[0], Some(1));
        assert_eq!(assignment[1], Some(0));
        assert_eq!(assignment[2], Some(2));
    }
}
