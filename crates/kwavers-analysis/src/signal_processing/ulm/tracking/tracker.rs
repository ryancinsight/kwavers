use super::types::{BubbleTrack, TrackingConfig};
use crate::signal_processing::ulm::microbubble_detection::BubbleDetection;

/// Multi-frame microbubble tracker using the Hungarian (Munkres) algorithm.
///
/// # References
///
/// - Kuhn, H. W. (1955). The Hungarian method for the assignment problem.
///   *Naval Research Logistics Quarterly* 2(1–2):83–97.
/// - Munkres, J. (1957). Algorithms for the assignment and transportation problems.
///   *J. SIAM* 5(1):32–38.
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
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn update(&mut self, detections: &[BubbleDetection]) {
        let m = self.active_tracks.len();
        let n = detections.len();

        if m == 0 {
            for det in detections {
                self.active_tracks
                    .push(BubbleTrack::new(self.next_id, det.clone()));
                self.next_id += 1;
            }
            return;
        }

        if n == 0 {
            self.increment_gaps_no_detections();
            return;
        }

        let dmax2 = self.config.max_displacement * self.config.max_displacement;
        let big_m = 1e12_f64;
        // Row-major cost matrix, stride `n`: one allocation instead of one per
        // track, and the solver below then walks it in memory order.
        let mut cost = vec![big_m; m * n];
        for (i, track) in self.active_tracks.iter().enumerate() {
            let last = track
                .detections
                .last()
                .expect("invariant: an active track carries at least one detection");
            for (j, detection) in detections.iter().enumerate() {
                let dx = last.x - detection.x;
                let dz = last.z - detection.z;
                let d2 = dx.mul_add(dx, dz * dz);
                cost[i * n + j] = if d2 <= dmax2 { d2 } else { big_m };
            }
        }

        let assignment = hungarian(&cost, m, n, big_m);

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

        for (track, &is_matched) in self.active_tracks.iter_mut().zip(&matched_tracks) {
            if !is_matched {
                track.gap += 1;
            }
        }

        for (j, &matched) in matched_dets.iter().enumerate() {
            if !matched {
                self.active_tracks
                    .push(BubbleTrack::new(self.next_id, detections[j].clone()));
                self.next_id += 1;
            }
        }

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
    #[must_use]
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

/// Solve the linear assignment problem on a rectangular cost matrix (m tracks × n detections).
///
/// `cost` is row-major with stride `n`.
///
/// # Theorem (Munkres 1957; Kuhn 1955)
///
/// The minimum-cost perfect matching on an n×n cost matrix can be found in O(n³) time
/// by successive row/column reductions and alternating-path augmentation.
///
/// # Returns
/// `assignment[i]` = `Some(j)` if track `i` was matched to detection `j`,
/// or `None` if the cost was ≥ `big_m` (gated out) or no matching was possible.
pub(super) fn hungarian(cost: &[f64], m: usize, n: usize, big_m: f64) -> Vec<Option<usize>> {
    let sz = m.max(n);
    if sz == 0 {
        return Vec::new();
    }
    // One dense row-major block instead of a `Vec` per row: the reductions and
    // the alternating-path scans below touch it `O(sz²)` times per iteration,
    // and the padded square keeps every gated entry at `big_m`.
    let mut c = vec![big_m; sz * sz];
    for i in 0..m {
        c[i * sz..i * sz + n].copy_from_slice(&cost[i * n..(i + 1) * n]);
    }

    // Step 1: row reduce
    for row in c.chunks_exact_mut(sz) {
        let min_val = row.iter().copied().fold(f64::MAX, f64::min);
        for v in row.iter_mut() {
            *v -= min_val;
        }
    }

    // Step 2: column reduce
    for j in 0..sz {
        let min_val = (0..sz).map(|i| c[i * sz + j]).fold(f64::MAX, f64::min);
        for i in 0..sz {
            c[i * sz + j] -= min_val;
        }
    }

    let mut row_cover = vec![false; sz];
    let mut col_cover = vec![false; sz];
    let mut assignment = vec![usize::MAX; sz];
    let mut col_assignment = vec![usize::MAX; sz];

    loop {
        assignment.fill(usize::MAX);
        col_assignment.fill(usize::MAX);
        for i in 0..sz {
            for j in 0..sz {
                if c[i * sz + j].abs() < 1e-9
                    && assignment[i] == usize::MAX
                    && col_assignment[j] == usize::MAX
                {
                    assignment[i] = j;
                    col_assignment[j] = i;
                }
            }
        }

        let n_assigned = assignment.iter().filter(|&&a| a != usize::MAX).count();
        if n_assigned == sz {
            break;
        }

        row_cover.fill(false);
        col_cover.fill(false);

        let mut marked_rows: Vec<bool> = assignment.iter().map(|&a| a == usize::MAX).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..sz {
                if !marked_rows[i] {
                    continue;
                }
                for j in 0..sz {
                    if c[i * sz + j].abs() < 1e-9 && !col_cover[j] {
                        col_cover[j] = true;
                        if col_assignment[j] != usize::MAX && !marked_rows[col_assignment[j]] {
                            marked_rows[col_assignment[j]] = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        for i in 0..sz {
            row_cover[i] = !marked_rows[i];
        }

        let n_lines: usize =
            row_cover.iter().filter(|&&r| r).count() + col_cover.iter().filter(|&&c| c).count();

        if n_lines >= sz {
            break;
        }

        let mut delta = f64::MAX;
        for i in 0..sz {
            for j in 0..sz {
                if !row_cover[i] && !col_cover[j] && c[i * sz + j] < delta {
                    delta = c[i * sz + j];
                }
            }
        }

        if delta >= big_m / 2.0 {
            break;
        }

        for i in 0..sz {
            for j in 0..sz {
                if !row_cover[i] && !col_cover[j] {
                    c[i * sz + j] -= delta;
                } else if row_cover[i] && col_cover[j] {
                    c[i * sz + j] += delta;
                }
            }
        }
    }

    (0..m)
        .map(|i| {
            let j = assignment[i];
            if j < n && cost[i * n + j] < big_m / 2.0 {
                Some(j)
            } else {
                None
            }
        })
        .collect()
}
