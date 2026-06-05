use crate::signal_processing::ulm::microbubble_detection::BubbleDetection;

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
    pub(super) fn new(id: usize, det: BubbleDetection) -> Self {
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
