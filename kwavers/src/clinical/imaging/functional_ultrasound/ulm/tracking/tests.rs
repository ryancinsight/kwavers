use super::tracker::{hungarian, HungarianTracker};
use super::types::TrackingConfig;
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

    let (vx, vz) = tracks[0].velocity().unwrap();
    assert!((vx - 1.0).abs() < 1e-9, "vx={vx}");
    assert!((vz - 0.5).abs() < 1e-9, "vz={vz}");
}

#[test]
fn test_crossing_bubbles_no_identity_swap() {
    let cfg = TrackingConfig {
        max_displacement: 10.0,
        max_gap: 1,
        min_track_length: 2,
    };
    let mut tracker = HungarianTracker::new(cfg);

    tracker.update(&[make_det(10.0, 5.0, 0), make_det(0.0, 5.0, 0)]);
    tracker.update(&[make_det(7.0, 5.0, 1), make_det(3.0, 5.0, 1)]);
    tracker.update(&[make_det(4.0, 5.0, 2), make_det(6.0, 5.0, 2)]);

    let tracks = tracker.finalize();
    assert_eq!(tracks.len(), 2, "Should produce 2 tracks");

    for t in &tracks {
        assert_eq!(t.length(), 3, "Track {} has {} dets", t.id, t.length());
    }
}

#[test]
fn test_track_termination_after_max_gap() {
    let cfg = TrackingConfig {
        max_displacement: 5.0,
        max_gap: 2,
        min_track_length: 1,
    };
    let mut tracker = HungarianTracker::new(cfg);

    tracker.update(&[make_det(5.0, 5.0, 0)]);
    tracker.update(&[make_det(5.1, 5.0, 1)]);
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
    let n = 4;
    let mut cost = vec![vec![1e6_f64; n]; n];
    for (i, row) in cost.iter_mut().enumerate() {
        row[i] = 0.0;
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
    assert!(
        (total - 5.0).abs() < 1e-9,
        "Optimal cost should be 5, got {total}"
    );
    assert_eq!(assignment[0], Some(1));
    assert_eq!(assignment[1], Some(0));
    assert_eq!(assignment[2], Some(2));
}
