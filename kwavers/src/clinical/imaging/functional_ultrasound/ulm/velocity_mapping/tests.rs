//! Tests for velocity mapping.

use super::config::VelocityMapConfig;
use super::mapper::VelocityMapper;
use crate::clinical::imaging::functional_ultrasound::ulm::microbubble_detection::BubbleDetection;
use crate::clinical::imaging::functional_ultrasound::ulm::tracking::BubbleTrack;

fn make_track_at(positions: &[(f64, f64)]) -> BubbleTrack {
    let dets: Vec<BubbleDetection> = positions
        .iter()
        .enumerate()
        .map(|(i, &(x, z))| BubbleDetection {
            x,
            z,
            amplitude: 1.0,
            sigma: 1.0,
            background: 0.0,
            frame: i,
        })
        .collect();
    BubbleTrack {
        id: 0,
        detections: dets,
        last_frame: positions.len().saturating_sub(1),
        gap: 0,
        active: false,
    }
}

#[test]
fn test_velocity_known_lateral_motion() {
    // Bubble moving at Δx = 1 mm per frame, frame_dt = 1e-3 s → vx = 1 m/s.
    let config = VelocityMapConfig {
        x_extent: 5e-3,
        z_extent: 5e-3,
        pixel_size: 50e-6,
        frame_dt: 1e-3,
        min_count: 1,
        ..Default::default()
    };
    let mut mapper = VelocityMapper::new(config).unwrap();
    let track = make_track_at(&[(1e-3, 2e-3), (2e-3, 2e-3), (3e-3, 2e-3)]);
    mapper.accumulate(&[track]);
    let map = mapper.compute();

    // Midpoint of first segment: (1.5 mm, 2 mm)
    let ix = (1.5e-3_f64 / 50e-6) as usize;
    let iz = (2e-3_f64 / 50e-6) as usize;
    let vx = map.vx[[ix, iz]];
    assert!((vx - 1.0).abs() < 1e-9, "Expected vx=1.0 m/s, got {vx:.6}");
    assert!(map.vz[[ix, iz]].abs() < 1e-9, "vz should be 0");
    let s = map.speed[[ix, iz]];
    assert!((s - 1.0).abs() < 1e-9, "speed should be 1 m/s, got {s}");
}

#[test]
fn test_velocity_known_axial_motion() {
    // Bubble moving at Δz = 1 mm per frame → vz = 1 m/s.
    let config = VelocityMapConfig {
        x_extent: 5e-3,
        z_extent: 5e-3,
        pixel_size: 50e-6,
        frame_dt: 1e-3,
        min_count: 1,
        ..Default::default()
    };
    let mut mapper = VelocityMapper::new(config).unwrap();
    let track = make_track_at(&[(2e-3, 1e-3), (2e-3, 2e-3)]);
    mapper.accumulate(&[track]);
    let map = mapper.compute();

    let ix = (2e-3_f64 / 50e-6) as usize;
    let iz = (1.5e-3_f64 / 50e-6) as usize;
    let vz = map.vz[[ix, iz]];
    assert!((vz - 1.0).abs() < 1e-9, "Expected vz=1.0 m/s, got {vz}");
    assert!(map.vx[[ix, iz]].abs() < 1e-9, "vx should be 0");
}

#[test]
fn test_velocity_direction_diagonal() {
    // Bubble moving equal parts x and z → direction = atan2(1, 1) = π/4.
    let config = VelocityMapConfig {
        x_extent: 5e-3,
        z_extent: 5e-3,
        pixel_size: 50e-6,
        frame_dt: 1e-3,
        min_count: 1,
        ..Default::default()
    };
    let mut mapper = VelocityMapper::new(config).unwrap();
    let track = make_track_at(&[(1e-3, 1e-3), (2e-3, 2e-3)]);
    mapper.accumulate(&[track]);
    let map = mapper.compute();

    let ix = (1.5e-3_f64 / 50e-6) as usize;
    let iz = (1.5e-3_f64 / 50e-6) as usize;
    let dir = map.direction[[ix, iz]];
    assert!(
        (dir - std::f64::consts::FRAC_PI_4).abs() < 1e-9,
        "Expected π/4 rad, got {dir}"
    );
}

#[test]
fn test_min_count_mask() {
    // Cell with only 1 vote and min_count=3 must be NaN.
    let config = VelocityMapConfig {
        x_extent: 1e-3,
        z_extent: 1e-3,
        pixel_size: 50e-6,
        frame_dt: 1e-3,
        min_count: 3,
        ..Default::default()
    };
    let mut mapper = VelocityMapper::new(config).unwrap();
    let track = make_track_at(&[(100e-6, 200e-6), (150e-6, 200e-6)]);
    mapper.accumulate(&[track]);
    let map = mapper.compute();

    let ix = (125e-6_f64 / 50e-6) as usize;
    let iz = (200e-6_f64 / 50e-6) as usize;
    assert!(
        map.speed[[ix, iz]].is_nan(),
        "Cell with count < min_count must be NaN"
    );
}

#[test]
fn test_count_accumulation() {
    // 5 tracks all passing through the same cell — count should be 5.
    // Δx = 100 μm, frame_dt = 100 μs → vx = 100e-6 / 100e-6 = 1 m/s.
    let config = VelocityMapConfig {
        x_extent: 1e-3,
        z_extent: 1e-3,
        pixel_size: 100e-6,
        frame_dt: 100e-6,
        min_count: 1,
        ..Default::default()
    };
    let mut mapper = VelocityMapper::new(config).unwrap();
    for _ in 0..5 {
        let track = make_track_at(&[(200e-6, 400e-6), (300e-6, 400e-6)]);
        mapper.accumulate(&[track]);
    }
    let map = mapper.compute();
    let ix = (250e-6_f64 / 100e-6) as usize;
    let iz = (400e-6_f64 / 100e-6) as usize;
    assert_eq!(map.count[[ix, iz]], 5, "Expected 5 votes");
    // Mean velocity should be 1 m/s (Δx=100 μm, dt=100 μs)
    let vx = map.vx[[ix, iz]];
    assert!((vx - 1.0).abs() < 1e-9, "Mean vx should be 1 m/s, got {vx}");
}

#[test]
fn test_wall_shear_stress_uniform_field_is_zero() {
    // Uniform velocity everywhere → ∇speed = 0 → WSS = 0 in interior cells.
    let config = VelocityMapConfig {
        x_extent: 1e-3,
        z_extent: 1e-3,
        pixel_size: 50e-6,
        frame_dt: 1e-3,
        min_count: 1,
        ..Default::default()
    };
    let mut mapper = VelocityMapper::new(config).unwrap();
    // Dense uniform-motion tracks covering a 5×5 block.
    let mut tracks = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            let x0 = i as f64 * 50e-6 + 100e-6;
            let z0 = j as f64 * 50e-6 + 100e-6;
            // 4 frames of uniform x-motion at 50 μm/frame
            tracks.push(make_track_at(&[
                (x0, z0),
                (x0 + 50e-6, z0),
                (x0 + 100e-6, z0),
                (x0 + 150e-6, z0),
            ]));
        }
    }
    mapper.accumulate(&tracks);
    let map = mapper.compute();

    // Check a fully interior cell of the block.
    let ix = (275e-6_f64 / 50e-6) as usize;
    let iz = (275e-6_f64 / 50e-6) as usize;
    let wss = map.wall_shear_stress[[ix, iz]];
    if !wss.is_nan() {
        assert!(
            wss.abs() < 1e-6,
            "WSS for uniform flow must be ≈0, got {wss}"
        );
    }
}

#[test]
fn test_zero_length_track_skipped() {
    // A track with only one detection has no velocity segments — must not panic.
    let config = VelocityMapConfig {
        x_extent: 1e-3,
        z_extent: 1e-3,
        pixel_size: 50e-6,
        frame_dt: 1e-3,
        min_count: 1,
        ..Default::default()
    };
    let mut mapper = VelocityMapper::new(config).unwrap();
    let track = make_track_at(&[(500e-6, 500e-6)]);
    mapper.accumulate(&[track]); // Should not panic
    let total: u32 = mapper.count.iter().sum();
    assert_eq!(
        total, 0,
        "Single-detection track contributes no velocity segments"
    );
}
