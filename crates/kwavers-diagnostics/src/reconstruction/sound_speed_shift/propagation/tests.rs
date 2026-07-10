use leto::Array2;

use super::{path, row_entries};
use crate::reconstruction::sound_speed_shift::{
    ShiftPropagation, ShiftSensitivity, SoundSpeedShiftConfig, SoundSpeedShiftSample,
};
use kwavers_solver::inverse::same_aperture::PlanarPoint;

#[test]
fn circular_arc_path_is_longer_than_its_chord() {
    let sample = SoundSpeedShiftSample::new(
        PlanarPoint {
            x_m: -0.002,
            y_m: 0.0,
        },
        PlanarPoint {
            x_m: 0.002,
            y_m: 0.0,
        },
        0.0,
    );

    let path = path::build_path(
        &sample,
        ShiftPropagation::CircularArc {
            sagitta_m: 0.001,
            segments: 16,
        },
    );
    let length = path.iter().map(|segment| segment.length_m).sum::<f64>();

    assert!(length > 0.004);
}

#[test]
fn finite_frequency_entries_conserve_segment_length_on_full_mask() {
    let mask = Array2::from_elem((9, 9), true);
    let active_lookup = full_lookup({
        let [rows, cols] = mask.shape();
        (rows, cols)
    });
    let sample = SoundSpeedShiftSample::new(
        PlanarPoint {
            x_m: -0.002,
            y_m: 0.0,
        },
        PlanarPoint {
            x_m: 0.002,
            y_m: 0.0,
        },
        0.0,
    );
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        sensitivity: ShiftSensitivity::FiniteFrequency {
            wavelength_m: 0.001,
            support_radius_m: 0.002,
        },
        ..Default::default()
    };

    let entries = row_entries(&sample, &active_lookup, {
        let [rows, cols] = mask.shape();
        (rows, cols)
    }, config);
    let total = entries.iter().map(|(_, weight)| *weight).sum::<f64>();

    assert!((total - 0.004).abs() <= 1.0e-14);
}

fn full_lookup(shape: (usize, usize)) -> Vec<Option<usize>> {
    let (_, ny) = shape;
    (0..shape.0 * shape.1)
        .map(|idx| Some((idx / ny) * ny + idx % ny))
        .collect()
}
