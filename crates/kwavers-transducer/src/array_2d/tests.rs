use super::array::TransducerArray2D;
use super::types::{ApodizationType, ArrayCurvature, TransducerArray2DConfig};
use aequitas::systems::si::quantities::{Angle, Frequency, Length, Velocity};
use aequitas::systems::si::units::{Degree, Hertz, Meter, MeterPerSecond};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_grid::Grid;
use kwavers_source::Source;
use leto::Array3;

fn create_test_config() -> TransducerArray2DConfig {
    TransducerArray2DConfig {
        number_elements: 16,
        element_width: Length::from_unit::<Meter>(0.3e-3),
        element_length: Length::from_unit::<Meter>(10e-3),
        element_spacing: Length::from_unit::<Meter>(0.5e-3),
        curvature: ArrayCurvature::Flat,
        center_position: [Length::from_unit::<Meter>(0.0); 3],
    }
}

#[test]
fn test_array_creation() {
    let config = create_test_config();
    let array = TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .unwrap();

    assert_eq!(array.num_elements(), 16);
    assert!(array.satisfies_nyquist());
}

#[test]
fn test_focus_and_steering() {
    let config = create_test_config();
    let mut array = TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .unwrap();

    array
        .set_focus_distance(Length::from_unit::<Meter>(20e-3))
        .unwrap();
    array
        .set_steering_angle(Angle::from_unit::<Degree>(10.0))
        .unwrap();

    assert!((array.focus_distance().unwrap().in_unit::<Meter>() - 20e-3).abs() < 1e-10);
    assert!((array.steering_angle().in_unit::<Degree>() - 10.0).abs() < 1e-10);

    let positions = array.element_positions();
    assert_eq!(positions.len(), 16);
    assert!(array.elevation_focus_distance().is_none());
    array.clear_focus_distance();
    assert!(array.focus_distance().is_none());
}

#[test]
fn cylindrical_curvature_preserves_radius_and_sag() {
    let config = TransducerArray2DConfig {
        number_elements: 3,
        element_width: Length::from_unit::<Meter>(0.5e-3),
        element_length: Length::from_unit::<Meter>(2.0e-3),
        element_spacing: Length::from_unit::<Meter>(1.0e-3),
        curvature: ArrayCurvature::Cylindrical {
            radius: Length::from_unit::<Meter>(10.0e-3),
        },
        center_position: [Length::from_unit::<Meter>(0.0); 3],
    };
    let array = TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .unwrap();

    assert_eq!(
        array.curvature().radius().unwrap().in_unit::<Meter>(),
        10.0e-3
    );
    let positions = array.element_positions();
    let sag = positions[0][1].in_unit::<Meter>();
    let expected_sag = 10.0e-3 * (1.0 - (0.001_f64 / 0.01_f64).cos());
    assert!((sag - expected_sag).abs() < 1.0e-15);
}

#[test]
fn curvature_rejects_nonfinite_radius() {
    let config = TransducerArray2DConfig {
        curvature: ArrayCurvature::Cylindrical {
            radius: Length::from_unit::<Meter>(f64::INFINITY),
        },
        ..create_test_config()
    };
    assert!(TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .is_err());
}

#[test]
fn test_apodization() {
    let config = create_test_config();
    let mut array = TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .unwrap();

    array.set_transmit_apodization(ApodizationType::Hanning);
    array.set_receive_apodization(ApodizationType::Hamming);
}

#[test]
fn test_active_elements() {
    let config = create_test_config();
    let mut array = TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .unwrap();

    let mut mask = vec![true; 16];
    for i in (0..16).step_by(2) {
        mask[i] = false;
    }

    array.set_active_elements(&mask).unwrap();

    let active = array.get_active_elements();
    assert_eq!(active.len(), 16);
    for i in (0..16).step_by(2) {
        assert!(!active[i]);
    }
}

#[test]
fn test_invalid_config() {
    let config = TransducerArray2DConfig {
        number_elements: 0,
        ..create_test_config()
    };

    assert!(TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .is_err());
}

#[test]
fn test_aperture_calculation() {
    let config = create_test_config();
    let array = TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .unwrap();

    let expected = 15.0 * 0.5e-3 + 0.3e-3;
    assert!((array.aperture_width().in_unit::<Meter>() - expected).abs() < 1e-10);
    let positions = array.element_positions();
    let first = positions[0][0].in_unit::<Meter>();
    let second = positions[1][0].in_unit::<Meter>();
    assert!((second - first - 0.5e-3).abs() < 1e-12);
}

#[test]
fn add_mask_into_accumulates_cached_mask() {
    let config = create_test_config();
    let mut array = TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .unwrap();
    let grid = Grid::new(4, 3, 2, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let grid_id = (&grid as *const Grid) as u64;

    array.cached_grid_id = Some(grid_id);
    array.cached_mask = Some(Array3::from_elem([grid.nx, grid.ny, grid.nz], 2.0));

    let mut mask = Array3::from_elem([grid.nx, grid.ny, grid.nz], 1.0);
    array.add_mask_into(&grid, &mut mask);

    assert_eq!(mask, Array3::from_elem([grid.nx, grid.ny, grid.nz], 3.0));
}

#[test]
fn position_visitor_matches_owned_positions() {
    let config = create_test_config();
    let array = TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .unwrap();

    let mut visited = Vec::new();
    array.for_each_position(&mut |position| visited.push(position));
    assert_eq!(visited, array.positions());
    assert_eq!(visited.len(), 16);
}

#[test]
fn position_visitor_tracks_active_element_mask() {
    let config = create_test_config();
    let mut array = TransducerArray2D::new(
        config,
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    )
    .unwrap();

    let mut mask = vec![true; 16];
    for i in (0..16).step_by(2) {
        mask[i] = false;
    }
    array.set_active_elements(&mask).unwrap();

    let mut visited = Vec::new();
    array.for_each_position(&mut |position| visited.push(position));
    assert_eq!(visited, array.positions());
    assert_eq!(visited.len(), 8);
}
