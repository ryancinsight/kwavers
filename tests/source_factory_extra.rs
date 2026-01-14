use kwavers::domain::grid::Grid;
use kwavers::domain::source::{SourceFactory, SourceModel, SourceParameters};

#[test]
fn test_create_linear_array() {
    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
    let mut config = SourceParameters::default();
    config.model = SourceModel::LinearArray;
    config.radius = 0.01;
    config.num_elements = Some(16);
    config.position = [0.0, 0.032, 0.032];
    config.focus = Some([0.01, 0.032, 0.042]); // Focused slightly offset

    let source = SourceFactory::create_source(&config, &grid).unwrap();
    // Check positions
    let positions = source.positions();
    assert_eq!(positions.len(), 16);

    // Check signal
    assert!(source.amplitude(0.0).abs() < 1e10); // just check it doesn't crash
}

#[test]
fn test_create_matrix_array() {
    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
    let mut config = SourceParameters::default();
    config.model = SourceModel::MatrixArray;
    config.radius = 0.01;
    config.num_elements = Some(16); // Should result in 4x4=16 elements (sqrt(16)=4)

    let source = SourceFactory::create_source(&config, &grid).unwrap();
    let positions = source.positions();
    assert_eq!(positions.len(), 16);
}

#[test]
fn test_create_focused_source() {
    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
    let mut config = SourceParameters::default();
    config.model = SourceModel::Focused;
    config.radius = 0.01; // diameter 0.02
    config.position = [0.032, 0.032, 0.0];
    config.focus = Some([0.032, 0.032, 0.03]); // focus at z=0.03
    config.frequency = 1.0e6;

    let source = SourceFactory::create_source(&config, &grid).unwrap();
    let positions = source.positions();
    assert!(!positions.is_empty());

    // Check get_source_term logic
    let p = positions[0];
    let val = source.get_source_term(0.0, p.0, p.1, p.2, &grid);
    // Should run without error
    println!("Source term at element 0: {}", val);
}
