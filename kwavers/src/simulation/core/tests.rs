use super::*;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::domain::signal::SineWave;
use crate::domain::source::{PointSource, Source};
use crate::solver::feature::SolverFeature;
use std::sync::Arc;

#[test]
fn test_simulation_creation() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

    let signal = Arc::new(SineWave::new(1e6, 1.0, 0.0));
    let source = PointSource::new((0.032, 0.032, 0.032), signal);
    let sources: Vec<Arc<dyn Source>> = vec![Arc::new(source)];

    let simulation = CoreSimulation::new(
        grid,
        &medium,
        sources,
        vec![],
        Box::new(crate::solver::progress::ConsoleProgressReporter::default()),
    )
    .unwrap();

    assert_eq!(simulation.statistics().num_sources, 1);
    assert_eq!(simulation.statistics().num_sensors, 0);
}

#[test]
fn test_feature_management() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

    let mut simulation = CoreSimulation::new(
        grid,
        &medium,
        vec![],
        vec![],
        Box::new(crate::solver::progress::ConsoleProgressReporter::default()),
    )
    .unwrap();

    // Test enabling features
    assert!(simulation
        .enable_feature(SolverFeature::Reconstruction)
        .is_ok());
    assert!(simulation.is_feature_enabled(SolverFeature::Reconstruction));

    assert!(simulation
        .enable_feature(SolverFeature::GpuAcceleration)
        .is_ok());
    assert!(simulation.is_feature_enabled(SolverFeature::GpuAcceleration));
}

#[test]
fn test_simulation_builder() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

    let signal = Arc::new(SineWave::new(1e6, 1.0, 0.0));
    let source = PointSource::new((0.016, 0.016, 0.016), signal);
    let source: Arc<dyn Source> = Arc::new(source);

    let simulation = SimulationBuilder::new()
        .with_grid(grid)
        .with_medium(&medium)
        .with_source(source)
        .with_feature(SolverFeature::Reconstruction)
        .build();

    assert!(simulation.is_ok());
    let simulation = simulation.unwrap();
    assert!(simulation.is_feature_enabled(SolverFeature::Reconstruction));
}
