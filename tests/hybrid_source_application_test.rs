use kwavers::domain::grid::Grid;
use kwavers::domain::medium::homogeneous::HomogeneousMedium;
use kwavers::domain::plugin::{Plugin, PluginContext, PluginFields};
use kwavers::domain::source::{GaussianBuilder, Source};
use kwavers::domain::signal::SineWave;
use kwavers::solver::hybrid::{
    DecompositionStrategy, HybridConfig, HybridPlugin,
};
use kwavers::solver::hybrid::domain_decomposition::{DomainRegion, DomainType};
use kwavers::solver::forward::pstd::config::{BoundaryConfig as PSTDBoundaryConfig, PSTDConfig};
use kwavers::domain::boundary::PMLConfig;
use ndarray::{Array3, Array4};
use std::sync::Arc;

#[test]
fn test_hybrid_source_application() {
    // 1. Setup Grid
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).expect("Grid creation failed");
    let medium = HomogeneousMedium::water(&grid);

    // 2. Setup Config with Full FDTD
    let region = DomainRegion::new(
        (0, 0, 0),
        (grid.nx, grid.ny, grid.nz),
        DomainType::FDTD,
        1.0,
    );

    let mut pstd_config = PSTDConfig::default();
    pstd_config.boundary = PSTDBoundaryConfig::PML(PMLConfig {
        thickness: 4,
        ..Default::default()
    });

    let config = HybridConfig {
        decomposition_strategy: DecompositionStrategy::UserDefined(vec![region]),
        pstd_config,
        ..Default::default()
    };

    // 3. Create Plugin
    let mut plugin = HybridPlugin::new(config, &grid).expect("Plugin creation failed");
    plugin
        .initialize(&grid, &medium)
        .expect("Plugin init failed");

    // 4. Create Source
    let center = (
        grid.dx * grid.nx as f64 / 2.0,
        grid.dy * grid.ny as f64 / 2.0,
        grid.dz * grid.nz as f64 / 2.0,
    );
    let signal = Arc::new(SineWave::new(1.0, 1e6, 0.0));
    let source = GaussianBuilder::new()
        .focal_point(center)
        .waist_radius(2e-3)
        .wavelength(1.5e-3)
        .build(signal);

    // 5. Update
    let mut fields = Array4::zeros((17, grid.nx, grid.ny, grid.nz));
    let dt = 1e-7;
    let t = 0.25e-6; // Quarter period of 1MHz sine wave, amplitude should be 1.0

    // Create dummy boundary and sources list for context
    let pml_config = PMLConfig {
        thickness: 4,
        ..Default::default()
    };
    let mut boundary = kwavers::domain::boundary::PMLBoundary::new(pml_config).unwrap();
    let sources: Vec<Box<dyn Source>> = vec![Box::new(source)];
    let extra_fields = PluginFields::new(Array3::zeros((1, 1, 1)));

    let mut context = PluginContext {
        boundary: &mut boundary,
        sources: &sources,
        extra_fields: &extra_fields,
    };

    plugin.update(&mut fields, &grid, &medium, dt, t, &mut context).expect("Update failed");

    // 6. Assert
    let p_idx = 0; // Pressure
    // Check max pressure
    let max_p = fields.index_axis(ndarray::Axis(0), p_idx).iter().fold(0.0f64, |a, &b| a.max(b.abs()));

    println!("Max pressure: {}", max_p);
    assert!(max_p > 1e-10, "Pressure field should be non-zero after applying source (expected ~1.0)");
}
