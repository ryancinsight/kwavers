use super::*;
use kwavers_core::constants::fundamental::{
    DENSITY_WATER, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER,
};
use kwavers_medium::homogeneous::HomogeneousMedium;

#[test]
fn grid_descriptor_preserves_grid_dimensions_and_spacing() {
    let grid = Grid::new(4, 5, 6, 1.0e-3, 2.0e-3, 3.0e-3).unwrap();
    let descriptor = GridDescriptor(&grid);

    assert_eq!(descriptor.nx(), 4);
    assert_eq!(descriptor.ny(), 5);
    assert_eq!(descriptor.nz(), 6);
    assert_eq!(descriptor.total_points(), 120);
    assert_eq!(descriptor.dx(), 1.0e-3);
    assert_eq!(descriptor.dy(), 2.0e-3);
    assert_eq!(descriptor.dz(), 3.0e-3);
    assert_eq!(
        descriptor.characteristic_size().to_bits(),
        (6.0_f64 * 3.0e-3).to_bits()
    );
}

#[test]
fn medium_descriptor_preserves_acoustic_values() {
    let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER, SOUND_SPEED_WATER, &grid);
    let descriptor = MediumDescriptor {
        medium: &medium,
        grid: &grid,
    };

    assert_eq!(descriptor.density(0.0, 0.0, 0.0), DENSITY_WATER);
    assert_eq!(descriptor.sound_speed(0.0, 0.0, 0.0), SOUND_SPEED_WATER);
    assert_eq!(descriptor.heterogeneity(), 0.0);
}

#[test]
fn assembles_kspace_solver_through_full_kspace_pstd() {
    let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::KSpace,
        max_steps: 2,
        dt: 1.0e-8,
        ..SolverConfiguration::default()
    };

    let mut solver =
        SimulationSolverFactory::create_solver(SolverType::KSpace, config, &grid, &medium).unwrap();

    assert_eq!(solver.name(), "PSTD");
    solver.run(0).unwrap();
    assert_eq!(solver.pressure_field().shape(), [4, 4, 4]);
}

#[test]
fn rejects_hephaestus_for_factory_kspace_before_cpu_assembly() {
    let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::KSpace,
        fft_backend: FftBackend::Hephaestus,
        ..SolverConfiguration::default()
    };

    let error = SimulationSolverFactory::create_solver(SolverType::KSpace, config, &grid, &medium)
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "Feature not available: KSpace does not yet support FftBackend::Hephaestus; select SolverType::PSTD and configure k-space correction"
    );
}

#[test]
fn rejects_hephaestus_for_factory_hybrid_before_cpu_assembly() {
    let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::Hybrid,
        fft_backend: FftBackend::Hephaestus,
        ..SolverConfiguration::default()
    };

    let error = SimulationSolverFactory::create_solver(SolverType::Hybrid, config, &grid, &medium)
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "Feature not available: Hybrid does not yet support FftBackend::Hephaestus"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn assembles_the_gpu_adapter_for_hephaestus_pstd() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let solver =
        SimulationSolverFactory::create_solver(SolverType::PSTD, config, &grid, &medium).unwrap();

    // The Leto solver built for `FftBackend::Leto` reports "PSTD".
    assert_eq!(solver.name(), "GpuPstd");
}

#[cfg(not(feature = "gpu"))]
#[test]
fn rejects_hephaestus_pstd_without_the_gpu_feature() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let error = SimulationSolverFactory::create_solver(SolverType::PSTD, config, &grid, &medium)
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "Feature not available: FftBackend::Hephaestus requires the `gpu` Cargo feature"
    );
}

#[test]
fn assembles_discontinuous_galerkin_solver_for_valid_layout() {
    let grid = Grid::new(3, 3, 1, 1.0e-3, 1.0, 1.0).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, 10.0, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::DiscontinuousGalerkin,
        spatial_order: 2,
        dt: 1.0e-6,
        ..SolverConfiguration::default()
    };

    let mut solver = SimulationSolverFactory::create_solver(
        SolverType::DiscontinuousGalerkin,
        config,
        &grid,
        &medium,
    )
    .unwrap();

    assert_eq!(solver.name(), "DiscontinuousGalerkin");
    solver.run(1).unwrap();
    assert_eq!(solver.statistics().current_step, 1);
    assert_eq!(solver.pressure_field().shape(), [3, 3, 1]);
}

#[test]
fn assembles_fdtd_with_cpml_via_absorbing_boundary_config() {
    use kwavers_boundary::cpml::CPMLConfig;
    use kwavers_solver::config::AbsorbingBoundaryConfig;

    // Grid must be large enough for the CPML thickness.
    // CPMLConfig::with_thickness(4) → 4 cells each side.
    // 16 × 16 × 16 with dx = 1e-3 is well-conditioned.
    let grid = Grid::new(16, 16, 16, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);

    // dt satisfies CFL: dx / (sqrt(3) * c) ≈ 1e-3 / (1.732 * 1500) ≈ 3.85e-7 s.
    // Use 1e-7 to stay well inside the stability region.
    let dt = 1.0e-7_f64;
    let config = SolverConfiguration {
        solver_type: SolverType::FDTD,
        max_steps: 2,
        dt,
        cfl: 0.3,
        absorbing_boundary: Some(AbsorbingBoundaryConfig {
            cpml: CPMLConfig::with_thickness(4),
            max_sound_speed: SOUND_SPEED_WATER,
        }),
        ..SolverConfiguration::default()
    };

    let mut solver =
        SimulationSolverFactory::create_solver(SolverType::FDTD, config, &grid, &medium).unwrap();

    // Run two steps through the CPML-enabled FDTD to confirm the boundary
    // code path executes without panic or NaN propagation.
    solver.run(2).unwrap();

    let p = solver.pressure_field();
    // Value-semantic: every pressure cell must be finite after 2 steps
    // in a medium with no source and CPML boundaries.
    assert!(
        p.iter().all(|v| v.is_finite()),
        "CPML FDTD produced non-finite pressure after 2 steps"
    );
    // Shape must match the grid dimensions unchanged.
    assert_eq!(p.shape(), [16, 16, 16]);
}

#[test]
fn reports_unavailable_fem_grid_assembly_without_not_implemented() {
    let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration::default();

    let error = SimulationSolverFactory::create_solver(SolverType::FEM, config, &grid, &medium)
        .unwrap_err();

    assert!(matches!(error, KwaversError::FeatureNotAvailable(_)));
}
