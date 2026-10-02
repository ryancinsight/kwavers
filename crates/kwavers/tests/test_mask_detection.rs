//! Test mask detection for source injection mode determination
//!
//! This test verifies that boundary plane detection works correctly.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_signal::SineWave;
use kwavers_source::{InjectionMode, PlaneWaveSource, PlaneWaveSourceConfig, Source, SourceField};
use std::sync::Arc;

#[test]
fn test_boundary_plane_mask_detection() -> KwaversResult<()> {
    eprintln!("\n=== Testing Boundary Plane Mask Detection ===\n");

    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 0.1e-3;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;

    // Create plane wave source with BoundaryOnly mode
    let signal = Arc::new(SineWave::new(1e6, 1e5, 0.0));
    let config = PlaneWaveSourceConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: 1.5e-3,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let source = PlaneWaveSource::new(config, signal);

    // Create mask
    let mask = source.create_mask(&grid);

    eprintln!("Mask shape: {:?}", mask.shape());
    eprintln!("Mask sum: {}", mask.iter().sum::<f64>());

    // Analyze mask structure
    let mut num_nonzero = 0;
    let mut indices = Vec::new();
    let mut i_values = std::collections::HashSet::new();
    let mut j_values = std::collections::HashSet::new();
    let mut k_values = std::collections::HashSet::new();

    for ([i, j, k], &val) in mask.indexed_iter() {
        if val.abs() > 1e-12 {
            num_nonzero += 1;
            if indices.len() < 10 {
                indices.push((i, j, k, val));
            }
            i_values.insert(i);
            j_values.insert(j);
            k_values.insert(k);
        }
    }

    eprintln!("Number of non-zero elements: {}", num_nonzero);
    eprintln!("First few non-zero indices:");
    for (i, j, k, val) in &indices {
        eprintln!("  [{}, {}, {}] = {}", i, j, k, val);
    }

    eprintln!("\nUnique i indices: {}", i_values.len());
    eprintln!("Unique j indices: {}", j_values.len());
    eprintln!("Unique k indices: {}", k_values.len());

    if i_values.len() == 1 {
        eprintln!("All points share i = {:?}", i_values.iter().next());
    }
    if j_values.len() == 1 {
        eprintln!("All points share j = {:?}", j_values.iter().next());
    }
    if k_values.len() == 1 {
        eprintln!("All points share k = {:?}", k_values.iter().next());
    }

    // Check if it's a boundary plane
    let is_boundary_plane_i =
        i_values.len() == 1 && (i_values.contains(&0) || i_values.contains(&(nx - 1)));
    let is_boundary_plane_j =
        j_values.len() == 1 && (j_values.contains(&0) || j_values.contains(&(ny - 1)));
    let is_boundary_plane_k =
        k_values.len() == 1 && (k_values.contains(&0) || k_values.contains(&(nz - 1)));

    let is_boundary_plane = is_boundary_plane_i || is_boundary_plane_j || is_boundary_plane_k;

    eprintln!("\nBoundary plane detection:");
    eprintln!("  Is boundary plane (i): {}", is_boundary_plane_i);
    eprintln!("  Is boundary plane (j): {}", is_boundary_plane_j);
    eprintln!("  Is boundary plane (k): {}", is_boundary_plane_k);
    eprintln!("  Overall: {}", is_boundary_plane);

    // Expected: should be a boundary plane at k=0
    assert!(
        is_boundary_plane,
        "BoundaryOnly plane wave should create a boundary plane mask"
    );
    assert!(
        k_values.len() == 1 && k_values.contains(&0),
        "Plane wave in +z direction should have all points at k=0"
    );
    assert_eq!(
        num_nonzero,
        nx * ny,
        "Boundary plane should have nx*ny non-zero points"
    );

    Ok(())
}

#[test]
fn test_fullgrid_mask_detection() -> KwaversResult<()> {
    eprintln!("\n=== Testing FullGrid Mask Detection ===\n");

    let nx = 16;
    let ny = 16;
    let nz = 16;
    let dx = 0.1e-3;

    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;

    // Create plane wave source with FullGrid mode
    let signal = Arc::new(SineWave::new(1e6, 1e5, 0.0));
    let config = PlaneWaveSourceConfig {
        direction: (0.0, 0.0, 1.0),
        wavelength: 1.5e-3,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::FullGrid,
    };
    let source = PlaneWaveSource::new(config, signal);

    // Create mask
    let mask = source.create_mask(&grid);

    eprintln!("Mask shape: {:?}", mask.shape());

    // Analyze mask structure
    let mut num_nonzero = 0;
    let mut k_values = std::collections::HashSet::new();

    for ([_i, _j, k], &val) in mask.indexed_iter() {
        if val.abs() > 1e-12 {
            num_nonzero += 1;
            k_values.insert(k);
        }
    }

    eprintln!("Number of non-zero elements: {}", num_nonzero);
    eprintln!("Number of unique k indices: {}", k_values.len());

    // Expected: should NOT be a boundary plane (should have points at all k values)
    let is_boundary_plane_k =
        k_values.len() == 1 && (k_values.contains(&0) || k_values.contains(&(nz - 1)));

    eprintln!("\nBoundary plane detection:");
    eprintln!("  Is boundary plane (k): {}", is_boundary_plane_k);

    assert!(
        !is_boundary_plane_k,
        "FullGrid mode should not create a boundary plane mask"
    );
    assert!(
        k_values.len() > 1,
        "FullGrid mode should have points at multiple k indices"
    );

    Ok(())
}
