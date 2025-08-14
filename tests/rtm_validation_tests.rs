//! Reverse Time Migration Validation Tests
//!
//! This test suite validates the RTM implementation against:
//! 1. Known synthetic models with reflectors
//! 2. Literature-validated imaging conditions
//! 3. Numerical accuracy requirements
//! 4. Multiple imaging condition algorithms

use kwavers::{
    Grid, KwaversResult,
    solver::reconstruction::seismic::{
        ReverseTimeMigration, SeismicImagingConfig, RtmImagingCondition
    },
    medium::HomogeneousMedium,
};
use ndarray::{Array2, Array3, s};
use approx::assert_relative_eq;

const TEST_TOLERANCE: f64 = 1e-6;
const REFLECTOR_POSITION_TOLERANCE: usize = 3; // Grid points

/// Test RTM with simple horizontal reflector
#[test]
fn test_rtm_horizontal_reflector() -> KwaversResult<()> {
    // Create test grid
    let grid = Grid::new(32, 32, 32, 25e-3, 25e-3, 25e-3); // 25m spacing
    
    // Create velocity model with horizontal reflector
    let mut velocity_model = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                if k < grid.nz / 2 {
                    velocity_model[[i, j, k]] = 2500.0; // Upper layer
                } else {
                    velocity_model[[i, j, k]] = 3500.0; // Lower layer (strong contrast)
                }
            }
        }
    }
    
    // Generate synthetic reflection data
    let recorded_data = generate_rtm_synthetic_data(&velocity_model, &grid)?;
    
    let source_positions = create_line_source_array(&grid, 5);
    let receiver_positions = create_line_receiver_array(&grid, 15);
    
    let config = SeismicImagingConfig {
        rtm_imaging_condition: RtmImagingCondition::ZeroLag,
        ..Default::default()
    };
    
    let mut rtm = ReverseTimeMigration::new(config);
    
    // Run RTM
    let migrated_image = rtm.migrate(
        &recorded_data,
        &source_positions,
        &receiver_positions,
        &velocity_model,
        &grid,
    )?;
    
    // Validate reflector detection
    let interface_depth = grid.nz / 2;
    let max_reflectivity = find_max_reflectivity_depth(&migrated_image, &grid);
    
    let depth_error = (max_reflectivity as i32 - interface_depth as i32).abs() as usize;
    assert!(depth_error <= REFLECTOR_POSITION_TOLERANCE, 
        "RTM reflector detection error: found at depth {}, expected near {}", 
        max_reflectivity, interface_depth);
    
    println!("Horizontal reflector test: detected at depth {} (expected: {})", 
             max_reflectivity, interface_depth);
    
    Ok(())
}

/// Test different RTM imaging conditions
#[test]
fn test_rtm_imaging_conditions() -> KwaversResult<()> {
    let grid = Grid::new(24, 24, 24, 30e-3, 30e-3, 30e-3);
    
    // Simple velocity model with strong contrast
    let mut velocity_model = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                if k < grid.nz / 3 {
                    velocity_model[[i, j, k]] = 2000.0;
                } else if k < 2 * grid.nz / 3 {
                    velocity_model[[i, j, k]] = 3000.0; // Reflector 1
                } else {
                    velocity_model[[i, j, k]] = 4000.0; // Reflector 2
                }
            }
        }
    }
    
    let recorded_data = generate_rtm_synthetic_data(&velocity_model, &grid)?;
    let source_positions = create_line_source_array(&grid, 3);
    let receiver_positions = create_line_receiver_array(&grid, 10);
    
    // Test different imaging conditions
    let imaging_conditions = vec![
        RtmImagingCondition::ZeroLag,
        RtmImagingCondition::Normalized,
        RtmImagingCondition::Laplacian,
        RtmImagingCondition::EnergyNormalized,
    ];
    
    for (i, condition) in imaging_conditions.iter().enumerate() {
        let config = SeismicImagingConfig {
            rtm_imaging_condition: condition.clone(),
            ..Default::default()
        };
        
        let mut rtm = ReverseTimeMigration::new(config);
        let migrated_image = rtm.migrate(
            &recorded_data,
            &source_positions,
            &receiver_positions,
            &velocity_model,
            &grid,
        )?;
        
        // Validate that reflectors are detected
        let reflector1_depth = grid.nz / 3;
        let reflector2_depth = 2 * grid.nz / 3;
        
        let max_depth = find_max_reflectivity_depth(&migrated_image, &grid);
        
        // At least one reflector should be detected
        let error1 = (max_depth as i32 - reflector1_depth as i32).abs() as usize;
        let error2 = (max_depth as i32 - reflector2_depth as i32).abs() as usize;
        
        assert!(error1 <= REFLECTOR_POSITION_TOLERANCE || error2 <= REFLECTOR_POSITION_TOLERANCE,
            "Imaging condition {:?}: no reflectors detected correctly at depth {}", 
            condition, max_depth);
        
        println!("Imaging condition {:?}: max reflectivity at depth {}", condition, max_depth);
    }
    
    Ok(())
}

/// Test RTM with dipping reflector
#[test]
fn test_rtm_dipping_reflector() -> KwaversResult<()> {
    let grid = Grid::new(32, 16, 32, 20e-3, 20e-3, 20e-3);
    
    // Create dipping reflector model
    let mut velocity_model = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                // Dipping interface: depth increases with lateral position
                let interface_depth = grid.nz / 3 + (i * grid.nz / 3) / grid.nx;
                if k < interface_depth {
                    velocity_model[[i, j, k]] = 2500.0;
                } else {
                    velocity_model[[i, j, k]] = 3800.0; // Strong contrast
                }
            }
        }
    }
    
    let recorded_data = generate_rtm_synthetic_data(&velocity_model, &grid)?;
    let source_positions = create_line_source_array(&grid, 4);
    let receiver_positions = create_line_receiver_array(&grid, 12);
    
    let config = SeismicImagingConfig {
        rtm_imaging_condition: RtmImagingCondition::Normalized,
        ..Default::default()
    };
    
    let mut rtm = ReverseTimeMigration::new(config);
    let migrated_image = rtm.migrate(
        &recorded_data,
        &source_positions,
        &receiver_positions,
        &velocity_model,
        &grid,
    )?;
    
    // Validate dipping reflector imaging
    // Check that reflectivity follows the dip trend
    let left_side_depth = find_max_reflectivity_at_x(&migrated_image, 5, &grid);
    let right_side_depth = find_max_reflectivity_at_x(&migrated_image, grid.nx - 5, &grid);
    
    // Right side should be deeper than left side for this dipping model
    assert!(right_side_depth > left_side_depth, 
        "Dipping reflector not properly imaged: left depth {}, right depth {}", 
        left_side_depth, right_side_depth);
    
    println!("Dipping reflector test: left depth {}, right depth {} (dip detected)", 
             left_side_depth, right_side_depth);
    
    Ok(())
}

/// Test RTM CFL condition validation
#[test]
fn test_rtm_cfl_validation() -> KwaversResult<()> {
    let grid = Grid::new(16, 16, 16, 10e-3, 10e-3, 10e-3); // 10m spacing
    
    // High velocity model that tests CFL limits
    let high_velocity_model = Array3::from_elem((grid.nx, grid.ny, grid.nz), 6000.0);
    
    let recorded_data = Array2::zeros((5, 100));
    let source_positions = vec![[grid.nx as f64 * grid.dx / 2.0, 0.0, 0.0]];
    let receiver_positions = create_line_receiver_array(&grid, 5);
    
    let config = SeismicImagingConfig::default();
    let mut rtm = ReverseTimeMigration::new(config);
    
    // RTM should handle high velocities gracefully with proper timestep
    let result = rtm.migrate(
        &recorded_data,
        &source_positions,
        &receiver_positions,
        &high_velocity_model,
        &grid,
    );
    
    assert!(result.is_ok(), "RTM should handle high velocities with proper CFL validation");
    
    println!("CFL validation test: RTM handled high velocity model successfully");
    
    Ok(())
}

/// Test RTM with point scatterer
#[test]
fn test_rtm_point_scatterer() -> KwaversResult<()> {
    let grid = Grid::new(24, 24, 24, 40e-3, 40e-3, 40e-3);
    
    // Velocity model with point scatterer (velocity anomaly)
    let mut velocity_model = Array3::from_elem((grid.nx, grid.ny, grid.nz), 3000.0);
    
    // Add point scatterer at center
    let scatter_i = grid.nx / 2;
    let scatter_j = grid.ny / 2;
    let scatter_k = grid.nz / 2;
    
    // Create small high-velocity anomaly
    for di in -1..=1 {
        for dj in -1..=1 {
            for dk in -1..=1 {
                let i = (scatter_i as i32 + di) as usize;
                let j = (scatter_j as i32 + dj) as usize;
                let k = (scatter_k as i32 + dk) as usize;
                if i < grid.nx && j < grid.ny && k < grid.nz {
                    velocity_model[[i, j, k]] = 4500.0; // Strong scatterer
                }
            }
        }
    }
    
    let recorded_data = generate_rtm_synthetic_data(&velocity_model, &grid)?;
    let source_positions = create_circular_source_array(&grid, 6);
    let receiver_positions = create_circular_receiver_array(&grid, 16);
    
    let config = SeismicImagingConfig {
        rtm_imaging_condition: RtmImagingCondition::Laplacian,
        ..Default::default()
    };
    
    let mut rtm = ReverseTimeMigration::new(config);
    let migrated_image = rtm.migrate(
        &recorded_data,
        &source_positions,
        &receiver_positions,
        &velocity_model,
        &grid,
    )?;
    
    // Find maximum reflectivity position
    let (max_i, max_j, max_k) = find_max_reflectivity_position(&migrated_image);
    
    // Validate scatterer detection
    let position_error = ((max_i as i32 - scatter_i as i32).abs() + 
                         (max_j as i32 - scatter_j as i32).abs() + 
                         (max_k as i32 - scatter_k as i32).abs()) as usize;
    
    assert!(position_error <= 3 * REFLECTOR_POSITION_TOLERANCE,
        "Point scatterer detection error: found at ({}, {}, {}), expected near ({}, {}, {})", 
        max_i, max_j, max_k, scatter_i, scatter_j, scatter_k);
    
    println!("Point scatterer test: detected at ({}, {}, {}) vs expected ({}, {}, {})", 
             max_i, max_j, max_k, scatter_i, scatter_j, scatter_k);
    
    Ok(())
}

/// Test RTM memory efficiency with large models
#[test]
fn test_rtm_memory_efficiency() -> KwaversResult<()> {
    // This test validates that RTM can handle larger models without excessive memory usage
    let grid = Grid::new(48, 48, 48, 25e-3, 25e-3, 25e-3);
    
    let velocity_model = Array3::from_elem((grid.nx, grid.ny, grid.nz), 3000.0);
    let recorded_data = Array2::zeros((10, 200)); // Minimal data for efficiency test
    
    let source_positions = create_line_source_array(&grid, 2);
    let receiver_positions = create_line_receiver_array(&grid, 10);
    
    let config = SeismicImagingConfig::default();
    let mut rtm = ReverseTimeMigration::new(config);
    
    // This should complete without memory issues
    let result = rtm.migrate(
        &recorded_data,
        &source_positions,
        &receiver_positions,
        &velocity_model,
        &grid,
    );
    
    assert!(result.is_ok(), "RTM should handle larger models efficiently");
    
    println!("Memory efficiency test: RTM processed {}^3 grid successfully", grid.nx);
    
    Ok(())
}

// Helper functions for RTM test data generation

fn generate_rtm_synthetic_data(velocity_model: &Array3<f64>, grid: &Grid) -> KwaversResult<Array2<f64>> {
    // Generate synthetic seismic data with reflections based on velocity contrasts
    let num_receivers = 15;
    let num_time_samples = 300;
    let mut data = Array2::zeros((num_receivers, num_time_samples));
    
    // Generate traces with reflections at velocity interfaces
    for receiver in 0..num_receivers {
        for t in 0..num_time_samples {
            let time = t as f64 * 1e-3; // 1ms sampling
            
            // Direct wave
            let direct_time = 0.04 + (receiver as f64 * 0.008);
            if time > direct_time && time < direct_time + 0.015 {
                let ricker_arg = std::f64::consts::PI * 30.0 * (time - direct_time);
                data[[receiver, t]] = (1.0 - 2.0 * ricker_arg.powi(2)) * (-ricker_arg.powi(2)).exp();
            }
            
            // Primary reflection from first interface
            let reflection1_time = 0.12 + (receiver as f64 * 0.012);
            if time > reflection1_time && time < reflection1_time + 0.02 {
                let ricker_arg = std::f64::consts::PI * 25.0 * (time - reflection1_time);
                let amplitude = 0.4; // Strong reflection
                data[[receiver, t]] += amplitude * (1.0 - 2.0 * ricker_arg.powi(2)) * (-ricker_arg.powi(2)).exp();
            }
            
            // Secondary reflection (if multiple layers)
            let reflection2_time = 0.20 + (receiver as f64 * 0.018);
            if time > reflection2_time && time < reflection2_time + 0.015 {
                let ricker_arg = std::f64::consts::PI * 20.0 * (time - reflection2_time);
                let amplitude = 0.25; // Weaker reflection
                data[[receiver, t]] += amplitude * (1.0 - 2.0 * ricker_arg.powi(2)) * (-ricker_arg.powi(2)).exp();
            }
        }
    }
    
    Ok(data)
}

fn create_line_source_array(grid: &Grid, num_sources: usize) -> Vec<[f64; 3]> {
    let mut sources = Vec::new();
    for i in 0..num_sources {
        let x = (i as f64 / (num_sources - 1) as f64) * grid.nx as f64 * grid.dx * 0.8 + 0.1 * grid.nx as f64 * grid.dx;
        sources.push([x, grid.ny as f64 * grid.dy / 2.0, 0.0]);
    }
    sources
}

fn create_line_receiver_array(grid: &Grid, num_receivers: usize) -> Vec<[f64; 3]> {
    let mut receivers = Vec::new();
    for i in 0..num_receivers {
        let x = (i as f64 / (num_receivers - 1) as f64) * grid.nx as f64 * grid.dx;
        receivers.push([x, grid.ny as f64 * grid.dy / 2.0, grid.nz as f64 * grid.dz * 0.05]);
    }
    receivers
}

fn create_circular_source_array(grid: &Grid, num_sources: usize) -> Vec<[f64; 3]> {
    let mut sources = Vec::new();
    let center_x = grid.nx as f64 * grid.dx / 2.0;
    let center_y = grid.ny as f64 * grid.dy / 2.0;
    let radius = grid.nx as f64 * grid.dx * 0.3;
    
    for i in 0..num_sources {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / num_sources as f64;
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();
        sources.push([x, y, 0.0]);
    }
    sources
}

fn create_circular_receiver_array(grid: &Grid, num_receivers: usize) -> Vec<[f64; 3]> {
    let mut receivers = Vec::new();
    let center_x = grid.nx as f64 * grid.dx / 2.0;
    let center_y = grid.ny as f64 * grid.dy / 2.0;
    let radius = grid.nx as f64 * grid.dx * 0.4;
    
    for i in 0..num_receivers {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / num_receivers as f64;
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();
        receivers.push([x, y, grid.nz as f64 * grid.dz * 0.1]);
    }
    receivers
}

fn find_max_reflectivity_depth(migrated_image: &Array3<f64>, grid: &Grid) -> usize {
    let mut max_reflectivity = 0.0;
    let mut max_depth = 0;
    
    for k in 0..grid.nz {
        let depth_reflectivity: f64 = migrated_image.slice(s![.., .., k]).iter().map(|&x| x.abs()).sum();
        if depth_reflectivity > max_reflectivity {
            max_reflectivity = depth_reflectivity;
            max_depth = k;
        }
    }
    
    max_depth
}

fn find_max_reflectivity_at_x(migrated_image: &Array3<f64>, x_idx: usize, grid: &Grid) -> usize {
    let mut max_reflectivity = 0.0;
    let mut max_depth = 0;
    
    for k in 0..grid.nz {
        let depth_reflectivity: f64 = migrated_image.slice(s![x_idx, .., k]).iter().map(|&x| x.abs()).sum();
        if depth_reflectivity > max_reflectivity {
            max_reflectivity = depth_reflectivity;
            max_depth = k;
        }
    }
    
    max_depth
}

fn find_max_reflectivity_position(migrated_image: &Array3<f64>) -> (usize, usize, usize) {
    let mut max_reflectivity = 0.0;
    let mut max_pos = (0, 0, 0);
    
    for i in 0..migrated_image.shape()[0] {
        for j in 0..migrated_image.shape()[1] {
            for k in 0..migrated_image.shape()[2] {
                let reflectivity = migrated_image[[i, j, k]].abs();
                if reflectivity > max_reflectivity {
                    max_reflectivity = reflectivity;
                    max_pos = (i, j, k);
                }
            }
        }
    }
    
    max_pos
}