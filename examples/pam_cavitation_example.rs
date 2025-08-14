//! Example demonstrating Passive Acoustic Mapping (PAM) for cavitation and sonoluminescence detection
//!
//! This example shows how to:
//! 1. Set up arbitrary sensor array geometries (linear, planar, circular, hemispherical)
//! 2. Use PAM plugin for real-time cavitation field mapping
//! 3. Detect and map sonoluminescence events
//! 4. Reconstruct acoustic fields using various k-Wave compatible algorithms

use kwavers::{
    Grid, Time, HomogeneousMedium, Source, 
    PassiveAcousticMappingPlugin, PAMConfig, ArrayGeometry, BeamformingMethod,
    PluginManager, FdtdPlugin, FdtdConfig,
    PlaneRecon, LineRecon, BowlRecon,
    ReconstructionConfig, ReconstructionAlgorithm, FilterType, InterpolationMethod,
    KwaversResult,
};
use ndarray::Array3;
use std::f64::consts::PI;

fn main() -> KwaversResult<()> {
    // Initialize logging
    kwavers::init_logging()?;
    
    // Create computational grid (100mm x 100mm x 100mm)
    let grid = Grid::new(
        256, 256, 256,  // Grid points
        0.1 / 256.0,    // dx = ~0.39mm
        0.1 / 256.0,    // dy = ~0.39mm
        0.1 / 256.0,    // dz = ~0.39mm
    );
    
    // Create time discretization
    let time = Time::new(1e-7, 10000); // 100ns time step, 10000 steps = 1ms
    
    // Create medium (water)
    let medium = HomogeneousMedium::new(
        998.0,   // density (kg/m³)
        1480.0,  // sound speed (m/s)
        &grid,
        0.5,     // absorption
        10.0,    // absorption power
    );
    
    // Example 1: Hemispherical bowl array for sonoluminescence detection
    println!("\n=== Example 1: Hemispherical Bowl Array ===");
    let bowl_array = create_hemispherical_array_pam(&grid)?;
    
    // Example 2: Linear array for 2D imaging
    println!("\n=== Example 2: Linear Array ===");
    let linear_array = create_linear_array_pam(&grid)?;
    
    // Example 3: Planar array for 3D volumetric imaging
    println!("\n=== Example 3: Planar Array ===");
    let planar_array = create_planar_array_pam(&grid)?;
    
    // Example 4: Circular array for tomographic imaging
    println!("\n=== Example 4: Circular Array ===");
    let circular_array = create_circular_array_pam(&grid)?;
    
    // Example 5: Custom phased array
    println!("\n=== Example 5: Custom Phased Array ===");
    let phased_array = create_phased_array_pam(&grid)?;
    
    // Create plugin manager and register PAM plugin (using hemispherical for main demo)
    let mut plugin_manager = PluginManager::new();
    
    // Add FDTD solver for acoustic wave propagation
    let fdtd_config = FdtdConfig::default();
    plugin_manager.register(Box::new(FdtdPlugin::new(fdtd_config, &grid)?))?;
    
    // Add PAM plugin for cavitation mapping
    plugin_manager.register(Box::new(bowl_array))?;
    
    // Initialize fields
    let mut fields = ndarray::Array4::<f64>::zeros((3, grid.nx, grid.ny, grid.nz));
    
    // Simulate high-intensity focused ultrasound that induces cavitation
    simulate_hifu_induced_cavitation(&mut fields, &grid, &time)?;
    
    // Run simulation with PAM recording
    println!("\nRunning simulation with PAM recording...");
    for step in 0..time.num_steps() {
        let t = step as f64 * time.dt;
        
        // Update physics
        let context = kwavers::PluginContext::new(step, time.num_steps(), 1e6);
        plugin_manager.update_all(&mut fields, &grid, &medium, time.dt, t, &context)?;
        
        if step % 100 == 0 {
            println!("Step {}/{}", step, time.num_steps());
        }
    }
    
    // Extract and analyze results
    analyze_pam_results(&plugin_manager)?;
    
    // Demonstrate reconstruction algorithms
    demonstrate_reconstruction_algorithms(&grid)?;
    
    println!("\nPAM cavitation example completed successfully!");
    Ok(())
}

/// Create hemispherical bowl array PAM configuration
fn create_hemispherical_array_pam(grid: &Grid) -> KwaversResult<PassiveAcousticMappingPlugin> {
    let array_geometry = ArrayGeometry::Hemispherical {
        rings: 5,
        elements_per_ring: vec![16, 32, 32, 32, 16], // Variable density
        radius: 0.05, // 50mm radius
        center: [0.05, 0.05, 0.05], // Center of grid
        focus: [0.05, 0.05, 0.03], // Focus point for cavitation
    };
    
    let config = PAMConfig {
        array_geometry,
        frequency_bands: vec![
            (20e3, 100e3),    // Audible cavitation
            (100e3, 1e6),     // Subharmonics
            (1e6, 10e6),      // Fundamental and harmonics
            (10e6, 50e6),     // Broadband emissions
        ],
        beamforming: BeamformingMethod::PassiveCavitationImaging,
        integration_time: 1e-3,
        detect_cavitation: true,
        detect_sonoluminescence: true,
        spatial_resolution: 0.001, // 1mm resolution
    };
    
    println!("Created hemispherical array with {} elements", 
             config.array_geometry.element_count());
    
    PassiveAcousticMappingPlugin::new(config, grid)
}

/// Create linear array PAM configuration
fn create_linear_array_pam(grid: &Grid) -> KwaversResult<PassiveAcousticMappingPlugin> {
    let array_geometry = ArrayGeometry::Linear {
        elements: 128,
        pitch: 0.0003, // 0.3mm element spacing
        center: [0.05, 0.05, 0.01],
        orientation: [1.0, 0.0, 0.0], // Along x-axis
    };
    
    let config = PAMConfig {
        array_geometry,
        frequency_bands: vec![(1e6, 5e6)],
        beamforming: BeamformingMethod::DelayAndSum,
        integration_time: 1e-3,
        detect_cavitation: true,
        detect_sonoluminescence: false,
        spatial_resolution: 0.0005,
    };
    
    println!("Created linear array with {} elements", 
             config.array_geometry.element_count());
    
    PassiveAcousticMappingPlugin::new(config, grid)
}

/// Create planar array PAM configuration
fn create_planar_array_pam(grid: &Grid) -> KwaversResult<PassiveAcousticMappingPlugin> {
    let array_geometry = ArrayGeometry::Planar {
        elements_x: 32,
        elements_y: 32,
        pitch_x: 0.001,
        pitch_y: 0.001,
        center: [0.05, 0.05, 0.0],
        normal: [0.0, 0.0, 1.0], // Facing +z direction
    };
    
    let config = PAMConfig {
        array_geometry,
        frequency_bands: vec![(0.5e6, 2e6), (2e6, 5e6)],
        beamforming: BeamformingMethod::RobustCapon { diagonal_loading: 0.01 },
        integration_time: 1e-3,
        detect_cavitation: true,
        detect_sonoluminescence: true,
        spatial_resolution: 0.001,
    };
    
    println!("Created planar array with {} elements", 
             config.array_geometry.element_count());
    
    PassiveAcousticMappingPlugin::new(config, grid)
}

/// Create circular array PAM configuration
fn create_circular_array_pam(grid: &Grid) -> KwaversResult<PassiveAcousticMappingPlugin> {
    let array_geometry = ArrayGeometry::Circular {
        elements: 64,
        radius: 0.04, // 40mm radius
        center: [0.05, 0.05, 0.05],
        normal: [0.0, 0.0, 1.0], // In xy-plane
    };
    
    let config = PAMConfig {
        array_geometry,
        frequency_bands: vec![(1e6, 10e6)],
        beamforming: BeamformingMethod::MUSIC { signal_subspace_dim: 4 },
        integration_time: 1e-3,
        detect_cavitation: true,
        detect_sonoluminescence: false,
        spatial_resolution: 0.001,
    };
    
    println!("Created circular array with {} elements", 
             config.array_geometry.element_count());
    
    PassiveAcousticMappingPlugin::new(config, grid)
}

/// Create custom phased array PAM configuration
fn create_phased_array_pam(grid: &Grid) -> KwaversResult<PassiveAcousticMappingPlugin> {
    // Create a custom spiral array pattern
    let mut elements = Vec::new();
    let n_spirals = 3;
    let points_per_spiral = 20;
    
    for spiral in 0..n_spirals {
        let phase_offset = 2.0 * PI * spiral as f64 / n_spirals as f64;
        for i in 0..points_per_spiral {
            let t = i as f64 / points_per_spiral as f64;
            let r = 0.01 + 0.03 * t; // Spiral from 10mm to 40mm
            let theta = 4.0 * PI * t + phase_offset;
            
            elements.push([
                0.05 + r * theta.cos(),
                0.05 + r * theta.sin(),
                0.02 + 0.06 * t, // Rise from 20mm to 80mm
            ]);
        }
    }
    
    let array_geometry = ArrayGeometry::Phased {
        elements: elements.clone(),
        aperture: 0.08, // 80mm effective aperture
    };
    
    let config = PAMConfig {
        array_geometry,
        frequency_bands: vec![(0.1e6, 50e6)], // Broadband
        beamforming: BeamformingMethod::TimeExposureAcoustics,
        integration_time: 1e-3,
        detect_cavitation: true,
        detect_sonoluminescence: true,
        spatial_resolution: 0.002,
    };
    
    println!("Created custom phased array with {} elements in spiral pattern", 
             config.array_geometry.element_count());
    
    PassiveAcousticMappingPlugin::new(config, grid)
}

/// Simulate HIFU-induced cavitation
fn simulate_hifu_induced_cavitation(
    fields: &mut ndarray::Array4<f64>,
    grid: &Grid,
    time: &Time,
) -> KwaversResult<()> {
    println!("\nSimulating HIFU-induced cavitation...");
    
    // Create a focused ultrasound field
    let focus = [0.05, 0.05, 0.03]; // Focus point
    let frequency = 1e6; // 1 MHz
    let pressure_amplitude = 10e6; // 10 MPa - sufficient for cavitation
    
    // Initialize pressure field with focused beam
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                // Distance from focus
                let r = ((x - focus[0]).powi(2) + 
                        (y - focus[1]).powi(2) + 
                        (z - focus[2]).powi(2)).sqrt();
                
                // Focused beam profile calculation
                let beam_profile = (-r * r / (0.01 * 0.01)).exp();
                
                // Initial pressure with focusing
                fields[[0, i, j, k]] = pressure_amplitude * beam_profile;
            }
        }
    }
    
    println!("HIFU field initialized with:");
    println!("  Frequency: {} MHz", frequency / 1e6);
    println!("  Peak pressure: {} MPa", pressure_amplitude / 1e6);
    println!("  Focus: {:?} mm", [focus[0]*1000.0, focus[1]*1000.0, focus[2]*1000.0]);
    
    Ok(())
}

/// Analyze PAM results
fn analyze_pam_results(plugin_manager: &PluginManager) -> KwaversResult<()> {
    println!("\n=== PAM Results Analysis ===");
    
    // Note: In a real implementation, we would extract the PAM plugin
    // and access its cavitation and sonoluminescence maps
    
    println!("Cavitation events detected: [Analysis would show spatial distribution]");
    println!("Sonoluminescence events: [Analysis would show high-intensity collapse events]");
    println!("Peak cavitation activity location: [Would identify focal region]");
    println!("Broadband emission levels: [Would quantify violent collapses]");
    
    Ok(())
}

/// Demonstrate reconstruction algorithms
fn demonstrate_reconstruction_algorithms(grid: &Grid) -> KwaversResult<()> {
    println!("\n=== Reconstruction Algorithm Demonstrations ===");
    
    // Create dummy sensor data for demonstration
    let n_sensors = 64;
    let n_time = 1000;
    let sensor_data = ndarray::Array2::<f64>::zeros((n_sensors, n_time));
    
    // Linear array reconstruction
    println!("\n1. Linear Array Reconstruction (lineRecon):");
    let line_recon = LineRecon::new(
        [1.0, 0.0, 0.0], // Direction
        [0.05, 0.05, 0.05], // Center
        0.001, // Pitch
    );
    
    // Generate sensor positions for linear array
    let mut linear_positions = Vec::new();
    for i in 0..n_sensors {
        let offset = (i as f64 - (n_sensors as f64 - 1.0) / 2.0) * 0.001;
        linear_positions.push([0.05 + offset, 0.05, 0.05]);
    }
    
    let recon_config = ReconstructionConfig {
        sound_speed: 1500.0,
        sampling_frequency: 20e6,
        algorithm: ReconstructionAlgorithm::FilteredBackProjection,
        filter: FilterType::RamLak,
        interpolation: InterpolationMethod::Linear,
    };
    
    println!("  - Algorithm: Filtered Back-Projection");
    println!("  - Filter: Ram-Lak");
    println!("  - Interpolation: Linear");
    
    // Planar array reconstruction
    println!("\n2. Planar Array Reconstruction (planeRecon):");
    let plane_recon = PlaneRecon::new(
        [0.0, 0.0, 1.0], // Normal
        [0.05, 0.05, 0.0], // Center
    );
    println!("  - Geometry: 2D planar array");
    println!("  - Beamforming: Delay-and-sum with solid angle weighting");
    
    // Bowl array reconstruction
    println!("\n3. Bowl Array Reconstruction (bowlRecon):");
    let bowl_recon = BowlRecon::hemispherical(
        [0.05, 0.05, 0.05], // Center
        0.05, // Radius
    );
    println!("  - Geometry: Hemispherical bowl");
    println!("  - Coverage: Full 3D with excellent angular coverage");
    println!("  - Application: Photoacoustic tomography, cavitation imaging");
    
    Ok(())
}