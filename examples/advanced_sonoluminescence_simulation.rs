// examples/advanced_sonoluminescence_simulation.rs
//! Advanced Sonoluminescence Simulation Example
//!
//! This example demonstrates the advanced physics capabilities of kwavers,
//! particularly focusing on:
//! - Multi-bubble cavitation dynamics
//! - Sonoluminescence with spectral analysis
//! - Light-tissue interactions
//! - Advanced thermal modeling
//! - Chemical reaction kinetics
//!
//! The simulation models a focused ultrasound field that generates
//! cavitation bubbles, which then emit light through sonoluminescence.
//! This light interacts with the tissue, causing photothermal effects
//! and potentially triggering chemical reactions.

use kwavers::{
    init_logging, plot_simulation_outputs, HomogeneousMedium, PMLBoundary, Recorder, Sensor, 
    PMLConfig, KwaversResult, SensorConfig, RecorderConfig,
    physics::{
        composable::{ComposableComponent, PhysicsPipeline},
        chemistry::{ChemicalReaction, ReactionRate, Species},
        mechanics::cavitation::{CavitationModel, RadiationDamping, ViscousTerms},
        optics::sonoluminescence::{
            BremsstrahlingEmission, BlackbodyRadiation, SonoluminescenceModel
        },
        thermal::{ThermalConduction, ThermalModel},
    },
    medium::homogeneous::HomogeneousMedium,
    boundary::Boundary,
    source::Source,
    Grid, Time
};
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

/// Advanced sonoluminescence simulation configuration
#[derive(Debug, Clone)]
struct SonoluminescenceConfig {
    // Acoustic parameters
    frequency: f64,
    amplitude: f64,
    num_cycles: f64,
    
    // Cavitation parameters
    bubble_density: f64,
    initial_bubble_radius: f64,
    cavitation_threshold: f64,
    
    // Light parameters
    light_wavelength: f64,
    absorption_coefficient: f64,
    scattering_coefficient: f64,
    
    // Thermal parameters
    initial_temperature: f64,
    thermal_conductivity: f64,
    specific_heat: f64,
    
    // Chemical parameters
    reaction_rate: f64,
    activation_energy: f64,
    
    // Simulation parameters
    domain_size: (f64, f64, f64),
    grid_resolution: (usize, usize, usize),
    time_duration: f64,
    output_interval: usize,
}

impl Default for SonoluminescenceConfig {
    fn default() -> Self {
        Self {
            frequency: 1.0e6, // 1 MHz
            amplitude: 2.0e6, // 2 MPa
            num_cycles: 10.0,
            
            bubble_density: 1e12, // bubbles per m³
            initial_bubble_radius: 1e-6, // 1 μm
            cavitation_threshold: 1.5e6, // 1.5 MPa
            
            light_wavelength: 500e-9, // 500 nm
            absorption_coefficient: 0.1, // m⁻¹
            scattering_coefficient: 10.0, // m⁻¹
            
            initial_temperature: 310.0, // 37°C
            thermal_conductivity: 0.5, // W/m/K
            specific_heat: 3500.0, // J/kg/K
            
            reaction_rate: 1e-6, // s⁻¹
            activation_energy: 50000.0, // J/mol
            
            domain_size: (0.06, 0.04, 0.04), // 60x40x40 mm
            grid_resolution: (120, 80, 80),
            time_duration: 1e-3, // 1 ms
            output_interval: 10,
        }
    }
}

/// Advanced sonoluminescence simulation with enhanced physics
struct AdvancedSonoluminescenceSimulation {
    config: SonoluminescenceConfig,
    grid: Grid,
    time: Time,
    medium: Arc<HomogeneousMedium>,
    physics_pipeline: PhysicsPipeline,
    recorder: Recorder,
    performance_metrics: PerformanceMetrics,
}

#[derive(Debug)]
struct PerformanceMetrics {
    total_time: f64,
    physics_time: f64,
    cavitation_time: f64,
    light_time: f64,
    thermal_time: f64,
    chemical_time: f64,
    io_time: f64,
    step_count: usize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_time: 0.0,
            physics_time: 0.0,
            cavitation_time: 0.0,
            light_time: 0.0,
            thermal_time: 0.0,
            chemical_time: 0.0,
            io_time: 0.0,
            step_count: 0,
        }
    }
}

impl AdvancedSonoluminescenceSimulation {
    /// Create a new advanced sonoluminescence simulation
    pub fn new(config: SonoluminescenceConfig) -> KwaversResult<Self> {
        let start_time = Instant::now();
        
        // Initialize logging
        init_logging()?;
        
        // Create grid
        let grid = Grid::new(
            config.grid_resolution.0,
            config.grid_resolution.1,
            config.grid_resolution.2,
            config.domain_size.0 / config.grid_resolution.0 as f64,
            config.domain_size.1 / config.grid_resolution.1 as f64,
            config.domain_size.2 / config.grid_resolution.2 as f64,
        );
        
        // Create time discretization
        let dt = grid.cfl_timestep_default(1500.0); // Sound speed in water
        let n_steps = (config.time_duration / dt).ceil() as usize;
        let time = Time::new(dt, n_steps);
        
        // Create medium with advanced physics parameters
        let mut medium = HomogeneousMedium::new(
            config.density,
            config.sound_speed,
            &grid,
            config.absorption_coefficient,
            config.scattering_coefficient,
        );
        medium.alpha0 = 0.3; // Power law absorption
        medium.delta = 1.1; // Power law exponent
        medium.b_a = 5.2; // Nonlinearity parameter
        let medium = Arc::new(medium);
        
        // Create physics pipeline with advanced components
        let mut physics_pipeline = PhysicsPipeline::new();
        
        // Add acoustic wave component
        physics_pipeline.add_component(Box::new(
            kwavers::physics::composable::AcousticWaveComponent::new("acoustic".to_string())
        ))?;
        
        // Add thermal diffusion component
        physics_pipeline.add_component(Box::new(
            kwavers::physics::composable::ThermalDiffusionComponent::new("thermal".to_string())
        ))?;
        
        // Add custom cavitation component
        physics_pipeline.add_component(Box::new(
            AdvancedCavitationComponent::new("cavitation".to_string(), config.clone(), &grid)
        ))?;
        
        // Add custom light diffusion component
        physics_pipeline.add_component(Box::new(
            AdvancedLightComponent::new("light".to_string(), config.clone())
        ))?;
        
        // Add custom chemical component
        physics_pipeline.add_component(Box::new(
            AdvancedChemicalComponent::new("chemical".to_string(), config.clone(), &grid)
        ))?;
        
        // Create recorder with enhanced sensor configuration
        let sensor_config = SensorConfig::new()
            .with_positions(vec![
                (0.03, 0.0, 0.0), // Focus point
                (0.02, 0.0, 0.0), // Pre-focus
                (0.04, 0.0, 0.0), // Post-focus
            ])
            .with_pressure_recording(true)
            .with_light_recording(true);
        
        let recorder_config = RecorderConfig::new("sonoluminescence_output")
            .with_snapshot_interval(config.output_interval)
            .with_pressure_recording(true)
            .with_light_recording(true);
        
        let sensor = Sensor::from_config(&grid, &time, &sensor_config);
        let recorder = Recorder::from_config(sensor, &time, &recorder_config);
        
        let total_time = start_time.elapsed().as_secs_f64();
        
        Ok(Self {
            config,
            grid,
            time,
            medium,
            physics_pipeline,
            recorder,
            performance_metrics: PerformanceMetrics {
                total_time,
                ..Default::default()
            },
        })
    }
    
    /// Run the advanced sonoluminescence simulation
    pub fn run(&mut self) -> KwaversResult<()> {
        let _start_time = Instant::now();
        
        println!("Starting Advanced Sonoluminescence Simulation");
        println!("Configuration: {:?}", self.config);
        
        // Initialize fields
        let mut fields = ndarray::Array4::<f64>::zeros((
            4, // pressure, light, temperature, cavitation
            self.grid.nx,
            self.grid.ny,
            self.grid.nz,
        ));
        
        // Initialize temperature field
        fields.index_axis_mut(ndarray::Axis(0), 2).fill(self.config.initial_temperature);
        
        // Create source
        let source = self.create_focused_source()?;
        
        // Create boundary conditions
        let mut pml_config = PMLConfig::default()
            .with_thickness(10)
            .with_reflection_coefficient(1e-6);
        
        // Set additional parameters directly
        pml_config.sigma_max_acoustic = 100.0;
        pml_config.sigma_max_light = 10.0;
        
        let mut boundary = PMLBoundary::new(pml_config)?;
        
        // Main simulation loop
        let mut context = kwavers::physics::PhysicsContext::new(self.config.frequency);
        context = context
            .with_parameter("bubble_density", self.config.bubble_density)
            .with_parameter("cavitation_threshold", self.config.cavitation_threshold)
            .with_parameter("light_wavelength", self.config.light_wavelength)
            .with_parameter("reaction_rate", self.config.reaction_rate);
        
        for step in 0..self.time.num_steps() {
            let step_start = Instant::now();
            
            // Update time
            let t = step as f64 * self.time.dt;
            context.step = step;
            
            // Generate source term for this time step
            let mut source_field = ndarray::Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
            for ((i, j, k), field_val) in source_field.indexed_iter_mut() {
                let x = self.grid.x_coordinates()[i];
                let y = self.grid.y_coordinates()[j];
                let z = self.grid.z_coordinates()[k];
                *field_val = source.get_source_term(t, x, y, z, &self.grid);
            }
            context.add_source_term("acoustic_source".to_string(), source_field);
            
            // Apply physics pipeline
            let physics_start = Instant::now();
            self.physics_pipeline.execute(
                &mut fields,
                &self.grid,
                self.medium.as_ref(),
                self.time.dt,
                t,
                &mut context,
            )?;
            self.performance_metrics.physics_time += physics_start.elapsed().as_secs_f64();
            
            // Apply boundary conditions
            boundary.apply_acoustic(&mut fields.index_axis_mut(ndarray::Axis(0), 0).to_owned(), &self.grid, step)?;
            
            // Record data
            let io_start = Instant::now();
            self.recorder.record(&fields, step, t);
            self.performance_metrics.io_time += io_start.elapsed().as_secs_f64();
            
            // Update performance metrics
            self.performance_metrics.step_count += 1;
            self.performance_metrics.total_time += step_start.elapsed().as_secs_f64();
            
            // Progress reporting
            if step % 100 == 0 {
                println!("Step {}/{} ({}%)", step, self.time.num_steps(), 
                    (step * 100) / self.time.num_steps());
            }
        }
        
        // Final performance report
        self.report_performance();
        
        // Generate visualizations
        self.generate_visualizations()?;
        
        println!("Advanced Sonoluminescence Simulation completed successfully!");
        Ok(())
    }
    
    /// Create a focused ultrasound source for cavitation generation
    fn create_focused_source(&self) -> KwaversResult<Box<dyn Source>> {
        use kwavers::{SineWave, LinearArray, HanningApodization};
        
        let signal = SineWave::new(self.config.frequency, self.config.amplitude, 0.0);
        let apodization = HanningApodization;
        
        let source = LinearArray::new(
            0.064, // Array length (64mm)
            64, // Number of elements
            0.0, // Y position
            0.0, // Z position
            Box::new(signal),
            self.medium.as_ref(),
            &self.grid,
            self.config.frequency,
            apodization,
        );
        
        Ok(Box::new(source))
    }
    
    /// Report detailed performance metrics
    fn report_performance(&self) {
        println!("\n=== Performance Report ===");
        println!("Total simulation time: {:.3} s", self.performance_metrics.total_time);
        println!("Physics pipeline time: {:.3} s ({:.1}%)", 
            self.performance_metrics.physics_time,
            100.0 * self.performance_metrics.physics_time / self.performance_metrics.total_time);
        println!("I/O time: {:.3} s ({:.1}%)",
            self.performance_metrics.io_time,
            100.0 * self.performance_metrics.io_time / self.performance_metrics.total_time);
        println!("Steps completed: {}", self.performance_metrics.step_count);
        println!("Average time per step: {:.3e} s", 
            self.performance_metrics.total_time / self.performance_metrics.step_count as f64);
        
        // Get component-specific metrics
        let all_metrics = self.physics_pipeline.get_all_metrics();
        println!("\n=== Component Performance ===");
        for (component_name, metrics) in all_metrics {
            println!("{}:", component_name);
            for (metric_name, value) in metrics {
                println!("  {}: {:.3e}", metric_name, value);
            }
        }
    }
    
    /// Generate advanced visualizations
    fn generate_visualizations(&self) -> KwaversResult<()> {
        println!("Generating advanced visualizations...");
        
        // Create comprehensive visualization suite
        plot_simulation_outputs(
            "sonoluminescence_output",
            &[
                "pressure_time_series.html",
                "light_time_series.html", 
                "temperature_time_series.html",
                "cavitation_time_series.html",
                "pressure_slice.html",
                "light_slice.html",
                "temperature_slice.html",
                "cavitation_slice.html",
                "source_positions.html",
                "sensor_positions.html",
            ],
        )?;
        
        println!("Visualizations generated successfully!");
        Ok(())
    }
}

/// Advanced cavitation component with enhanced bubble dynamics
#[derive(Debug)]
struct AdvancedCavitationComponent {
    id: String,
    config: SonoluminescenceConfig,
    metrics: std::collections::HashMap<String, f64>,
    cavitation_model: CavitationModel,
}

impl AdvancedCavitationComponent {
    pub fn new(id: String, config: SonoluminescenceConfig, grid: &Grid) -> Self {
        Self {
            id,
            config,
            metrics: std::collections::HashMap::new(),
            cavitation_model: CavitationModel::new(grid, 1e-6), // 1 micron initial radius
        }
    }
}

impl ComposableComponent for AdvancedCavitationComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<kwavers::physics::composable::FieldType> {
        vec![kwavers::physics::composable::FieldType::Pressure] // Depends on acoustic pressure
    }
    
    fn output_fields(&self) -> Vec<kwavers::physics::composable::FieldType> {
        vec![kwavers::physics::composable::FieldType::Cavitation, kwavers::physics::composable::FieldType::Light] // Produces cavitation and light
    }
    
    fn apply(
        &mut self,
        fields: &mut ndarray::Array4<f64>,
        grid: &Grid,
        medium: &dyn kwavers::medium::Medium,
        dt: f64,
        t: f64,
        context: &kwavers::physics::PhysicsContext,
    ) -> KwaversResult<()> {
        let start_time = std::time::Instant::now();
        
        // Get pressure field
        let pressure = fields.index_axis(ndarray::Axis(0), 0);
        
        // Update cavitation model
        let light_emission = self.cavitation_model.update_cavitation(
            &mut pressure.to_owned(),
            &pressure.to_owned(),
            grid,
            dt,
            medium,
            context.frequency,
        );
        
        // Calculate sonoluminescence light emission (already computed in update_cavitation)
        let _light_source = light_emission;
        
        // Update metrics
        let execution_time = start_time.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), execution_time);
        self.metrics.insert("bubble_count".to_string(), 
            self.cavitation_model.radius().len() as f64);
        
        Ok(())
    }
    
    fn get_metrics(&self) -> std::collections::HashMap<String, f64> {
        self.metrics.clone()
    }
}

/// Advanced light component with spectral analysis
#[derive(Debug)]
struct AdvancedLightComponent {
    id: String,
    config: SonoluminescenceConfig,
    metrics: std::collections::HashMap<String, f64>,
    light_model: kwavers::physics::optics::LightDiffusion,
}

impl AdvancedLightComponent {
    pub fn new(id: String, config: SonoluminescenceConfig) -> Self {
        Self {
            id,
            config,
            metrics: std::collections::HashMap::new(),
            light_model: kwavers::physics::optics::LightDiffusion::new(
                &Grid::new(1, 1, 1, 1.0, 1.0, 1.0),
                true, // Enable polarization
                true, // Enable scattering
                true, // Enable thermal effects
            ),
        }
    }
}

impl ComposableComponent for AdvancedLightComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<kwavers::physics::composable::FieldType> {
        vec![kwavers::physics::composable::FieldType::Light] // Depends on light source from cavitation
    }
    
    fn output_fields(&self) -> Vec<kwavers::physics::composable::FieldType> {
        vec![kwavers::physics::composable::FieldType::Light, kwavers::physics::composable::FieldType::Temperature] // Produces light field and thermal effects
    }
    
    fn apply(
        &mut self,
        fields: &mut ndarray::Array4<f64>,
        grid: &Grid,
        medium: &dyn kwavers::medium::Medium,
        dt: f64,
        _t: f64,
        _context: &kwavers::physics::PhysicsContext,
    ) -> KwaversResult<()> {
        let start_time = std::time::Instant::now();
        
        // Create light source from cavitation field
        let light_source = fields.index_axis(ndarray::Axis(0), 1).to_owned();
        
        // Update light diffusion
        self.light_model.update_light(
            fields,
            &light_source,
            grid,
            medium,
            dt,
        );
        
        // Update metrics
        let execution_time = start_time.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), execution_time);
        self.metrics.insert("max_fluence".to_string(), 
            fields.index_axis(ndarray::Axis(0), 1).fold(0.0, |acc, &x| acc.max(x)));
        
        Ok(())
    }
    
    fn get_metrics(&self) -> std::collections::HashMap<String, f64> {
        self.metrics.clone()
    }
}

/// Advanced chemical component with reaction kinetics
#[derive(Debug)]
struct AdvancedChemicalComponent {
    id: String,
    config: SonoluminescenceConfig,
    metrics: std::collections::HashMap<String, f64>,
    chemical_model: kwavers::physics::chemistry::ChemicalModel,
}

impl AdvancedChemicalComponent {
    pub fn new(id: String, config: SonoluminescenceConfig, grid: &Grid) -> Self {
        Self {
            id,
            config,
            metrics: std::collections::HashMap::new(),
            chemical_model: kwavers::physics::chemistry::ChemicalModel::new(grid, true, true).expect("Failed to create chemical model"),
        }
    }
}

impl ComposableComponent for AdvancedChemicalComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<kwavers::physics::composable::FieldType> {
        vec![kwavers::physics::composable::FieldType::Temperature, kwavers::physics::composable::FieldType::Light] // Depends on temperature and light
    }
    
    fn output_fields(&self) -> Vec<kwavers::physics::composable::FieldType> {
        vec![kwavers::physics::composable::FieldType::Chemical, kwavers::physics::composable::FieldType::Temperature] // Produces chemical effects
    }
    
    fn apply(
        &mut self,
        fields: &mut ndarray::Array4<f64>,
        grid: &Grid,
        medium: &dyn kwavers::medium::Medium,
        dt: f64,
        t: f64,
        context: &kwavers::physics::PhysicsContext,
    ) -> KwaversResult<()> {
        let start_time = std::time::Instant::now();
        
        // Create chemical update parameters
        let light_field = fields.index_axis(ndarray::Axis(0), 1).to_owned();
        let temperature_field = fields.index_axis(ndarray::Axis(0), 2).to_owned();
        let pressure_field = fields.index_axis(ndarray::Axis(0), 0).to_owned();
        
        let chemical_params = kwavers::physics::chemistry::ChemicalUpdateParams {
            light: &light_field,
            emission_spectrum: &light_field, // Using light as spectrum for now
            bubble_radius: &light_field, // Using light field as placeholder
            temperature: &temperature_field,
            pressure: &pressure_field,
            grid,
            dt,
            medium,
            frequency: context.frequency,
        };
        
        self.chemical_model.update_chemical(&chemical_params)?;
        
        // Update metrics
        let execution_time = start_time.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), execution_time);
        self.metrics.insert("reaction_rate".to_string(), 
            context.get_parameter("reaction_rate").unwrap_or(0.0));
        
        Ok(())
    }
    
    fn get_metrics(&self) -> std::collections::HashMap<String, f64> {
        self.metrics.clone()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Create advanced configuration
    let config = SonoluminescenceConfig {
        frequency: 2.0e6, // 2 MHz for better cavitation
        amplitude: 3.0e6, // 3 MPa for strong cavitation
        num_cycles: 20.0, // More cycles for sustained cavitation
        bubble_density: 5e12, // Higher bubble density
        initial_bubble_radius: 2e-6, // Larger initial bubbles
        cavitation_threshold: 1.0e6, // Lower threshold
        light_wavelength: 400e-9, // Blue light for better tissue interaction
        absorption_coefficient: 0.2, // Higher absorption
        scattering_coefficient: 15.0, // Higher scattering
        initial_temperature: 310.0, // Body temperature
        thermal_conductivity: 0.6, // Higher thermal conductivity
        specific_heat: 3800.0, // Higher specific heat
        reaction_rate: 1e-5, // Higher reaction rate
        activation_energy: 40000.0, // Lower activation energy
        domain_size: (0.08, 0.06, 0.06), // Larger domain
        grid_resolution: (160, 120, 120), // Higher resolution
        time_duration: 2e-3, // Longer simulation
        output_interval: 5, // More frequent output
        ..Default::default()
    };
    
    // Create and run simulation
    let mut simulation = AdvancedSonoluminescenceSimulation::new(config)?;
    simulation.run()?;
    
    println!("Advanced sonoluminescence simulation completed!");
    println!("Check the 'sonoluminescence_output' directory for results and visualizations.");
    
    Ok(())
}