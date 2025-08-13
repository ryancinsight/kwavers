//! Example demonstrating the plugin architecture for extensible physics
//! 
//! This example shows how to:
//! 1. Create custom physics plugins
//! 2. Use the plugin manager for composition
//! 3. Adapt existing components as plugins

use kwavers::{
    KwaversResult,
    physics::{
        PhysicsPlugin, PluginManager, PluginMetadata, PluginContext,
        UnifiedFieldType,
    },
    Grid, HomogeneousMedium,
};
use ndarray::Array4;
use std::collections::{HashMap, BTreeMap};

/// Custom plugin for modeling frequency-dependent absorption
#[derive(Debug)]
struct FrequencyAbsorptionPlugin {
    metadata: PluginMetadata,
    absorption_coefficients: HashMap<u64, f64>, // frequency in Hz -> absorption coefficient
}

impl FrequencyAbsorptionPlugin {
    fn new() -> Self {
        let mut absorption_coefficients = HashMap::new();
        // Example tissue absorption at different frequencies
        absorption_coefficients.insert(1_000_000, 0.5);   // 1 MHz
        absorption_coefficients.insert(3_000_000, 1.5);   // 3 MHz
        absorption_coefficients.insert(5_000_000, 2.5);   // 5 MHz
        
        Self {
            metadata: PluginMetadata {
                id: "frequency_absorption".to_string(),
                name: "Frequency-Dependent Absorption".to_string(),
                version: "1.0.0".to_string(),
                description: "Models tissue absorption as a function of frequency".to_string(),
                author: "Example Author".to_string(),
                license: "MIT".to_string(),
            },
            absorption_coefficients,
        }
    }
}

impl PhysicsPlugin for FrequencyAbsorptionPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> kwavers::physics::plugin::PluginState {
        kwavers::physics::plugin::PluginState::Created
    }
    
    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }
    
    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![] // This plugin modifies pressure in-place
    }
    
    fn initialize(
        &mut self,
        _grid: &Grid,
        _medium: &dyn kwavers::medium::Medium,
    ) -> KwaversResult<()> {
        println!("Initializing frequency-dependent absorption plugin");
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        _medium: &dyn kwavers::medium::Medium,
        dt: f64,
        _t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()> {
        // Get absorption coefficient for current frequency
        let freq_hz = context.frequency as u64;
        let alpha = self.absorption_coefficients
            .get(&freq_hz)
            .copied()
            .unwrap_or(1.0);
        
        // Apply frequency-dependent absorption to pressure field
        let mut pressure = fields.index_axis_mut(ndarray::Axis(0), 0);
        pressure.mapv_inplace(|p| p * (-alpha * dt).exp());
        
        println!("Applied absorption: α = {} at f = {} Hz", alpha, context.frequency);
        Ok(())
    }
    
    fn performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("absorption_coefficient".to_string(), 1.0);
        metrics
    }
    
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
        Box::new(Self {
            metadata: self.metadata.clone(),
            absorption_coefficients: self.absorption_coefficients.clone(),
        })
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Custom plugin for monitoring simulation statistics
#[derive(Debug)]
struct StatisticsPlugin {
    metadata: PluginMetadata,
    max_pressure: f64,
    min_pressure: f64,
    update_count: usize,
}

impl StatisticsPlugin {
    fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "statistics".to_string(),
                name: "Statistics Monitor".to_string(),
                version: "1.0.0".to_string(),
                description: "Monitors field statistics during simulation".to_string(),
                author: "Example Author".to_string(),
                license: "MIT".to_string(),
            },
            max_pressure: f64::NEG_INFINITY,
            min_pressure: f64::INFINITY,
            update_count: 0,
        }
    }
}

impl PhysicsPlugin for StatisticsPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> kwavers::physics::plugin::PluginState {
        kwavers::physics::plugin::PluginState::Created
    }
    
    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }
    
    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![]
    }
    
    fn initialize(
        &mut self,
        _grid: &Grid,
        _medium: &dyn kwavers::medium::Medium,
    ) -> KwaversResult<()> {
        println!("Initializing statistics monitor");
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        _grid: &Grid,
        _medium: &dyn kwavers::medium::Medium,
        _dt: f64,
        t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        let pressure = fields.index_axis(ndarray::Axis(0), 0);
        
        // Update statistics
        self.max_pressure = self.max_pressure.max(pressure.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        self.min_pressure = self.min_pressure.min(pressure.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        self.update_count += 1;
        
        if self.update_count % 100 == 0 {
            println!(
                "Statistics at t = {:.3e}: P_max = {:.3e}, P_min = {:.3e}",
                t, self.max_pressure, self.min_pressure
            );
        }
        
        Ok(())
    }
    
    fn performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("max_pressure".to_string(), self.max_pressure);
        metrics.insert("min_pressure".to_string(), self.min_pressure);
        metrics.insert("update_count".to_string(), self.update_count as f64);
        metrics
    }
    
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
        Box::new(Self {
            metadata: self.metadata.clone(),
            max_pressure: self.max_pressure,
            min_pressure: self.min_pressure,
            update_count: self.update_count,
        })
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn main() -> KwaversResult<()> {
    println!("=== Plugin Architecture Example ===\n");
    
    // Create simulation components
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);
    
    // Create plugin manager
    let mut plugin_manager = PluginManager::new();
    
    // Register existing components as plugins using adapters
    println!("Registering adapted components:");
    // Note: The factories module is not yet implemented
    // Example of how to register plugins when factories are available:
    // let acoustic_plugin = Box::new(factories::acoustic_wave_plugin("acoustic".to_string()));
    // let thermal_plugin = Box::new(factories::thermal_diffusion_plugin("thermal".to_string()));
    // plugin_manager.register(acoustic_plugin)?;
    // plugin_manager.register(thermal_plugin)?;
    
    // Register custom plugins
    println!("\nRegistering custom plugins:");
    let absorption_plugin = Box::new(FrequencyAbsorptionPlugin::new());
    let statistics_plugin = Box::new(StatisticsPlugin::new());
    
    plugin_manager.register(absorption_plugin)?;
    println!("  ✓ Frequency absorption plugin registered");
    
    plugin_manager.register(statistics_plugin)?;
    println!("  ✓ Statistics monitor plugin registered");
    
    // Show available fields
    println!("\nAvailable fields from all plugins:");
    for field in plugin_manager.available_fields() {
        println!("  - {}", field);
    }
    
    // Validate plugin configuration
    println!("\nValidating plugin configuration:");
    let validation = plugin_manager.validate_all(&grid, &medium);
    if validation.is_valid {
        println!("  ✓ All plugins validated successfully");
    } else {
        println!("  ✗ Validation errors:");
        for error in &validation.errors {
            println!("    - {}", error);
        }
    }
    
    // Initialize all plugins
    println!("\nInitializing plugins:");
    plugin_manager.initialize_all(&grid, &medium)?;
    
    // Simulate a few time steps
    println!("\nRunning simulation with plugin system:");
    let mut fields = Array4::zeros((4, grid.nx, grid.ny, grid.nz));
    
    // Initialize pressure field with a Gaussian pulse
    {
        let mut pressure = fields.index_axis_mut(ndarray::Axis(0), 0);
        let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let r2 = ((i as f64 - center.0 as f64).powi(2) +
                             (j as f64 - center.1 as f64).powi(2) +
                             (k as f64 - center.2 as f64).powi(2)) / 100.0;
                    pressure[[i, j, k]] = 1000.0 * (-r2).exp();
                }
            }
        }
    }
    
    let dt = 1e-6;
    let context = PluginContext::new(0, 100, 3e6);
    
    for step in 0..10 {
        let t = step as f64 * dt;
        let step_context = PluginContext::new(step, 100, 3e6);
        
        plugin_manager.update_all(&mut fields, &grid, &medium, dt, t, &step_context)?;
    }
    
    // Display final metrics
    println!("\nFinal plugin metrics:");
    let all_metrics = plugin_manager.get_all_metrics();
    for (plugin_id, metrics) in all_metrics {
        println!("  Plugin '{}' metrics:", plugin_id);
        for (key, value) in metrics {
            println!("    - {}: {:.3e}", key, value);
        }
    }
    
    println!("\n=== Plugin system demonstration complete ===");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_custom_plugin_creation() {
        let plugin = FrequencyAbsorptionPlugin::new();
        assert_eq!(plugin.metadata().id, "frequency_absorption");
        assert_eq!(plugin.required_fields(), vec![UnifiedFieldType::Pressure]);
        assert!(plugin.provided_fields().is_empty());
    }
    
    #[test]
    fn test_plugin_manager_workflow() -> KwaversResult<()> {
        let mut manager = PluginManager::new();
        let plugin = Box::new(StatisticsPlugin::new());
        
        manager.register(plugin)?;
        assert_eq!(manager.available_fields().len(), 0); // Statistics plugin provides no fields
        
        Ok(())
    }
}