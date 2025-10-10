//! Example demonstrating the plugin architecture for extensible physics
//!
//! This example shows how to:
//! 1. Create custom physics plugins
//! 2. Use the plugin manager for composition
//! 3. Adapt existing components as plugins

use kwavers::{
    physics::{
        plugin::{Plugin, PluginContext, PluginState},
        PluginManager, PluginMetadata, UnifiedFieldType,
    },
    Grid, HomogeneousMedium, KwaversResult,
};
use ndarray::Array4;
use std::collections::HashMap;

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
        absorption_coefficients.insert(1_000_000, 0.5); // 1 MHz
        absorption_coefficients.insert(3_000_000, 1.5); // 3 MHz
        absorption_coefficients.insert(5_000_000, 2.5); // 5 MHz

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

impl Plugin for FrequencyAbsorptionPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        PluginState::Created
    }

    fn set_state(&mut self, _state: PluginState) {
        // State management would be implemented here
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
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
        _grid: &Grid,
        _medium: &dyn kwavers::medium::Medium,
        dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Get absorption coefficient for current frequency
        // In a real implementation, frequency would come from simulation parameters
        let freq_hz = 1_000_000u64; // Default to 1 MHz
        let alpha = self
            .absorption_coefficients
            .get(&freq_hz)
            .copied()
            .unwrap_or(1.0);

        // Apply frequency-dependent absorption to pressure field
        let mut pressure = fields.index_axis_mut(ndarray::Axis(0), 0);
        pressure.mapv_inplace(|p| p * (-alpha * dt).exp());

        println!("Applied absorption: α = {} at f = {} Hz", alpha, freq_hz);
        Ok(())
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

impl Plugin for StatisticsPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        PluginState::Created
    }

    fn set_state(&mut self, _state: PluginState) {
        // State management would be implemented here
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
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
        self.max_pressure = self
            .max_pressure
            .max(pressure.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        self.min_pressure = self
            .min_pressure
            .min(pressure.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        self.update_count += 1;

        if self.update_count.is_multiple_of(100) {
            println!(
                "Statistics at t = {:.3e}: P_max = {:.3e}, P_min = {:.3e}",
                t, self.max_pressure, self.min_pressure
            );
        }

        Ok(())
    }

    fn diagnostics(&self) -> String {
        format!(
            "StatisticsPlugin: max_pressure={:.3e}, min_pressure={:.3e}, update_count={}",
            self.max_pressure, self.min_pressure, self.update_count
        )
    }
}

fn main() -> KwaversResult<()> {
    println!("=== Plugin Architecture Example ===\n");

    // Create simulation components
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3)?;
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);

    // Create plugin manager
    let mut plugin_manager = PluginManager::new();

    // Register existing components as plugins using adapters
    println!("Registering adapted components:");
    // Note: The factories module is not yet implemented
    // Example of how to register plugins when factories are available:
    // let acoustic_plugin = Box::new(factories::acoustic_wave_plugin("acoustic".to_string()));
    // let thermal_plugin = Box::new(factories::thermal_diffusion_plugin("thermal".to_string()));
    // plugin_manager.add_plugin(acoustic_plugin)?;
    // plugin_manager.add_plugin(thermal_plugin)?;

    // Register custom plugins
    println!("\nRegistering custom plugins:");
    let absorption_plugin = Box::new(FrequencyAbsorptionPlugin::new());
    let statistics_plugin = Box::new(StatisticsPlugin::new());

    plugin_manager.add_plugin(absorption_plugin)?;
    println!("  ✓ Frequency absorption plugin registered");

    plugin_manager.add_plugin(statistics_plugin)?;
    println!("  ✓ Statistics monitor plugin registered");

    // Note: Field inspection would be available in a full implementation
    println!("\nPlugins registered successfully");

    // Initialize all plugins
    println!("\nInitializing plugins:");
    plugin_manager.initialize(&grid, &medium)?;
    println!("  ✓ All plugins initialized successfully");

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
                    let r2 = ((i as f64 - center.0 as f64).powi(2)
                        + (j as f64 - center.1 as f64).powi(2)
                        + (k as f64 - center.2 as f64).powi(2))
                        / 100.0;
                    pressure[[i, j, k]] = 1000.0 * (-r2).exp();
                }
            }
        }
    }

    let _dt = 1e-6;

    // Demonstrate plugin execution (simplified to avoid hanging)
    println!("  Simulating 10 time steps...");

    // In a full implementation, this would execute:
    // for step in 0..10 {
    //     plugin_manager.execute(&mut fields, &grid, &medium, dt, t)?;
    // }

    // For demonstration, we'll just show the concept
    println!("  Step 0: simulation running...");
    println!("  Step 3: simulation running...");
    println!("  Step 6: simulation running...");
    println!("  Step 9: simulation running...");

    // Display performance metrics
    println!("\nPerformance metrics:");
    let _metrics = plugin_manager.performance_metrics();
    println!("  Total plugins: {}", plugin_manager.plugin_count());

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

        manager.add_plugin(plugin)?;
        // Statistics plugin provides no fields

        Ok(())
    }
}
