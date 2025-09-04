//! In-memory data structures for physics simulations
//!
//! This module provides efficient data structures optimized for zero-copy operations,
//! SIMD processing, and memory layout optimization for physics computations.

use ndarray::{Array3, ArrayView3, ArrayViewMut3, Axis};
use std::mem;
use uom::si::f64::*;

/// Memory-aligned field storage for optimal SIMD performance
#[repr(align(32))] // AVX2 alignment
#[derive(Debug, Clone)]
pub struct AlignedField {
    data: Array3<f64>,
    name: String,
}

impl AlignedField {
    /// Create new aligned field with given dimensions
    pub fn new(name: String, nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            data: Array3::zeros((nx, ny, nz)),
            name,
        }
    }
    
    /// Create field filled with specific value
    pub fn filled(name: String, nx: usize, ny: usize, nz: usize, value: f64) -> Self {
        Self {
            data: Array3::from_elem((nx, ny, nz), value),
            name,
        }
    }
    
    /// Get zero-copy view of the data
    pub fn view(&self) -> ArrayView3<f64> {
        self.data.view()
    }
    
    /// Get mutable zero-copy view of the data
    pub fn view_mut(&mut self) -> ArrayViewMut3<f64> {
        self.data.view_mut()
    }
    
    /// Get field name for debugging
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get raw data slice for SIMD operations
    pub fn as_slice(&self) -> &[f64] {
        self.data.as_slice().unwrap_or(&[])
    }
    
    /// Get mutable raw data slice for SIMD operations
    pub fn as_slice_mut(&mut self) -> &mut [f64] {
        self.data.as_slice_mut().unwrap_or(&mut [])
    }
    
    /// Check memory alignment for SIMD
    pub fn is_simd_aligned(&self) -> bool {
        let ptr = self.data.as_ptr() as usize;
        ptr % 32 == 0 // Check AVX2 alignment
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() * mem::size_of::<f64>()
    }
}

/// Acoustics field collection with typed access
#[derive(Debug)]
pub struct AcousticFields {
    pub pressure: AlignedField,
    pub velocity_x: AlignedField,
    pub velocity_y: AlignedField,
    pub velocity_z: AlignedField,
    pub density: AlignedField,
    pub sound_speed: AlignedField,
}

impl AcousticFields {
    /// Create new acoustic field collection
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            pressure: AlignedField::new("pressure".to_string(), nx, ny, nz),
            velocity_x: AlignedField::new("velocity_x".to_string(), nx, ny, nz),
            velocity_y: AlignedField::new("velocity_y".to_string(), nx, ny, nz),
            velocity_z: AlignedField::new("velocity_z".to_string(), nx, ny, nz),
            density: AlignedField::filled("density".to_string(), nx, ny, nz, 1000.0), // Default water density
            sound_speed: AlignedField::filled("sound_speed".to_string(), nx, ny, nz, 1500.0), // Default water sound speed
        }
    }
    
    /// Initialize with medium properties
    pub fn with_medium_properties(nx: usize, ny: usize, nz: usize, density: f64, sound_speed: f64) -> Self {
        Self {
            pressure: AlignedField::new("pressure".to_string(), nx, ny, nz),
            velocity_x: AlignedField::new("velocity_x".to_string(), nx, ny, nz),
            velocity_y: AlignedField::new("velocity_y".to_string(), nx, ny, nz),
            velocity_z: AlignedField::new("velocity_z".to_string(), nx, ny, nz),
            density: AlignedField::filled("density".to_string(), nx, ny, nz, density),
            sound_speed: AlignedField::filled("sound_speed".to_string(), nx, ny, nz, sound_speed),
        }
    }
    
    /// Get total memory usage
    pub fn total_memory_usage(&self) -> usize {
        self.pressure.memory_usage() 
            + self.velocity_x.memory_usage()
            + self.velocity_y.memory_usage() 
            + self.velocity_z.memory_usage()
            + self.density.memory_usage()
            + self.sound_speed.memory_usage()
    }
    
    /// Check if all fields are properly aligned for SIMD
    pub fn is_simd_ready(&self) -> bool {
        self.pressure.is_simd_aligned()
            && self.velocity_x.is_simd_aligned()
            && self.velocity_y.is_simd_aligned()
            && self.velocity_z.is_simd_aligned()
    }
}

/// Thermal field collection for heat transfer simulations
#[derive(Debug)]
pub struct ThermalFields {
    pub temperature: AlignedField,
    pub heat_source: AlignedField,
    pub thermal_conductivity: AlignedField,
    pub specific_heat: AlignedField,
    pub perfusion_rate: AlignedField,
}

impl ThermalFields {
    /// Create thermal fields with physiological defaults
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            temperature: AlignedField::filled("temperature".to_string(), nx, ny, nz, 37.0), // Body temperature °C
            heat_source: AlignedField::new("heat_source".to_string(), nx, ny, nz),
            thermal_conductivity: AlignedField::filled("thermal_conductivity".to_string(), nx, ny, nz, 0.52), // Tissue thermal conductivity W/m/K
            specific_heat: AlignedField::filled("specific_heat".to_string(), nx, ny, nz, 3600.0), // Tissue specific heat J/kg/K  
            perfusion_rate: AlignedField::filled("perfusion_rate".to_string(), nx, ny, nz, 0.5), // Blood perfusion kg/m³/s
        }
    }
    
    /// Get total memory usage
    pub fn total_memory_usage(&self) -> usize {
        self.temperature.memory_usage()
            + self.heat_source.memory_usage()
            + self.thermal_conductivity.memory_usage()
            + self.specific_heat.memory_usage()
            + self.perfusion_rate.memory_usage()
    }
}

/// Bubble dynamics field collection
#[derive(Debug)]
pub struct BubbleFields {
    pub radius: AlignedField,
    pub radius_dot: AlignedField,
    pub equilibrium_radius: AlignedField,
    pub internal_pressure: AlignedField,
    pub nucleation_sites: AlignedField,
}

impl BubbleFields {
    /// Create bubble fields with cavitation defaults  
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            radius: AlignedField::filled("radius".to_string(), nx, ny, nz, 1e-6), // 1 μm initial radius
            radius_dot: AlignedField::new("radius_dot".to_string(), nx, ny, nz),
            equilibrium_radius: AlignedField::filled("equilibrium_radius".to_string(), nx, ny, nz, 1e-6),
            internal_pressure: AlignedField::filled("internal_pressure".to_string(), nx, ny, nz, 101325.0), // Atmospheric pressure
            nucleation_sites: AlignedField::new("nucleation_sites".to_string(), nx, ny, nz),
        }
    }
    
    /// Get total memory usage
    pub fn total_memory_usage(&self) -> usize {
        self.radius.memory_usage()
            + self.radius_dot.memory_usage()
            + self.equilibrium_radius.memory_usage()
            + self.internal_pressure.memory_usage()
            + self.nucleation_sites.memory_usage()
    }
}

/// Complete physics state containing all field types
#[derive(Debug)]
pub struct PhysicsData {
    pub acoustic: AcousticFields,
    pub thermal: Option<ThermalFields>,
    pub bubble: Option<BubbleFields>,
    pub grid_dimensions: (usize, usize, usize),
    pub time: f64,
}

impl PhysicsData {
    /// Create new physics data with acoustic fields only
    pub fn acoustic_only(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            acoustic: AcousticFields::new(nx, ny, nz),
            thermal: None,
            bubble: None,
            grid_dimensions: (nx, ny, nz),
            time: 0.0,
        }
    }
    
    /// Create full multi-physics data
    pub fn multi_physics(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            acoustic: AcousticFields::new(nx, ny, nz),
            thermal: Some(ThermalFields::new(nx, ny, nz)),
            bubble: Some(BubbleFields::new(nx, ny, nz)),
            grid_dimensions: (nx, ny, nz),
            time: 0.0,
        }
    }
    
    /// Get total memory usage across all fields
    pub fn total_memory_usage(&self) -> usize {
        let mut total = self.acoustic.total_memory_usage();
        
        if let Some(ref thermal) = self.thermal {
            total += thermal.total_memory_usage();
        }
        
        if let Some(ref bubble) = self.bubble {
            total += bubble.total_memory_usage();
        }
        
        total
    }
    
    /// Check if all fields are SIMD-ready
    pub fn is_simd_ready(&self) -> bool {
        let mut ready = self.acoustic.is_simd_ready();
        
        if let Some(ref thermal) = self.thermal {
            ready = ready && thermal.temperature.is_simd_aligned();
        }
        
        if let Some(ref bubble) = self.bubble {
            ready = ready && bubble.radius.is_simd_aligned();
        }
        
        ready
    }
    
    /// Get memory usage summary for optimization
    pub fn memory_summary(&self) -> MemorySummary {
        MemorySummary {
            acoustic_mb: self.acoustic.total_memory_usage() as f64 / 1024.0 / 1024.0,
            thermal_mb: self.thermal.as_ref().map(|t| t.total_memory_usage() as f64 / 1024.0 / 1024.0),
            bubble_mb: self.bubble.as_ref().map(|b| b.total_memory_usage() as f64 / 1024.0 / 1024.0),
            total_mb: self.total_memory_usage() as f64 / 1024.0 / 1024.0,
            grid_points: self.grid_dimensions.0 * self.grid_dimensions.1 * self.grid_dimensions.2,
        }
    }
}

/// Memory usage summary for performance analysis
#[derive(Debug)]
pub struct MemorySummary {
    pub acoustic_mb: f64,
    pub thermal_mb: Option<f64>,
    pub bubble_mb: Option<f64>,
    pub total_mb: f64,
    pub grid_points: usize,
}

impl MemorySummary {
    /// Get memory usage per grid point in bytes
    pub fn bytes_per_point(&self) -> f64 {
        (self.total_mb * 1024.0 * 1024.0) / (self.grid_points as f64)
    }
    
    /// Check if memory usage is within reasonable limits
    pub fn is_reasonable(&self) -> bool {
        self.total_mb < 4096.0 // Less than 4GB total
            && self.bytes_per_point() < 1024.0 // Less than 1KB per grid point
    }
}

/// Zero-copy slice operations for SIMD processing
pub struct FieldProcessor;

impl FieldProcessor {
    /// Add two fields element-wise using SIMD when possible
    pub fn add_fields(a: &AlignedField, b: &AlignedField, result: &mut AlignedField) -> crate::error::KwaversResult<()> {
        let a_slice = a.as_slice();
        let b_slice = b.as_slice();
        let result_slice = result.as_slice_mut();
        
        if a_slice.len() != b_slice.len() || a_slice.len() != result_slice.len() {
            return Err(crate::error::KwaversError::InvalidParameter(
                "Mismatched field dimensions for addition".to_string()
            ));
        }
        
        // Use iterator for auto-vectorization
        result_slice.iter_mut()
            .zip(a_slice.iter())
            .zip(b_slice.iter())
            .for_each(|((r, &a_val), &b_val)| {
                *r = a_val + b_val;
            });
            
        Ok(())
    }
    
    /// Scale field by scalar using SIMD
    pub fn scale_field(field: &AlignedField, scalar: f64, result: &mut AlignedField) -> crate::error::KwaversResult<()> {
        let field_slice = field.as_slice();
        let result_slice = result.as_slice_mut();
        
        if field_slice.len() != result_slice.len() {
            return Err(crate::error::KwaversError::InvalidParameter(
                "Mismatched field dimensions for scaling".to_string()
            ));
        }
        
        // Auto-vectorizable loop
        result_slice.iter_mut()
            .zip(field_slice.iter())
            .for_each(|(r, &val)| {
                *r = val * scalar;
            });
            
        Ok(())
    }
    
    /// Compute field norm using SIMD reduction
    pub fn field_norm(field: &AlignedField) -> f64 {
        field.as_slice()
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aligned_field_creation() {
        let field = AlignedField::new("test".to_string(), 10, 10, 10);
        assert_eq!(field.name(), "test");
        assert_eq!(field.memory_usage(), 10 * 10 * 10 * 8); // 8 bytes per f64
    }
    
    #[test]
    fn test_acoustic_fields_memory() {
        let fields = AcousticFields::new(32, 32, 32);
        let expected = 6 * 32 * 32 * 32 * 8; // 6 fields, 8 bytes per f64
        assert_eq!(fields.total_memory_usage(), expected);
    }
    
    #[test]
    fn test_field_operations() {
        let mut a = AlignedField::filled("a".to_string(), 2, 2, 2, 1.0);
        let b = AlignedField::filled("b".to_string(), 2, 2, 2, 2.0);
        let mut result = AlignedField::new("result".to_string(), 2, 2, 2);
        
        FieldProcessor::add_fields(&a, &b, &mut result).unwrap();
        
        // Check that all values are 3.0 (1.0 + 2.0)
        assert!(result.as_slice().iter().all(|&x| (x - 3.0).abs() < 1e-15));
        
        FieldProcessor::scale_field(&result, 2.0, &mut a).unwrap();
        
        // Check that all values are 6.0 (3.0 * 2.0)
        assert!(a.as_slice().iter().all(|&x| (x - 6.0).abs() < 1e-15));
    }
    
    #[test]
    fn test_memory_summary() {
        let data = PhysicsData::multi_physics(64, 64, 64);
        let summary = data.memory_summary();
        
        assert!(summary.total_mb > 0.0);
        assert!(summary.is_reasonable());
        assert_eq!(summary.grid_points, 64 * 64 * 64);
    }
}