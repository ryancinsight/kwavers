//! Fallback Visualization Module
//!
//! Provides CPU-based fallback rendering when GPU is not available.

use crate::domain::core::error::KwaversResult;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use log::{info, warn};
use ndarray::Array3;

/// Fallback renderer for CPU-based visualization
#[derive(Debug)]
pub struct FallbackRenderer {
    /// Whether fallback mode is active
    active: bool,
}

impl FallbackRenderer {
    /// Create a new fallback renderer
    pub fn new() -> Self {
        Self { active: true }
    }

    /// Render field using CPU fallback
    pub fn render_field(
        &self,
        field: &Array3<f64>,
        field_type: UnifiedFieldType,
        grid: &Grid,
    ) -> KwaversResult<()> {
        if !self.active {
            return Ok(());
        }

        warn!("Using CPU fallback rendering - GPU visualization not available");

        // Calculate field statistics
        let min_val = field.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = field.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_val = field.iter().sum::<f64>() / field.len() as f64;

        info!(
            "Field {:?} statistics: min={:.3e}, max={:.3e}, mean={:.3e}",
            field_type, min_val, max_val, mean_val
        );

        // Generate ASCII visualization for small grids
        if grid.nx <= 40 && grid.ny <= 40 {
            self.render_ascii_slice(field, grid.nz / 2)?;
        }

        Ok(())
    }

    /// Render a 2D slice as ASCII art
    fn render_ascii_slice(&self, field: &Array3<f64>, z_slice: usize) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();

        if z_slice >= nz {
            return Ok(());
        }

        // Normalize field values to 0-9 range for ASCII display
        let slice = field.slice(ndarray::s![.., .., z_slice]);
        let min_val = slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < 1e-10 {
            return Ok(());
        }

        println!("\n=== Field Slice at z={} ===", z_slice);

        // ASCII gradient characters
        let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '@', '█'];

        for j in 0..ny.min(40) {
            for i in 0..nx.min(40) {
                let val = slice[[i, j]];
                let normalized = ((val - min_val) / (max_val - min_val) * 9.0) as usize;
                let char_idx = normalized.min(9);
                print!("{}", chars[char_idx]);
            }
            println!();
        }

        println!("=== End Slice ===\n");

        Ok(())
    }

    /// Export field data to file
    pub fn export_field(
        &self,
        field: &Array3<f64>,
        field_type: UnifiedFieldType,
        filename: &str,
    ) -> KwaversResult<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename)?;

        // Write header
        writeln!(file, "# Field: {:?}", field_type)?;
        writeln!(file, "# Shape: {:?}", field.dim())?;

        // Write data in simple format
        for ((i, j, k), &value) in field.indexed_iter() {
            writeln!(file, "{} {} {} {:.6e}", i, j, k, value)?;
        }

        info!("Exported field to {}", filename);
        Ok(())
    }
}

impl Default for FallbackRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fallback function for simple field rendering
pub fn render_field(
    field: &Array3<f64>,
    field_type: UnifiedFieldType,
    grid: &Grid,
) -> KwaversResult<()> {
    let renderer = FallbackRenderer::new();
    renderer.render_field(field, field_type, grid)
}

/// Fallback function for field export
pub fn export_field(
    field: &Array3<f64>,
    field_type: UnifiedFieldType,
    filename: &str,
) -> KwaversResult<()> {
    let renderer = FallbackRenderer::new();
    renderer.export_field(field, field_type, filename)
}
