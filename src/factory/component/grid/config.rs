//! Grid configuration types and validation
//!
//! Domain-focused configuration management following SOLID principles

use serde::{Deserialize, Serialize};
use crate::error::KwaversResult;

/// Grid configuration with comprehensive validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl GridConfig {
    /// Create new grid configuration with validation
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self { nx, ny, nz, dx, dy, dz }
    }
    
    /// Validate configuration - maintains backward compatibility
    pub fn validate(&self) -> KwaversResult<()> {
        super::validator::GridValidator::validate(self)
    }
    
    /// Calculate total grid points
    pub fn total_points(&self) -> usize {
        self.nx * self.ny * self.nz
    }
    
    /// Calculate memory footprint estimate in bytes
    pub fn memory_estimate(&self) -> usize {
        // Estimate: 8 fields * 8 bytes (f64) per point
        self.total_points() * 64
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            nx: 128,
            ny: 128, 
            nz: 128,
            dx: 1e-4,
            dy: 1e-4,
            dz: 1e-4,
        }
    }
}