//! Enhanced Adaptive Mesh Refinement Module
//!
//! This module provides advanced AMR capabilities with:
//! - Dynamic refinement criteria based on physics
//! - Load balancing for parallel execution
//! - Multi-criteria refinement (gradient, curvature, feature-based)
//! - Predictive refinement based on wave propagation
//! - Memory-aware refinement strategies
//!
//! # Design Principles
//! - **SOLID**: Each refinement criterion is a separate component
//! - **CUPID**: Clear interfaces for criteria and load balancing
//! - **DRY**: Reusable refinement patterns
//! - **KISS**: Simple API despite complex algorithms

use crate::{
    error::{KwaversResult, KwaversError, NumericalError},
    grid::Grid,
    solver::amr::{AMRConfig, WaveletType, InterpolationScheme, CellStatus},
};
use ndarray::{Array3, Array4, Axis, s};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use log::{info, debug, warn};
use std::sync::{Arc, Mutex};

/// Dynamic refinement criterion trait
pub trait RefinementCriterion: Send + Sync {
    /// Evaluate the criterion and return refinement priority (0.0 to 1.0)
    fn evaluate(&self, field: &Array3<f64>, position: (usize, usize, usize)) -> f64;
    
    /// Get criterion name for logging
    fn name(&self) -> &str;
    
    /// Check if criterion requires gradient computation
    fn requires_gradient(&self) -> bool { false }
    
    /// Check if criterion requires curvature computation
    fn requires_curvature(&self) -> bool { false }
}

/// Gradient-based refinement criterion
#[derive(Debug, Clone)]
pub struct GradientCriterion {
    /// Gradient threshold for refinement
    pub threshold: f64,
    /// Normalization factor
    pub normalization: f64,
}

impl RefinementCriterion for GradientCriterion {
    fn evaluate(&self, field: &Array3<f64>, position: (usize, usize, usize)) -> f64 {
        let (i, j, k) = position;
        let (nx, ny, nz) = field.dim();
        
        // Compute gradient magnitude using central differences
        let mut grad_mag = 0.0;
        
        // X-direction
        if i > 0 && i < nx - 1 {
            let dx = (field[[i + 1, j, k]] - field[[i - 1, j, k]]) / 2.0;
            grad_mag += dx * dx;
        }
        
        // Y-direction
        if j > 0 && j < ny - 1 {
            let dy = (field[[i, j + 1, k]] - field[[i, j - 1, k]]) / 2.0;
            grad_mag += dy * dy;
        }
        
        // Z-direction
        if k > 0 && k < nz - 1 {
            let dz = (field[[i, j, k + 1]] - field[[i, j, k - 1]]) / 2.0;
            grad_mag += dz * dz;
        }
        
        grad_mag = grad_mag.sqrt();
        
        // Normalize and threshold
        let normalized = grad_mag / self.normalization;
        (normalized / self.threshold).min(1.0)
    }
    
    fn name(&self) -> &str { "Gradient" }
    fn requires_gradient(&self) -> bool { true }
}

/// Curvature-based refinement criterion
#[derive(Debug, Clone)]
pub struct CurvatureCriterion {
    /// Curvature threshold
    pub threshold: f64,
    /// Weight for different curvature components
    pub laplacian_weight: f64,
    pub hessian_weight: f64,
}

impl RefinementCriterion for CurvatureCriterion {
    fn evaluate(&self, field: &Array3<f64>, position: (usize, usize, usize)) -> f64 {
        let (i, j, k) = position;
        let (nx, ny, nz) = field.dim();
        
        // Compute Laplacian (second derivatives)
        let mut laplacian = 0.0;
        
        if i > 0 && i < nx - 1 {
            laplacian += field[[i + 1, j, k]] - 2.0 * field[[i, j, k]] + field[[i - 1, j, k]];
        }
        
        if j > 0 && j < ny - 1 {
            laplacian += field[[i, j + 1, k]] - 2.0 * field[[i, j, k]] + field[[i, j - 1, k]];
        }
        
        if k > 0 && k < nz - 1 {
            laplacian += field[[i, j, k + 1]] - 2.0 * field[[i, j, k]] + field[[i, j, k - 1]];
        }
        
        // Normalize curvature
        let curvature_measure = laplacian.abs() * self.laplacian_weight;
        (curvature_measure / self.threshold).min(1.0)
    }
    
    fn name(&self) -> &str { "Curvature" }
    fn requires_curvature(&self) -> bool { true }
}

/// Feature-based refinement criterion (e.g., shock detection)
#[derive(Debug, Clone)]
pub struct FeatureCriterion {
    /// Feature detection threshold
    pub threshold: f64,
    /// Feature type
    pub feature_type: FeatureType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureType {
    Shock,
    Interface,
    Vortex,
    HighFrequency,
}

impl RefinementCriterion for FeatureCriterion {
    fn evaluate(&self, field: &Array3<f64>, position: (usize, usize, usize)) -> f64 {
        match self.feature_type {
            FeatureType::Shock => self.detect_shock(field, position),
            FeatureType::Interface => self.detect_interface(field, position),
            FeatureType::Vortex => self.detect_vortex(field, position),
            FeatureType::HighFrequency => self.detect_high_frequency(field, position),
        }
    }
    
    fn name(&self) -> &str {
        match self.feature_type {
            FeatureType::Shock => "Shock",
            FeatureType::Interface => "Interface",
            FeatureType::Vortex => "Vortex",
            FeatureType::HighFrequency => "HighFrequency",
        }
    }
}

impl FeatureCriterion {
    fn detect_shock(&self, field: &Array3<f64>, position: (usize, usize, usize)) -> f64 {
        // Simplified shock detector using pressure ratio
        let (i, j, k) = position;
        let center = field[[i, j, k]];
        
        let mut max_ratio = 1.0;
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    if di == 0 && dj == 0 && dk == 0 { continue; }
                    
                    let ni = (i as i32 + di).max(0) as usize;
                    let nj = (j as i32 + dj).max(0) as usize;
                    let nk = (k as i32 + dk).max(0) as usize;
                    
                    if ni < field.dim().0 && nj < field.dim().1 && nk < field.dim().2 {
                        let neighbor = field[[ni, nj, nk]];
                        let ratio = (center / neighbor).abs().max(neighbor / center).abs();
                        max_ratio = f64::max(max_ratio, ratio);
                    }
                }
            }
        }
        
        ((max_ratio - 1.0) / self.threshold).min(1.0)
    }
    
    fn detect_interface(&self, field: &Array3<f64>, position: (usize, usize, usize)) -> f64 {
        // Detect material interfaces using gradient discontinuity
        self.detect_shock(field, position) * 0.7 // Similar but with lower weight
    }
    
    fn detect_vortex(&self, field: &Array3<f64>, position: (usize, usize, usize)) -> f64 {
        // Simplified vorticity detection
        // In real implementation, would need velocity components
        0.0 // Placeholder
    }
    
    fn detect_high_frequency(&self, field: &Array3<f64>, position: (usize, usize, usize)) -> f64 {
        // Detect high-frequency content using local oscillations
        let (i, j, k) = position;
        let center = field[[i, j, k]];
        
        let mut oscillation = 0.0;
        let mut count = 0;
        
        // Check oscillations in each direction
        for (di, dj, dk) in &[(2, 0, 0), (0, 2, 0), (0, 0, 2)] {
            let i1 = (i as i32 - di).max(0) as usize;
            let j1 = (j as i32 - dj).max(0) as usize;
            let k1 = (k as i32 - dk).max(0) as usize;
            
            let i2 = (i as i32 + di).min(field.dim().0 as i32 - 1) as usize;
            let j2 = (j as i32 + dj).min(field.dim().1 as i32 - 1) as usize;
            let k2 = (k as i32 + dk).min(field.dim().2 as i32 - 1) as usize;
            
            let val1 = field[[i1, j1, k1]];
            let val2 = field[[i2, j2, k2]];
            
            oscillation += (val1 + val2 - 2.0 * center).abs();
            count += 1;
        }
        
        if count > 0 {
            let avg_oscillation = oscillation / count as f64;
            (avg_oscillation / self.threshold).min(1.0)
        } else {
            0.0
        }
    }
}

/// Predictive refinement criterion based on wave propagation
#[derive(Debug, Clone)]
pub struct PredictiveCriterion {
    /// Wave speed for prediction
    pub wave_speed: f64,
    /// Look-ahead time
    pub prediction_time: f64,
    /// Refinement buffer distance
    pub buffer_distance: f64,
}

impl RefinementCriterion for PredictiveCriterion {
    fn evaluate(&self, field: &Array3<f64>, position: (usize, usize, usize)) -> f64 {
        // Check if there's a feature nearby that will propagate to this location
        let search_radius = (self.wave_speed * self.prediction_time + self.buffer_distance) as usize;
        let (i, j, k) = position;
        
        let mut max_influence = 0.0;
        
        for di in -(search_radius as i32)..=(search_radius as i32) {
            for dj in -(search_radius as i32)..=(search_radius as i32) {
                for dk in -(search_radius as i32)..=(search_radius as i32) {
                    let ni = (i as i32 + di).max(0) as usize;
                    let nj = (j as i32 + dj).max(0) as usize;
                    let nk = (k as i32 + dk).max(0) as usize;
                    
                    if ni < field.dim().0 && nj < field.dim().1 && nk < field.dim().2 {
                        let distance = ((di * di + dj * dj + dk * dk) as f64).sqrt();
                        if distance <= search_radius as f64 {
                            let feature_strength = field[[ni, nj, nk]].abs();
                            let influence = feature_strength * (1.0 - distance / search_radius as f64);
                                                            max_influence = f64::max(max_influence, influence);
                        }
                    }
                }
            }
        }
        
        f64::min(max_influence, 1.0)
    }
    
    fn name(&self) -> &str { "Predictive" }
}

/// Load balancer for parallel AMR execution
#[derive(Debug)]
pub struct LoadBalancer {
    /// Number of available threads
    num_threads: usize,
    /// Work distribution strategy
    strategy: LoadBalancingStrategy,
    /// Performance metrics
    metrics: LoadBalancingMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Static block distribution
    Static,
    /// Dynamic work stealing
    Dynamic,
    /// Guided self-scheduling
    Guided,
    /// Space-filling curve based
    SpaceFillingCurve,
}

#[derive(Debug, Default)]
struct LoadBalancingMetrics {
    total_cells_processed: usize,
    load_imbalance_factor: f64,
    redistribution_count: usize,
    average_cells_per_thread: f64,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            strategy,
            metrics: LoadBalancingMetrics::default(),
        }
    }
    
    /// Distribute work among threads
    pub fn distribute_work<T: Send + Sync>(
        &mut self,
        cells: Vec<(usize, usize, usize)>,
        work_estimates: Option<&HashMap<(usize, usize, usize), f64>>,
    ) -> Vec<Vec<(usize, usize, usize)>> {
        match self.strategy {
            LoadBalancingStrategy::Static => self.static_distribution(cells),
            LoadBalancingStrategy::Dynamic => self.dynamic_distribution(cells, work_estimates),
            LoadBalancingStrategy::Guided => self.guided_distribution(cells, work_estimates),
            LoadBalancingStrategy::SpaceFillingCurve => self.sfc_distribution(cells),
        }
    }
    
    fn static_distribution(&self, cells: Vec<(usize, usize, usize)>) -> Vec<Vec<(usize, usize, usize)>> {
        let chunk_size = (cells.len() + self.num_threads - 1) / self.num_threads;
        cells.chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
    
    fn dynamic_distribution(
        &self,
        cells: Vec<(usize, usize, usize)>,
        work_estimates: Option<&HashMap<(usize, usize, usize), f64>>,
    ) -> Vec<Vec<(usize, usize, usize)>> {
        if let Some(estimates) = work_estimates {
            // Sort cells by estimated work (descending)
            let mut weighted_cells: Vec<_> = cells.into_iter()
                .map(|cell| (cell, estimates.get(&cell).copied().unwrap_or(1.0)))
                .collect();
            weighted_cells.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Distribute using greedy algorithm
            let mut thread_work: Vec<(Vec<(usize, usize, usize)>, f64)> = 
                vec![(Vec::new(), 0.0); self.num_threads];
            
            for (cell, work) in weighted_cells {
                // Find thread with least work
                let min_thread = thread_work.iter_mut()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                
                min_thread.0.push(cell);
                min_thread.1 += work;
            }
            
            thread_work.into_iter().map(|(cells, _)| cells).collect()
        } else {
            self.static_distribution(cells)
        }
    }
    
    fn guided_distribution(
        &self,
        cells: Vec<(usize, usize, usize)>,
        _work_estimates: Option<&HashMap<(usize, usize, usize), f64>>,
    ) -> Vec<Vec<(usize, usize, usize)>> {
        // Guided self-scheduling: larger chunks first, then smaller
        let mut distributions = vec![Vec::new(); self.num_threads];
        let mut remaining = cells.len();
        let mut cells_iter = cells.into_iter();
        let mut thread_idx = 0;
        
        while remaining > 0 {
            let chunk_size = (remaining / (2 * self.num_threads)).max(1);
            let chunk: Vec<_> = cells_iter.by_ref().take(chunk_size).collect();
            
            distributions[thread_idx].extend(chunk);
            remaining -= chunk_size;
            thread_idx = (thread_idx + 1) % self.num_threads;
        }
        
        distributions
    }
    
    fn sfc_distribution(&self, mut cells: Vec<(usize, usize, usize)>) -> Vec<Vec<(usize, usize, usize)>> {
        // Sort cells by Morton code (Z-order curve)
        cells.sort_by_key(|&(i, j, k)| morton_encode_3d(i, j, k));
        
        // Distribute consecutive cells to threads
        self.static_distribution(cells)
    }
    
    /// Update load balancing metrics
    pub fn update_metrics(&mut self, thread_loads: &[usize]) {
        let total: usize = thread_loads.iter().sum();
        let avg = total as f64 / thread_loads.len() as f64;
        let max_load = thread_loads.iter().max().copied().unwrap_or(0) as f64;
        
        self.metrics.total_cells_processed += total;
        self.metrics.average_cells_per_thread = avg;
        self.metrics.load_imbalance_factor = if avg > 0.0 { max_load / avg } else { 1.0 };
        self.metrics.redistribution_count += 1;
    }
}

/// Morton encoding for 3D space-filling curve
fn morton_encode_3d(x: usize, y: usize, z: usize) -> u64 {
    let mut morton = 0u64;
    
    for i in 0..21 { // 21 bits per dimension for 64-bit result
        morton |= ((x as u64 >> i) & 1) << (3 * i);
        morton |= ((y as u64 >> i) & 1) << (3 * i + 1);
        morton |= ((z as u64 >> i) & 1) << (3 * i + 2);
    }
    
    morton
}

// EnhancedAMRManager functionality has been integrated into the main AMRManager
// The enhanced features are now available through the standard AMRManager API

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gradient_criterion() {
        let criterion = GradientCriterion {
            threshold: 0.1,
            normalization: 1.0,
        };
        
        let mut field = Array3::zeros((5, 5, 5));
        field[[2, 2, 2]] = 1.0;
        field[[3, 2, 2]] = 0.5;
        
        let priority = criterion.evaluate(&field, (2, 2, 2));
        assert!(priority > 0.0);
    }
    
    #[test]
    fn test_morton_encoding() {
        assert_eq!(morton_encode_3d(0, 0, 0), 0);
        assert_eq!(morton_encode_3d(1, 0, 0), 1);
        assert_eq!(morton_encode_3d(0, 1, 0), 2);
        assert_eq!(morton_encode_3d(0, 0, 1), 4);
        assert_eq!(morton_encode_3d(1, 1, 1), 7);
    }
    
    #[test]
    fn test_load_balancer() {
        let mut balancer = LoadBalancer::new(LoadBalancingStrategy::Static);
        let cells = vec![(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)];
        
        let distribution = balancer.distribute_work::<(usize, usize, usize)>(cells, None);
        assert!(!distribution.is_empty());
        
        let total_cells: usize = distribution.iter().map(|d| d.len()).sum();
        assert_eq!(total_cells, 4);
    }
}