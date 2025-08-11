// src/solver/amr/refinement.rs
//! Refinement strategies and criteria for AMR
//! 
//! This module implements advanced refinement strategies for Adaptive Mesh Refinement,
//! including gradient-based, wavelet-based, and physics-aware refinement criteria.
//! 
//! # References
//! - Berger, M. J., & Oliger, J. (1984). "Adaptive mesh refinement for hyperbolic partial differential equations"
//! - Harten, A. (1995). "Multiresolution algorithms for the numerical solution of hyperbolic conservation laws"

use ndarray::{Array3, Zip};

/// Named constants for refinement thresholds
const DEFAULT_GRADIENT_THRESHOLD: f64 = 0.1;
const DEFAULT_WAVELET_THRESHOLD: f64 = 0.01;
const DEFAULT_CURVATURE_THRESHOLD: f64 = 0.05;
const MIN_REFINEMENT_LEVEL: usize = 0;
const MAX_REFINEMENT_LEVEL: usize = 8;

/// Refinement criterion type
#[derive(Debug, Clone, Copy)]
pub enum RefinementCriterion {
    /// Gradient-based refinement (Löhner, 1987)
    Gradient { threshold: f64 },
    /// Wavelet-based refinement (Harten, 1995)
    Wavelet { threshold: f64 },
    /// Curvature-based refinement
    Curvature { threshold: f64 },
    /// Physics-based refinement (e.g., shock detection)
    PhysicsBased { shock_threshold: f64 },
    /// Combined criteria with weights
    Combined { 
        gradient_weight: f64,
        wavelet_weight: f64,
        curvature_weight: f64,
    },
}

/// Refinement strategy for AMR
pub struct RefinementStrategy {
    criterion: RefinementCriterion,
    min_level: usize,
    max_level: usize,
    buffer_cells: usize,
}

impl Default for RefinementStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl RefinementStrategy {
    pub fn new() -> Self {
        Self {
            criterion: RefinementCriterion::Gradient { 
                threshold: DEFAULT_GRADIENT_THRESHOLD 
            },
            min_level: MIN_REFINEMENT_LEVEL,
            max_level: MAX_REFINEMENT_LEVEL,
            buffer_cells: 2,
        }
    }
    
    /// Create gradient-based refinement strategy
    pub fn gradient(threshold: f64) -> Self {
        Self {
            criterion: RefinementCriterion::Gradient { threshold },
            ..Self::new()
        }
    }
    
    /// Create wavelet-based refinement strategy
    pub fn wavelet(threshold: f64) -> Self {
        Self {
            criterion: RefinementCriterion::Wavelet { threshold },
            ..Self::new()
        }
    }
    
    /// Create physics-based refinement strategy
    pub fn physics_based(shock_threshold: f64) -> Self {
        Self {
            criterion: RefinementCriterion::PhysicsBased { shock_threshold },
            ..Self::new()
        }
    }
    
    /// Set refinement level limits
    pub fn with_levels(mut self, min: usize, max: usize) -> Self {
        self.min_level = min.min(MAX_REFINEMENT_LEVEL);
        self.max_level = max.min(MAX_REFINEMENT_LEVEL);
        self
    }
    
    /// Set buffer cells around refined regions
    pub fn with_buffer(mut self, buffer: usize) -> Self {
        self.buffer_cells = buffer;
        self
    }
    
    /// Compute refinement indicator for a field
    pub fn compute_indicator(&self, field: &Array3<f64>) -> Array3<f64> {
        match self.criterion {
            RefinementCriterion::Gradient { threshold } => {
                self.gradient_indicator(field, threshold)
            }
            RefinementCriterion::Wavelet { threshold } => {
                self.wavelet_indicator(field, threshold)
            }
            RefinementCriterion::Curvature { threshold } => {
                self.curvature_indicator(field, threshold)
            }
            RefinementCriterion::PhysicsBased { shock_threshold } => {
                self.physics_indicator(field, shock_threshold)
            }
            RefinementCriterion::Combined { gradient_weight, wavelet_weight, curvature_weight } => {
                self.combined_indicator(field, gradient_weight, wavelet_weight, curvature_weight)
            }
        }
    }
    
    /// Gradient-based refinement indicator (Löhner, 1987)
    fn gradient_indicator(&self, field: &Array3<f64>, threshold: f64) -> Array3<f64> {
        let shape = field.shape();
        let mut indicator = Array3::zeros((shape[0], shape[1], shape[2]));
        
        // Compute gradient magnitude using central differences
        Zip::indexed(&mut indicator)
            .for_each(|(i, j, k), ind| {
                let grad_x = if i > 0 && i < shape[0] - 1 {
                    (field[[i + 1, j, k]] - field[[i - 1, j, k]]) * 0.5
                } else { 0.0 };
                
                let grad_y = if j > 0 && j < shape[1] - 1 {
                    (field[[i, j + 1, k]] - field[[i, j - 1, k]]) * 0.5
                } else { 0.0 };
                
                let grad_z = if k > 0 && k < shape[2] - 1 {
                    (field[[i, j, k + 1]] - field[[i, j, k - 1]]) * 0.5
                } else { 0.0 };
                
                let grad_mag = (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();
                *ind = if grad_mag > threshold { 1.0 } else { 0.0 };
            });
        
        // Apply buffer cells
        if self.buffer_cells > 0 {
            self.apply_buffer(&mut indicator);
        }
        
        indicator
    }
    
    /// Wavelet-based refinement indicator (Harten, 1995)
    fn wavelet_indicator(&self, field: &Array3<f64>, threshold: f64) -> Array3<f64> {
        let shape = field.shape();
        let mut indicator = Array3::zeros((shape[0], shape[1], shape[2]));
        
        // Simple Haar wavelet coefficients
        Zip::indexed(&mut indicator)
            .for_each(|(i, j, k), ind| {
                if i > 0 && j > 0 && k > 0 && 
                   i < shape[0] - 1 && j < shape[1] - 1 && k < shape[2] - 1 {
                    // Compute Haar wavelet detail coefficients
                    let avg = (field[[i-1, j-1, k-1]] + field[[i-1, j-1, k]] +
                               field[[i-1, j, k-1]] + field[[i-1, j, k]] +
                               field[[i, j-1, k-1]] + field[[i, j-1, k]] +
                               field[[i, j, k-1]] + field[[i, j, k]]) * 0.125;
                    
                    let detail = (field[[i, j, k]] - avg).abs();
                    *ind = if detail > threshold { 1.0 } else { 0.0 };
                }
            });
        
        // Apply buffer cells
        if self.buffer_cells > 0 {
            self.apply_buffer(&mut indicator);
        }
        
        indicator
    }
    
    /// Curvature-based refinement indicator
    fn curvature_indicator(&self, field: &Array3<f64>, threshold: f64) -> Array3<f64> {
        let shape = field.shape();
        let mut indicator = Array3::zeros((shape[0], shape[1], shape[2]));
        
        // Compute second derivatives (Laplacian as curvature measure)
        Zip::indexed(&mut indicator)
            .for_each(|(i, j, k), ind| {
                if i > 0 && i < shape[0] - 1 &&
                   j > 0 && j < shape[1] - 1 &&
                   k > 0 && k < shape[2] - 1 {
                    let d2_dx2 = field[[i+1, j, k]] - 2.0 * field[[i, j, k]] + field[[i-1, j, k]];
                    let d2_dy2 = field[[i, j+1, k]] - 2.0 * field[[i, j, k]] + field[[i, j-1, k]];
                    let d2_dz2 = field[[i, j, k+1]] - 2.0 * field[[i, j, k]] + field[[i, j, k-1]];
                    
                    let laplacian = (d2_dx2.abs() + d2_dy2.abs() + d2_dz2.abs()) / 3.0;
                    *ind = if laplacian > threshold { 1.0 } else { 0.0 };
                }
            });
        
        // Apply buffer cells
        if self.buffer_cells > 0 {
            self.apply_buffer(&mut indicator);
        }
        
        indicator
    }
    
    /// Physics-based refinement indicator (e.g., shock detection)
    fn physics_indicator(&self, field: &Array3<f64>, shock_threshold: f64) -> Array3<f64> {
        let shape = field.shape();
        let mut indicator = Array3::zeros((shape[0], shape[1], shape[2]));
        
        // Persson & Peraire (2006) shock detection
        Zip::indexed(&mut indicator)
            .for_each(|(i, j, k), ind| {
                if i > 1 && i < shape[0] - 2 &&
                   j > 1 && j < shape[1] - 2 &&
                   k > 1 && k < shape[2] - 2 {
                    // Compute smoothness indicator
                    let center = field[[i, j, k]];
                    let neighbors = [
                        field[[i-1, j, k]], field[[i+1, j, k]],
                        field[[i, j-1, k]], field[[i, j+1, k]],
                        field[[i, j, k-1]], field[[i, j, k+1]]
                    ];
                    
                    let max_jump = neighbors.iter()
                        .map(|&n| (n - center).abs())
                        .fold(0.0, f64::max);
                    
                    let avg_val = neighbors.iter().sum::<f64>() / 6.0;
                    let normalized_jump = if avg_val.abs() > 1e-10 {
                        max_jump / avg_val.abs()
                    } else { 0.0 };
                    
                    *ind = if normalized_jump > shock_threshold { 1.0 } else { 0.0 };
                }
            });
        
        // Apply buffer cells
        if self.buffer_cells > 0 {
            self.apply_buffer(&mut indicator);
        }
        
        indicator
    }
    
    /// Combined refinement indicator with weighted criteria
    fn combined_indicator(&self, field: &Array3<f64>, 
                         gradient_weight: f64, wavelet_weight: f64, curvature_weight: f64) -> Array3<f64> {
        let grad = self.gradient_indicator(field, DEFAULT_GRADIENT_THRESHOLD);
        let wave = self.wavelet_indicator(field, DEFAULT_WAVELET_THRESHOLD);
        let curv = self.curvature_indicator(field, DEFAULT_CURVATURE_THRESHOLD);
        
        let mut combined = Array3::zeros(field.dim());
        
        Zip::from(&mut combined)
            .and(&grad)
            .and(&wave)
            .and(&curv)
            .for_each(|c, &g, &w, &cv| {
                let weighted = gradient_weight * g + wavelet_weight * w + curvature_weight * cv;
                let total_weight = gradient_weight + wavelet_weight + curvature_weight;
                *c = if total_weight > 0.0 {
                    (weighted / total_weight).min(1.0)
                } else { 0.0 };
            });
        
        combined
    }
    
    /// Apply buffer cells around marked regions
    fn apply_buffer(&self, indicator: &mut Array3<f64>) {
        let shape = indicator.shape();
        let mut buffer_indicator = indicator.clone();
        
        for _ in 0..self.buffer_cells {
            Zip::indexed(&mut buffer_indicator)
                .for_each(|(i, j, k), buf| {
                    if *buf < 0.5 {  // Not already marked
                        // Check if any neighbor is marked
                        let mut has_marked_neighbor = false;
                        
                        for di in -1i32..=1 {
                            for dj in -1i32..=1 {
                                for dk in -1i32..=1 {
                                    if di == 0 && dj == 0 && dk == 0 { continue; }
                                    
                                    let ni = (i as i32 + di) as usize;
                                    let nj = (j as i32 + dj) as usize;
                                    let nk = (k as i32 + dk) as usize;
                                    
                                    if ni < shape[0] && nj < shape[1] && nk < shape[2] {
                                        if indicator[[ni, nj, nk]] > 0.5 {
                                            has_marked_neighbor = true;
                                            break;
                                        }
                                    }
                                }
                                if has_marked_neighbor { break; }
                            }
                            if has_marked_neighbor { break; }
                        }
                        
                        if has_marked_neighbor {
                            *buf = 1.0;
                        }
                    }
                });
            
            indicator.assign(&buffer_indicator);
        }
    }
    
    /// Get current refinement level limits
    pub fn level_range(&self) -> (usize, usize) {
        (self.min_level, self.max_level)
    }
}