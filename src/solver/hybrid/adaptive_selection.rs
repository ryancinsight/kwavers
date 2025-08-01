//! Adaptive Selection for Hybrid Numerical Methods
//!
//! This module implements intelligent adaptive selection algorithms that analyze
//! field properties in real-time and dynamically choose the optimal numerical
//! method (PSTD, FDTD, or hybrid) for each region of the computational domain.
//!
//! # Selection Criteria:
//!
//! ## Spectral Analysis Metrics:
//! - **Local smoothness**: Gradient magnitude and curvature analysis
//! - **Frequency content**: Local FFT analysis for dominant frequencies
//! - **Dispersion characteristics**: Phase error and numerical dispersion
//!
//! ## Material Property Metrics:
//! - **Homogeneity**: Spatial variation in material parameters
//! - **Interface proximity**: Distance to material boundaries
//! - **Acoustic impedance contrast**: Strength of material discontinuities
//!
//! ## Computational Efficiency Metrics:
//! - **Grid resolution requirements**: Points per wavelength analysis
//! - **Stability constraints**: CFL conditions for each method
//! - **Memory access patterns**: Cache efficiency considerations
//!
//! # Design Principles Applied:
//! - **SOLID**: Single responsibility for quality assessment
//! - **CUPID**: Composable metrics, predictable behavior
//! - **GRASP**: Information expert pattern for method selection
//! - **DRY**: Reusable analysis utilities

use crate::grid::Grid;
use crate::error::KwaversResult;
use ndarray::{Array4, Axis, s};
use serde::{Serialize, Deserialize};
use log::{debug, info};

/// Selection criteria for adaptive method choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    /// Weight for smoothness in decision making (0-1)
    pub smoothness_weight: f64,
    /// Weight for frequency content in decision making (0-1)
    pub frequency_weight: f64,
    /// Weight for material properties in decision making (0-1)
    pub material_weight: f64,
    /// Weight for computational efficiency in decision making (0-1)
    pub efficiency_weight: f64,
    /// Threshold for switching methods (0-1)
    pub switch_threshold: f64,
    /// Hysteresis factor to prevent oscillation (0-1)
    pub hysteresis_factor: f64,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            smoothness_weight: 0.3,
            frequency_weight: 0.25,
            material_weight: 0.25,
            efficiency_weight: 0.2,
            switch_threshold: 0.1,
            hysteresis_factor: 0.05,
        }
    }
}

/// Quality metrics for field analysis
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Overall smoothness score (0-1, higher is smoother)
    pub smoothness_score: f64,
    /// Frequency content score (0-1, higher is more high-frequency)
    pub frequency_score: f64,
    /// Material homogeneity score (0-1, higher is more homogeneous)
    pub homogeneity_score: f64,
    /// Computational efficiency score (0-1, higher is more efficient)
    pub efficiency_score: f64,
    /// Composite quality score (0-1)
    pub composite_score: f64,
    /// Confidence in the assessment (0-1)
    pub confidence: f64,
    /// Detailed metrics for advanced analysis
    pub detailed_metrics: DetailedMetrics,
}

/// Detailed metrics for comprehensive analysis
#[derive(Debug, Clone, Default)]
pub struct DetailedMetrics {
    /// Local gradient magnitude statistics
    pub gradient_stats: StatisticalMetrics,
    /// Local curvature statistics
    pub curvature_stats: StatisticalMetrics,
    /// Frequency spectrum analysis
    pub frequency_spectrum: FrequencySpectrum,
    /// Material property variations
    pub material_variations: MaterialVariationMetrics,
    /// Grid resolution adequacy
    pub resolution_adequacy: ResolutionMetrics,
}

/// Statistical metrics for field analysis
#[derive(Debug, Clone, Default)]
pub struct StatisticalMetrics {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Frequency spectrum analysis results
#[derive(Debug, Clone, Default)]
pub struct FrequencySpectrum {
    /// Dominant frequency (Hz)
    pub dominant_frequency: f64,
    /// High frequency content ratio (0-1)
    pub high_freq_ratio: f64,
    /// Spectral centroid (Hz)
    pub spectral_centroid: f64,
    /// Spectral bandwidth (Hz)
    pub spectral_bandwidth: f64,
    /// Dispersion error estimate
    pub dispersion_error: f64,
}

/// Material variation metrics
#[derive(Debug, Clone, Default)]
pub struct MaterialVariationMetrics {
    /// Density variation coefficient
    pub density_variation: f64,
    /// Sound speed variation coefficient
    pub sound_speed_variation: f64,
    /// Interface sharpness metric
    pub interface_sharpness: f64,
    /// Impedance contrast ratio
    pub impedance_contrast: f64,
}

/// Grid resolution adequacy metrics
#[derive(Debug, Clone, Default)]
pub struct ResolutionMetrics {
    /// Points per wavelength (minimum)
    pub points_per_wavelength: f64,
    /// Grid dispersion number
    pub grid_dispersion_number: f64,
    /// Stability number (CFL-like)
    pub stability_number: f64,
    /// Resolution adequacy score (0-1)
    pub adequacy_score: f64,
}

/// Adaptive selector for numerical methods
pub struct AdaptiveSelector {
    /// Selection criteria
    criteria: SelectionCriteria,
    /// Historical quality metrics for hysteresis
    history: Vec<QualityMetrics>,
    /// Maximum history length
    max_history: usize,
    /// Analysis window size for local metrics
    analysis_window: usize,
}

impl AdaptiveSelector {
    /// Create a new adaptive selector
    pub fn new(criteria: SelectionCriteria) -> KwaversResult<Self> {
        info!("Initializing adaptive selector with criteria: {:?}", criteria);
        
        // Validate criteria
        Self::validate_criteria(&criteria)?;
        
        Ok(Self {
            criteria,
            history: Vec::new(),
            max_history: 100,
            analysis_window: 8,
        })
    }
    
    /// Validate selection criteria
    fn validate_criteria(criteria: &SelectionCriteria) -> KwaversResult<()> {
        let total_weight = criteria.smoothness_weight + 
                          criteria.frequency_weight + 
                          criteria.material_weight + 
                          criteria.efficiency_weight;
        
        if (total_weight - 1.0).abs() > 1e-6 {
            return Err(crate::error::KwaversError::Config(crate::error::ConfigError::InvalidValue {
                parameter: "selection_weights".to_string(),
                value: total_weight.to_string(),
                constraint: "weights must sum to 1.0".to_string(),
            }));
        }
        
        if criteria.switch_threshold < 0.0 || criteria.switch_threshold > 1.0 {
            return Err(crate::error::KwaversError::Config(crate::error::ConfigError::InvalidValue {
                parameter: "switch_threshold".to_string(),
                value: criteria.switch_threshold.to_string(),
                constraint: "must be between 0.0 and 1.0".to_string(),
            }));
        }
        
        Ok(())
    }
    
    /// Analyze field quality and generate metrics
    pub fn analyze_field_quality(
        &mut self,
        fields: &Array4<f64>,
        grid: &Grid,
    ) -> KwaversResult<QualityMetrics> {
        debug!("Analyzing field quality for adaptive selection");
        
        // Extract pressure field (assuming index 0)
        let pressure = fields.index_axis(Axis(0), 0);
        
        // Compute smoothness metrics
        let smoothness_score = self.compute_smoothness_metrics(&pressure, grid)?;
        
        // Compute frequency content metrics
        let frequency_score = self.compute_frequency_metrics(&pressure, grid)?;
        
        // Compute detailed metrics
        let detailed_metrics = self.compute_detailed_metrics(&pressure, grid)?;
        
        // Create quality metrics
        let mut quality_metrics = QualityMetrics {
            smoothness_score,
            frequency_score,
            homogeneity_score: 0.8, // Placeholder - would need medium information
            efficiency_score: self.compute_efficiency_score(grid)?,
            composite_score: 0.0,
            confidence: 0.9, // Placeholder
            detailed_metrics,
        };
        
        // Compute composite score
        quality_metrics.composite_score = self.compute_composite_score(&quality_metrics)?;
        
        // Add to history
        self.add_to_history(quality_metrics.clone());
        
        debug!("Field quality analysis completed: composite score = {:.3}", 
               quality_metrics.composite_score);
        
        Ok(quality_metrics)
    }
    
    /// Compute smoothness metrics from field data
    fn compute_smoothness_metrics(
        &self,
        field: &ndarray::ArrayView3<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let mut gradient_magnitudes = Vec::new();
        let mut curvature_values = Vec::new();
        
        // Analyze local smoothness using sliding window
        let window = self.analysis_window;
        
        for i in window..grid.nx-window {
            for j in window..grid.ny-window {
                for k in window..grid.nz-window {
                    // Extract local window
                    let local_region = field.slice(s![
                        i-window/2..i+window/2,
                        j-window/2..j+window/2,
                        k-window/2..k+window/2
                    ]);
                    
                    // Compute local gradient and curvature
                    let (grad_mag, curvature) = self.compute_local_differential_metrics(
                        &local_region, grid
                    )?;
                    
                    gradient_magnitudes.push(grad_mag);
                    curvature_values.push(curvature);
                }
            }
        }
        
        // Compute smoothness score from statistics
        let grad_stats = compute_statistics(&gradient_magnitudes);
        let curv_stats = compute_statistics(&curvature_values);
        
        // Smoothness = 1 / (1 + normalized_gradient + normalized_curvature)
        let normalized_gradient = grad_stats.mean / (grad_stats.max + 1e-12);
        let normalized_curvature = curv_stats.mean / (curv_stats.max + 1e-12);
        
        let smoothness = 1.0 / (1.0 + normalized_gradient + normalized_curvature);
        
        Ok(smoothness.max(0.0).min(1.0))
    }
    
    /// Compute local differential metrics (gradient and curvature)
    fn compute_local_differential_metrics(
        &self,
        region: &ndarray::ArrayView3<f64>,
        grid: &Grid,
    ) -> KwaversResult<(f64, f64)> {
        let (nx, ny, nz) = region.dim();
        let center = (nx/2, ny/2, nz/2);
        
        if center.0 == 0 || center.1 == 0 || center.2 == 0 ||
           center.0 >= nx-1 || center.1 >= ny-1 || center.2 >= nz-1 {
            return Ok((0.0, 0.0));
        }
        
        // First derivatives (gradient)
        let grad_x = (region[[center.0+1, center.1, center.2]] - 
                     region[[center.0-1, center.1, center.2]]) / (2.0 * grid.dx);
        let grad_y = (region[[center.0, center.1+1, center.2]] - 
                     region[[center.0, center.1-1, center.2]]) / (2.0 * grid.dy);
        let grad_z = (region[[center.0, center.1, center.2+1]] - 
                     region[[center.0, center.1, center.2-1]]) / (2.0 * grid.dz);
        
        let grad_magnitude = (grad_x*grad_x + grad_y*grad_y + grad_z*grad_z).sqrt();
        
        // Second derivatives (curvature)
        let lapl_x = (region[[center.0+1, center.1, center.2]] - 
                     2.0 * region[[center.0, center.1, center.2]] + 
                     region[[center.0-1, center.1, center.2]]) / (grid.dx * grid.dx);
        let lapl_y = (region[[center.0, center.1+1, center.2]] - 
                     2.0 * region[[center.0, center.1, center.2]] + 
                     region[[center.0, center.1-1, center.2]]) / (grid.dy * grid.dy);
        let lapl_z = (region[[center.0, center.1, center.2+1]] - 
                     2.0 * region[[center.0, center.1, center.2]] + 
                     region[[center.0, center.1, center.2-1]]) / (grid.dz * grid.dz);
        
        let curvature = (lapl_x*lapl_x + lapl_y*lapl_y + lapl_z*lapl_z).sqrt();
        
        Ok((grad_magnitude, curvature))
    }
    
    /// Compute frequency content metrics
    fn compute_frequency_metrics(
        &self,
        field: &ndarray::ArrayView3<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        // Use local spectral analysis to determine frequency content
        let mut high_freq_indicators = Vec::new();
        let window = self.analysis_window;
        
        for i in window..field.shape()[0]-window {
            for j in window..field.shape()[1]-window {
                for k in window..field.shape()[2]-window {
                    // Extract local window
                    let local_region = field.slice(s![
                        i-window/2..i+window/2,
                        j-window/2..j+window/2,
                        k-window/2..k+window/2
                    ]);
                    
                    // Compute high-frequency indicator using variance of differences
                    let high_freq_indicator = self.compute_high_frequency_content(&local_region, grid)?;
                    high_freq_indicators.push(high_freq_indicator);
                }
            }
        }
        
        // Compute frequency score from indicators
        let freq_stats = compute_statistics(&high_freq_indicators);
        
        // Normalize frequency score (higher values indicate more high-frequency content)
        let frequency_score = (freq_stats.mean / (freq_stats.max + 1e-12)).min(1.0);
        
        Ok(frequency_score)
    }
    
    /// Compute high-frequency content indicator
    fn compute_high_frequency_content(
        &self,
        region: &ndarray::ArrayView3<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let (nx, ny, nz) = region.dim();
        let mut high_freq_indicator = 0.0;
        let mut count = 0;
        
        // Use second differences as a measure of high-frequency content
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Compute discrete Laplacian
                    let laplacian = 
                        (region[[i+1, j, k]] - 2.0 * region[[i, j, k]] + region[[i-1, j, k]]) / (grid.dx * grid.dx) +
                        (region[[i, j+1, k]] - 2.0 * region[[i, j, k]] + region[[i, j-1, k]]) / (grid.dy * grid.dy) +
                        (region[[i, j, k+1]] - 2.0 * region[[i, j, k]] + region[[i, j, k-1]]) / (grid.dz * grid.dz);
                    
                    high_freq_indicator += laplacian.abs();
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            Ok(high_freq_indicator / count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute efficiency score based on grid properties
    fn compute_efficiency_score(&self, grid: &Grid) -> KwaversResult<f64> {
        // Simple efficiency metric based on grid size and uniformity
        let total_points = (grid.nx * grid.ny * grid.nz) as f64;
        let max_reasonable_points = 1e7; // 10M points as a reference
        
        // Efficiency decreases with grid size (memory/compute cost)
        let size_efficiency = (max_reasonable_points / total_points).min(1.0);
        
        // Grid uniformity (uniform grids are more efficient for spectral methods)
        let dx_ratio = grid.dx / grid.dy;
        let dy_ratio = grid.dy / grid.dz;
        let uniformity = 1.0 - ((dx_ratio - 1.0).abs() + (dy_ratio - 1.0).abs()).min(1.0);
        
        let efficiency_score = 0.7 * size_efficiency + 0.3 * uniformity;
        
        Ok(efficiency_score.max(0.0).min(1.0))
    }
    
    /// Compute detailed metrics for comprehensive analysis
    fn compute_detailed_metrics(
        &self,
        field: &ndarray::ArrayView3<f64>,
        grid: &Grid,
    ) -> KwaversResult<DetailedMetrics> {
        // Placeholder implementation
        // In practice, this would compute comprehensive spectral analysis,
        // material property analysis, etc.
        
        Ok(DetailedMetrics::default())
    }
    
    /// Compute composite quality score
    fn compute_composite_score(&self, metrics: &QualityMetrics) -> KwaversResult<f64> {
        let composite = 
            self.criteria.smoothness_weight * metrics.smoothness_score +
            self.criteria.frequency_weight * metrics.frequency_score +
            self.criteria.material_weight * metrics.homogeneity_score +
            self.criteria.efficiency_weight * metrics.efficiency_score;
        
        Ok(composite.max(0.0).min(1.0))
    }
    
    /// Add metrics to history for hysteresis analysis
    fn add_to_history(&mut self, metrics: QualityMetrics) {
        self.history.push(metrics);
        
        // Limit history size
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }
    
    /// Determine if method switching should occur based on hysteresis
    pub fn should_switch_method(
        &self,
        current_metrics: &QualityMetrics,
        previous_composite_score: f64,
    ) -> KwaversResult<bool> {
        let score_change = (current_metrics.composite_score - previous_composite_score).abs();
        
        // Apply hysteresis: only switch if change exceeds threshold + hysteresis
        let switch_threshold = self.criteria.switch_threshold + self.criteria.hysteresis_factor;
        
        Ok(score_change > switch_threshold)
    }
    
    /// Get trend analysis from historical data
    pub fn get_trend_analysis(&self) -> TrendAnalysis {
        if self.history.len() < 3 {
            return TrendAnalysis::default();
        }
        
        let recent_scores: Vec<f64> = self.history.iter()
            .rev()
            .take(5)
            .map(|m| m.composite_score)
            .collect();
        
        let trend_slope = if recent_scores.len() >= 2 {
            let last = recent_scores[0];
            let first = recent_scores[recent_scores.len() - 1];
            (last - first) / (recent_scores.len() - 1) as f64
        } else {
            0.0
        };
        
        let trend_direction = if trend_slope > 0.01 {
            TrendDirection::Improving
        } else if trend_slope < -0.01 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };
        
        TrendAnalysis {
            direction: trend_direction,
            slope: trend_slope,
            confidence: 0.8, // Placeholder
        }
    }
    
    /// Update selection criteria based on performance feedback
    pub fn update_criteria_from_feedback(&mut self, performance_metrics: &PerformanceMetrics) {
        // Adaptive learning: adjust criteria based on actual performance
        // This is a simplified implementation
        
        if performance_metrics.accuracy_error > 0.01 {
            // If accuracy is poor, increase emphasis on smoothness and frequency
            self.criteria.smoothness_weight *= 1.1;
            self.criteria.frequency_weight *= 1.1;
            self.criteria.efficiency_weight *= 0.9;
        }
        
        if performance_metrics.computational_time > performance_metrics.target_time {
            // If computation is too slow, increase emphasis on efficiency
            self.criteria.efficiency_weight *= 1.1;
            self.criteria.smoothness_weight *= 0.95;
        }
        
        // Renormalize weights
        let total = self.criteria.smoothness_weight + 
                   self.criteria.frequency_weight + 
                   self.criteria.material_weight + 
                   self.criteria.efficiency_weight;
        
        self.criteria.smoothness_weight /= total;
        self.criteria.frequency_weight /= total;
        self.criteria.material_weight /= total;
        self.criteria.efficiency_weight /= total;
        
        debug!("Updated selection criteria based on performance feedback");
    }
}

/// Trend analysis results
#[derive(Debug, Clone, Default)]
pub struct TrendAnalysis {
    pub direction: TrendDirection,
    pub slope: f64,
    pub confidence: f64,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Default)]
pub enum TrendDirection {
    #[default]
    Stable,
    Improving,
    Degrading,
}

/// Performance metrics for feedback
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub accuracy_error: f64,
    pub computational_time: f64,
    pub target_time: f64,
    pub memory_usage: f64,
}

/// Compute statistical metrics for a dataset
fn compute_statistics(data: &[f64]) -> StatisticalMetrics {
    if data.is_empty() {
        return StatisticalMetrics::default();
    }
    
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    let variance = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / n;
    let std_dev = variance.sqrt();
    
    // Simplified skewness and kurtosis calculation
    let skewness = if std_dev > 1e-12 {
        data.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / n
    } else {
        0.0
    };
    
    let kurtosis = if std_dev > 1e-12 {
        data.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>() / n - 3.0
    } else {
        0.0
    };
    
    StatisticalMetrics {
        mean,
        std_dev,
        min,
        max,
        skewness,
        kurtosis,
    }
}