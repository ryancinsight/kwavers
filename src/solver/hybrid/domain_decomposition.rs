//! Domain Decomposition for Hybrid PSTD/FDTD Solver
//!
//! This module implements intelligent domain decomposition algorithms that analyze
//! the computational domain and automatically partition it into regions where
//! either PSTD, FDTD, or hybrid methods provide optimal performance.
//!
//! # Physics-Based Selection Criteria:
//!
//! ## PSTD Optimal Regions:
//! - **Homogeneous media**: Constant material properties
//! - **Smooth fields**: Low spatial gradients, high spectral content
//! - **Far-field regions**: Distance from sources and boundaries
//! - **High-frequency dominance**: Where spectral accuracy is critical
//!
//! ## FDTD Optimal Regions:
//! - **Heterogeneous media**: Material interfaces and discontinuities
//! - **Complex geometries**: Curved boundaries, fine structures
//! - **Near-field regions**: Close to sources and scatterers
//! - **Shock formation**: Steep gradients and nonlinear effects
//!
//! ## Hybrid Regions:
//! - **Transition zones**: Gradual changes in material properties
//! - **Intermediate smoothness**: Neither fully smooth nor highly discontinuous
//! - **Multi-scale features**: Mixed frequency content
//!
//! # Design Principles Applied:
//! - **SOLID**: Single responsibility for domain analysis and partitioning
//! - **CUPID**: Composable algorithms, predictable partitioning
//! - **GRASP**: Information expert pattern for quality assessment
//! - **DRY**: Reusable analysis and partitioning algorithms

use crate::grid::Grid;
use crate::medium::Medium;
use crate::error::KwaversResult;
use crate::solver::hybrid::DecompositionStrategy;
use crate::solver::hybrid::adaptive_selection::QualityMetrics;
use ndarray::{Array3, Array4, Axis, Zip, s};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use log::{debug, info};

/// Domain region with associated solver type
#[derive(Debug, Clone)]
pub struct DomainRegion {
    /// Starting indices (inclusive)
    pub start: (usize, usize, usize),
    /// Ending indices (exclusive)
    pub end: (usize, usize, usize),
    /// Optimal solver type for this region
    pub domain_type: DomainType,
    /// Quality score for this assignment (0-1, higher score indicates more suitable)
    pub quality_score: f64,
    /// Buffer zones for coupling with adjacent regions
    pub buffer_zones: BufferZones,
}

impl DomainRegion {
    /// Calculate the volume of this region in cells
    pub fn volume(&self) -> usize {
        (self.end.0 - self.start.0) * 
        (self.end.1 - self.start.1) * 
        (self.end.2 - self.start.2)
    }
}

/// Types of computational domains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainType {
    /// Use PSTD (spectral) method
    Spectral,
    /// Use FDTD (finite difference) method
    FiniteDifference,
    /// Use hybrid approach with both methods
    Hybrid,
}

/// Buffer zones for inter-domain coupling
#[derive(Debug, Clone)]
pub struct BufferZones {
    /// Buffer width in each direction (nx-, nx+, ny-, ny+, nz-, nz+)
    pub widths: [usize; 6],
    /// Overlap regions with adjacent domains
    pub overlaps: Vec<OverlapRegion>,
}

/// Overlap region between two domains
#[derive(Debug, Clone)]
pub struct OverlapRegion {
    /// Starting indices of overlap
    pub start: (usize, usize, usize),
    /// Ending indices of overlap
    pub end: (usize, usize, usize),
    /// Type of the adjacent domain
    pub adjacent_type: DomainType,
    /// Interpolation weight for this domain (0-1)
    pub weight: f64,
}

impl Default for BufferZones {
    fn default() -> Self {
        Self {
            widths: [2, 2, 2, 2, 2, 2], // 2 cells on each face
            overlaps: Vec::new(),
        }
    }
}

/// Domain decomposer for hybrid solver
#[derive(Clone, Debug)]
pub struct DomainDecomposer {
    /// Decomposition strategy
    strategy: DecompositionStrategy,
    /// Minimum domain size in each dimension
    min_domain_size: (usize, usize, usize),
    /// Analysis parameters
    analysis_params: AnalysisParameters,
    /// Cached analysis results
    analysis_cache: HashMap<String, AnalysisResult>,
}

/// Parameters for domain analysis
#[derive(Debug, Clone)]
struct AnalysisParameters {
    /// Smoothness threshold for PSTD (lower = more PSTD)
    smoothness_threshold: f64,
    /// Heterogeneity threshold for FDTD (higher = more FDTD)
    heterogeneity_threshold: f64,
    /// Frequency cutoff for method selection
    frequency_cutoff: f64,
    /// Minimum region size for decomposition
    min_region_size: usize,
    /// Buffer zone overlap factor
    overlap_factor: f64,
}

impl Default for AnalysisParameters {
    fn default() -> Self {
        use crate::constants::domain_decomposition::*;
        Self {
            smoothness_threshold: SMOOTHNESS_THRESHOLD,
            heterogeneity_threshold: HETEROGENEITY_THRESHOLD,
            frequency_cutoff: FREQUENCY_CUTOFF_FRACTION, // Fraction of Nyquist frequency
            min_region_size: 8,     // Minimum 8x8x8 regions
            overlap_factor: 0.2,    // 20% overlap
        }
    }
}

/// Results of domain analysis
#[derive(Debug, Clone)]
struct AnalysisResult {
    /// Smoothness map (higher = smoother, favor PSTD)
    smoothness: Array3<f64>,
    /// Heterogeneity map (higher = more heterogeneous, favor FDTD)
    heterogeneity: Array3<f64>,
    /// Frequency content map (higher = high frequency, favor PSTD)
    frequency_content: Array3<f64>,
    /// Boundary proximity map (higher = closer to boundary, favor FDTD)
    boundary_proximity: Array3<f64>,
    /// Composite quality score for each solver type
    quality_maps: HashMap<DomainType, Array3<f64>>,
}

impl DomainDecomposer {
    /// Create a new domain decomposer
    pub fn new(strategy: DecompositionStrategy, grid: &Grid) -> KwaversResult<Self> {
        let min_domain_size = (
            (grid.nx / 8).max(4),
            (grid.ny / 8).max(4),
            (grid.nz / 8).max(4),
        );
        
        info!("Initializing domain decomposer with strategy: {:?}", strategy);
        debug!("Minimum domain size: {:?}", min_domain_size);
        
        Ok(Self {
            strategy,
            min_domain_size,
            analysis_params: AnalysisParameters::default(),
            analysis_cache: HashMap::new(),
        })
    }
    
    /// Decompose the computational domain based on field properties
    pub fn decompose_domain(
        &mut self,
        fields: &Array4<f64>,
        medium: &dyn Medium,
        quality_metrics: &QualityMetrics,
        grid: &Grid,
    ) -> KwaversResult<Vec<DomainRegion>> {
        info!("Starting domain decomposition");
        
        // Analyze field properties and material distribution
        let analysis = self.analyze_domain(fields, medium, grid)?;
        
        // Apply decomposition strategy
        let regions = match self.strategy {
            DecompositionStrategy::Fixed => {
                self.fixed_decomposition(grid)?
            }
            DecompositionStrategy::Adaptive => {
                self.adaptive_decomposition(&analysis, quality_metrics, grid)?
            }
            DecompositionStrategy::GradientBased => {
                self.gradient_based_decomposition(&analysis, grid)?
            }
            DecompositionStrategy::FrequencyBased => {
                self.frequency_based_decomposition(&analysis, grid)?
            }
            DecompositionStrategy::MaterialBased => {
                self.material_based_decomposition(&analysis, medium, grid)?
            }
        };
        
        // Configure buffer zones and overlaps
        let configured_regions = self.configure_buffer_zones(regions, grid)?;
        
        info!("Domain decomposition completed: {} regions", configured_regions.len());
        self.log_decomposition_summary(&configured_regions);
        
        Ok(configured_regions)
    }
    
    /// Analyze domain properties for decomposition
    fn analyze_domain(
        &mut self,
        fields: &Array4<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<AnalysisResult> {
        debug!("Analyzing domain properties");
        
        // Compute smoothness using spectral analysis
        let smoothness = self.compute_smoothness_map(fields, grid)?;
        
        // Compute heterogeneity from material properties
        let heterogeneity = self.compute_heterogeneity_map(medium, grid)?;
        
        // Compute frequency content using local FFT analysis
        let frequency_content = self.compute_frequency_content_map(fields, grid)?;
        
        // Compute boundary proximity
        let boundary_proximity = self.compute_boundary_proximity_map(grid)?;
        
        // Generate quality maps for each solver type
        let quality_maps = self.compute_quality_maps(
            &smoothness,
            &heterogeneity,
            &frequency_content,
            &boundary_proximity,
        )?;
        
        Ok(AnalysisResult {
            smoothness,
            heterogeneity,
            frequency_content,
            boundary_proximity,
            quality_maps,
        })
    }
    
    /// Compute smoothness map using local gradient analysis
    fn compute_smoothness_map(
        &self,
        fields: &Array4<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let mut smoothness = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        // Use pressure field for smoothness analysis
        let pressure = fields.index_axis(Axis(0), 0); // Assuming pressure is at index 0
        
        // Compute local gradients using central differences
        for i in 1..grid.nx-1 {
            for j in 1..grid.ny-1 {
                for k in 1..grid.nz-1 {
                    // Compute gradients in all directions
                    let grad_x = (pressure[[i+1, j, k]] - pressure[[i-1, j, k]]) / (2.0 * grid.dx);
                    let grad_y = (pressure[[i, j+1, k]] - pressure[[i, j-1, k]]) / (2.0 * grid.dy);
                    let grad_z = (pressure[[i, j, k+1]] - pressure[[i, j, k-1]]) / (2.0 * grid.dz);
                    
                    // Compute gradient magnitude
                    let grad_mag = (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();
                    
                    // Compute second derivatives for curvature
                    let lap_x = (pressure[[i+1, j, k]] - 2.0 * pressure[[i, j, k]] + pressure[[i-1, j, k]]) / (grid.dx * grid.dx);
                    let lap_y = (pressure[[i, j+1, k]] - 2.0 * pressure[[i, j, k]] + pressure[[i, j-1, k]]) / (grid.dy * grid.dy);
                    let lap_z = (pressure[[i, j, k+1]] - 2.0 * pressure[[i, j, k]] + pressure[[i, j, k-1]]) / (grid.dz * grid.dz);
                    
                    let curvature = (lap_x * lap_x + lap_y * lap_y + lap_z * lap_z).sqrt();
                    
                    // Smoothness metric: low gradients and low curvature = high smoothness
                    let local_smoothness = 1.0 / (1.0 + grad_mag + curvature);
                    smoothness[[i, j, k]] = local_smoothness;
                }
            }
        }
        
        // Apply smoothing filter to reduce noise
        self.apply_smoothing_filter(&mut smoothness, grid)?;
        
        Ok(smoothness)
    }
    
    /// Compute heterogeneity map from material properties
    fn compute_heterogeneity_map(
        &self,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let mut heterogeneity = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        // Sample material properties at grid points
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    
                    // Get material properties at this point
                    let density = medium.density(x, y, z, grid);
                    let sound_speed = medium.sound_speed(x, y, z, grid);
                    
                    // Compute local variation by comparing with neighbors
                    let mut local_variation = 0.0;
                    let mut count = 0;
                    
                    for di in -1i32..=1 {
                        for dj in -1i32..=1 {
                            for dk in -1i32..=1 {
                                if di == 0 && dj == 0 && dk == 0 { continue; }
                                
                                let ni = (i as i32 + di) as usize;
                                let nj = (j as i32 + dj) as usize;
                                let nk = (k as i32 + dk) as usize;
                                
                                if ni < grid.nx && nj < grid.ny && nk < grid.nz {
                                    let nx = ni as f64 * grid.dx;
                                    let ny = nj as f64 * grid.dy;
                                    let nz = nk as f64 * grid.dz;
                                    
                                    let n_density = medium.density(nx, ny, nz, grid);
                                    let n_sound_speed = medium.sound_speed(nx, ny, nz, grid);
                                    
                                    // Relative variation in material properties
                                    let density_var = ((n_density - density) / density).abs();
                                    let speed_var = ((n_sound_speed - sound_speed) / sound_speed).abs();
                                    
                                    local_variation += density_var + speed_var;
                                    count += 1;
                                }
                            }
                        }
                    }
                    
                    if count > 0 {
                        heterogeneity[[i, j, k]] = local_variation / count as f64;
                    }
                }
            }
        }
        
        Ok(heterogeneity)
    }
    
    /// Compute frequency content map using local spectral analysis
    fn compute_frequency_content_map(
        &self,
        fields: &Array4<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let mut frequency_content = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        let pressure = fields.index_axis(Axis(0), 0);
        let window_size = 8; // Analysis window size
        
        // Perform local spectral analysis using sliding window
        for i in window_size/2..grid.nx-window_size/2 {
            for j in window_size/2..grid.ny-window_size/2 {
                for k in window_size/2..grid.nz-window_size/2 {
                    // Extract local window
                    let local_window = pressure.slice(s![
                        i-window_size/2..i+window_size/2,
                        j-window_size/2..j+window_size/2,
                        k-window_size/2..k+window_size/2
                    ]);
                    
                    // Compute local frequency content
                    let high_freq_content = self.analyze_local_frequency_content(&local_window, grid)?;
                    frequency_content[[i, j, k]] = high_freq_content;
                }
            }
        }
        
        Ok(frequency_content)
    }
    
    /// Analyze frequency content in a local window
    fn analyze_local_frequency_content(
        &self,
        window: &ndarray::ArrayView3<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        // Simple frequency analysis using variance of finite differences
        let (nx, ny, nz) = window.dim();
        let mut high_freq_indicator = 0.0;
        let mut count = 0;
        
        // Compute high-frequency content using second derivatives
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let laplacian = 
                        (window[[i+1, j, k]] - 2.0 * window[[i, j, k]] + window[[i-1, j, k]]) / (grid.dx * grid.dx) +
                        (window[[i, j+1, k]] - 2.0 * window[[i, j, k]] + window[[i, j-1, k]]) / (grid.dy * grid.dy) +
                        (window[[i, j, k+1]] - 2.0 * window[[i, j, k]] + window[[i, j, k-1]]) / (grid.dz * grid.dz);
                    
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
    
    /// Compute boundary proximity map
    fn compute_boundary_proximity_map(&self, grid: &Grid) -> KwaversResult<Array3<f64>> {
        let mut proximity = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    // Distance to nearest boundary (normalized)
                    let dist_x = (i.min(grid.nx - 1 - i)) as f64 / (grid.nx as f64 / 2.0);
                    let dist_y = (j.min(grid.ny - 1 - j)) as f64 / (grid.ny as f64 / 2.0);
                    let dist_z = (k.min(grid.nz - 1 - k)) as f64 / (grid.nz as f64 / 2.0);
                    
                    let min_dist = dist_x.min(dist_y).min(dist_z);
                    
                    // Proximity = 1 - distance (higher near boundaries)
                    proximity[[i, j, k]] = 1.0 - min_dist;
                }
            }
        }
        
        Ok(proximity)
    }
    
    /// Compute quality maps for each solver type
    fn compute_quality_maps(
        &self,
        smoothness: &Array3<f64>,
        heterogeneity: &Array3<f64>,
        frequency_content: &Array3<f64>,
        boundary_proximity: &Array3<f64>,
    ) -> KwaversResult<HashMap<DomainType, Array3<f64>>> {
        let mut quality_maps = HashMap::new();
        
        let (nx, ny, nz) = smoothness.dim();
        
        // PSTD quality: favor smooth, homogeneous, high-frequency, far-from-boundary regions
        let mut pstd_quality = Array3::zeros((nx, ny, nz));
        Zip::from(&mut pstd_quality)
            .and(smoothness)
            .and(heterogeneity)
            .and(frequency_content)
            .and(boundary_proximity)
            .for_each(|quality, &smooth, &hetero, &freq, &boundary| {
                // PSTD benefits from smoothness, low heterogeneity, high frequency, low boundary proximity
                *quality = smooth * (1.0 - hetero) * freq * (1.0 - boundary);
            });
        
        // FDTD quality: favor rough, heterogeneous, low-frequency, near-boundary regions
        let mut fdtd_quality = Array3::zeros((nx, ny, nz));
        Zip::from(&mut fdtd_quality)
            .and(smoothness)
            .and(heterogeneity)
            .and(frequency_content)
            .and(boundary_proximity)
            .for_each(|quality, &smooth, &hetero, &freq, &boundary| {
                // FDTD benefits from low smoothness, high heterogeneity, low frequency, high boundary proximity
                *quality = (1.0 - smooth) * hetero * (1.0 - freq) * boundary;
            });
        
        // Hybrid quality: favor intermediate regions
        let mut hybrid_quality = Array3::zeros((nx, ny, nz));
        Zip::from(&mut hybrid_quality)
            .and(&pstd_quality)
            .and(&fdtd_quality)
            .for_each(|quality, &pstd, &fdtd| {
                // Hybrid is good when neither PSTD nor FDTD clearly dominates
                let max_quality = pstd.max(fdtd);
                let min_quality = pstd.min(fdtd);
                *quality = min_quality / (max_quality + 1e-10); // Avoid division by zero
            });
        
        quality_maps.insert(DomainType::Spectral, pstd_quality);
        quality_maps.insert(DomainType::FiniteDifference, fdtd_quality);
        quality_maps.insert(DomainType::Hybrid, hybrid_quality);
        
        Ok(quality_maps)
    }
    
    /// Apply smoothing filter to reduce noise in analysis maps
    fn apply_smoothing_filter(
        &self,
        field: &mut Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        // Simple 3x3x3 averaging filter
        let original = field.clone();
        
        for i in 1..grid.nx-1 {
            for j in 1..grid.ny-1 {
                for k in 1..grid.nz-1 {
                    let mut sum = 0.0;
                    let mut count = 0;
                    
                    for di in -1i32..=1 {
                        for dj in -1i32..=1 {
                            for dk in -1i32..=1 {
                                let ni = (i as i32 + di) as usize;
                                let nj = (j as i32 + dj) as usize;
                                let nk = (k as i32 + dk) as usize;
                                
                                sum += original[[ni, nj, nk]];
                                count += 1;
                            }
                        }
                    }
                    
                    field[[i, j, k]] = sum / count as f64;
                }
            }
        }
        
        Ok(())
    }
    
    /// Fixed decomposition strategy
    fn fixed_decomposition(&self, grid: &Grid) -> KwaversResult<Vec<DomainRegion>> {
        // Simple fixed decomposition: PSTD in center, FDTD near boundaries
        let mut regions = Vec::new();
        
        let boundary_width = self.min_domain_size.0;
        
        // Central PSTD region
        if grid.nx > 2 * boundary_width && grid.ny > 2 * boundary_width && grid.nz > 2 * boundary_width {
            regions.push(DomainRegion {
                start: (boundary_width, boundary_width, boundary_width),
                end: (grid.nx - boundary_width, grid.ny - boundary_width, grid.nz - boundary_width),
                domain_type: DomainType::Spectral,
                quality_score: 0.8,
                buffer_zones: BufferZones::default(),
            });
        }
        
        // Boundary FDTD regions
        // X boundaries
        regions.push(DomainRegion {
            start: (0, 0, 0),
            end: (boundary_width, grid.ny, grid.nz),
            domain_type: DomainType::FiniteDifference,
            quality_score: 0.7,
            buffer_zones: BufferZones::default(),
        });
        
        regions.push(DomainRegion {
            start: (grid.nx - boundary_width, 0, 0),
            end: (grid.nx, grid.ny, grid.nz),
            domain_type: DomainType::FiniteDifference,
            quality_score: 0.7,
            buffer_zones: BufferZones::default(),
        });
        
        // Add Y and Z boundary regions...
        // (simplified for brevity)
        
        Ok(regions)
    }
    
    /// Adaptive decomposition based on quality maps
    fn adaptive_decomposition(
        &self,
        analysis: &AnalysisResult,
        _quality_metrics: &QualityMetrics,
        grid: &Grid,
    ) -> KwaversResult<Vec<DomainRegion>> {
        // Use k-means-like clustering on quality maps
        let regions = self.cluster_regions_by_quality(&analysis.quality_maps, grid)?;
        Ok(regions)
    }
    
    /// Cluster regions by quality using simplified algorithm
    fn cluster_regions_by_quality(
        &self,
        quality_maps: &HashMap<DomainType, Array3<f64>>,
        grid: &Grid,
    ) -> KwaversResult<Vec<DomainRegion>> {
        let mut regions = Vec::new();
        
        // Find the best method for each grid point
        let mut best_method = Array3::from_elem((grid.nx, grid.ny, grid.nz), DomainType::Spectral);
        let mut best_quality = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        for (&method, quality_map) in quality_maps {
            ndarray::Zip::from(&mut best_method)
                .and(&mut best_quality)
                .and(quality_map)
                .for_each(|best_m, best_q, &quality| {
                    if quality > *best_q {
                        *best_m = method;
                        *best_q = quality;
                    }
                });
        }
        
        // Group contiguous regions of the same method (simplified)
        let mut visited = Array3::from_elem((grid.nx, grid.ny, grid.nz), false);
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if !visited[[i, j, k]] {
                        let region = self.extract_connected_region(
                            &best_method,
                            &mut visited,
                            (i, j, k),
                            grid,
                        )?;
                        if let Some(r) = region {
                            regions.push(r);
                        }
                    }
                }
            }
        }
        
        Ok(regions)
    }
    
    /// Extract a connected region of the same domain type
    fn extract_connected_region(
        &self,
        method_map: &Array3<DomainType>,
        visited: &mut Array3<bool>,
        start: (usize, usize, usize),
        grid: &Grid,
    ) -> KwaversResult<Option<DomainRegion>> {
        let target_method = method_map[start];
        let mut region_points = Vec::new();
        let mut stack = vec![start];
        
        while let Some((i, j, k)) = stack.pop() {
            if visited[[i, j, k]] || method_map[[i, j, k]] != target_method {
                continue;
            }
            
            visited[[i, j, k]] = true;
            region_points.push((i, j, k));
            
            // Add neighbors
            for (di, dj, dk) in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                let nk = k as i32 + dk;
                
                if ni >= 0 && ni < grid.nx as i32 && 
                   nj >= 0 && nj < grid.ny as i32 && 
                   nk >= 0 && nk < grid.nz as i32 {
                    stack.push((ni as usize, nj as usize, nk as usize));
                }
            }
        }
        
        // Create bounding box for the region
        if region_points.len() >= self.analysis_params.min_region_size {
            let min_i = region_points.iter().map(|(i, _, _)| *i).min().unwrap();
            let max_i = region_points.iter().map(|(i, _, _)| *i).max().unwrap();
            let min_j = region_points.iter().map(|(_, j, _)| *j).min().unwrap();
            let max_j = region_points.iter().map(|(_, j, _)| *j).max().unwrap();
            let min_k = region_points.iter().map(|(_, _, k)| *k).min().unwrap();
            let max_k = region_points.iter().map(|(_, _, k)| *k).max().unwrap();
            
            // Compute quality score based on region size and method type
            let region_size = region_points.len();
            let quality_score = match target_method {
                DomainType::Spectral => 0.9,    // High quality for spectral
                DomainType::FiniteDifference => 0.8, // Good quality for FDTD
                DomainType::Hybrid => 0.75,     // Medium quality for hybrid
            } * (1.0 - (region_size as f64 / (grid.nx * grid.ny * grid.nz) as f64).min(0.5));
            
            Ok(Some(DomainRegion {
                start: (min_i, min_j, min_k),
                end: (max_i + 1, max_j + 1, max_k + 1),
                domain_type: target_method,
                quality_score,
                buffer_zones: BufferZones::default(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Gradient-based decomposition
    fn gradient_based_decomposition(
        &self,
        analysis: &AnalysisResult,
        grid: &Grid,
    ) -> KwaversResult<Vec<DomainRegion>> {
        // Use gradient magnitude to determine region boundaries
        let regions = self.segment_by_gradients(&analysis.smoothness, grid)?;
        Ok(regions)
    }
    
    /// Frequency-based decomposition
    fn frequency_based_decomposition(
        &self,
        analysis: &AnalysisResult,
        grid: &Grid,
    ) -> KwaversResult<Vec<DomainRegion>> {
        // Use frequency content to determine optimal method
        let regions = self.segment_by_frequency(&analysis.frequency_content, grid)?;
        Ok(regions)
    }
    
    /// Material-based decomposition
    fn material_based_decomposition(
        &self,
        analysis: &AnalysisResult,
        _medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Vec<DomainRegion>> {
        // Use heterogeneity map for material-based segmentation
        let regions = self.segment_by_materials(&analysis.heterogeneity, grid)?;
        Ok(regions)
    }
    
    /// Gradient-based segmentation for domain decomposition
    fn segment_by_gradients(&self, smoothness: &Array3<f64>, grid: &Grid) -> KwaversResult<Vec<DomainRegion>> {
        let (nx, ny, nz) = smoothness.dim();
        let mut regions = Vec::new();
        let mut processed = Array3::<bool>::default((nx, ny, nz));
        
        // Threshold for gradient magnitude (regions with gradient above this use FDTD)
        let gradient_threshold = 0.1;
        
        // Compute gradient magnitude
        let gradient_mag = self.compute_gradient_magnitude(smoothness);
        
        // Region growing algorithm
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if !processed[[i, j, k]] {
                        // Start a new region
                        let seed_gradient = gradient_mag[[i, j, k]];
                        let method = if seed_gradient > gradient_threshold {
                            DomainType::FiniteDifference
                        } else {
                            DomainType::Spectral
                        };
                        
                        // Grow region from seed
                        let region = self.grow_region_by_gradient(
                            (i, j, k),
                            &gradient_mag,
                            &mut processed,
                            method,
                            gradient_threshold,
                            grid
                        )?;
                        
                        regions.push(region);
                    }
                }
            }
        }
        
        // If no regions found, create a single spectral region
        if regions.is_empty() {
            // Compute average smoothness as quality score
            let quality_score = smoothness.mean().unwrap_or(0.5);
            
            regions.push(DomainRegion {
                start: (0, 0, 0),
                end: (nx-1, ny-1, nz-1),
                domain_type: DomainType::Spectral,
                quality_score,
                buffer_zones: BufferZones::default(),
            });
        }
        
        // Set up neighbor relationships
        self.setup_neighbors(&mut regions);
        
        Ok(regions)
    }
    
    /// Compute gradient magnitude of a field
    fn compute_gradient_magnitude(&self, field: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = field.dim();
        let mut gradient_mag = Array3::zeros((nx, ny, nz));
        
        // Use ndarray's windows for gradient computation
        // Process interior points with central differences
        gradient_mag
            .slice_mut(s![1..nx-1, 1..ny-1, 1..nz-1])
            .indexed_iter_mut()
            .for_each(|((i, j, k), grad)| {
                let i = i + 1;
                let j = j + 1;
                let k = k + 1;
                
                // Central differences
                let dx = (field[[i+1, j, k]] - field[[i-1, j, k]]) / 2.0;
                let dy = (field[[i, j+1, k]] - field[[i, j-1, k]]) / 2.0;
                let dz = (field[[i, j, k+1]] - field[[i, j, k-1]]) / 2.0;
                
                *grad = (dx*dx + dy*dy + dz*dz).sqrt();
            });
        
        // Handle boundaries with one-sided differences using slices
        // X boundaries
        let temp = gradient_mag.slice(s![1, .., ..]).to_owned();
        gradient_mag.slice_mut(s![0, .., ..]).assign(&temp);
        let temp = gradient_mag.slice(s![nx-2, .., ..]).to_owned();
        gradient_mag.slice_mut(s![nx-1, .., ..]).assign(&temp);
        
        // Y boundaries
        let temp = gradient_mag.slice(s![.., 1, ..]).to_owned();
        gradient_mag.slice_mut(s![.., 0, ..]).assign(&temp);
        let temp = gradient_mag.slice(s![.., ny-2, ..]).to_owned();
        gradient_mag.slice_mut(s![.., ny-1, ..]).assign(&temp);
        
        // Z boundaries
        let temp = gradient_mag.slice(s![.., .., 1]).to_owned();
        gradient_mag.slice_mut(s![.., .., 0]).assign(&temp);
        let temp = gradient_mag.slice(s![.., .., nz-2]).to_owned();
        gradient_mag.slice_mut(s![.., .., nz-1]).assign(&temp);
        
        gradient_mag
    }
    
    /// Grow a region from a seed point based on gradient similarity
    fn grow_region_by_gradient(
        &self,
        seed: (usize, usize, usize),
        gradient_mag: &Array3<f64>,
        processed: &mut Array3<bool>,
        method: DomainType,
        threshold: f64,
        grid: &Grid,
    ) -> KwaversResult<DomainRegion> {
        let (nx, ny, nz) = gradient_mag.dim();
        let mut region_indices = Vec::new();
        let mut stack = vec![seed];
        let mut bounds = (seed, seed);
        
        // Region growing with gradient-based criteria
        while let Some((i, j, k)) = stack.pop() {
            if processed[[i, j, k]] {
                continue;
            }
            
            let grad = gradient_mag[[i, j, k]];
            let matches_criteria = match method {
                DomainType::Spectral => grad <= threshold,
                DomainType::FiniteDifference => grad > threshold,
                DomainType::Hybrid => grad > threshold * 0.5 && grad <= threshold, // Mid-range gradients
            };
            
            if matches_criteria {
                processed[[i, j, k]] = true;
                region_indices.push((i, j, k));
                
                // Update bounds
                bounds.0 = (bounds.0.0.min(i), bounds.0.1.min(j), bounds.0.2.min(k));
                bounds.1 = (bounds.1.0.max(i), bounds.1.1.max(j), bounds.1.2.max(k));
                
                // Add neighbors to stack
                for di in -1i32..=1 {
                    for dj in -1i32..=1 {
                        for dk in -1i32..=1 {
                            if di == 0 && dj == 0 && dk == 0 {
                                continue;
                            }
                            
                            let ni = (i as i32 + di) as usize;
                            let nj = (j as i32 + dj) as usize;
                            let nk = (k as i32 + dk) as usize;
                            
                            if ni < nx && nj < ny && nk < nz && !processed[[ni, nj, nk]] {
                                stack.push((ni, nj, nk));
                            }
                        }
                    }
                }
            }
        }
        
        // Compute average quality of the clustered region
        let total_quality: f64 = region_indices.iter()
            .map(|&idx| gradient_mag[[idx.0, idx.1, idx.2]])
            .sum();
        let quality_score = total_quality / region_indices.len() as f64;
        
        Ok(DomainRegion {
            start: bounds.0,
            end: bounds.1,
            domain_type: method,
            quality_score,
            buffer_zones: BufferZones::default(),
        })
    }
    
    /// Segment domain based on frequency content
    fn segment_by_frequency(&self, frequency_content: &Array3<f64>, grid: &Grid) -> KwaversResult<Vec<DomainRegion>> {
        let mut regions = Vec::new();
        
        // Compute local frequency characteristics using windowed FFT
        let window_size = 16; // Window size for local frequency analysis
        let overlap = 8;      // Overlap between windows
        
        // Threshold for high-frequency content (use FDTD for high frequencies)
        let frequency_threshold = 0.3; // Normalized frequency threshold
        
        // Scan the domain with overlapping windows
        for i in (0..grid.nx).step_by(window_size - overlap) {
            for j in (0..grid.ny).step_by(window_size - overlap) {
                for k in (0..grid.nz).step_by(window_size - overlap) {
                    // Define window bounds
                    let i_end = (i + window_size).min(grid.nx);
                    let j_end = (j + window_size).min(grid.ny);
                    let k_end = (k + window_size).min(grid.nz);
                    
                    // Extract window data
                    let window = frequency_content.slice(ndarray::s![i..i_end, j..j_end, k..k_end]);
                    
                    // Compute frequency metric (simplified - ratio of high to low frequencies)
                    let high_freq_energy: f64 = window.iter()
                        .filter(|&&v| v.abs() > frequency_threshold)
                        .map(|v| v * v)
                        .sum();
                    
                    let total_energy: f64 = window.iter()
                        .map(|v| v * v)
                        .sum();
                    
                    let high_freq_ratio = if total_energy > 1e-10 {
                        high_freq_energy / total_energy
                    } else {
                        0.0
                    };
                    
                    // Choose domain type based on frequency content
                    let domain_type = if high_freq_ratio > 0.5 {
                        DomainType::FiniteDifference // FDTD for high frequencies
                    } else {
                        DomainType::Spectral // PSTD for smooth fields
                    };
                    
                    // Create region
                    regions.push(DomainRegion {
                        start: (i, j, k),
                        end: (i_end, j_end, k_end),
                        domain_type,
                        buffer_zones: BufferZones::default(),
                        quality_score: 0.0,
                    });
                }
            }
        }
        
        // Merge adjacent regions of the same type
        regions = self.merge_adjacent_regions(regions)?;
        
        // Optimize buffer zones
        self.optimize_buffer_zones(regions, grid)
    }
    
    fn segment_by_materials(&self, heterogeneity: &Array3<f64>, grid: &Grid) -> KwaversResult<Vec<DomainRegion>> {
        let mut regions = Vec::new();
        
        // Threshold for material heterogeneity
        let heterogeneity_threshold = 0.1; // 10% variation threshold
        
        // Compute local heterogeneity using gradient magnitude
        let mut visited = Array3::<bool>::default(heterogeneity.dim());
        
        // Region growing algorithm for material-based segmentation
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if visited[[i, j, k]] {
                        continue;
                    }
                    
                    // Start new region
                    let seed_value = heterogeneity[[i, j, k]];
                    let mut region_points = Vec::new();
                    let mut queue = vec![(i, j, k)];
                    
                    // Grow region using flood-fill
                    while let Some((ci, cj, ck)) = queue.pop() {
                        if visited[[ci, cj, ck]] {
                            continue;
                        }
                        
                        visited[[ci, cj, ck]] = true;
                        region_points.push((ci, cj, ck));
                        
                        // Check neighbors using precomputed offsets
                        const NEIGHBOR_OFFSETS: [(i32, i32, i32); 26] = [
                            (-1, -1, -1), (-1, -1,  0), (-1, -1,  1),
                            (-1,  0, -1), (-1,  0,  0), (-1,  0,  1),
                            (-1,  1, -1), (-1,  1,  0), (-1,  1,  1),
                            ( 0, -1, -1), ( 0, -1,  0), ( 0, -1,  1),
                            ( 0,  0, -1),               ( 0,  0,  1),
                            ( 0,  1, -1), ( 0,  1,  0), ( 0,  1,  1),
                            ( 1, -1, -1), ( 1, -1,  0), ( 1, -1,  1),
                            ( 1,  0, -1), ( 1,  0,  0), ( 1,  0,  1),
                            ( 1,  1, -1), ( 1,  1,  0), ( 1,  1,  1),
                        ];
                        for &(di, dj, dk) in NEIGHBOR_OFFSETS.iter() {
                            let ni = ci as i32 + di;
                            let nj = cj as i32 + dj;
                            let nk = ck as i32 + dk;
                            if ni < 0 || nj < 0 || nk < 0 {
                                continue;
                            }
                            let ni = ni as usize;
                            let nj = nj as usize;
                            let nk = nk as usize;
                            if ni < grid.nx && nj < grid.ny && nk < grid.nz && !visited[[ni, nj, nk]] {
                                let neighbor_value = heterogeneity[[ni, nj, nk]];
                                let relative_diff = (neighbor_value - seed_value).abs() / seed_value.abs().max(1e-10);
                                if relative_diff < heterogeneity_threshold {
                                    queue.push((ni, nj, nk));
                                }
                            }
                        }
                    }
                    
                    // Create bounding box for region
                    if !region_points.is_empty() {
                        let min_i = region_points.iter().map(|p| p.0).min().unwrap();
                        let max_i = region_points.iter().map(|p| p.0).max().unwrap() + 1;
                        let min_j = region_points.iter().map(|p| p.1).min().unwrap();
                        let max_j = region_points.iter().map(|p| p.1).max().unwrap() + 1;
                        let min_k = region_points.iter().map(|p| p.2).min().unwrap();
                        let max_k = region_points.iter().map(|p| p.2).max().unwrap() + 1;
                        
                        // Choose domain type based on region size and heterogeneity
                        let region_size = (max_i - min_i) * (max_j - min_j) * (max_k - min_k);
                        let domain_type = if region_size < 64 {
                            DomainType::FiniteDifference // Small regions use FDTD
                        } else {
                            DomainType::Spectral // Large homogeneous regions use PSTD
                        };
                        
                        regions.push(DomainRegion {
                            start: (min_i, min_j, min_k),
                            end: (max_i, max_j, max_k),
                            domain_type,
                            buffer_zones: BufferZones::default(),
                            quality_score: 0.0,
                        });
                    }
                }
            }
        }
        
        // Configure buffer zones
        self.configure_buffer_zones(regions, grid)
    }
    
    /// Configure buffer zones and overlaps between regions
    fn configure_buffer_zones(
        &self,
        mut regions: Vec<DomainRegion>,
        grid: &Grid,
    ) -> KwaversResult<Vec<DomainRegion>> {
        // Compute optimal buffer zone sizes based on wave physics
        let wavelength = grid.dx.max(grid.dy).max(grid.dz) * 10.0; // Estimate typical wavelength
        let min_buffer = (wavelength / grid.dx).ceil() as usize;
        let optimal_buffer = (min_buffer * 2).max(4); // At least 4 cells
        
        // For each region, optimize its buffer zones
        for i in 0..regions.len() {
            let region = &regions[i];
            
            // Find neighboring regions
            let mut neighbors = Vec::new();
            for j in 0..regions.len() {
                if i != j {
                    let other = &regions[j];
                    if self.regions_are_adjacent(region, other) {
                        neighbors.push(j);
                    }
                }
            }
            
            // Set buffer sizes based on domain type transitions
            let mut buffer_widths = [optimal_buffer; 6]; // [x_min, x_max, y_min, y_max, z_min, z_max]
            
            // Adjust buffer sizes based on neighbors
            for &j in &neighbors {
                let neighbor = &regions[j];
                
                // Larger buffers for spectral-FDTD interfaces
                if region.domain_type != neighbor.domain_type {
                    let interface_buffer = optimal_buffer * 2;
                    
                    // Determine which faces are adjacent
                    if region.end.0 == neighbor.start.0 {
                        buffer_widths[1] = buffer_widths[1].max(interface_buffer); // x_max
                    }
                    if region.start.0 == neighbor.end.0 {
                        buffer_widths[0] = buffer_widths[0].max(interface_buffer); // x_min
                    }
                    if region.end.1 == neighbor.start.1 {
                        buffer_widths[3] = buffer_widths[3].max(interface_buffer); // y_max
                    }
                    if region.start.1 == neighbor.end.1 {
                        buffer_widths[2] = buffer_widths[2].max(interface_buffer); // y_min
                    }
                    if region.end.2 == neighbor.start.2 {
                        buffer_widths[5] = buffer_widths[5].max(interface_buffer); // z_max
                    }
                    if region.start.2 == neighbor.end.2 {
                        buffer_widths[4] = buffer_widths[4].max(interface_buffer); // z_min
                    }
                }
            }
            
            // Reduce buffer sizes at domain boundaries
            if region.start.0 == 0 {
                buffer_widths[0] = min_buffer; // x_min
            }
            if region.end.0 == grid.nx {
                buffer_widths[1] = min_buffer; // x_max
            }
            if region.start.1 == 0 {
                buffer_widths[2] = min_buffer; // y_min
            }
            if region.end.1 == grid.ny {
                buffer_widths[3] = min_buffer; // y_max
            }
            if region.start.2 == 0 {
                buffer_widths[4] = min_buffer; // z_min
            }
            if region.end.2 == grid.nz {
                buffer_widths[5] = min_buffer; // z_max
            }
            
            // Compute overlap regions with neighbors
            let mut overlaps = Vec::new();
            for &j in &neighbors {
                let neighbor = &regions[j];
                overlaps.push(OverlapRegion {
                    start: (
                        region.start.0.max(neighbor.start.0),
                        region.start.1.max(neighbor.start.1),
                        region.start.2.max(neighbor.start.2),
                    ),
                    end: (
                        region.end.0.min(neighbor.end.0),
                        region.end.1.min(neighbor.end.1),
                        region.end.2.min(neighbor.end.2),
                    ),
                    adjacent_type: neighbor.domain_type,
                    weight: 0.5, // Equal weight for now
                });
            }
            
            regions[i].buffer_zones = BufferZones {
                widths: buffer_widths,
                overlaps,
            };
        }
        
        Ok(regions)
    }
    
    /// Check if two regions are adjacent
    fn regions_are_adjacent(&self, region1: &DomainRegion, region2: &DomainRegion) -> bool {
        // Check if regions share a face
        let x_adjacent = (region1.end.0 == region2.start.0 || region1.start.0 == region2.end.0) &&
                        !(region1.end.1 <= region2.start.1 || region1.start.1 >= region2.end.1) &&
                        !(region1.end.2 <= region2.start.2 || region1.start.2 >= region2.end.2);
                        
        let y_adjacent = (region1.end.1 == region2.start.1 || region1.start.1 == region2.end.1) &&
                        !(region1.end.0 <= region2.start.0 || region1.start.0 >= region2.end.0) &&
                        !(region1.end.2 <= region2.start.2 || region1.start.2 >= region2.end.2);
                        
        let z_adjacent = (region1.end.2 == region2.start.2 || region1.start.2 == region2.end.2) &&
                        !(region1.end.0 <= region2.start.0 || region1.start.0 >= region2.end.0) &&
                        !(region1.end.1 <= region2.start.1 || region1.start.1 >= region2.end.1);
                        
        x_adjacent || y_adjacent || z_adjacent
    }
    
    /// Merge adjacent regions of the same type
    fn merge_adjacent_regions(&self, regions: Vec<DomainRegion>) -> KwaversResult<Vec<DomainRegion>> {
        let mut merged = Vec::new();
        let mut used = vec![false; regions.len()];
        
        for i in 0..regions.len() {
            if used[i] {
                continue;
            }
            
            let mut current = regions[i].clone();
            used[i] = true;
            
            // Try to merge with adjacent regions of the same type
            let mut changed = true;
            while changed {
                changed = false;
                for j in 0..regions.len() {
                    if used[j] || regions[j].domain_type != current.domain_type {
                        continue;
                    }
                    
                    // Check if regions can be merged (adjacent and form a rectangle)
                    if self.can_merge(&current, &regions[j]) {
                        current = self.merge_two_regions(&current, &regions[j]);
                        used[j] = true;
                        changed = true;
                    }
                }
            }
            
            merged.push(current);
        }
        
        Ok(merged)
    }
    
    /// Check if two regions can be merged
    fn can_merge(&self, r1: &DomainRegion, r2: &DomainRegion) -> bool {
        // Regions must be adjacent and of the same type
        if r1.domain_type != r2.domain_type || !self.regions_are_adjacent(r1, r2) {
            return false;
        }
        
        // Check if they form a rectangular region when merged
        let x_aligned = (r1.start.0 == r2.start.0 && r1.end.0 == r2.end.0) ||
                       (r1.start.1 == r2.start.1 && r1.end.1 == r2.end.1 && 
                        r1.start.2 == r2.start.2 && r1.end.2 == r2.end.2);
                        
        let y_aligned = (r1.start.1 == r2.start.1 && r1.end.1 == r2.end.1) ||
                       (r1.start.0 == r2.start.0 && r1.end.0 == r2.end.0 && 
                        r1.start.2 == r2.start.2 && r1.end.2 == r2.end.2);
                        
        let z_aligned = (r1.start.2 == r2.start.2 && r1.end.2 == r2.end.2) ||
                       (r1.start.0 == r2.start.0 && r1.end.0 == r2.end.0 && 
                        r1.start.1 == r2.start.1 && r1.end.1 == r2.end.1);
                        
        x_aligned || y_aligned || z_aligned
    }
    
    /// Merge two regions into one
    fn merge_two_regions(&self, r1: &DomainRegion, r2: &DomainRegion) -> DomainRegion {
        DomainRegion {
            start: (
                r1.start.0.min(r2.start.0),
                r1.start.1.min(r2.start.1),
                r1.start.2.min(r2.start.2),
            ),
            end: (
                r1.end.0.max(r2.end.0),
                r1.end.1.max(r2.end.1),
                r1.end.2.max(r2.end.2),
            ),
            domain_type: r1.domain_type,
            buffer_zones: BufferZones::default(),
            quality_score: (r1.quality_score + r2.quality_score) / 2.0,
        }
    }
    
    /// Log summary of decomposition results
    fn log_decomposition_summary(&self, regions: &[DomainRegion]) {
        let mut pstd_count = 0;
        let mut fdtd_count = 0;
        let mut hybrid_count = 0;
        
        for region in regions {
            match region.domain_type {
                DomainType::Spectral => pstd_count += 1,
                DomainType::FiniteDifference => fdtd_count += 1,
                DomainType::Hybrid => hybrid_count += 1,
            }
        }
        
        info!("Domain decomposition summary:");
        info!("  PSTD regions: {}", pstd_count);
        info!("  FDTD regions: {}", fdtd_count);
        info!("  Hybrid regions: {}", hybrid_count);
        info!("  Total regions: {}", regions.len());
    }

    /// Set up neighbor relationships between regions
    fn setup_neighbors(&self, regions: &mut Vec<DomainRegion>) {
        let n = regions.len();
        
        for i in 0..n {
            for j in i+1..n {
                // Check if regions are adjacent
                let (r1_start, r1_end, r2_start, r2_end) = {
                    let r1 = &regions[i];
                    let r2 = &regions[j];
                    (r1.start, r1.end, r2.start, r2.end)
                };
                
                let adjacent = 
                    // Check x-adjacency
                    (r1_end.0 == r2_start.0 || r2_end.0 == r1_start.0) &&
                    !(r1_end.1 <= r2_start.1 || r2_end.1 <= r1_start.1) &&
                    !(r1_end.2 <= r2_start.2 || r2_end.2 <= r1_start.2) ||
                    // Check y-adjacency
                    (r1_end.1 == r2_start.1 || r2_end.1 == r1_start.1) &&
                    !(r1_end.0 <= r2_start.0 || r2_end.0 <= r1_start.0) &&
                    !(r1_end.2 <= r2_start.2 || r2_end.2 <= r1_start.2) ||
                    // Check z-adjacency
                    (r1_end.2 == r2_start.2 || r2_end.2 == r1_start.2) &&
                    !(r1_end.0 <= r2_start.0 || r2_end.0 <= r1_start.0) &&
                    !(r1_end.1 <= r2_start.1 || r2_end.1 <= r1_start.1);
                
                if adjacent {
                    // Set up buffer zones
                    let overlap = (self.analysis_params.overlap_factor * self.min_domain_size.0 as f64) as usize;
                    
                    // Update buffer zones for both regions
                    // widths array: [nx-, nx+, ny-, ny+, nz-, nz+]
                    if r1_end.0 == r2_start.0 {
                        regions[i].buffer_zones.widths[1] = overlap; // nx+
                        regions[j].buffer_zones.widths[0] = overlap; // nx-
                    } else if r2_end.0 == r1_start.0 {
                        regions[j].buffer_zones.widths[1] = overlap; // nx+
                        regions[i].buffer_zones.widths[0] = overlap; // nx-
                    }
                    
                    if r1_end.1 == r2_start.1 {
                        regions[i].buffer_zones.widths[3] = overlap; // ny+
                        regions[j].buffer_zones.widths[2] = overlap; // ny-
                    } else if r2_end.1 == r1_start.1 {
                        regions[j].buffer_zones.widths[3] = overlap; // ny+
                        regions[i].buffer_zones.widths[2] = overlap; // ny-
                    }
                    
                    if r1_end.2 == r2_start.2 {
                        regions[i].buffer_zones.widths[5] = overlap; // nz+
                        regions[j].buffer_zones.widths[4] = overlap; // nz-
                    } else if r2_end.2 == r1_start.2 {
                        regions[j].buffer_zones.widths[5] = overlap; // nz+
                        regions[i].buffer_zones.widths[4] = overlap; // nz-
                    }
                }
            }
        }
    }
}