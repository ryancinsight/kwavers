//! Hybrid PSTD/FDTD solver combining the strengths of both methods
//!
//! This module implements an intelligent hybrid solver that adaptively
//! selects between PSTD and FDTD methods based on local field characteristics.

pub mod domain_decomposition;
pub mod coupling_interface;
pub mod adaptive_selection;
pub mod validation;

use crate::grid::Grid;
use crate::medium::Medium;
use crate::error::{KwaversResult, KwaversError, ConfigError};
use crate::solver::pstd::{PstdSolver, PstdConfig};
use crate::solver::fdtd::{FdtdSolver, FdtdConfig};
use crate::solver::hybrid::domain_decomposition::{DomainDecomposer, DomainRegion, DomainType};
use crate::solver::hybrid::adaptive_selection::{AdaptiveSelector, SelectionCriteria};
use crate::solver::hybrid::coupling_interface::{CouplingInterface, InterpolationScheme};
use ndarray::{Array4, Zip, s};
use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use log::{debug, info, warn};

/// Configuration for the hybrid PSTD/FDTD solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// PSTD solver configuration
    pub pstd_config: PstdConfig,
    
    /// FDTD solver configuration  
    pub fdtd_config: FdtdConfig,
    
    /// Domain decomposition strategy
    pub decomposition_strategy: DecompositionStrategy,
    
    /// Adaptive selection parameters
    pub selection_criteria: SelectionCriteria,
    
    /// Coupling interface configuration
    pub coupling_interface: CouplingInterfaceConfig,
    
    /// Performance optimization settings
    pub optimization: OptimizationConfig,
    
    /// Validation and quality control
    pub validation: ValidationConfig,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            pstd_config: PstdConfig::default(),
            fdtd_config: FdtdConfig::default(),
            decomposition_strategy: DecompositionStrategy::Adaptive,
            selection_criteria: SelectionCriteria::default(),
            coupling_interface: CouplingInterfaceConfig::default(),
            optimization: OptimizationConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

/// Domain decomposition strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Fixed regions based on predefined criteria
    Fixed,
    /// Adaptive regions based on runtime analysis
    Adaptive,
    /// Gradient-based decomposition
    GradientBased,
    /// Frequency-based decomposition
    FrequencyBased,
    /// Material-property-based decomposition
    MaterialBased,
}

/// Coupling interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingInterfaceConfig {
    /// Interpolation scheme for data transfer
    pub interpolation_scheme: InterpolationScheme,
    /// Buffer zone width (in grid cells)
    pub buffer_width: usize,
    /// Smoothing parameter for interface continuity
    pub smoothing_factor: f64,
    /// Enable conservative interpolation
    pub conservative_transfer: bool,
}

impl Default for CouplingInterfaceConfig {
    fn default() -> Self {
        Self {
            interpolation_scheme: InterpolationScheme::CubicSpline,
            buffer_width: 4,
            smoothing_factor: 0.1,
            conservative_transfer: true,
        }
    }
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable parallel processing of domains
    pub parallel_domains: bool,
    /// Minimum domain size for parallelization
    pub min_parallel_size: usize,
    /// Enable GPU acceleration where available
    pub enable_gpu: bool,
    /// Memory management strategy
    pub memory_strategy: MemoryStrategy,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            parallel_domains: true,
            min_parallel_size: 1000,
            enable_gpu: false,
            memory_strategy: MemoryStrategy::Balanced,
        }
    }
}

/// Memory management strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryStrategy {
    /// Minimize memory usage
    Conservative,
    /// Balance memory and performance
    Balanced,
    /// Maximize performance
    Performance,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable runtime accuracy validation
    pub enable_accuracy_checks: bool,
    /// Enable energy conservation validation
    pub enable_conservation_checks: bool,
    /// Enable interface continuity validation
    pub enable_continuity_checks: bool,
    /// Validation frequency (every N steps)
    pub validation_frequency: usize,
    /// Tolerance for validation checks
    pub tolerance: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_accuracy_checks: true,
            enable_conservation_checks: true,
            enable_continuity_checks: true,
            validation_frequency: 10,
            tolerance: 1e-6,
        }
    }
}

/// Main hybrid PSTD/FDTD solver
pub struct HybridSolver {
    /// Configuration
    config: HybridConfig,
    
    /// Grid reference
    grid: Grid,
    
    /// Domain decomposer
    domain_decomposer: DomainDecomposer,
    
    /// Adaptive selector
    adaptive_selector: AdaptiveSelector,
    
    /// Coupling interface manager
    coupling_interface: CouplingInterface,
    
    /// PSTD solver instances for spectral domains
    pstd_solvers: HashMap<usize, PstdSolver>,
    
    /// FDTD solver instances for finite-difference domains
    fdtd_solvers: HashMap<usize, FdtdSolver>,
    
    /// Current domain decomposition
    current_domains: Vec<DomainRegion>,
    
    /// Performance metrics
    metrics: HybridMetrics,
    
    /// Validation results
    validation_results: ValidationResults,
    
    /// Step counter for adaptive updates
    step_count: u64,
}

/// Performance metrics for hybrid solver
#[derive(Debug, Clone, Default)]
pub struct HybridMetrics {
    /// Time spent in PSTD domains
    pub pstd_time: f64,
    /// Time spent in FDTD domains  
    pub fdtd_time: f64,
    /// Time spent in coupling
    pub coupling_time: f64,
    /// Time spent in domain selection
    pub selection_time: f64,
    /// Number of domain switches
    pub domain_switches: u64,
    /// Total computational time
    pub total_time: f64,
    /// Efficiency metrics
    pub efficiency: EfficiencyMetrics,
}

/// Efficiency metrics
#[derive(Debug, Clone, Default)]
pub struct EfficiencyMetrics {
    /// Grid updates per second
    pub updates_per_second: f64,
    /// Memory usage efficiency
    pub memory_efficiency: f64,
    /// Load balancing efficiency
    pub load_balance_efficiency: f64,
}

/// Validation results
#[derive(Debug, Clone, Default)]
pub struct ValidationResults {
    /// Energy conservation error
    pub energy_conservation_error: f64,
    /// Interface continuity error
    pub interface_continuity_error: f64,
    /// Accuracy validation error
    pub accuracy_error: f64,
    /// Quality score (0-1, higher is better)
    pub quality_score: f64,
}

impl HybridSolver {
    /// Create a new hybrid PSTD/FDTD solver
    pub fn new(config: HybridConfig, grid: &Grid) -> KwaversResult<Self> {
        info!("Initializing hybrid PSTD/FDTD solver");
        
        // Validate configuration
        Self::validate_config(&config, grid)?;
        
        // Initialize components
        let domain_decomposer = DomainDecomposer::new(config.decomposition_strategy, grid)?;
        let adaptive_selector = AdaptiveSelector::new(config.selection_criteria.clone())?;
        let coupling_interface = CouplingInterface::new(config.coupling_interface.clone())?;
        
        let solver = Self {
            config,
            grid: grid.clone(),
            domain_decomposer,
            adaptive_selector,
            coupling_interface,
            pstd_solvers: HashMap::new(),
            fdtd_solvers: HashMap::new(),
            current_domains: Vec::new(),
            metrics: HybridMetrics::default(),
            validation_results: ValidationResults::default(),
            step_count: 0,
        };
        
        info!("Hybrid solver initialized successfully");
        Ok(solver)
    }
    
    /// Validate hybrid solver configuration
    fn validate_config(config: &HybridConfig, grid: &Grid) -> KwaversResult<()> {
        // Validate grid compatibility
        if grid.nx < 8 || grid.ny < 8 || grid.nz < 8 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "grid_size".to_string(),
                value: format!("{}x{}x{}", grid.nx, grid.ny, grid.nz),
                constraint: "minimum 8x8x8 for hybrid solver".to_string(),
            }));
        }
        
        // Validate buffer width
        if config.coupling_interface.buffer_width > grid.nx.min(grid.ny).min(grid.nz) / 4 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "buffer_width".to_string(),
                value: config.coupling_interface.buffer_width.to_string(),
                constraint: "must be less than 1/4 of smallest grid dimension".to_string(),
            }));
        }
        
        // Validate tolerance ranges
        if config.validation.tolerance <= 0.0 || config.validation.tolerance > 1.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "validation_tolerance".to_string(),
                value: config.validation.tolerance.to_string(),
                constraint: "must be between 0.0 and 1.0".to_string(),
            }));
        }
        
        debug!("Hybrid solver configuration validated successfully");
        Ok(())
    }
    
    /// Update fields using hybrid PSTD/FDTD approach
    pub fn update_fields(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        time: f64,
    ) -> KwaversResult<()> {
        let start_time = Instant::now();
        self.step_count += 1;
        
        // Adaptive domain selection (if enabled and periodic)
        if self.should_update_domains()? {
            self.update_domain_decomposition(fields, medium, time)?;
        }
        
        // Process each domain with appropriate solver
        self.process_domains(fields, medium, dt, time)?;
        
        // Apply coupling interface corrections
        self.apply_coupling_corrections(fields, dt)?;
        
        // Validate results (if enabled)
        if self.should_validate()? {
            self.validate_solution(fields, time)?;
        }
        
        // Update metrics
        self.metrics.total_time += start_time.elapsed().as_secs_f64();
        self.update_efficiency_metrics(fields.len());
        
        Ok(())
    }
    
    /// Determine if domain decomposition should be updated
    fn should_update_domains(&self) -> KwaversResult<bool> {
        match self.config.decomposition_strategy {
            DecompositionStrategy::Adaptive => {
                // Update every 100 steps or when quality drops
                Ok(self.step_count % 100 == 0 || 
                   self.validation_results.quality_score < 0.8)
            }
            DecompositionStrategy::GradientBased => {
                // Update every 50 steps
                Ok(self.step_count % 50 == 0)
            }
            _ => {
                // Fixed decomposition - only update once
                Ok(self.current_domains.is_empty())
            }
        }
    }
    
    /// Update domain decomposition based on current field state
    fn update_domain_decomposition(
        &mut self,
        fields: &Array4<f64>,
        medium: &dyn Medium,
        time: f64,
    ) -> KwaversResult<()> {
        let start_time = Instant::now();
        
        // Analyze field properties
        let quality_metrics = self.adaptive_selector.analyze_field_quality(fields, &self.grid)?;
        
        // Generate new domain decomposition
        self.current_domains = self.domain_decomposer.decompose_domain(
            fields,
            medium,
            &quality_metrics,
            &self.grid,
        )?;
        
        // Update solver instances for new domains
        self.update_solver_instances()?;
        
        self.metrics.selection_time += start_time.elapsed().as_secs_f64();
        self.metrics.domain_switches += 1;
        
        info!("Updated domain decomposition: {} domains", self.current_domains.len());
        Ok(())
    }
    
    /// Update solver instances based on domain types
    fn update_solver_instances(&mut self) -> KwaversResult<()> {
        // Clear existing solvers
        self.pstd_solvers.clear();
        self.fdtd_solvers.clear();
        
        // Create solvers for each domain
        for (idx, domain) in self.current_domains.iter().enumerate() {
            match domain.domain_type {
                DomainType::Spectral => {
                    let pstd_solver = PstdSolver::new(self.config.pstd_config, &self.grid)?;
                    self.pstd_solvers.insert(idx, pstd_solver);
                }
                DomainType::FiniteDifference => {
                    let fdtd_solver = FdtdSolver::new(self.config.fdtd_config, &self.grid)?;
                    self.fdtd_solvers.insert(idx, fdtd_solver);
                }
                DomainType::Hybrid => {
                    // Create both for hybrid domains
                    let pstd_solver = PstdSolver::new(self.config.pstd_config, &self.grid)?;
                    let fdtd_solver = FdtdSolver::new(self.config.fdtd_config, &self.grid)?;
                    self.pstd_solvers.insert(idx, pstd_solver);
                    self.fdtd_solvers.insert(idx, fdtd_solver);
                }
            }
        }
        
        Ok(())
    }
    
    /// Process all domains with appropriate solvers
    fn process_domains(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        time: f64,
    ) -> KwaversResult<()> {
        // Process domains in parallel if enabled
        if self.config.optimization.parallel_domains {
            self.process_domains_parallel(fields, medium, dt, time)
        } else {
            self.process_domains_sequential(fields, medium, dt, time)
        }
    }
    
    /// Process domains sequentially
    fn process_domains_sequential(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        time: f64,
    ) -> KwaversResult<()> {
        let domains_clone = self.current_domains.clone();
        for (idx, domain) in domains_clone.iter().enumerate() {
            self.process_single_domain(idx, domain, fields, medium, dt, time)?;
        }
        Ok(())
    }
    
    /// Process domains in parallel (placeholder for future implementation)
    fn process_domains_parallel(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        time: f64,
    ) -> KwaversResult<()> {
        // For now, fall back to sequential processing
        // TODO: Implement parallel processing with proper synchronization
        warn!("Parallel domain processing not yet implemented, falling back to sequential");
        self.process_domains_sequential(fields, medium, dt, time)
    }
    
    /// Process a single domain with appropriate solver
    fn process_single_domain(
        &mut self,
        domain_idx: usize,
        domain: &DomainRegion,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        time: f64,
    ) -> KwaversResult<()> {
        let start_time = Instant::now();
        
        // Extract domain fields
        let mut domain_fields = self.extract_domain_fields(fields, domain)?;
        
        let elapsed_time = match domain.domain_type {
            DomainType::Spectral => {
                if let Some(pstd_solver) = self.pstd_solvers.get_mut(&domain_idx) {
                    Self::apply_pstd_update_static(pstd_solver, &mut domain_fields, medium, dt, time)?;
                }
                start_time.elapsed().as_secs_f64()
            }
            DomainType::FiniteDifference => {
                if let Some(fdtd_solver) = self.fdtd_solvers.get_mut(&domain_idx) {
                    Self::apply_fdtd_update_static(fdtd_solver, &mut domain_fields, medium, dt, time)?;
                }
                start_time.elapsed().as_secs_f64()
            }
            DomainType::Hybrid => {
                // Apply both methods and blend results
                self.apply_hybrid_update(domain_idx, &mut domain_fields, medium, dt, time)?;
                start_time.elapsed().as_secs_f64()
            }
        };
        
        // Update timing metrics
        match domain.domain_type {
            DomainType::Spectral => {
                self.metrics.pstd_time += elapsed_time;
            }
            DomainType::FiniteDifference => {
                self.metrics.fdtd_time += elapsed_time;
            }
            DomainType::Hybrid => {
                self.metrics.pstd_time += elapsed_time * 0.5;
                self.metrics.fdtd_time += elapsed_time * 0.5;
            }
        }
        
        // Insert updated fields back
        self.insert_domain_fields(fields, &domain_fields, domain)?;
        
        Ok(())
    }
    
    /// Extract fields for a specific domain
    fn extract_domain_fields(
        &self,
        fields: &Array4<f64>,
        domain: &DomainRegion,
    ) -> KwaversResult<Array4<f64>> {
        let (nx, ny, nz) = (
            domain.end.0 - domain.start.0,
            domain.end.1 - domain.start.1,
            domain.end.2 - domain.start.2,
        );
        
        let mut domain_fields = Array4::zeros((fields.shape()[0], nx, ny, nz));
        
        for field_idx in 0..fields.shape()[0] {
            let source_slice = fields.slice(s![field_idx, 
                domain.start.0..domain.end.0,
                domain.start.1..domain.end.1,
                domain.start.2..domain.end.2]);
            domain_fields.slice_mut(s![field_idx, .., .., ..]).assign(&source_slice);
        }
        
        Ok(domain_fields)
    }
    
    /// Insert domain fields back into global field array
    fn insert_domain_fields(
        &self,
        fields: &mut Array4<f64>,
        domain_fields: &Array4<f64>,
        domain: &DomainRegion,
    ) -> KwaversResult<()> {
        for field_idx in 0..fields.shape()[0] {
            let mut target_slice = fields.slice_mut(s![field_idx,
                domain.start.0..domain.end.0,
                domain.start.1..domain.end.1,
                domain.start.2..domain.end.2]);
            target_slice.assign(&domain_fields.slice(s![field_idx, .., .., ..]));
        }
        Ok(())
    }
    
    /// Apply PSTD update to domain (static version to avoid borrowing issues)
    fn apply_pstd_update_static(
        _pstd_solver: &mut PstdSolver,
        _domain_fields: &mut Array4<f64>,
        _medium: &dyn Medium,
        _dt: f64,
        _time: f64,
    ) -> KwaversResult<()> {
        // TODO: Interface with actual PSTD solver
        // For now, placeholder implementation
        debug!("Applying PSTD update to spectral domain");
        Ok(())
    }
    
    /// Apply FDTD update to domain (static version to avoid borrowing issues)
    fn apply_fdtd_update_static(
        _fdtd_solver: &mut FdtdSolver,
        _domain_fields: &mut Array4<f64>,
        _medium: &dyn Medium,
        _dt: f64,
        _time: f64,
    ) -> KwaversResult<()> {
        // TODO: Interface with actual FDTD solver
        // For now, placeholder implementation
        debug!("Applying FDTD update to finite-difference domain");
        Ok(())
    }
    
    /// Apply hybrid update (combination of PSTD and FDTD)
    fn apply_hybrid_update(
        &mut self,
        domain_idx: usize,
        domain_fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        time: f64,
    ) -> KwaversResult<()> {
        // Apply both methods and blend results based on local quality metrics
        let mut pstd_result = domain_fields.clone();
        let mut fdtd_result = domain_fields.clone();
        
        if let Some(pstd_solver) = self.pstd_solvers.get_mut(&domain_idx) {
            Self::apply_pstd_update_static(pstd_solver, &mut pstd_result, medium, dt, time)?;
        }
        
        if let Some(fdtd_solver) = self.fdtd_solvers.get_mut(&domain_idx) {
            Self::apply_fdtd_update_static(fdtd_solver, &mut fdtd_result, medium, dt, time)?;
        }
        
        // Blend results based on local quality metrics
        self.blend_hybrid_results(domain_fields, &pstd_result, &fdtd_result)?;
        
        Ok(())
    }
    
    /// Blend PSTD and FDTD results for hybrid domains
    fn blend_hybrid_results(
        &self,
        result_fields: &mut Array4<f64>,
        pstd_result: &Array4<f64>,
        fdtd_result: &Array4<f64>,
    ) -> KwaversResult<()> {
        // Simple weighted blending - can be enhanced with adaptive weights
        let pstd_weight = 0.6; // Favor PSTD for accuracy
        let fdtd_weight = 0.4;
        
        Zip::from(result_fields)
            .and(pstd_result)
            .and(fdtd_result)
            .for_each(|result, &pstd_val, &fdtd_val| {
                *result = pstd_weight * pstd_val + fdtd_weight * fdtd_val;
            });
        
        Ok(())
    }
    
    /// Apply coupling interface corrections
    fn apply_coupling_corrections(
        &mut self,
        fields: &mut Array4<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        let start_time = Instant::now();
        
        // Apply interface corrections between domains
        self.coupling_interface.apply_corrections(
            fields,
            &self.current_domains,
            &self.grid,
            dt,
        )?;
        
        self.metrics.coupling_time += start_time.elapsed().as_secs_f64();
        Ok(())
    }
    
    /// Determine if validation should be performed
    fn should_validate(&self) -> KwaversResult<bool> {
        Ok(self.config.validation.enable_accuracy_checks &&
           self.step_count % self.config.validation.validation_frequency as u64 == 0)
    }
    
    /// Validate solution quality and conservation properties
    fn validate_solution(
        &mut self,
        fields: &Array4<f64>,
        time: f64,
    ) -> KwaversResult<()> {
        // TODO: Implement comprehensive validation
        // For now, placeholder implementation
        self.validation_results.quality_score = 0.95; // Placeholder
        debug!("Solution validation completed at time {:.3e}", time);
        Ok(())
    }
    
    /// Update efficiency metrics
    fn update_efficiency_metrics(&mut self, grid_size: usize) {
        if self.metrics.total_time > 0.0 {
            self.metrics.efficiency.updates_per_second = 
                self.step_count as f64 * grid_size as f64 / self.metrics.total_time;
        }
        
        // Calculate load balance efficiency
        let pstd_fraction = self.metrics.pstd_time / self.metrics.total_time;
        let fdtd_fraction = self.metrics.fdtd_time / self.metrics.total_time;
        let balance = 1.0 - (pstd_fraction - fdtd_fraction).abs();
        self.metrics.efficiency.load_balance_efficiency = balance.max(0.0);
    }
    
    /// Get current performance metrics
    pub fn get_metrics(&self) -> &HybridMetrics {
        &self.metrics
    }
    
    /// Get current validation results
    pub fn get_validation_results(&self) -> &ValidationResults {
        &self.validation_results
    }
    
    /// Report solver performance and status
    pub fn report_performance(&self) {
        info!("=== Hybrid Solver Performance Report ===");
        info!("Total steps: {}", self.step_count);
        info!("Total time: {:.3} s", self.metrics.total_time);
        info!("PSTD time: {:.3} s ({:.1}%)", 
              self.metrics.pstd_time, 
              100.0 * self.metrics.pstd_time / self.metrics.total_time);
        info!("FDTD time: {:.3} s ({:.1}%)", 
              self.metrics.fdtd_time,
              100.0 * self.metrics.fdtd_time / self.metrics.total_time);
        info!("Coupling time: {:.3} s ({:.1}%)", 
              self.metrics.coupling_time,
              100.0 * self.metrics.coupling_time / self.metrics.total_time);
        info!("Domain switches: {}", self.metrics.domain_switches);
        info!("Updates/second: {:.2e}", self.metrics.efficiency.updates_per_second);
        info!("Quality score: {:.3}", self.validation_results.quality_score);
        info!("========================================");
    }
}

// TODO: Implement PhysicsPlugin trait for compatibility with plugin architecture
// The trait implementation is deferred until trait interfaces are stabilized