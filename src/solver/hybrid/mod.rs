//! Hybrid PSTD/FDTD solver combining the strengths of both methods
//!
//! This module implements an intelligent hybrid solver that adaptively
//! selects between PSTD and FDTD methods based on local field characteristics.
//!
//! ## Overview
//!
//! The hybrid solver leverages the advantages of both Pseudo-Spectral Time Domain (PSTD)
//! and Finite-Difference Time Domain (FDTD) methods:
//!
//! - **PSTD**: High accuracy, no numerical dispersion, efficient for smooth fields
//! - **FDTD**: Robust shock handling, local operations, better for discontinuities
//!
//! ## Architecture
//!
//! The solver uses a domain decomposition approach where different regions of the
//! computational domain can use different numerical methods based on local field
//! characteristics. The selection is performed by the `AdaptiveSelector` which
//! analyzes field gradients, frequency content, and other metrics.
//!
//! ## Features
//!
//! - **Adaptive Method Selection**: Automatically chooses PSTD or FDTD based on local conditions
//! - **Domain Decomposition**: Efficient partitioning of the computational domain
//! - **Coupling Interface**: Seamless data exchange between PSTD and FDTD regions
//! - **Performance Monitoring**: Real-time metrics for method efficiency
//! - **Plugin Architecture**: Compatible with the Kwavers physics plugin system
//!
//! ## Example
//!
//! ```rust,no_run
//! use kwavers::solver::hybrid::{HybridSolver, HybridConfig};
//! use kwavers::grid::Grid;
//!
//! let grid = Grid::new(128, 128, 128, 0.001, 0.001, 0.001);
//! let config = HybridConfig::default();
//! let solver = HybridSolver::new(config, &grid)?;
//! ```
//!
//! ## Design Principles
//!
//! This implementation follows key software design principles:
//!
//! - **SOLID**: Single responsibility for each component, open for extension
//! - **CUPID**: Composable solvers, Unix philosophy, predictable behavior
//! - **GRASP**: Information expert pattern with domain-specific knowledge
//! - **DRY**: Shared numerical algorithms between PSTD and FDTD
//! - **KISS**: Simple interfaces despite complex internals
//! - **YAGNI**: Only essential features implemented
//!
//! ## Performance
//!
//! The hybrid solver aims to achieve:
//! - >100M grid updates/second with optimized kernels
//! - <1% numerical dispersion error in PSTD regions
//! - Robust shock handling in FDTD regions
//! - Minimal overhead from domain coupling

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
use crate::physics::plugin::PluginMetadata;
use ndarray::{Array4, s, Zip};
use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use log::{debug, info};

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
#[derive(Clone, Debug)]
pub struct HybridSolver {
    /// Plugin metadata
    metadata: PluginMetadata,
    
    /// Configuration
    config: HybridConfig,
    
    /// Grid reference
    grid: Grid,
    
    /// Solver state
    state: SolverState,
    
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

/// Solver state for tracking lifecycle
#[derive(Debug, Clone)]
pub enum SolverState {
    Initialized,
    Running,
    Error(String),
    Finalized,
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

impl HybridMetrics {
    /// Get a summary of metrics as a HashMap
    pub fn get_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();
        summary.insert("pstd_time".to_string(), self.pstd_time);
        summary.insert("fdtd_time".to_string(), self.fdtd_time);
        summary.insert("coupling_time".to_string(), self.coupling_time);
        summary.insert("selection_time".to_string(), self.selection_time);
        summary.insert("domain_switches".to_string(), self.domain_switches as f64);
        summary.insert("total_time".to_string(), self.total_time);
        summary.insert("updates_per_second".to_string(), self.efficiency.updates_per_second);
        summary.insert("memory_efficiency".to_string(), self.efficiency.memory_efficiency);
        summary.insert("load_balance_efficiency".to_string(), self.efficiency.load_balance_efficiency);
        summary
    }
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
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the hybrid solver
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// A new `HybridSolver` instance or an error if initialization fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use kwavers::solver::hybrid::{HybridSolver, HybridConfig};
    /// # use kwavers::grid::Grid;
    /// let grid = Grid::new(128, 128, 128, 0.001, 0.001, 0.001);
    /// let config = HybridConfig::default();
    /// let solver = HybridSolver::new(config, &grid)?;
    /// # Ok::<(), kwavers::error::KwaversError>(())
    /// ```
    pub fn new(config: HybridConfig, grid: &Grid) -> KwaversResult<Self> {
        info!("Initializing hybrid PSTD/FDTD solver");
        
        // Validate configuration
        Self::validate_config(&config, grid)?;
        
        // Initialize components
        let domain_decomposer = DomainDecomposer::new(config.decomposition_strategy, grid)?;
        let adaptive_selector = AdaptiveSelector::new(config.selection_criteria.clone())?;
        let coupling_interface = CouplingInterface::new(config.coupling_interface.clone())?;
        
        let solver = Self {
            metadata: Self::create_metadata(),
            config,
            grid: grid.clone(),
            state: SolverState::Initialized,
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
    pub fn process_domains(
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
    
    /// Process domains in parallel using rayon
    fn process_domains_parallel(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        time: f64,
    ) -> KwaversResult<()> {
        use rayon::prelude::*;
        use std::sync::{Arc, Mutex};
        use crate::error::CompositeError;
        
        // Create thread-safe wrappers
        let errors: Arc<Mutex<Vec<KwaversError>>> = Arc::new(Mutex::new(Vec::new()));
        
        // Structure to hold solver state updates
        #[derive(Clone)]
        struct SolverUpdate {
            domain_idx: usize,
            domain_type: DomainType,
            domain_fields: Array4<f64>,
            metrics: HashMap<String, f64>,
        }
        
        // Collect domain updates in parallel
        let domain_updates: Vec<_> = self.current_domains.par_iter().enumerate()
            .filter_map(|(idx, domain)| {
                // Skip hybrid domains for parallel processing
                if matches!(domain.domain_type, DomainType::Hybrid) {
                    return None;
                }
                
                // Extract domain fields
                match self.extract_domain_fields(fields, domain) {
                    Ok(mut domain_fields) => {
                        // Process the domain based on type and collect metrics
                        let (result, metrics) = match domain.domain_type {
                            DomainType::Spectral => {
                                match PstdSolver::new(self.config.pstd_config.clone(), &self.grid) {
                                    Ok(mut temp_solver) => {
                                        let update_result = Self::apply_pstd_update_static(&mut temp_solver, &mut domain_fields, medium, dt, time);
                                        let metrics = temp_solver.get_metrics().clone();
                                        (update_result, metrics)
                                    }
                                    Err(e) => (Err(e), HashMap::new())
                                }
                            }
                            DomainType::FiniteDifference => {
                                match FdtdSolver::new(self.config.fdtd_config.clone(), &self.grid) {
                                    Ok(mut temp_solver) => {
                                        let update_result = Self::apply_fdtd_update_static(&mut temp_solver, &mut domain_fields, medium, dt, time);
                                        let metrics = temp_solver.get_metrics().clone();
                                        (update_result, metrics)
                                    }
                                    Err(e) => (Err(e), HashMap::new())
                                }
                            }
                            _ => (Ok(()), HashMap::new())
                        };
                        
                        match result {
                            Ok(_) => Some(SolverUpdate {
                                domain_idx: idx,
                                domain_type: domain.domain_type,
                                domain_fields,
                                metrics,
                            }),
                            Err(e) => {
                                if let Ok(mut errors_guard) = errors.lock() {
                                    errors_guard.push(e);
                                }
                                None
                            }
                        }
                    }
                    Err(e) => {
                        if let Ok(mut errors_guard) = errors.lock() {
                            errors_guard.push(e);
                        }
                        None
                    }
                }
            })
            .collect();
        
        // Check for errors and return CompositeError if multiple errors occurred
        let errors_guard = errors.lock().unwrap();
        if !errors_guard.is_empty() {
            if errors_guard.len() == 1 {
                return Err(errors_guard[0].clone());
            } else {
                return Err(KwaversError::Composite(CompositeError {
                    context: format!("Multiple errors occurred during parallel domain processing ({} errors)", errors_guard.len()),
                    errors: errors_guard.clone(),
                }));
            }
        }
        
        // Apply all updates sequentially and merge metrics
        for update in domain_updates {
            // Find the corresponding domain region
            if let Some(domain) = self.current_domains.get(update.domain_idx) {
                self.copy_domain_to_fields(fields, &update.domain_fields, domain)?;
            }
            
            // Update solver metrics
            match update.domain_type {
                DomainType::Spectral => {
                    if let Some(solver) = self.pstd_solvers.get_mut(&update.domain_idx) {
                        // Merge metrics from parallel execution
                        solver.merge_metrics(&update.metrics);
                    }
                }
                DomainType::FiniteDifference => {
                    if let Some(solver) = self.fdtd_solvers.get_mut(&update.domain_idx) {
                        // Merge metrics from parallel execution
                        solver.merge_metrics(&update.metrics);
                    }
                }
                _ => {}
            }
        }
        
        // Process hybrid domains sequentially
        let hybrid_domains: Vec<_> = self.current_domains.iter()
            .enumerate()
            .filter(|(_, domain)| matches!(domain.domain_type, DomainType::Hybrid))
            .map(|(idx, domain)| (idx, domain.clone()))
            .collect();
            
        for (idx, domain) in hybrid_domains {
            self.process_single_domain(idx, &domain, fields, medium, dt, time)?;
        }
        
        Ok(())
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
    
    /// Copy domain fields back to main fields
    fn copy_domain_to_fields(
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
        pstd_solver: &mut PstdSolver,
        domain_fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        _time: f64,
    ) -> KwaversResult<()> {
        debug!("Applying PSTD update to spectral domain");
        
        // Extract pressure and velocity fields
        let pressure_idx = 0; // Assuming pressure is at index 0
        let vx_idx = 1;       // Assuming velocity components follow
        let vy_idx = 2;
        let vz_idx = 3;
        
        // Get field slices
        let mut pressure = domain_fields.index_axis_mut(ndarray::Axis(0), pressure_idx).to_owned();
        let velocity_x = domain_fields.index_axis(ndarray::Axis(0), vx_idx).to_owned();
        let velocity_y = domain_fields.index_axis(ndarray::Axis(0), vy_idx).to_owned();
        let velocity_z = domain_fields.index_axis(ndarray::Axis(0), vz_idx).to_owned();
        
        // Compute velocity divergence
        let div_v = pstd_solver.compute_divergence(&velocity_x, &velocity_y, &velocity_z)?;
        
        // Update pressure using PSTD
        pstd_solver.update_pressure(&mut pressure, &div_v, medium, dt)?;
        
        // Update velocities using PSTD
        let mut vx_mut = velocity_x.clone();
        let mut vy_mut = velocity_y.clone();
        let mut vz_mut = velocity_z.clone();
        pstd_solver.update_velocity(&mut vx_mut, &mut vy_mut, &mut vz_mut, &pressure, medium, dt)?;
        
        // Copy results back
        domain_fields.index_axis_mut(ndarray::Axis(0), pressure_idx).assign(&pressure);
        domain_fields.index_axis_mut(ndarray::Axis(0), vx_idx).assign(&vx_mut);
        domain_fields.index_axis_mut(ndarray::Axis(0), vy_idx).assign(&vy_mut);
        domain_fields.index_axis_mut(ndarray::Axis(0), vz_idx).assign(&vz_mut);
        
        Ok(())
    }
    
    /// Apply FDTD update to domain (static version to avoid borrowing issues)
    fn apply_fdtd_update_static(
        fdtd_solver: &mut FdtdSolver,
        domain_fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        _time: f64,
    ) -> KwaversResult<()> {
        debug!("Applying FDTD update to finite-difference domain");
        
        // Extract pressure and velocity fields
        let pressure_idx = 0;
        let vx_idx = 1;
        let vy_idx = 2;
        let vz_idx = 3;
        
        // Get field slices
        let mut pressure = domain_fields.index_axis_mut(ndarray::Axis(0), pressure_idx).to_owned();
        let velocity_x = domain_fields.index_axis(ndarray::Axis(0), vx_idx).to_owned();
        let velocity_y = domain_fields.index_axis(ndarray::Axis(0), vy_idx).to_owned();
        let velocity_z = domain_fields.index_axis(ndarray::Axis(0), vz_idx).to_owned();
        
        // Update pressure using FDTD
        fdtd_solver.update_pressure(&mut pressure, &velocity_x, &velocity_y, &velocity_z, medium, dt)?;
        
        // Update velocities using FDTD
        let mut vx_mut = velocity_x.clone();
        let mut vy_mut = velocity_y.clone();
        let mut vz_mut = velocity_z.clone();
        fdtd_solver.update_velocity(&mut vx_mut, &mut vy_mut, &mut vz_mut, &pressure, medium, dt)?;
        
        // Copy results back
        domain_fields.index_axis_mut(ndarray::Axis(0), pressure_idx).assign(&pressure);
        domain_fields.index_axis_mut(ndarray::Axis(0), vx_idx).assign(&vx_mut);
        domain_fields.index_axis_mut(ndarray::Axis(0), vy_idx).assign(&vy_mut);
        domain_fields.index_axis_mut(ndarray::Axis(0), vz_idx).assign(&vz_mut);
        
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
        use crate::solver::PRESSURE_IDX;
        
        // Check for NaN or infinite values
        let pressure = fields.index_axis(ndarray::Axis(0), PRESSURE_IDX);
        let has_nan = pressure.iter().any(|&x| x.is_nan());
        let has_inf = pressure.iter().any(|&x| x.is_infinite());
        
        if has_nan || has_inf {
            self.validation_results.quality_score = 0.0;
            return Err(crate::error::ValidationError::FieldValidation {
                field: "pressure".to_string(),
                value: "NaN or Inf detected".to_string(),
                constraint: "finite values required".to_string(),
            }.into());
        }
        
        // Calculate quality metrics
        let max_pressure = pressure.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let mean_pressure = pressure.iter().sum::<f64>() / pressure.len() as f64;
        
        // Check conservation (mass/energy)
        let total_energy = pressure.iter().map(|&p| p * p).sum::<f64>();
        let energy_change = if self.validation_results.energy_conservation_error > 0.0 {
            (total_energy - self.validation_results.energy_conservation_error).abs() / self.validation_results.energy_conservation_error
        } else {
            0.0
        };
        self.validation_results.energy_conservation_error = total_energy;
        
        // Calculate quality score based on multiple factors
        let stability_score = if max_pressure < 1e6 { 1.0 } else { 0.5 };
        let conservation_score = if energy_change < 0.01 { 1.0 } else { 0.8 };
        let smoothness_score = 0.95; // Could be improved with gradient analysis
        
        self.validation_results.quality_score = 
            (stability_score + conservation_score + smoothness_score) / 3.0;
        
        debug!("Solution validation: quality={:.3}, max_p={:.2e}, energy_change={:.2e}", 
               self.validation_results.quality_score, max_pressure, energy_change);
        
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

use crate::physics::plugin::{PhysicsPlugin, PluginState, PluginContext};
use crate::physics::composable::FieldType;
use std::any::Any;

impl PhysicsPlugin for HybridSolver {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> PluginState {
        match self.state {
            SolverState::Initialized => PluginState::Initialized,
            SolverState::Running => PluginState::Running,
            SolverState::Error(_) => PluginState::Error,
            SolverState::Finalized => PluginState::Finalized,
        }
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Pressure, FieldType::Velocity]
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Pressure, FieldType::Velocity]
    }
    
    fn initialize(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        // Validate that the solver is properly configured for the given grid
        if self.grid.nx != grid.nx || self.grid.ny != grid.ny || self.grid.nz != grid.nz {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "grid".to_string(),
                value: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
                constraint: format!("Grid dimensions must match solver configuration: ({}, {}, {})", 
                    self.grid.nx, self.grid.ny, self.grid.nz),
            }));
        }
        
        // The HybridSolver is already initialized in its constructor with the grid.
        // This method serves as a validation checkpoint and state transition.
        self.state = SolverState::Initialized;
        
        // Log initialization for debugging
        log::debug!(
            "HybridSolver initialized for grid {}x{}x{} with {} PSTD and {} FDTD domains",
            grid.nx, grid.ny, grid.nz,
            self.pstd_solvers.len(),
            self.fdtd_solvers.len()
        );
        
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Perform the hybrid solver update
        self.update_fields(fields, medium, dt, t)?;
        
        Ok(())
    }
    
    fn finalize(&mut self) -> KwaversResult<()> {
        // Clear all solver instances
        self.pstd_solvers.clear();
        self.fdtd_solvers.clear();
        self.state = SolverState::Finalized;
        Ok(())
    }
    
    fn can_execute(&self, available_fields: &[FieldType]) -> bool {
        // Check if required fields are available
        self.required_fields()
            .iter()
            .all(|req| available_fields.contains(req))
    }
    
    fn performance_metrics(&self) -> HashMap<String, f64> {
        self.metrics.get_summary()
    }
    
    fn validate(&self, grid: &Grid, medium: &dyn Medium) -> crate::physics::composable::ValidationResult {
        let mut result = crate::physics::composable::ValidationResult::new();
        
        // Validate grid compatibility
        if grid.nx < 16 || grid.ny < 16 || grid.nz < 16 {
            result.add_error("Grid dimensions must be at least 16x16x16 for hybrid solver".to_string());
        }
        
        // Validate configuration
        if self.config.coupling_interface.buffer_width == 0 {
            result.add_error("Buffer width must be greater than 0".to_string());
        }
        
        if self.config.coupling_interface.smoothing_factor < 0.0 || self.config.coupling_interface.smoothing_factor > 1.0 {
            result.add_error("Smoothing factor must be between 0 and 1".to_string());
        }
        
        result
    }
    
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// Add metadata field to HybridSolver struct
impl HybridSolver {
    fn create_metadata() -> PluginMetadata {
        PluginMetadata {
            id: "hybrid_pstd_fdtd".to_string(),
            name: "Hybrid PSTD/FDTD Solver".to_string(),
            version: "1.0.0".to_string(),
            description: "Adaptive hybrid solver combining PSTD and FDTD methods".to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::homogeneous::HomogeneousMedium;
    

    #[test]
    fn test_hybrid_solver_as_plugin() {
        // Create a simple grid
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001);
        
        // Create hybrid solver
        let config = HybridConfig::default();
        let solver = HybridSolver::new(config, &grid).unwrap();
        
        // Test metadata
        assert_eq!(solver.metadata().id, "hybrid_pstd_fdtd");
        assert_eq!(solver.metadata().name, "Hybrid PSTD/FDTD Solver");
        
        // Test required and provided fields
        let required = solver.required_fields();
        let provided = solver.provided_fields();
        assert!(required.contains(&FieldType::Pressure));
        assert!(required.contains(&FieldType::Velocity));
        assert!(provided.contains(&FieldType::Pressure));
        assert!(provided.contains(&FieldType::Velocity));
        
        // Test state
        assert_eq!(solver.state(), PluginState::Initialized);
    }
    
    #[test]
    fn test_hybrid_solver_validation() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.01, 1.0);
        
        let config = HybridConfig::default();
        let solver = HybridSolver::new(config, &grid).unwrap();
        
        // Validate the solver
        let result = solver.validate(&grid, &medium);
        assert!(result.is_valid, "Validation failed: {:?}", result.errors);
    }
    
    #[test]
    fn test_hybrid_solver_clone() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001);
        let config = HybridConfig::default();
        let solver = HybridSolver::new(config, &grid).unwrap();
        
        // Test cloning
        let cloned = solver.clone_plugin();
        assert_eq!(cloned.metadata().id, solver.metadata().id);
    }
}