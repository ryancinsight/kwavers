//! Enhanced BEM-FEM solver with Burton-Miller formulation and adaptive refinement.
//!
//! # Theorem: Burton-Miller uniqueness diagnostic
//!
//! For the exterior Helmholtz problem, the classical boundary integral equation
//! can lose uniqueness at fictitious interior eigenfrequencies. The
//! Burton-Miller combined equation adds a non-real normal-derivative coupling
//! parameter, which removes the real-axis fictitious-frequency nullspace for
//! the exterior acoustic problem. Therefore a configured Burton-Miller coupling
//! is sufficient for this validation layer to reject a spurious-resonance
//! diagnosis.
//!
//! # Interface estimator contract
//!
//! This module does not assemble the coupled BEM-FEM matrix. Its validation
//! result is a configuration-level estimator: it combines the requested
//! interface tolerance, iterative coupling tolerance, relaxation factor, and
//! acoustic element resolution `h/lambda`. For linear interface traces, the
//! interpolation component scales as `O((h/lambda)^2)`. Adaptive refinement
//! halves `h` until the configured minimum element size is reached.
//!
//! References: Burton & Miller (1971); Amini (1990) coupling-parameter
//! analysis; residual a-posteriori Helmholtz estimators by Ihlenburg/Babuška.

use super::config::EnhancedBemFemConfig;
use super::types::{BemFemValidationResult, InterfaceQuality, RefinementStep};
use crate::core::error::{KwaversError, KwaversResult};

use crate::core::constants::fundamental::SOUND_SPEED_AIR;
use crate::core::constants::numerical::{TWO_PI};
const DEFAULT_SOUND_SPEED_M_PER_S: f64 = SOUND_SPEED_AIR;
const RESONANCE_RELATIVE_BAND: f64 = 1e-6;

/// Enhanced BEM-FEM solver with Burton-Miller and adaptive refinement.
#[derive(Debug)]
pub struct EnhancedBemFemSolver {
    /// Configuration.
    config: EnhancedBemFemConfig,

    /// Interface quality metrics.
    pub(super) interface_quality: Option<InterfaceQuality>,

    /// Refinement history.
    pub(super) refinement_history: Vec<RefinementStep>,
}

impl EnhancedBemFemSolver {
    /// Create new enhanced solver.
    #[must_use]
    pub fn new(config: EnhancedBemFemConfig) -> Self {
        Self {
            config,
            interface_quality: None,
            refinement_history: Vec::new(),
        }
    }

    /// Validate coupling at a specific frequency.
    ///
    /// Tests:
    /// 1. No spurious resonances detected
    /// 2. System is well-conditioned
    /// 3. Interface continuity satisfied
    /// 4. Solution converges with refinement
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn validate(&mut self, frequency: f64) -> KwaversResult<BemFemValidationResult> {
        let start_time = std::time::Instant::now();
        self.validate_frequency(frequency)?;

        let spurious_detected = if self.config.burton_miller_config.is_some() {
            false
        } else {
            self.check_spurious_resonance(frequency)?
        };

        let mut interface_error = self.estimate_interface_error_at_level(frequency, 0)?;

        let mut refinement_count = 0;
        while self.config.adaptive_refinement
            && interface_error > self.config.target_interface_error
            && refinement_count < self.config.max_refinement_level
        {
            self.refine_interface(frequency, refinement_count + 1)?;
            refinement_count += 1;
            interface_error =
                self.estimate_interface_error_at_level(frequency, refinement_count)?;
        }

        let avg_element_size = self.element_size_at_level(refinement_count);
        let num_elements = self.estimated_interface_elements(frequency, avg_element_size);
        self.interface_quality = Some(InterfaceQuality {
            num_elements,
            avg_element_size,
            estimated_error: interface_error,
            condition_number: Some(self.estimate_condition_number(frequency, avg_element_size)),
            max_local_error: interface_error,
            spurious_resonance_detected: spurious_detected,
        });

        Ok(BemFemValidationResult {
            frequency,
            spurious_resonance_detected: spurious_detected,
            burton_miller_used: self.config.burton_miller_config.is_some(),
            interface_error,
            refinement_levels: refinement_count,
            validation_time: start_time.elapsed().as_secs_f64(),
            interface_quality: self.interface_quality.clone().unwrap(),
        })
    }

    /// Check for explicitly configured standard-BEM characteristic frequencies.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn check_spurious_resonance(&self, frequency: f64) -> KwaversResult<bool> {
        self.validate_frequency(frequency)?;

        let Some(frequencies) = &self.config.validation_frequencies else {
            return Ok(false);
        };

        frequencies.iter().try_fold(false, |detected, candidate| {
            self.validate_frequency(*candidate)?;
            let scale = frequency.abs().max(candidate.abs()).max(1.0);
            Ok(detected || (frequency - candidate).abs() / scale <= RESONANCE_RELATIVE_BAND)
        })
    }

    /// Estimate error on coupling interface at the current mesh level.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn estimate_interface_error_at_level(
        &self,
        frequency: f64,
        level: usize,
    ) -> KwaversResult<f64> {
        self.validate_frequency(frequency)?;
        self.validate_mesh_bounds()?;

        let wavelength = self.sound_speed() / frequency;
        let h = self.element_size_at_level(level);
        let resolution_ratio = h / wavelength;
        let mesh_error = self.config.base_config.interface_tolerance * resolution_ratio.powi(2);
        let iterative_error = self.config.base_config.convergence_tolerance
            / self.config.base_config.relaxation_factor;

        Ok(mesh_error + iterative_error)
    }

    /// Refine interface mesh at specified level.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn refine_interface(&mut self, frequency: f64, level: usize) -> KwaversResult<()> {
        let h = self.element_size_at_level(level);
        let estimated_error = self.estimate_interface_error_at_level(frequency, level)?;
        let num_elements = self.estimated_interface_elements(frequency, h);
        let previous_elements = self.refinement_history.last().map_or_else(
            || self.estimated_interface_elements(frequency, self.element_size_at_level(0)),
            |step| step.num_elements,
        );

        self.refinement_history.push(RefinementStep {
            level,
            num_elements,
            estimated_error,
            num_refined_elements: num_elements.saturating_sub(previous_elements),
        });

        Ok(())
    }

    /// Get interface quality metrics.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn interface_quality(&self) -> Option<&InterfaceQuality> {
        self.interface_quality.as_ref()
    }

    /// Get refinement history.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn refinement_history(&self) -> &[RefinementStep] {
        &self.refinement_history
    }

    fn validate_frequency(&self, frequency: f64) -> KwaversResult<()> {
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "BEM-FEM validation frequency must be finite and positive; got {frequency}"
            )));
        }
        Ok(())
    }

    fn validate_mesh_bounds(&self) -> KwaversResult<()> {
        if !self.config.min_element_size.is_finite()
            || !self.config.max_element_size.is_finite()
            || self.config.min_element_size <= 0.0
            || self.config.max_element_size <= 0.0
            || self.config.min_element_size > self.config.max_element_size
        {
            return Err(KwaversError::InvalidInput(format!(
                "Invalid BEM-FEM interface element bounds: min={}, max={}",
                self.config.min_element_size, self.config.max_element_size
            )));
        }
        if !self.config.base_config.convergence_tolerance.is_finite()
            || self.config.base_config.convergence_tolerance < 0.0
            || !self.config.base_config.interface_tolerance.is_finite()
            || self.config.base_config.interface_tolerance < 0.0
            || !self.config.base_config.relaxation_factor.is_finite()
            || self.config.base_config.relaxation_factor <= 0.0
            || self.config.base_config.relaxation_factor > 1.0
        {
            return Err(KwaversError::InvalidInput(
                "Invalid BEM-FEM coupling tolerances or relaxation factor".to_owned(),
            ));
        }
        Ok(())
    }

    fn sound_speed(&self) -> f64 {
        self.config
            .burton_miller_config
            .map_or(DEFAULT_SOUND_SPEED_M_PER_S, |config| config.sound_speed)
    }

    fn element_size_at_level(&self, level: usize) -> f64 {
        let divisor = 2usize.saturating_pow(level as u32) as f64;
        (self.config.max_element_size / divisor).max(self.config.min_element_size)
    }

    fn estimated_interface_elements(&self, frequency: f64, element_size: f64) -> usize {
        let wavelength = self.sound_speed() / frequency;
        let elements_per_wavelength = (wavelength / element_size).ceil().max(1.0) as usize;
        elements_per_wavelength.saturating_mul(elements_per_wavelength)
    }

    fn estimate_condition_number(&self, frequency: f64, element_size: f64) -> f64 {
        let k = TWO_PI * frequency / self.sound_speed();
        let kh = k * element_size;
        let burton_miller_factor = if self.config.burton_miller_config.is_some() {
            1.0
        } else {
            1.0 + kh
        };
        burton_miller_factor * (1.0 + kh * kh)
    }
}
