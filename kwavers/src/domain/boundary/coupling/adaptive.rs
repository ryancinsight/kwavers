//! Adaptive absorbing boundary condition
//!
//! Dynamically adjusts absorption strength based on field energy levels.
//! Useful for preventing reflections while maintaining computational efficiency.

use crate::core::error::KwaversResult;
use crate::domain::boundary::traits::BoundaryCondition;
use crate::domain::grid::GridTopology;
use ndarray::ArrayViewMut3;
use rustfft::num_complex::Complex;

use super::types::BoundaryDirections;

/// Adaptive absorbing boundary
///
/// Dynamically adjusts absorption strength based on field energy levels.
/// This boundary condition monitors the energy in the boundary region and
/// increases absorption when high-energy waves approach, preventing reflections
/// while maintaining computational efficiency during low-energy periods.
///
/// # Physics
///
/// The absorption coefficient α(t) adapts according to:
///
/// ```text
/// α_target = α_base                           if E < E_threshold
/// α_target = α_base × (1 + log(E/E_threshold)) if E ≥ E_threshold
/// ```
///
/// The actual absorption evolves smoothly via exponential smoothing:
///
/// ```text
/// α(t+Δt) = α(t) × (1 - β) + α_target × β
/// ```
///
/// where β = 1 - exp(-λ·Δt) is the adaptation factor.
///
/// # Algorithm
///
/// 1. **Energy Monitoring**: Computes field energy E = ⟨|u|²⟩ in boundary region
/// 2. **Threshold Comparison**: Determines if energy exceeds threshold
/// 3. **Target Calculation**: Computes target absorption coefficient
/// 4. **Smooth Adaptation**: Updates current absorption via exponential smoothing
/// 5. **Application**: Applies absorption as exp(-α·Δt) damping factor
///
/// # Example
///
/// ```no_run
/// use kwavers::domain::boundary::coupling::AdaptiveBoundary;
/// use kwavers::domain::boundary::traits::BoundaryDirections;
///
/// let mut boundary = AdaptiveBoundary::new(
///     0.1,   // Base absorption coefficient
///     1.0,   // Energy threshold
///     1.0,   // Maximum absorption coefficient
///     1.0,   // Adaptation rate λ
///     BoundaryDirections::all(),
/// );
///
/// // During simulation, energy triggers adaptation
/// let field_energy = 5.0; // High energy
/// let dt = 1e-6;
/// boundary.adapt_to_energy(field_energy, dt);
///
/// // Check current absorption
/// let current_alpha = boundary.current_absorption();
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveBoundary {
    /// Base absorption coefficient (minimum level)
    pub base_absorption: f64,
    /// Energy threshold for triggering adaptation
    pub energy_threshold: f64,
    /// Maximum absorption coefficient (capped)
    pub max_absorption: f64,
    /// Adaptation time constant λ (smaller = faster adaptation)
    pub adaptation_rate: f64,
    /// Current absorption level α(t)
    pub current_absorption: f64,
    /// Directions to apply boundary
    pub directions: BoundaryDirections,
}

impl AdaptiveBoundary {
    /// Create a new adaptive boundary
    ///
    /// # Arguments
    ///
    /// * `base_absorption` - Minimum absorption coefficient (active at all times)
    /// * `energy_threshold` - Energy level that triggers increased absorption
    /// * `max_absorption` - Maximum absorption coefficient (upper bound)
    /// * `adaptation_rate` - Rate of adaptation λ (higher = faster response)
    /// * `directions` - Boundary directions to apply
    ///
    /// # Returns
    ///
    /// New `AdaptiveBoundary` initialized at base absorption level
    pub fn new(
        base_absorption: f64,
        energy_threshold: f64,
        max_absorption: f64,
        adaptation_rate: f64,
        directions: BoundaryDirections,
    ) -> Self {
        Self {
            base_absorption,
            energy_threshold,
            max_absorption,
            adaptation_rate,
            current_absorption: base_absorption,
            directions,
        }
    }

    /// Update absorption based on current field energy
    ///
    /// # Arguments
    ///
    /// * `field_energy` - Current field energy in boundary region
    /// * `dt` - Time step in seconds
    ///
    /// # Algorithm
    ///
    /// 1. Compare field_energy to energy_threshold
    /// 2. If above threshold: compute adaptive factor based on log(E/E_threshold)
    /// 3. Apply exponential smoothing with time constant 1/λ
    /// 4. Clamp to [base_absorption, max_absorption]
    pub fn adapt_to_energy(&mut self, field_energy: f64, dt: f64) {
        let target_absorption = if field_energy > self.energy_threshold {
            // Scale absorption based on energy level
            let energy_ratio = (field_energy / self.energy_threshold).ln();
            let adaptive_factor = 1.0 + energy_ratio.min(10.0); // Cap at 10x increase
            (self.base_absorption * adaptive_factor).min(self.max_absorption)
        } else {
            self.base_absorption
        };

        // Exponential smoothing for stability
        // α(t+Δt) = α(t) × (1-β) + α_target × β
        // where β = 1 - exp(-λ·Δt)
        let alpha = 1.0 - (-self.adaptation_rate * dt).exp();
        self.current_absorption =
            self.current_absorption * (1.0 - alpha) + target_absorption * alpha;
    }

    /// Get current absorption coefficient
    ///
    /// # Returns
    ///
    /// Current absorption coefficient α(t)
    pub fn current_absorption(&self) -> f64 {
        self.current_absorption
    }
}

impl BoundaryCondition for AdaptiveBoundary {
    fn name(&self) -> &str {
        "AdaptiveBoundary"
    }

    fn active_directions(&self) -> BoundaryDirections {
        self.directions
    }

    fn apply_scalar_spatial(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Apply adaptive absorption based on current field energy

        // Compute field energy in boundary region
        let field_energy = field.iter().map(|&x| x * x).sum::<f64>() / field.len() as f64;

        // Adapt absorption coefficient
        self.adapt_to_energy(field_energy, _dt);

        // Apply absorption: u(t+Δt) = u(t) × exp(-α·Δt)
        let absorption = self.current_absorption();
        field.mapv_inplace(|x| x * (-absorption * _dt).exp());

        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut ndarray::Array3<Complex<f64>>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Adaptive boundary in frequency domain
        Ok(())
    }

    fn reset(&mut self) {
        self.current_absorption = self.base_absorption;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_boundary() {
        let mut boundary = AdaptiveBoundary::new(
            0.1, // base absorption
            1.0, // energy threshold
            1.0, // max absorption
            1.0, // adaptation rate
            BoundaryDirections::all(),
        );

        // Low energy - should stay at base absorption
        boundary.adapt_to_energy(0.1, 0.001);
        assert!((boundary.current_absorption() - 0.1).abs() < 0.01);

        // High energy - should increase absorption
        boundary.adapt_to_energy(10.0, 0.001);
        let absorption = boundary.current_absorption();
        assert!(absorption > 0.1);
        assert!((0.0..=1.0).contains(&absorption));
    }

    #[test]
    fn test_adaptive_boundary_energy_threshold() {
        let mut boundary = AdaptiveBoundary::new(
            0.2,
            5.0, // threshold
            1.0,
            1.0,
            BoundaryDirections::all(),
        );

        // Below threshold
        boundary.adapt_to_energy(2.0, 0.01);
        assert!((boundary.current_absorption() - 0.2).abs() < 0.01);

        // Above threshold
        boundary.adapt_to_energy(15.0, 0.01);
        assert!(boundary.current_absorption() > 0.2);
    }

    #[test]
    fn test_adaptive_boundary_max_capping() {
        let mut boundary = AdaptiveBoundary::new(
            0.1,
            1.0,
            0.5,  // max absorption
            10.0, // fast adaptation
            BoundaryDirections::all(),
        );

        // Very high energy
        for _ in 0..100 {
            boundary.adapt_to_energy(1000.0, 0.01);
        }

        // Should cap at max_absorption
        assert!(boundary.current_absorption() <= 0.5);
        assert!(boundary.current_absorption() >= 0.45); // Should be close to max after many steps
    }

    #[test]
    fn test_adaptive_boundary_smooth_adaptation() {
        let mut boundary = AdaptiveBoundary::new(
            0.1,
            1.0,
            1.0,
            0.1, // slow adaptation
            BoundaryDirections::all(),
        );

        let initial = boundary.current_absorption();

        // Single step with high energy
        boundary.adapt_to_energy(10.0, 0.001);
        let after_one_step = boundary.current_absorption();

        // Should increase, but not jump instantly
        assert!(after_one_step > initial);
        assert!(after_one_step < 0.5); // Smooth increase, not instant jump
    }

    #[test]
    fn test_adaptive_boundary_reset() {
        let mut boundary = AdaptiveBoundary::new(0.1, 1.0, 1.0, 1.0, BoundaryDirections::all());

        // Increase absorption
        boundary.adapt_to_energy(10.0, 0.01);
        assert!(boundary.current_absorption() > 0.1);

        // Reset should return to base
        boundary.reset();
        assert!((boundary.current_absorption() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_boundary_exponential_smoothing() {
        let mut boundary = AdaptiveBoundary::new(0.0, 1.0, 1.0, 1.0, BoundaryDirections::all());

        // Track absorption over multiple steps with constant high energy
        let mut previous = boundary.current_absorption();
        for _ in 0..10 {
            boundary.adapt_to_energy(10.0, 0.01);
            let current = boundary.current_absorption();
            // Should increase monotonically
            assert!(current >= previous);
            previous = current;
        }
    }
}
