//! Radiation Forces on Microbubbles
//!
//! Implementation of acoustic radiation forces acting on oscillating microbubbles
//! in ultrasound fields. These forces are responsible for bubble translation,
//! concentration, and therapeutic effects.
//!
//! ## Physical Mechanisms
//!
//! ### Primary Bjerknes Force
//!
//! The primary Bjerknes force arises from the interaction between an oscillating
//! bubble and a spatially varying acoustic pressure field. It drives bubbles
//! toward pressure nodes (standing waves) or antinodes (traveling waves).
//!
//! **Time-averaged force**:
//! ```text
//! F_Bjerknes = -⟨V(t)⟩ · ∇P_acoustic
//!            = -(4π/3)⟨R³⟩ · ∇P_acoustic
//! ```
//!
//! For small-amplitude oscillations:
//! ```text
//! F_Bjerknes ≈ -(4πR₀³/3) · ∇P_acoustic
//! ```
//!
//! For large-amplitude oscillations, time-averaging is required over the
//! acoustic period.
//!
//! ### Secondary Bjerknes Force
//!
//! Secondary Bjerknes forces arise from bubble-bubble interactions. For two
//! bubbles separated by distance d:
//!
//! ```text
//! F_secondary = (ρ R₁³ R₂³ / d²) · [R̈₁R₂ + 2Ṙ₁Ṙ₂]
//! ```
//!
//! This force is attractive when bubbles oscillate in phase and repulsive
//! when out of phase. (Currently deferred to P1 priority)
//!
//! ### Acoustic Streaming
//!
//! Acoustic streaming is the steady flow induced by viscous attenuation of
//! acoustic waves. Around oscillating bubbles, microstreaming creates local
//! fluid circulation that contributes to transport and mixing.
//!
//! **Streaming velocity** (simplified):
//! ```text
//! v_streaming ∝ (R₀²ω/ν) · (U/c)²
//! ```
//! where:
//! - ω: Angular frequency
//! - ν: Kinematic viscosity
//! - U: Bubble wall velocity amplitude
//! - c: Sound speed
//!
//! ## References
//!
//! - Leighton (1994): "The Acoustic Bubble", Academic Press
//! - Doinikov (1994): "Acoustic radiation force on a spherical particle"
//! - Blake (1986): "Bjerknes forces in stationary sound fields"
//! - Elder (1959): "Steady flow produced by vibrating cylinders"
//! - Marmottant & Hilgenfeldt (2003): "Controlled vesicle deformation"

use crate::core::error::KwaversResult;

/// Primary Bjerknes force on oscillating bubble
///
/// Value object representing the time-averaged radiation force in 3D space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiationForce {
    /// Force components [N]
    pub fx: f64,
    pub fy: f64,
    pub fz: f64,
}

impl RadiationForce {
    /// Create new radiation force
    #[must_use]
    pub fn new(fx: f64, fy: f64, fz: f64) -> Self {
        Self { fx, fy, fz }
    }

    /// Zero force
    #[must_use]
    pub fn zero() -> Self {
        Self {
            fx: 0.0,
            fy: 0.0,
            fz: 0.0,
        }
    }

    /// Force magnitude [N]
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.fx * self.fx + self.fy * self.fy + self.fz * self.fz).sqrt()
    }

    /// Normalize to unit direction (returns zero if magnitude is zero)
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        if mag > 0.0 {
            Self {
                fx: self.fx / mag,
                fy: self.fy / mag,
                fz: self.fz / mag,
            }
        } else {
            Self::zero()
        }
    }

    /// Add two forces
    #[must_use]
    pub fn add(&self, other: &RadiationForce) -> Self {
        Self {
            fx: self.fx + other.fx,
            fy: self.fy + other.fy,
            fz: self.fz + other.fz,
        }
    }

    /// Scale force by scalar
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            fx: self.fx * factor,
            fy: self.fy * factor,
            fz: self.fz * factor,
        }
    }
}

/// Calculate primary Bjerknes force
///
/// Computes the time-averaged radiation force on an oscillating bubble
/// in a spatially varying acoustic pressure field.
///
/// # Arguments
///
/// - `radius`: Current bubble radius [m]
/// - `radius_equilibrium`: Equilibrium radius [m]
/// - `pressure_gradient`: Spatial gradient of acoustic pressure [Pa/m]
///   in (∂P/∂x, ∂P/∂y, ∂P/∂z) components
///
/// # Returns
///
/// Primary Bjerknes force [N]
///
/// # Mathematical Specification
///
/// For small-amplitude oscillations (R ≈ R₀):
/// ```text
/// F_Bjerknes = -(4π/3)R₀³ · ∇P_acoustic
/// ```
///
/// For large-amplitude oscillations:
/// ```text
/// F_Bjerknes = -(4π/3)⟨R³⟩ · ∇P_acoustic
/// ```
/// where ⟨R³⟩ is the time-averaged cube of the instantaneous radius.
///
/// # Approximation
///
/// This implementation uses the instantaneous radius R(t) rather than
/// time-averaging ⟨R³⟩. For accurate time-averaged forces, the caller
/// should average this force over multiple acoustic periods.
///
/// # Example
///
/// ```rust,no_run
/// use kwavers::domain::therapy::microbubble::forces::calculate_primary_bjerknes_force;
///
/// let radius = 1.5e-6; // 1.5 μm
/// let r0 = 1.0e-6;     // 1.0 μm equilibrium
/// let grad_p = (1e5, 0.0, 0.0); // Pressure gradient [Pa/m]
///
/// let force = calculate_primary_bjerknes_force(radius, r0, grad_p).unwrap();
/// // Force pulls bubble toward lower pressure (negative x direction)
/// ```
pub fn calculate_primary_bjerknes_force(
    radius: f64,
    _radius_equilibrium: f64,
    pressure_gradient: (f64, f64, f64),
) -> KwaversResult<RadiationForce> {
    // Use instantaneous radius for volume
    // For time-averaged force, caller should average over acoustic period
    let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);

    // F = -V · ∇P
    let fx = -volume * pressure_gradient.0;
    let fy = -volume * pressure_gradient.1;
    let fz = -volume * pressure_gradient.2;

    Ok(RadiationForce::new(fx, fy, fz))
}

/// Calculate time-averaged primary Bjerknes force
///
/// Computes the primary Bjerknes force using the time-averaged bubble volume.
/// This is more accurate for large-amplitude oscillations.
///
/// # Arguments
///
/// - `radius_avg`: Time-averaged radius [m]
/// - `radius_equilibrium`: Equilibrium radius [m]
/// - `pressure_gradient`: Spatial gradient of acoustic pressure [Pa/m]
///
/// # Returns
///
/// Time-averaged primary Bjerknes force [N]
///
/// # Note
///
/// The time-averaged volume ⟨V⟩ = (4π/3)⟨R³⟩ requires integration over
/// the acoustic period. For simplicity, this uses ⟨R⟩³ as approximation.
/// For accurate results, pass the cube-root of ⟨R³⟩ as radius_avg.
pub fn calculate_primary_bjerknes_force_averaged(
    radius_avg: f64,
    _radius_equilibrium: f64,
    pressure_gradient: (f64, f64, f64),
) -> KwaversResult<RadiationForce> {
    let volume_avg = (4.0 / 3.0) * std::f64::consts::PI * radius_avg.powi(3);

    let fx = -volume_avg * pressure_gradient.0;
    let fy = -volume_avg * pressure_gradient.1;
    let fz = -volume_avg * pressure_gradient.2;

    Ok(RadiationForce::new(fx, fy, fz))
}

/// Acoustic streaming velocity around oscillating bubble
///
/// Value object representing the steady streaming velocity field induced
/// by viscous dissipation near an oscillating bubble.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingVelocity {
    /// Velocity components [m/s]
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

impl StreamingVelocity {
    /// Create new streaming velocity
    #[must_use]
    pub fn new(vx: f64, vy: f64, vz: f64) -> Self {
        Self { vx, vy, vz }
    }

    /// Zero velocity
    #[must_use]
    pub fn zero() -> Self {
        Self {
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
        }
    }

    /// Speed (magnitude)
    #[must_use]
    pub fn speed(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz).sqrt()
    }
}

/// Calculate acoustic streaming velocity contribution
///
/// Estimates the local fluid velocity induced by acoustic streaming near
/// an oscillating bubble. This is a simplified model capturing the primary
/// scaling relationships.
///
/// # Arguments
///
/// - `radius_equilibrium`: Bubble equilibrium radius [m]
/// - `wall_velocity_amplitude`: Amplitude of bubble wall velocity [m/s]
/// - `frequency`: Acoustic frequency [Hz]
/// - `distance`: Distance from bubble center [m]
/// - `direction`: Unit direction vector (nx, ny, nz) from bubble to point
///
/// # Returns
///
/// Streaming velocity at the specified point [m/s]
///
/// # Mathematical Model
///
/// Simplified streaming velocity (Elder 1959, Marmottant & Hilgenfeldt 2003):
/// ```text
/// v_streaming ≈ (R₀²ω/ν) · (U/c)² · f(r/R₀)
/// ```
/// where:
/// - R₀: Equilibrium radius
/// - ω = 2πf: Angular frequency
/// - ν: Kinematic viscosity
/// - U: Wall velocity amplitude
/// - c: Sound speed
/// - f(r/R₀): Distance-dependent decay function
///
/// # Approximations
///
/// 1. Assumes axisymmetric streaming pattern
/// 2. Uses simplified scaling (actual pattern is complex)
/// 3. Valid for distances r > R₀ (far from bubble surface)
/// 4. Does not account for wall effects or bubble interactions
pub fn calculate_acoustic_streaming_velocity(
    radius_equilibrium: f64,
    wall_velocity_amplitude: f64,
    frequency: f64,
    distance: f64,
    direction: (f64, f64, f64),
) -> KwaversResult<StreamingVelocity> {
    // Physical constants
    const KINEMATIC_VISCOSITY: f64 = 1e-6; // Water at 37°C [m²/s]
    const SOUND_SPEED: f64 = 1540.0; // Soft tissue [m/s]

    if distance <= radius_equilibrium {
        // Inside or at bubble surface: return zero (not valid)
        return Ok(StreamingVelocity::zero());
    }

    // Angular frequency
    let omega = 2.0 * std::f64::consts::PI * frequency;

    // Mach number squared (U/c)²
    let mach_sq = (wall_velocity_amplitude / SOUND_SPEED).powi(2);

    // Reynolds number (inertial to viscous ratio)
    let re = (radius_equilibrium.powi(2) * omega) / KINEMATIC_VISCOSITY;

    // Distance ratio
    let r_ratio = distance / radius_equilibrium;

    // Decay function: 1/r² falloff (simplified)
    let decay = 1.0 / r_ratio.powi(2);

    // Streaming velocity magnitude (scaling estimate)
    let v_magnitude = re * mach_sq * radius_equilibrium * omega * decay;

    // Normalize direction
    let dir_mag =
        (direction.0 * direction.0 + direction.1 * direction.1 + direction.2 * direction.2).sqrt();
    if dir_mag < 1e-10 {
        return Ok(StreamingVelocity::zero());
    }

    let nx = direction.0 / dir_mag;
    let ny = direction.1 / dir_mag;
    let nz = direction.2 / dir_mag;

    // Apply direction
    Ok(StreamingVelocity::new(
        v_magnitude * nx,
        v_magnitude * ny,
        v_magnitude * nz,
    ))
}

/// Calculate drag force on bubble from streaming flow
///
/// Computes the Stokes drag force on a bubble moving through fluid
/// (or fluid moving past bubble).
///
/// # Arguments
///
/// - `radius`: Bubble radius [m]
/// - `relative_velocity`: Velocity of bubble relative to fluid [m/s]
///
/// # Returns
///
/// Drag force [N]
///
/// # Formula (Stokes drag)
///
/// ```text
/// F_drag = 6πμRv
/// ```
/// where μ is dynamic viscosity.
pub fn calculate_drag_force(
    radius: f64,
    relative_velocity: (f64, f64, f64),
) -> KwaversResult<RadiationForce> {
    const DYNAMIC_VISCOSITY: f64 = 0.001; // Water at 37°C [Pa·s]

    let drag_coeff = 6.0 * std::f64::consts::PI * DYNAMIC_VISCOSITY * radius;

    let fx = -drag_coeff * relative_velocity.0;
    let fy = -drag_coeff * relative_velocity.1;
    let fz = -drag_coeff * relative_velocity.2;

    Ok(RadiationForce::new(fx, fy, fz))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radiation_force_magnitude() {
        let force = RadiationForce::new(3.0, 4.0, 0.0);
        assert_eq!(force.magnitude(), 5.0);
    }

    #[test]
    fn test_radiation_force_normalized() {
        let force = RadiationForce::new(3.0, 4.0, 0.0);
        let norm = force.normalized();
        assert!((norm.magnitude() - 1.0).abs() < 1e-10);
        assert!((norm.fx - 0.6).abs() < 1e-10);
        assert!((norm.fy - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_primary_bjerknes_force_basic() {
        let radius = 1.0e-6; // 1 μm
        let r0 = 1.0e-6;
        let grad_p = (1e5, 0.0, 0.0); // 100 kPa/m gradient in x

        let force = calculate_primary_bjerknes_force(radius, r0, grad_p).unwrap();

        // Force should be negative x (toward lower pressure)
        assert!(force.fx < 0.0);
        assert_eq!(force.fy, 0.0);
        assert_eq!(force.fz, 0.0);

        // Check magnitude order
        let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
        let expected_magnitude = volume * 1e5;
        assert!((force.magnitude() - expected_magnitude).abs() < 1e-15);
    }

    #[test]
    fn test_primary_bjerknes_expanded_bubble() {
        let radius = 2.0e-6; // Expanded to 2 μm
        let r0 = 1.0e-6; // Equilibrium 1 μm
        let grad_p = (1e5, 0.0, 0.0);

        let force = calculate_primary_bjerknes_force(radius, r0, grad_p).unwrap();

        // Force should be 8x larger (volume scales as R³)
        let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
        let expected = -volume * 1e5;
        assert!((force.fx - expected).abs() < 1e-10);
    }

    #[test]
    fn test_primary_bjerknes_3d_gradient() {
        let radius = 1.0e-6;
        let r0 = 1.0e-6;
        let grad_p = (1e5, 2e5, 3e5); // Gradient in all directions

        let force = calculate_primary_bjerknes_force(radius, r0, grad_p).unwrap();

        let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
        assert!((force.fx + volume * 1e5).abs() < 1e-10);
        assert!((force.fy + volume * 2e5).abs() < 1e-10);
        assert!((force.fz + volume * 3e5).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_velocity_zero_at_surface() {
        let r0 = 1.0e-6;
        let u_wall = 10.0; // m/s
        let freq = 1e6; // 1 MHz
        let distance = r0; // At surface
        let direction = (1.0, 0.0, 0.0);

        let v =
            calculate_acoustic_streaming_velocity(r0, u_wall, freq, distance, direction).unwrap();

        assert_eq!(v.vx, 0.0);
        assert_eq!(v.vy, 0.0);
        assert_eq!(v.vz, 0.0);
    }

    #[test]
    fn test_streaming_velocity_far_field() {
        let r0 = 1.0e-6;
        let u_wall = 10.0;
        let freq = 1e6;
        let distance = 10.0 * r0; // 10 radii away
        let direction = (1.0, 0.0, 0.0);

        let v =
            calculate_acoustic_streaming_velocity(r0, u_wall, freq, distance, direction).unwrap();

        // Should have velocity in x direction
        assert!(v.vx > 0.0);
        assert_eq!(v.vy, 0.0);
        assert_eq!(v.vz, 0.0);

        // Should decay with distance
        let distance_far = 20.0 * r0;
        let v_far =
            calculate_acoustic_streaming_velocity(r0, u_wall, freq, distance_far, direction)
                .unwrap();
        assert!(v_far.vx < v.vx);
    }

    #[test]
    fn test_drag_force() {
        let radius = 1.0e-6;
        let velocity = (1.0, 0.0, 0.0); // 1 m/s in x

        let force = calculate_drag_force(radius, velocity).unwrap();

        // Drag opposes motion (negative x)
        assert!(force.fx < 0.0);
        assert_eq!(force.fy, 0.0);
        assert_eq!(force.fz, 0.0);

        // Check Stokes formula: F = 6πμRv
        const MU: f64 = 0.001;
        let expected = -6.0 * std::f64::consts::PI * MU * radius * 1.0;
        assert!((force.fx - expected).abs() < 1e-15);
    }

    #[test]
    fn test_force_addition() {
        let f1 = RadiationForce::new(1.0, 2.0, 3.0);
        let f2 = RadiationForce::new(4.0, 5.0, 6.0);
        let sum = f1.add(&f2);

        assert_eq!(sum.fx, 5.0);
        assert_eq!(sum.fy, 7.0);
        assert_eq!(sum.fz, 9.0);
    }

    #[test]
    fn test_force_scaling() {
        let force = RadiationForce::new(1.0, 2.0, 3.0);
        let scaled = force.scale(2.0);

        assert_eq!(scaled.fx, 2.0);
        assert_eq!(scaled.fy, 4.0);
        assert_eq!(scaled.fz, 6.0);
    }
}
