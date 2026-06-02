use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};
use kwavers_domain::therapy::microbubble::{
    calculate_primary_bjerknes_force, DrugPayload, MarmottantShellProperties, MicrobubbleState,
};
use kwavers_physics::acoustics::bubble_dynamics::adaptive_integration::integrate_bubble_dynamics_adaptive;

use super::MicrobubbleDynamicsService;

impl MicrobubbleDynamicsService {
    /// Update single microbubble dynamics for one timestep
    ///
    /// # Algorithm
    ///
    /// 1. Update shell state based on current radius
    /// 2. Convert domain state → Keller-Miksis state
    /// 3. Solve Keller-Miksis ODE with radiation-damping term R/c · dP_ac/dt
    /// 4. Calculate radiation force from ∇P
    /// 5. Update bubble position (Euler step: v += F·dt/m, x += v·dt)
    /// 6. Update drug release kinetics
    /// 7. Convert back to domain state
    /// 8. Check for cavitation events
    ///
    /// ## Radiation-Damping Term
    ///
    /// Passing `pressure_time_derivative = 0.0` is valid when the applied pressure
    /// varies slowly compared to the bubble dynamics timescale.
    ///
    /// # Reference
    ///
    /// Keller JB, Miksis M (1980). *J Acoust Soc Am* 68(2):628–633.
    /// # Errors
    /// - Returns [`KwaversError::Physics`] if the precondition for a Physics-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn update_bubble_dynamics(
        &self,
        bubble: &mut MicrobubbleState,
        shell: &mut MarmottantShellProperties,
        drug: &mut DrugPayload,
        acoustic_pressure: f64,
        pressure_gradient: (f64, f64, f64),
        // dP_ac/dt [Pa/s]. Pass 0.0 when waveform is slowly varying or unknown.
        pressure_time_derivative: f64,
        time: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        if dt <= 0.0 {
            return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
                parameter: "dt".to_owned(),
                value: dt,
                reason: "timestep must be positive".to_owned(),
            }));
        }

        // 1. Update shell state
        shell.update_state(bubble.radius);

        // 2. Convert domain → KM state
        let mut km_state = Self::domain_to_km_state(bubble, shell)?;

        // 3. Solve Keller-Miksis ODE
        integrate_bubble_dynamics_adaptive(
            &self.keller_miksis,
            &mut km_state,
            acoustic_pressure,
            pressure_time_derivative,
            dt,
            time,
        )?;

        // 4. Calculate radiation force
        let radiation_force = calculate_primary_bjerknes_force(
            km_state.radius,
            bubble.radius_equilibrium,
            pressure_gradient,
        )?;

        // 5. Update bubble position (Euler: v += F·dt/m, x += v·dt)
        let bubble_mass = Self::effective_bubble_mass(bubble.radius_equilibrium);
        let acceleration = (
            radiation_force.fx / bubble_mass,
            radiation_force.fy / bubble_mass,
            radiation_force.fz / bubble_mass,
        );

        bubble.velocity.vx += acceleration.0 * dt;
        bubble.velocity.vy += acceleration.1 * dt;
        bubble.velocity.vz += acceleration.2 * dt;

        bubble.position.x += bubble.velocity.vx * dt;
        bubble.position.y += bubble.velocity.vy * dt;
        bubble.position.z += bubble.velocity.vz * dt;

        // 6. Update drug release
        let shell_strain = shell.strain(km_state.radius);
        let volume = (4.0 / 3.0) * std::f64::consts::PI * km_state.radius.powi(3);
        let released = drug.update_release(volume, shell.state, shell_strain, dt)?;

        bubble.drug_released_total += released;

        // 7. Convert back to domain state
        Self::km_to_domain_state(&km_state, bubble, shell);

        // 8. Check for cavitation
        if bubble.is_cavitating() && !bubble.has_cavitated {
            bubble.has_cavitated = true;
        }

        bubble.time = time + dt;

        Ok(())
    }
}
