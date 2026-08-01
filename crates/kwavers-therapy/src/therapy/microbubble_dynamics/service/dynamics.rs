use crate::therapy::microbubble_dynamics::DrugPayload;
use aequitas::systems::si::quantities::{Length, Mass, Pressure, PressureRate, Time, Velocity};
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};
use kwavers_physics::acoustics::bubble_dynamics::adaptive_integration::integrate_bubble_dynamics_adaptive;
use kwavers_physics::therapy::microbubble::{
    calculate_primary_bjerknes_force, MarmottantShellProperties, MicrobubbleState,
    PressureGradient3D,
};

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
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn update_bubble_dynamics(
        &self,
        bubble: &mut MicrobubbleState,
        shell: &mut MarmottantShellProperties,
        drug: &mut DrugPayload,
        acoustic_pressure: Pressure<f64>,
        pressure_gradient: PressureGradient3D,
        // Pass zero when the waveform is slowly varying or unknown.
        pressure_time_derivative: PressureRate<f64>,
        time: Time<f64>,
        dt: Time<f64>,
    ) -> KwaversResult<()> {
        let dt_value = dt.into_base();
        if dt_value <= 0.0 {
            return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
                parameter: "dt".to_owned(),
                value: dt_value,
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
            acoustic_pressure.into_base(),
            pressure_time_derivative.into_base(),
            dt_value,
            time.into_base(),
        )?;

        // 4. Calculate radiation force
        let radiation_force = calculate_primary_bjerknes_force(
            Length::from_base(km_state.radius),
            bubble.radius_equilibrium,
            pressure_gradient,
        )?;

        // 5. Update bubble position (Euler: v += F·dt/m, x += v·dt)
        let bubble_mass = Self::effective_bubble_mass(bubble.radius_equilibrium.into_base());
        let acceleration = (
            radiation_force.fx.into_base() / bubble_mass,
            radiation_force.fy.into_base() / bubble_mass,
            radiation_force.fz.into_base() / bubble_mass,
        );

        bubble.velocity.vx =
            Velocity::from_base(bubble.velocity.vx.into_base() + acceleration.0 * dt_value);
        bubble.velocity.vy =
            Velocity::from_base(bubble.velocity.vy.into_base() + acceleration.1 * dt_value);
        bubble.velocity.vz =
            Velocity::from_base(bubble.velocity.vz.into_base() + acceleration.2 * dt_value);

        bubble.position.x = Length::from_base(
            bubble.position.x.into_base() + bubble.velocity.vx.into_base() * dt_value,
        );
        bubble.position.y = Length::from_base(
            bubble.position.y.into_base() + bubble.velocity.vy.into_base() * dt_value,
        );
        bubble.position.z = Length::from_base(
            bubble.position.z.into_base() + bubble.velocity.vz.into_base() * dt_value,
        );

        // 6. Update drug release
        let shell_strain = shell.strain(Length::from_base(km_state.radius));
        let volume = (4.0 / 3.0) * std::f64::consts::PI * km_state.radius.powi(3);
        let released =
            drug.update_release(volume, shell.state, shell_strain.into_base(), dt_value)?;

        bubble.drug_released_total =
            Mass::from_base(bubble.drug_released_total.into_base() + released);

        // 7. Convert back to domain state
        Self::km_to_domain_state(&km_state, bubble, shell);

        // 8. Check for cavitation
        if bubble.is_cavitating() && !bubble.has_cavitated {
            bubble.has_cavitated = true;
        }

        bubble.time = Time::from_base(time.into_base() + dt_value);

        Ok(())
    }
}
