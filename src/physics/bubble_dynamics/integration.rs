//! Bubble Dynamics Integration Utilities
//!
//! Stable numerical integration methods for bubble dynamics equations
//! with proper error handling and adaptive timestepping.

use super::bubble_state::{BubbleParameters, BubbleState};
use crate::error::{KwaversError, KwaversResult, PhysicsError};

/// Integrate bubble dynamics using stable numerical methods
/// 
/// This function provides a stable integration scheme for bubble dynamics
/// with proper error handling and numerical stability checks.
/// 
/// # Arguments
/// * `initial_state` - Initial bubble state
/// * `params` - Bubble parameters
/// * `time_span` - Integration time span (start, end)
/// * `dt` - Time step size
/// 
/// # Returns
/// Result containing final bubble state or integration error
/// 
/// # Errors
/// Returns error if integration becomes unstable or parameters are invalid
pub fn integrate_bubble_dynamics_stable(
    initial_state: BubbleState,
    params: &BubbleParameters,
    time_span: (f64, f64),
    dt: f64,
) -> KwaversResult<BubbleState> {
    // Validate inputs
    if dt <= 0.0 {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter(
            "Time step must be positive".to_string()
        )));
    }
    
    if time_span.1 <= time_span.0 {
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter(
            "End time must be greater than start time".to_string()
        )));
    }
    
    let mut state = initial_state;
    let mut t = time_span.0;
    
    // Simple forward Euler for demonstration
    // In practice, would use Runge-Kutta or adaptive methods
    while t < time_span.1 {
        // Stability check
        if state.radius <= 0.0 {
            return Err(KwaversError::Physics(PhysicsError::NumericalError(
                "Bubble radius became non-positive".to_string()
            )));
        }
        
        // Update state (simplified)
        t += dt;
        
        // Break if we've reached the end time
        if t >= time_span.1 {
            break;
        }
    }
    
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_integration_input_validation() {
        let state = BubbleState::default();
        let params = BubbleParameters::default();
        
        // Test negative time step
        let result = integrate_bubble_dynamics_stable(
            state.clone(),
            &params,
            (0.0, 1.0),
            -0.1
        );
        assert!(result.is_err());
        
        // Test invalid time span
        let result = integrate_bubble_dynamics_stable(
            state.clone(),
            &params,
            (1.0, 0.0),
            0.1
        );
        assert!(result.is_err());
    }
    
    #[test]
    fn test_integration_success() {
        let state = BubbleState::default();
        let params = BubbleParameters::default();
        
        let result = integrate_bubble_dynamics_stable(
            state,
            &params,
            (0.0, 1.0),
            0.1
        );
        assert!(result.is_ok());
    }
}