use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

///
/// Contains the material properties and geometric information required for
/// interface condition enforcement.
#[derive(Debug, Clone)]
pub struct FsiInterface {
    /// Fluid density [kg/m³]
    pub fluid_density: f64,
    /// Fluid sound speed [m/s]
    pub fluid_sound_speed: f64,
    /// Solid density [kg/m³]
    pub solid_density: f64,
    /// Solid longitudinal wave speed [m/s]
    pub solid_c_l: f64,
    /// Solid transverse wave speed [m/s]
    pub solid_c_t: f64,
    /// Interface normal vector (pointing from fluid to solid)
    pub normal: [f64; 3],
    /// Interface position in grid coordinates
    pub interface_mask: Array3<bool>,
}

impl FsiInterface {
    /// Create new fluid-structure interface
    ///
    /// # Mathematical Verification
    ///
    /// Material properties must satisfy wave speed ordering:
    /// c_t < c_l (shear slower than longitudinal)
    /// c_f < c_l (acoustic slower than solid compression)
    ///
    /// This ensures physically valid wave propagation modes.
    /// # Errors
    /// - Returns [`KwaversError::InternalError`] if the precondition for a InternalError-class constraint is violated.
    ///
    pub fn new(
        fluid_density: f64,
        fluid_sound_speed: f64,
        solid_density: f64,
        solid_c_l: f64,
        solid_c_t: f64,
        normal: [f64; 3],
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<Self> {
        // Validate material properties
        if solid_c_t >= solid_c_l {
            return Err(KwaversError::InternalError(format!(
                "Invalid solid wave speeds: c_t ({}) must be < c_l ({})",
                solid_c_t, solid_c_l
            )));
        }

        if fluid_density <= 0.0 || solid_density <= 0.0 {
            return Err(KwaversError::InternalError(
                "Densities must be positive".to_string(),
            ));
        }

        if fluid_sound_speed <= 0.0 || solid_c_l <= 0.0 || solid_c_t <= 0.0 {
            return Err(KwaversError::InternalError(
                "Wave speeds must be positive".to_string(),
            ));
        }

        // Normalize normal vector
        let len = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
        if len < 1e-10 {
            return Err(KwaversError::InternalError(
                "Interface normal vector cannot be zero".to_string(),
            ));
        }
        let normal = [normal[0] / len, normal[1] / len, normal[2] / len];

        // Initialize interface mask (false = no interface)
        let interface_mask = Array3::from_elem((nx, ny, nz), false);

        Ok(Self {
            fluid_density,
            fluid_sound_speed,
            solid_density,
            solid_c_l,
            solid_c_t,
            normal,
            interface_mask,
        })
    }

    /// Set interface location from spatial predicate
    ///
    /// Uses a level set function to determine interface location.
    pub fn set_interface_from_level_set<F>(&mut self, level_set: F)
    where
        F: Fn(usize, usize, usize) -> f64 + Sync,
    {
        ndarray::Zip::indexed(&mut self.interface_mask).par_for_each(|(i, j, k), mask| {
            // Interface where level set changes sign
            let phi_ijk = level_set(i, j, k);
            let is_interface = if i > 0 && j > 0 && k > 0 {
                let phi_im = level_set(i - 1, j, k);
                let phi_jm = level_set(i, j - 1, k);
                let phi_km = level_set(i, j, k - 1);
                (phi_ijk * phi_im < 0.0) || (phi_ijk * phi_jm < 0.0) || (phi_ijk * phi_km < 0.0)
            } else {
                false
            };
            *mask = is_interface;
        });
    }

    /// Acoustic impedance of fluid [Pa·s/m]
    pub fn fluid_impedance(&self) -> f64 {
        self.fluid_density * self.fluid_sound_speed
    }

    /// Longitudinal impedance of solid [Pa·s/m]
    pub fn solid_longitudinal_impedance(&self) -> f64 {
        self.solid_density * self.solid_c_l
    }

    /// Transverse impedance of solid [Pa·s/m]
    pub fn solid_transverse_impedance(&self) -> f64 {
        self.solid_density * self.solid_c_t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Test material property validation
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_fsi_interface_creation() {
        let interface = FsiInterface::new(
            1000.0,          // water density
            1500.0,          // water sound speed
            7850.0,          // steel density
            5960.0,          // steel longitudinal speed
            3240.0,          // steel transverse speed
            [1.0, 0.0, 0.0], // x-normal interface
            64,
            64,
            64,
        );
        let i = interface.unwrap();
        assert!((i.normal[0] - 1.0).abs() < 1e-10); // Normalized
    }

    /// Test wave speed ordering validation
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_invalid_wave_speeds() {
        let interface = FsiInterface::new(
            1000.0,
            1500.0,
            7850.0,
            3000.0, // c_l < c_t - invalid!
            3240.0,
            [1.0, 0.0, 0.0],
            64,
            64,
            64,
        );
        assert!(interface.is_err());
    }
}
