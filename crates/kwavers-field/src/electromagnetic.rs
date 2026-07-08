//! Electromagnetic field structures and energy calculations
//!
//! This module defines the field components and energy-related calculations
//! for electromagnetic wave propagation.

use kwavers_core::constants::fundamental::{VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY};
use leto::{ArrayD, VecStorage};

/// Electromagnetic field components
#[derive(Debug, Clone)]
pub struct EMFields {
    /// Electric field E (V/m) - [Ex, Ey, Ez] or [Ex, Ey] for 2D
    pub electric: ArrayD<f64, VecStorage<f64>>,
    /// Magnetic field H (A/m) - [Hx, Hy, Hz] or [Hx, Hy] for 2D
    pub magnetic: ArrayD<f64, VecStorage<f64>>,
    /// Electric displacement D (C/m²)
    pub displacement: Option<ArrayD<f64, VecStorage<f64>>>,
    /// Magnetic flux density B (T)
    pub flux_density: Option<ArrayD<f64, VecStorage<f64>>>,
}

impl EMFields {
    /// Create new EM fields from electric and magnetic components
    #[must_use]
    pub fn new(electric: ArrayD<f64, VecStorage<f64>>, magnetic: ArrayD<f64, VecStorage<f64>>) -> Self {
        Self {
            electric,
            magnetic,
            displacement: None,
            flux_density: None,
        }
    }

    /// Create fields with auxiliary components
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_auxiliary(
        electric: ArrayD<f64, VecStorage<f64>>,
        magnetic: ArrayD<f64, VecStorage<f64>>,
        displacement: ArrayD<f64, VecStorage<f64>>,
        flux_density: ArrayD<f64, VecStorage<f64>>,
    ) -> Self {
        Self {
            electric,
            magnetic,
            displacement: Some(displacement),
            flux_density: Some(flux_density),
        }
    }

    /// Validate field shapes are consistent
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn validate_shapes(&self) -> Result<(), String> {
        let e_shape = self.electric.shape();
        let h_shape = self.magnetic.shape();

        if e_shape != h_shape {
            return Err(format!(
                "Electric field shape {:?} does not match magnetic field shape {:?}",
                e_shape, h_shape
            ));
        }

        if let Some(ref d) = self.displacement {
            if d.shape() != e_shape {
                return Err(format!(
                    "Displacement field shape {:?} does not match electric field shape {:?}",
                    d.shape(),
                    e_shape
                ));
            }
        }

        if let Some(ref b) = self.flux_density {
            if b.shape() != h_shape {
                return Err(format!(
                    "Flux density shape {:?} does not match magnetic field shape {:?}",
                    b.shape(),
                    h_shape
                ));
            }
        }

        Ok(())
    }
}

/// Poynting vector (energy flux density)
#[derive(Debug, Clone)]
pub struct PoyntingVector {
    /// S = E × H (W/m²) - [Sx, Sy, Sz]
    pub vector: ArrayD<f64, VecStorage<f64>>,
    /// Magnitude |S| (W/m²)
    pub magnitude: ArrayD<f64, VecStorage<f64>>,
    /// Energy density u = (1/2)(ε|E|² + μ|H|²) (J/m³)
    pub energy_density: ArrayD<f64, VecStorage<f64>>,
}

impl PoyntingVector {
    /// Compute Poynting vector from EM fields
    ///
    /// S = E × H (in direction of energy flow)
    ///
    /// # Arguments
    ///
    /// * `electric` - Electric field E [Nx, Ny, 2] or [Nx, Ny, Nz, 3]
    /// * `magnetic` - Magnetic field H [Nx, Ny, 2] or [Nx, Ny, Nz, 3]
    /// * `permittivity` - Relative permittivity ε_r
    /// * `permeability` - Relative permeability μ_r
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn from_fields(
        electric: &ArrayD<f64, VecStorage<f64>>,
        magnetic: &ArrayD<f64, VecStorage<f64>>,
        permittivity: f64,
        permeability: f64,
    ) -> Result<Self, String> {
        let shape = electric.shape();
        if shape != magnetic.shape() {
            return Err("Electric and magnetic field shapes must match".to_owned());
        }

        let ndim = shape.len();
        if ndim < 2 {
            return Err("Fields must be at least 2D".to_owned());
        }

        let field_components = *shape.last().unwrap();
        if field_components != 2 && field_components != 3 {
            return Err(format!(
                "Last dimension must be 2 or 3 for field components, got {}",
                field_components
            ));
        }

        let mut s_vector_shape = shape[..ndim - 1].to_vec();
        s_vector_shape.push(3);
        let mut s_vector =
            ArrayD::<f64, VecStorage<f64>>::zeros(&s_vector_shape).map_err(|e| e.to_string())?;
        let mut s_magnitude =
            ArrayD::<f64, VecStorage<f64>>::zeros(&shape[..ndim - 1]).map_err(|e| e.to_string())?;
        let mut energy_density =
            ArrayD::<f64, VecStorage<f64>>::zeros(&shape[..ndim - 1]).map_err(|e| e.to_string())?;

        // For each spatial point
        let spatial_size = shape.iter().take(ndim - 1).product();
        for idx in 0..spatial_size {
            let mut indices = vec![0; ndim];
            let mut temp = idx;

            // Convert linear index to multi-dimensional indices
            for d in (0..ndim - 1).rev() {
                indices[d] = temp % shape[d];
                temp /= shape[d];
            }

            let spatial = &indices[..ndim - 1];

            if field_components == 2 {
                let mut e_idx0 = spatial.to_vec();
                e_idx0.push(0);
                let mut e_idx1 = spatial.to_vec();
                e_idx1.push(1);
                let mut h_idx0 = spatial.to_vec();
                h_idx0.push(0);
                let mut h_idx1 = spatial.to_vec();
                h_idx1.push(1);

                let ex = electric[&e_idx0[..]];
                let ey = electric[&e_idx1[..]];
                let hx = magnetic[&h_idx0[..]];
                let hy = magnetic[&h_idx1[..]];

                let sz = ex.mul_add(hy, -(ey * hx));
                let mut s_idx0 = spatial.to_vec();
                s_idx0.push(0);
                let mut s_idx1 = spatial.to_vec();
                s_idx1.push(1);
                let mut s_idx2 = spatial.to_vec();
                s_idx2.push(2);
                s_vector[&s_idx0[..]] = 0.0;
                s_vector[&s_idx1[..]] = 0.0;
                s_vector[&s_idx2[..]] = sz;

                s_magnitude[spatial] = sz.abs();
            } else {
                let mut e_idx0 = spatial.to_vec();
                e_idx0.push(0);
                let mut e_idx1 = spatial.to_vec();
                e_idx1.push(1);
                let mut e_idx2 = spatial.to_vec();
                e_idx2.push(2);
                let mut h_idx0 = spatial.to_vec();
                h_idx0.push(0);
                let mut h_idx1 = spatial.to_vec();
                h_idx1.push(1);
                let mut h_idx2 = spatial.to_vec();
                h_idx2.push(2);

                let ex = electric[&e_idx0[..]];
                let ey = electric[&e_idx1[..]];
                let ez = electric[&e_idx2[..]];
                let hx = magnetic[&h_idx0[..]];
                let hy = magnetic[&h_idx1[..]];
                let hz = magnetic[&h_idx2[..]];

                let sx = ey.mul_add(hz, -(ez * hy));
                let sy = ez.mul_add(hx, -(ex * hz));
                let sz = ex.mul_add(hy, -(ey * hx));

                let mut s_idx0 = spatial.to_vec();
                s_idx0.push(0);
                let mut s_idx1 = spatial.to_vec();
                s_idx1.push(1);
                let mut s_idx2 = spatial.to_vec();
                s_idx2.push(2);
                s_vector[&s_idx0[..]] = sx;
                s_vector[&s_idx1[..]] = sy;
                s_vector[&s_idx2[..]] = sz;

                s_magnitude[spatial] = sz.mul_add(sz, sx.mul_add(sx, sy * sy)).sqrt();
            }

            // Energy density: u = (1/2)(ε₀ε_r|E|² + μ₀μ_r|H|²)
            let e_squared: f64 = if field_components == 2 {
                let mut e_idx0 = spatial.to_vec();
                e_idx0.push(0);
                let mut e_idx1 = spatial.to_vec();
                e_idx1.push(1);
                let ex = electric[&e_idx0[..]];
                let ey = electric[&e_idx1[..]];
                ex.mul_add(ex, ey * ey)
            } else {
                let mut e_idx0 = spatial.to_vec();
                e_idx0.push(0);
                let mut e_idx1 = spatial.to_vec();
                e_idx1.push(1);
                let mut e_idx2 = spatial.to_vec();
                e_idx2.push(2);
                let ex = electric[&e_idx0[..]];
                let ey = electric[&e_idx1[..]];
                let ez = electric[&e_idx2[..]];
                ez.mul_add(ez, ex.mul_add(ex, ey * ey))
            };

            let h_squared: f64 = if field_components == 2 {
                let mut h_idx0 = spatial.to_vec();
                h_idx0.push(0);
                let mut h_idx1 = spatial.to_vec();
                h_idx1.push(1);
                let hx = magnetic[&h_idx0[..]];
                let hy = magnetic[&h_idx1[..]];
                hx.mul_add(hx, hy * hy)
            } else {
                let mut h_idx0 = spatial.to_vec();
                h_idx0.push(0);
                let mut h_idx1 = spatial.to_vec();
                h_idx1.push(1);
                let mut h_idx2 = spatial.to_vec();
                h_idx2.push(2);
                let hx = magnetic[&h_idx0[..]];
                let hy = magnetic[&h_idx1[..]];
                let hz = magnetic[&h_idx2[..]];
                hz.mul_add(hz, hx.mul_add(hx, hy * hy))
            };

            // u_em = ½(ε E² + μ H²) with ε = ε₀·ε_r and μ = μ₀·μ_r (SI).
            energy_density[spatial] = 0.5
                * (VACUUM_PERMITTIVITY * permittivity)
                    .mul_add(e_squared, VACUUM_PERMEABILITY * permeability * h_squared);
        }

        Ok(Self {
            vector: s_vector,
            magnitude: s_magnitude,
            energy_density,
        })
    }

    /// Get total power flow through a surface
    ///
    /// Computes ∫ S · dA over the specified surface
    #[must_use]
    pub fn total_power(&self, surface_normal: &[f64], surface_area: f64) -> f64 {
        let ndim = self.vector.ndim();
        let components = if ndim >= 1 {
            self.vector.shape()[ndim - 1]
        } else {
            0
        };

        if components == 0 || surface_normal.is_empty() || surface_area <= 0.0 {
            return 0.0;
        }

        let spatial_len = self.vector.size() / components;
        if spatial_len == 0 {
            return 0.0;
        }

        let normal_len = surface_normal.len().min(components);

        // Iterate over each spatial position and compute S · n̂.
        let spatial_shape = &self.vector.shape()[..ndim - 1];
        let mut sum_s_dot_n = 0.0f64;
        let mut spatial_idx = vec![0usize; ndim - 1];
        for _ in 0..spatial_len {
            let mut dot = 0.0;
            for comp in 0..normal_len {
                let mut idx = spatial_idx.clone();
                idx.push(comp);
                dot += self.vector[&idx[..]] * surface_normal[comp];
            }
            sum_s_dot_n += dot;

            // Increment spatial_idx in row-major order.
            let mut carry = true;
            for d in (0..ndim - 1).rev() {
                if carry {
                    spatial_idx[d] += 1;
                    if spatial_idx[d] >= spatial_shape[d] {
                        spatial_idx[d] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }

        let avg_s_dot_n = sum_s_dot_n / spatial_len as f64;
        avg_s_dot_n * surface_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_em_fields_creation() {
        let electric = ArrayD::<f64, VecStorage<f64>>::zeros(&[10, 10, 2]).unwrap();
        let magnetic = ArrayD::<f64, VecStorage<f64>>::zeros(&[10, 10, 2]).unwrap();

        let fields = EMFields::new(electric, magnetic);
        fields.validate_shapes().unwrap();
    }

    #[test]
    fn test_poynting_vector_2d() {
        // Simple 2D case: E = [1, 0], H = [0, 1] → S_z = 1*1 - 0*0 = 1
        let mut electric = ArrayD::<f64, VecStorage<f64>>::zeros(&[1, 1, 2]).unwrap();
        let mut magnetic = ArrayD::<f64, VecStorage<f64>>::zeros(&[1, 1, 2]).unwrap();

        *electric.get_mut(&[0, 0, 0]).unwrap() = 1.0; // Ex = 1
        *magnetic.get_mut(&[0, 0, 1]).unwrap() = 1.0; // Hy = 1

        let poynting = PoyntingVector::from_fields(&electric, &magnetic, 1.0, 1.0).unwrap();

        // For 2D, S_z = Ex*Hy - Ey*Hx = 1*1 - 0*0 = 1 → magnitude = 1
        assert_eq!(*poynting.magnitude.get(&[0, 0]).unwrap(), 1.0);
    }

    #[test]
    fn test_field_validation() {
        let electric = ArrayD::<f64, VecStorage<f64>>::zeros(&[5, 5, 3]).unwrap();
        let magnetic = ArrayD::<f64, VecStorage<f64>>::zeros(&[5, 5, 2]).unwrap(); // Wrong shape

        let fields = EMFields::new(electric, magnetic);
        assert!(fields.validate_shapes().is_err());
    }
}
