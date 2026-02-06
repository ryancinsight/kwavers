//! Electromagnetic field structures and energy calculations
//!
//! This module defines the field components and energy-related calculations
//! for electromagnetic wave propagation.

use ndarray::ArrayD;

/// Electromagnetic field components
#[derive(Debug, Clone)]
pub struct EMFields {
    /// Electric field E (V/m) - [Ex, Ey, Ez] or [Ex, Ey] for 2D
    pub electric: ArrayD<f64>,
    /// Magnetic field H (A/m) - [Hx, Hy, Hz] or [Hx, Hy] for 2D
    pub magnetic: ArrayD<f64>,
    /// Electric displacement D (C/m²)
    pub displacement: Option<ArrayD<f64>>,
    /// Magnetic flux density B (T)
    pub flux_density: Option<ArrayD<f64>>,
}

impl EMFields {
    /// Create new EM fields from electric and magnetic components
    pub fn new(electric: ArrayD<f64>, magnetic: ArrayD<f64>) -> Self {
        Self {
            electric,
            magnetic,
            displacement: None,
            flux_density: None,
        }
    }

    /// Create fields with auxiliary components
    pub fn with_auxiliary(
        electric: ArrayD<f64>,
        magnetic: ArrayD<f64>,
        displacement: ArrayD<f64>,
        flux_density: ArrayD<f64>,
    ) -> Self {
        Self {
            electric,
            magnetic,
            displacement: Some(displacement),
            flux_density: Some(flux_density),
        }
    }

    /// Validate field shapes are consistent
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
    pub vector: ArrayD<f64>,
    /// Magnitude |S| (W/m²)
    pub magnitude: ArrayD<f64>,
    /// Energy density u = (1/2)(ε|E|² + μ|H|²) (J/m³)
    pub energy_density: ArrayD<f64>,
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
    pub fn from_fields(
        electric: &ArrayD<f64>,
        magnetic: &ArrayD<f64>,
        permittivity: f64,
        permeability: f64,
    ) -> Result<Self, String> {
        let shape = electric.shape();
        if shape != magnetic.shape() {
            return Err("Electric and magnetic field shapes must match".to_string());
        }

        let ndim = shape.len();
        if ndim < 2 {
            return Err("Fields must be at least 2D".to_string());
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
        let mut s_vector = ArrayD::zeros(ndarray::IxDyn(&s_vector_shape));
        let mut s_magnitude = ArrayD::zeros(ndarray::IxDyn(&shape[..ndim - 1]));
        let mut energy_density = ArrayD::zeros(ndarray::IxDyn(&shape[..ndim - 1]));

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

                let sz = ex * hy - ey * hx;
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

                let sx = ey * hz - ez * hy;
                let sy = ez * hx - ex * hz;
                let sz = ex * hy - ey * hx;

                let mut s_idx0 = spatial.to_vec();
                s_idx0.push(0);
                let mut s_idx1 = spatial.to_vec();
                s_idx1.push(1);
                let mut s_idx2 = spatial.to_vec();
                s_idx2.push(2);
                s_vector[&s_idx0[..]] = sx;
                s_vector[&s_idx1[..]] = sy;
                s_vector[&s_idx2[..]] = sz;

                s_magnitude[spatial] = (sx * sx + sy * sy + sz * sz).sqrt();
            }

            // Energy density: u = (1/2)(ε₀ε_r|E|² + μ₀μ_r|H|²)
            let e_squared: f64 = if field_components == 2 {
                let mut e_idx0 = spatial.to_vec();
                e_idx0.push(0);
                let mut e_idx1 = spatial.to_vec();
                e_idx1.push(1);
                let ex = electric[&e_idx0[..]];
                let ey = electric[&e_idx1[..]];
                ex * ex + ey * ey
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
                ex * ex + ey * ey + ez * ez
            };

            let h_squared: f64 = if field_components == 2 {
                let mut h_idx0 = spatial.to_vec();
                h_idx0.push(0);
                let mut h_idx1 = spatial.to_vec();
                h_idx1.push(1);
                let hx = magnetic[&h_idx0[..]];
                let hy = magnetic[&h_idx1[..]];
                hx * hx + hy * hy
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
                hx * hx + hy * hy + hz * hz
            };

            let epsilon0 = 8.854e-12;
            let mu0 = 4.0 * std::f64::consts::PI * 1e-7;

            energy_density[spatial] =
                0.5 * (epsilon0 * permittivity * e_squared + mu0 * permeability * h_squared);
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

        let spatial_len = self.vector.len() / components;
        if spatial_len == 0 {
            return 0.0;
        }

        let normal_len = surface_normal.len().min(components);
        let mut sum_s_dot_n = 0.0f64;

        for lane in self.vector.lanes(ndarray::Axis(ndim - 1)) {
            let dot = lane
                .iter()
                .zip(surface_normal.iter())
                .take(normal_len)
                .map(|(s, n)| s * n)
                .sum::<f64>();
            sum_s_dot_n += dot;
        }

        let avg_s_dot_n = sum_s_dot_n / spatial_len as f64;
        avg_s_dot_n * surface_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_em_fields_creation() {
        let electric = ArrayD::zeros(ndarray::IxDyn(&[10, 10, 2]));
        let magnetic = ArrayD::zeros(ndarray::IxDyn(&[10, 10, 2]));

        let fields = EMFields::new(electric, magnetic);
        assert!(fields.validate_shapes().is_ok());
    }

    #[test]
    fn test_poynting_vector_2d() {
        // Simple 2D case: E = [1, 0], H = [0, 1] → S_z = 1*1 - 0*0 = 1
        let mut electric = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2]));
        let mut magnetic = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2]));

        electric[[0, 0, 0]] = 1.0; // Ex = 1
        magnetic[[0, 0, 1]] = 1.0; // Hy = 1

        let poynting = PoyntingVector::from_fields(&electric, &magnetic, 1.0, 1.0).unwrap();

        // For 2D, S should be [0, 0, 1] with magnitude 1
        assert_eq!(poynting.magnitude[[0, 0]], 1.0);
    }

    #[test]
    fn test_field_validation() {
        let electric = ArrayD::zeros(ndarray::IxDyn(&[5, 5, 3]));
        let magnetic = ArrayD::zeros(ndarray::IxDyn(&[5, 5, 2])); // Wrong shape

        let fields = EMFields::new(electric, magnetic);
        assert!(fields.validate_shapes().is_err());
    }
}
