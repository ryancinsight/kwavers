// src/solver/amr/wavelet.rs
//! Wavelet transforms for AMR error estimation
//! 
//! Implements various wavelet transforms for detecting regions
//! requiring refinement based on solution smoothness.

use crate::error::KwaversResult;
use ndarray::{Array1, Array3, s};
use super::WaveletType;

/// Wavelet transform for error estimation
#[derive(Debug)]
pub struct WaveletTransform {
    wavelet_type: WaveletType,
    filter_bank: FilterBank,
}

/// Filter bank for wavelet transforms
#[derive(Debug)]
struct FilterBank {
    /// Low-pass decomposition filter
    h0: Array1<f64>,
    /// High-pass decomposition filter  
    h1: Array1<f64>,
    /// Low-pass reconstruction filter
    g0: Array1<f64>,
    /// High-pass reconstruction filter
    g1: Array1<f64>,
}

impl WaveletTransform {
    /// Create a new wavelet transform
    pub fn new(wavelet_type: WaveletType) -> Self {
        let filter_bank = match wavelet_type {
            WaveletType::Haar => FilterBank::haar(),
            WaveletType::Daubechies4 => FilterBank::daubechies4(),
            WaveletType::Daubechies6 => FilterBank::daubechies6(),
            WaveletType::Coiflet6 => FilterBank::coiflet6(),
        };
        
        Self {
            wavelet_type,
            filter_bank,
        }
    }
    
    /// Compute wavelet coefficients for a 3D field
    pub fn forward_transform(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut coeffs = field.clone();
        
        // Apply 1D transforms along each dimension
        // X-direction
        for j in 0..ny {
            for k in 0..nz {
                let slice_len = coeffs.dim().0;
                let mut temp = vec![0.0; slice_len];
                for i in 0..slice_len {
                    temp[i] = coeffs[[i, j, k]];
                }
                self.dwt_1d_forward(&mut temp);
                for i in 0..slice_len {
                    coeffs[[i, j, k]] = temp[i];
                }
            }
        }
        
        // Y-direction
        for i in 0..nx {
            for k in 0..nz {
                let slice_len = coeffs.dim().1;
                let mut temp = vec![0.0; slice_len];
                for j in 0..slice_len {
                    temp[j] = coeffs[[i, j, k]];
                }
                self.dwt_1d_forward(&mut temp);
                for j in 0..slice_len {
                    coeffs[[i, j, k]] = temp[j];
                }
            }
        }
        
        // Z-direction
        for i in 0..nx {
            for j in 0..ny {
                let slice_len = coeffs.dim().2;
                let mut temp = vec![0.0; slice_len];
                for k in 0..slice_len {
                    temp[k] = coeffs[[i, j, k]];
                }
                self.dwt_1d_forward(&mut temp);
                for k in 0..slice_len {
                    coeffs[[i, j, k]] = temp[k];
                }
            }
        }
        
        Ok(coeffs)
    }
    
    /// Compute detail coefficients magnitude (error indicator)
    pub fn detail_magnitude(&self, coeffs: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = coeffs.dim();
        let mut magnitude = Array3::zeros((nx, ny, nz));
        
        // Compute magnitude of high-frequency components
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Consider a coefficient as detail if any dimension index is in high-freq band
                    let is_detail = i >= nx/2 || j >= ny/2 || k >= nz/2;
                    
                    if is_detail {
                        magnitude[[i, j, k]] = coeffs[[i, j, k]].abs();
                    }
                }
            }
        }
        
        magnitude
    }
    
    /// 1D forward discrete wavelet transform
    fn dwt_1d_forward(&self, data: &mut [f64]) {
        let n = data.len();
        if n < 2 {
            return;
        }
        
        let mut temp = vec![0.0; n];
        let h0 = &self.filter_bank.h0;
        let h1 = &self.filter_bank.h1;
        let filter_len = h0.len();
        
        // Apply filters
        for i in 0..n/2 {
            let mut sum_low = 0.0;
            let mut sum_high = 0.0;
            
            for j in 0..filter_len {
                let idx = (2 * i + j) % n;
                sum_low += h0[j] * data[idx];
                sum_high += h1[j] * data[idx];
            }
            
            temp[i] = sum_low;
            temp[n/2 + i] = sum_high;
        }
        
        // Copy back
        data.copy_from_slice(&temp);
    }
    
    /// Multi-level wavelet decomposition for better error estimation
    pub fn multi_level_decomposition(
        &self,
        field: &Array3<f64>,
        levels: usize,
    ) -> KwaversResult<Vec<Array3<f64>>> {
        let mut decompositions = Vec::with_capacity(levels);
        let mut current = field.clone();
        
        for level in 0..levels {
            // Get current level size
            let (nx, ny, nz) = current.dim();
            let new_nx = nx / 2_usize.pow(level as u32);
            let new_ny = ny / 2_usize.pow(level as u32);
            let new_nz = nz / 2_usize.pow(level as u32);
            
            if new_nx < 2 || new_ny < 2 || new_nz < 2 {
                break;
            }
            
            // Apply transform to low-frequency part only
            let mut low_freq = current.slice(s![..new_nx, ..new_ny, ..new_nz]).to_owned();
            let coeffs = self.forward_transform(&low_freq)?;
            
            decompositions.push(coeffs);
        }
        
        Ok(decompositions)
    }
}

impl FilterBank {
    /// Haar wavelet filters
    fn haar() -> Self {
        let sqrt2 = 2.0_f64.sqrt();
        Self {
            h0: Array1::from_vec(vec![1.0/sqrt2, 1.0/sqrt2]),
            h1: Array1::from_vec(vec![1.0/sqrt2, -1.0/sqrt2]),
            g0: Array1::from_vec(vec![1.0/sqrt2, 1.0/sqrt2]),
            g1: Array1::from_vec(vec![1.0/sqrt2, -1.0/sqrt2]),
        }
    }
    
    /// Daubechies 4 wavelet filters
    fn daubechies4() -> Self {
        let sqrt2 = 2.0_f64.sqrt();
        let sqrt3 = 3.0_f64.sqrt();
        
        let h0_vals = vec![
            (1.0 + sqrt3) / (4.0 * sqrt2),
            (3.0 + sqrt3) / (4.0 * sqrt2),
            (3.0 - sqrt3) / (4.0 * sqrt2),
            (1.0 - sqrt3) / (4.0 * sqrt2),
        ];
        
        let h1_vals = vec![
            h0_vals[3],
            -h0_vals[2],
            h0_vals[1],
            -h0_vals[0],
        ];
        
        Self {
            h0: Array1::from_vec(h0_vals.clone()),
            h1: Array1::from_vec(h1_vals),
            g0: Array1::from_vec(h0_vals.clone()),
            g1: Array1::from_vec(h0_vals.into_iter().rev().collect()),
        }
    }
    
    /// Daubechies 6 wavelet filters
    fn daubechies6() -> Self {
        // Daubechies 6 coefficients
        let h0_vals = vec![
            0.332670552950083,
            0.806891509311093,
            0.459877502118491,
            -0.135011020010255,
            -0.085441273882027,
            0.035226291885710,
        ];
        
        let h1_vals: Vec<f64> = h0_vals.iter()
            .enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
            .rev()
            .collect();
        
        Self {
            h0: Array1::from_vec(h0_vals.clone()),
            h1: Array1::from_vec(h1_vals),
            g0: Array1::from_vec(h0_vals.clone()),
            g1: Array1::from_vec(h0_vals.into_iter().rev().collect()),
        }
    }
    
    /// Coiflet 6 wavelet filters
    fn coiflet6() -> Self {
        // Coiflet 6 coefficients (symmetric-like)
        let h0_vals = vec![
            -0.015655728135410,
            -0.072732619512854,
            0.384864846864204,
            0.852572020212255,
            0.337897662457809,
            -0.072732619512854,
        ];
        
        let h1_vals: Vec<f64> = h0_vals.iter()
            .enumerate()
            .map(|(i, &v)| if i % 2 == 0 { -v } else { v })
            .rev()
            .collect();
        
        Self {
            h0: Array1::from_vec(h0_vals.clone()),
            h1: Array1::from_vec(h1_vals),
            g0: Array1::from_vec(h0_vals.clone()),
            g1: Array1::from_vec(h0_vals.into_iter().rev().collect()),
        }
    }
}

/// Compute smoothness indicator using wavelet coefficients
pub fn smoothness_indicator(coeffs: &Array3<f64>) -> f64 {
    let (nx, ny, nz) = coeffs.dim();
    let mut detail_energy = 0.0;
    let mut total_energy = 0.0;
    
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let val = coeffs[[i, j, k]];
                let energy = val * val;
                total_energy += energy;
                
                // High-frequency components
                if i >= nx/2 || j >= ny/2 || k >= nz/2 {
                    detail_energy += energy;
                }
            }
        }
    }
    
    if total_energy > 0.0 {
        detail_energy / total_energy
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    
    #[test]
    fn test_haar_transform() {
        let transform = WaveletTransform::new(WaveletType::Haar);
        
        // Create simple test field
        let mut field = Array3::zeros((4, 4, 4));
        field[[0, 0, 0]] = 1.0;
        field[[1, 1, 1]] = 1.0;
        
        // Apply transform
        let coeffs = transform.forward_transform(&field).unwrap();
        
        // Check that we get non-zero coefficients
        assert!(coeffs.iter().any(|&x| x != 0.0));
    }
    
    #[test]
    fn test_smoothness_detection() {
        let transform = WaveletTransform::new(WaveletType::Daubechies4);
        
        // Smooth field (low detail)
        let smooth = Array3::from_shape_fn((8, 8, 8), |(i, j, k)| {
            (i as f64 / 8.0) + (j as f64 / 8.0) + (k as f64 / 8.0)
        });
        
        // Sharp field (high detail)
        let sharp = Array3::from_shape_fn((8, 8, 8), |(i, j, k)| {
            if i < 4 && j < 4 && k < 4 { 1.0 } else { 0.0 }
        });
        
        let smooth_coeffs = transform.forward_transform(&smooth).unwrap();
        let sharp_coeffs = transform.forward_transform(&sharp).unwrap();
        
        let smooth_indicator = smoothness_indicator(&smooth_coeffs);
        let sharp_indicator = smoothness_indicator(&sharp_coeffs);
        
        // Sharp field should have higher detail coefficient ratio
        assert!(sharp_indicator > smooth_indicator);
    }
}