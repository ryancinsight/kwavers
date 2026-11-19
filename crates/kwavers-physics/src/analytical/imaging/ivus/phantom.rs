use crate::analytical::imaging::ivus::geometry::angle_difference;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Rust-owned deterministic IVUS vessel phantom used by the Chapter 30 book path.
///
/// The arrays are row-major `(n, n)` grids over a square field of view. Labels:
/// `0` background, `1` catheter, `2` lumen, `3` vessel wall, `4` plaque,
/// `5` fibrous cap, `6` lipid core, `7` calcium.
#[derive(Debug, Clone)]
pub struct IvusVesselPhantom {
    /// Cartesian x coordinate `m`.
    pub x_m: Vec<f64>,
    /// Cartesian y coordinate `m`.
    pub y_m: Vec<f64>,
    /// Radius from catheter center `m`.
    pub radius_m: Vec<f64>,
    /// Polar angle `rad`.
    pub theta_rad: Vec<f64>,
    /// Tissue labels.
    pub labels: Vec<u8>,
    /// Sound speed [m/s].
    pub sound_speed_m_s: Vec<f64>,
    /// Density [kg/m^3].
    pub density_kg_m3: Vec<f64>,
    /// Attenuation [dB/(cm MHz)].
    pub attenuation_db_cm_mhz: Vec<f64>,
    /// Normalized reflectivity/backscatter amplitude.
    pub backscatter: Vec<f64>,
    /// Lumen mask.
    pub lumen_mask: Vec<bool>,
    /// External elastic lamina mask.
    pub eel_mask: Vec<bool>,
    /// Plaque mask.
    pub plaque_mask: Vec<bool>,
    /// Fibrous-cap mask.
    pub fibrous_cap_mask: Vec<bool>,
    /// Lipid-core mask.
    pub lipid_mask: Vec<bool>,
    /// Calcium mask.
    pub calcium_mask: Vec<bool>,
}

/// Generate a deterministic synthetic IVUS vessel phantom.
///
/// # Errors
/// Returns an error when `n < 2`, `fov_m <= 0`, or `catheter_radius_m <= 0`.
pub fn ivus_vessel_phantom(
    n: usize,
    fov_m: f64,
    catheter_radius_m: f64,
    therapy_azimuth_rad: f64,
    seed: u64,
) -> Result<IvusVesselPhantom, String> {
    if n < 2 {
        return Err("IVUS phantom grid size must be at least 2".to_owned());
    }
    if fov_m <= 0.0 {
        return Err("IVUS phantom field of view must be positive".to_owned());
    }
    if catheter_radius_m <= 0.0 {
        return Err("IVUS catheter radius must be positive".to_owned());
    }

    let len = n * n;
    let mut x_m = Vec::with_capacity(len);
    let mut y_m = Vec::with_capacity(len);
    let mut radius_m = Vec::with_capacity(len);
    let mut theta_rad = Vec::with_capacity(len);
    let mut labels = vec![0_u8; len];
    let mut sound_speed_m_s = vec![343.0; len];
    let mut density_kg_m3 = vec![1.2; len];
    let mut attenuation_db_cm_mhz = vec![0.02; len];
    let mut lumen_mask = vec![false; len];
    let mut eel_mask = vec![false; len];
    let mut plaque_mask = vec![false; len];
    let mut fibrous_cap_mask = vec![false; len];
    let mut lipid_mask = vec![false; len];
    let mut calcium_mask = vec![false; len];

    let spacing = fov_m / (n - 1) as f64;
    for row in 0..n {
        let x = -0.5 * fov_m + row as f64 * spacing;
        for col in 0..n {
            let idx = row * n + col;
            let y = -0.5 * fov_m + col as f64 * spacing;
            let radius = x.hypot(y);
            let theta = y.atan2(x);
            let plaque_angle = angle_difference(theta, therapy_azimuth_rad);
            let plaque_weight = (-0.5 * (plaque_angle / 0.60).powi(2)).exp();
            let lumen_boundary =
                1.70e-3 - 0.45e-3 * plaque_weight + 0.08e-3 * (3.0 * theta + 0.30).cos();
            let eel_boundary =
                3.20e-3 + 0.22e-3 * plaque_weight + 0.10e-3 * (2.0 * theta - 0.40).sin();

            let catheter = radius <= catheter_radius_m;
            let lumen = radius > catheter_radius_m && radius <= lumen_boundary;
            let wall = radius > lumen_boundary && radius <= eel_boundary;
            let plaque = wall && plaque_weight > 0.28;
            let cap = plaque && radius - lumen_boundary < 0.24e-3;
            let lipid = plaque && radius - lumen_boundary > 0.45e-3 && plaque_weight > 0.62;
            let calcium_angle = angle_difference(theta, 1.35);
            let calcium = wall && calcium_angle.abs() < 0.20 && radius > eel_boundary - 0.42e-3;
            let eel = radius <= eel_boundary;

            x_m.push(x);
            y_m.push(y);
            radius_m.push(radius);
            theta_rad.push(theta);
            lumen_mask[idx] = lumen;
            eel_mask[idx] = eel;
            plaque_mask[idx] = plaque;
            fibrous_cap_mask[idx] = cap;
            lipid_mask[idx] = lipid;
            calcium_mask[idx] = calcium;

            if catheter {
                labels[idx] = 1;
            }
            if lumen {
                labels[idx] = 2;
                sound_speed_m_s[idx] = 1570.0;
                density_kg_m3[idx] = 1060.0;
                attenuation_db_cm_mhz[idx] = 0.12;
            }
            if wall {
                labels[idx] = 3;
                sound_speed_m_s[idx] = 1585.0;
                density_kg_m3[idx] = 1080.0;
                attenuation_db_cm_mhz[idx] = 0.65;
            }
            if plaque {
                labels[idx] = 4;
                sound_speed_m_s[idx] = 1520.0;
                density_kg_m3[idx] = 1040.0;
                attenuation_db_cm_mhz[idx] = 0.95;
            }
            if cap {
                labels[idx] = 5;
                sound_speed_m_s[idx] = 1630.0;
                density_kg_m3[idx] = 1120.0;
                attenuation_db_cm_mhz[idx] = 0.70;
            }
            if lipid {
                labels[idx] = 6;
                sound_speed_m_s[idx] = 1450.0;
                density_kg_m3[idx] = 980.0;
                attenuation_db_cm_mhz[idx] = 1.15;
            }
            if calcium {
                labels[idx] = 7;
                sound_speed_m_s[idx] = 2900.0;
                density_kg_m3[idx] = 1850.0;
                attenuation_db_cm_mhz[idx] = 4.0;
            }
        }
    }

    let interface_echo = normalized_impedance_gradient(n, &sound_speed_m_s, &density_kg_m3);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut backscatter = vec![0.0; len];
    for idx in 0..len {
        let tissue_weight = if lumen_mask[idx] {
            0.15
        } else if lipid_mask[idx] {
            0.55
        } else if eel_mask[idx] && !lumen_mask[idx] {
            0.80
        } else {
            0.0
        };
        let u = rng.gen::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
        let speckle = 0.18 * (-2.0 * u.ln()).sqrt();
        backscatter[idx] = interface_echo[idx] + speckle * tissue_weight;
        if labels[idx] == 1 {
            backscatter[idx] = 0.0;
        }
    }
    let max_backscatter = backscatter.iter().copied().fold(1.0_f64, f64::max);
    for value in &mut backscatter {
        *value /= max_backscatter;
    }

    Ok(IvusVesselPhantom {
        x_m,
        y_m,
        radius_m,
        theta_rad,
        labels,
        sound_speed_m_s,
        density_kg_m3,
        attenuation_db_cm_mhz,
        backscatter,
        lumen_mask,
        eel_mask,
        plaque_mask,
        fibrous_cap_mask,
        lipid_mask,
        calcium_mask,
    })
}

fn normalized_impedance_gradient(n: usize, sound_speed: &[f64], density: &[f64]) -> Vec<f64> {
    let impedance: Vec<f64> = sound_speed
        .iter()
        .zip(density)
        .map(|(&c, &rho)| c * rho)
        .collect();
    let mut gradient = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            let idx = row * n + col;
            let gx = if row == 0 {
                impedance[(row + 1) * n + col] - impedance[idx]
            } else if row + 1 == n {
                impedance[idx] - impedance[(row - 1) * n + col]
            } else {
                0.5 * (impedance[(row + 1) * n + col] - impedance[(row - 1) * n + col])
            };
            let gy = if col == 0 {
                impedance[row * n + col + 1] - impedance[idx]
            } else if col + 1 == n {
                impedance[idx] - impedance[row * n + col - 1]
            } else {
                0.5 * (impedance[row * n + col + 1] - impedance[row * n + col - 1])
            };
            gradient[idx] = gx.hypot(gy);
        }
    }
    let max_gradient = gradient.iter().copied().fold(1.0_f64, f64::max);
    for value in &mut gradient {
        *value /= max_gradient;
    }
    gradient
}

#[cfg(test)]
mod tests;
