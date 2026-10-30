//! Continuous-wave angular-spectrum plane-to-plane propagator.
//!
//! Rust owner for the k-wave parity helper (`kwave_parity.py`): the spectral
//! propagator, Zeng–McGough absorption factor, and angular-restriction filter
//! of Zeng & McGough (2008, JASA 123:68–76), matching k-wave-python
//! `angular_spectrum_cw` semantics. Python keeps argument parsing and result
//! shaping; every floating-point operation of the propagation lives here.
//!
//! # Module layout
//!
//! - [`propagate_cw`]: forward projection of a source plane to `z` planes.
//! - [`backproject_cw`]: time-reversed reconstruction of a source plane.
//! - [`CwMedium`]: homogeneous-medium parameters (sound speed, Np/m absorption).

use apollo::{fft_2d_complex, ifft_2d_complex, Complex64};
use eunomia::FloatElement;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use leto::{Array2, Array3};

/// Homogeneous propagation medium for the CW angular-spectrum operator.
#[derive(Debug, Clone, Copy)]
pub struct CwMedium {
    /// Sound speed [m/s].
    pub sound_speed: f64,
    /// Power-law absorption at the source frequency [Np/m].
    pub alpha_np_per_m: f64,
}

impl CwMedium {
    /// Lossless medium at the given sound speed.
    #[must_use]
    pub fn lossless(sound_speed: f64) -> Self {
        Self {
            sound_speed,
            alpha_np_per_m: 0.0,
        }
    }
}

/// Reject a non-positive-finite scalar with a typed error naming it.
fn check_positive_finite(parameter: &'static str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: parameter.to_owned(),
            value: value.to_string(),
            constraint: "must be finite and positive".to_owned(),
        }))
    }
}

/// FFT length used by the propagator: twice the next power of two at or
/// above the largest padded dimension (k-wave-python:
/// `2 ** (ceil(log2(max)) + 1)`).
fn fft_len(max_dim: usize) -> usize {
    max_dim.next_power_of_two() * 2
}

/// Wavenumber-grid spacing for an FFT of length `n` at grid spacing `dx`.
fn dk(n: usize, dx: f64) -> f64 {
    TWO_PI / (f64::from_count(n) * dx)
}

/// FFT-order (zero frequency first) wavenumber coordinate of index `j`.
///
/// Mirrors the reference exactly: the centred vector holds
/// `(i - N/2)·dk`, the `N/2` entry is assigned `0.0`, and `ifftshift` moves
/// that entry to index 0 — so index 0 is exactly zero and index `N/2` keeps
/// the negative-Nyquist coordinate `-(N/2)·dk` (evanescent, decaying), never
/// zero. Zeroing the Nyquist bin instead would give it a unit propagator and
/// leak spectral energy into every pixel, growing with `z`.
fn fft_freq(j: usize, n: usize, dk: f64) -> f64 {
    if j < n / 2 {
        f64::from_count(j) * dk
    } else {
        -f64::from_count(n - j) * dk
    }
}

/// Complex axial wavenumber `kz = sqrt(k² − kx² − ky²)` (principal branch:
/// evanescent components get a positive imaginary part, as in numpy).
fn axial_wavenumber(k: f64, kx: f64, ky: f64) -> Complex64 {
    Complex64::new(k * k - kx * kx - ky * ky, 0.0).sqrt()
}

/// Zero-pad a plane by `pad` cells on every side.
fn zero_pad(input: &Array2<Complex64>, pad: usize) -> Array2<Complex64> {
    let [nx, ny] = input.shape();
    let mut padded = Array2::zeros((nx + 2 * pad, ny + 2 * pad));
    for i in 0..nx {
        for j in 0..ny {
            padded[[i + pad, j + pad]] = input[[i, j]];
        }
    }
    padded
}

/// Spectral propagator state shared by the forward and backward operators.
struct Propagator {
    n: usize,
    k: f64,
    kz: Array2<Complex64>,
    radial: Array2<f64>,
}

impl Propagator {
    /// Build the `n × n` spectral grids for spacing `dx` at wavenumber `k`.
    fn new(n: usize, dx: f64, k: f64) -> Self {
        let step = dk(n, dx);
        let mut kz = Array2::zeros((n, n));
        let mut radial = Array2::zeros((n, n));
        for i in 0..n {
            let kx = fft_freq(i, n, step);
            for j in 0..n {
                let ky = fft_freq(j, n, step);
                kz[[i, j]] = axial_wavenumber(k, kx, ky);
                radial[[i, j]] = (kx * kx + ky * ky).sqrt();
            }
        }
        Self { n, k, kz, radial }
    }

    /// Forward propagator `H = conj(exp(i·z·kz))`, with the Zeng–McGough
    /// absorption factor and optional angular restriction applied.
    fn forward(
        &self,
        z: f64,
        alpha_np_per_m: f64,
        angular_restriction: bool,
        dx: f64,
    ) -> Array2<Complex64> {
        let mut h = Array2::zeros((self.n, self.n));
        for i in 0..self.n {
            for j in 0..self.n {
                let kz = self.kz[[i, j]];
                let mut hij = (Complex64::new(0.0, z) * kz).exp().conj();
                if alpha_np_per_m != 0.0 {
                    hij *= (Complex64::new(-alpha_np_per_m * z * self.k, 0.0) / kz).exp();
                }
                h[[i, j]] = hij;
            }
        }
        if angular_restriction {
            let diameter = f64::from_count(self.n - 1) * dx;
            let kc =
                self.k * (0.5 * diameter * diameter / (0.5 * diameter * diameter + z * z)).sqrt();
            zero_beyond(&mut h, &self.radial, kc);
        }
        h
    }

    /// Backward (time-reversed) propagator `H = exp(i·z·kz)`. Without angular
    /// restriction, evanescent components beyond `k` are still suppressed
    /// rather than amplified.
    fn backward(&self, z: f64, angular_restriction: bool, dx: f64) -> Array2<Complex64> {
        let mut h = Array2::zeros((self.n, self.n));
        for i in 0..self.n {
            for j in 0..self.n {
                h[[i, j]] = (Complex64::new(0.0, z) * self.kz[[i, j]]).exp();
            }
        }
        let cutoff = if angular_restriction {
            let diameter = f64::from_count(self.n - 1) * dx;
            self.k * (0.5 * diameter * diameter / (0.5 * diameter * diameter + z * z)).sqrt()
        } else {
            self.k
        };
        zero_beyond(&mut h, &self.radial, cutoff);
        h
    }
}

/// Zero the propagator wherever the transverse wavenumber exceeds `cutoff`.
fn zero_beyond(h: &mut Array2<Complex64>, radial: &Array2<f64>, cutoff: f64) {
    let [n, _] = h.shape();
    for i in 0..n {
        for j in 0..n {
            if radial[[i, j]] > cutoff {
                h[[i, j]] = Complex64::new(0.0, 0.0);
            }
        }
    }
}

/// Validate the shared scalar inputs of both operators.
fn check_scalars(dx: f64, f0: f64, sound_speed: f64) -> KwaversResult<f64> {
    check_positive_finite("dx", dx)?;
    check_positive_finite("f0", f0)?;
    check_positive_finite("sound_speed", sound_speed)?;
    // The grid must sample the wavelength (Nyquist); otherwise the spectral
    // grid aliases propagating energy into evanescent bins.
    if dx > sound_speed / (2.0 * f0) {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "dx".to_owned(),
            value: dx.to_string(),
            constraint: "must not exceed sound_speed / (2 * f0) (Nyquist)".to_owned(),
        }));
    }
    Ok(TWO_PI * f0 / sound_speed)
}

/// Project a 2-D CW pressure plane to the requested propagation distances.
///
/// Mirrors k-wave-python `angular_spectrum_cw`: the input spectrum (zero
/// padded by `grid_expansion`, FFT length doubled past the next power of
/// two) is multiplied by the conjugate spectral propagator — plus the
/// Zeng–McGough absorption factor when `medium.alpha_np_per_m ≠ 0` and the
/// angular-restriction filter unless disabled — and returned cropped to the
/// input footprint. A `z` of exactly zero returns the (padded, cropped)
/// input plane with no FFT round-off.
///
/// # Errors
///
/// Returns `KwaversError::Config` when a scalar input is non-positive or
/// non-finite, `dx` violates Nyquist, the absorption is negative or
/// non-finite, a `z` plane is non-finite, or the input plane is empty.
#[allow(clippy::too_many_arguments)]
pub fn propagate_cw(
    input: &Array2<Complex64>,
    dx: f64,
    z_planes: &[f64],
    f0: f64,
    medium: CwMedium,
    angular_restriction: bool,
    grid_expansion: usize,
) -> KwaversResult<Array3<Complex64>> {
    let k = check_scalars(dx, f0, medium.sound_speed)?;
    if !medium.alpha_np_per_m.is_finite() || medium.alpha_np_per_m < 0.0 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "alpha_np_per_m".to_owned(),
            value: medium.alpha_np_per_m.to_string(),
            constraint: "must be finite and non-negative".to_owned(),
        }));
    }
    for (zi, z) in z_planes.iter().enumerate() {
        if !z.is_finite() {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "z_planes".to_owned(),
                value: format!("z_planes[{zi}] = {z}"),
                constraint: "every propagation distance must be finite".to_owned(),
            }));
        }
    }
    let [nx, ny] = input.shape();
    if nx == 0 || ny == 0 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "input".to_owned(),
            value: format!("shape = ({nx}, {ny})"),
            constraint: "input plane must be non-empty".to_owned(),
        }));
    }

    let padded = zero_pad(input, grid_expansion);
    let [px, py] = padded.shape();
    let n = fft_len(px.max(py));
    let propagator = Propagator::new(n, dx, k);

    // Embed the padded plane in the N×N FFT grid (top-left, as `fft2(x, (N, N))`).
    let mut grid = Array2::zeros((n, n));
    for i in 0..px {
        for j in 0..py {
            grid[[i, j]] = padded[[i, j]];
        }
    }
    let spectrum = fft_2d_complex(&grid);

    let mut out = Array3::zeros((nx, ny, z_planes.len()));
    for (zi, &z) in z_planes.iter().enumerate() {
        if z == 0.0 {
            for i in 0..nx {
                for j in 0..ny {
                    out[[i, j, zi]] = padded[[i + grid_expansion, j + grid_expansion]];
                }
            }
            continue;
        }
        let h = propagator.forward(z, medium.alpha_np_per_m, angular_restriction, dx);
        let mut product = spectrum.clone();
        for i in 0..n {
            for j in 0..n {
                product[[i, j]] *= h[[i, j]];
            }
        }
        let projected = ifft_2d_complex(&product);
        for i in 0..nx {
            for j in 0..ny {
                out[[i, j, zi]] = projected[[i + grid_expansion, j + grid_expansion]];
            }
        }
    }
    Ok(out)
}

/// Reconstruct a source plane from a CW measurement via the backward
/// (time-reversed) angular spectrum.
///
/// Mirrors k-wave-python `backward_angular_spectrum_cw`: the measurement
/// spectrum is multiplied by `exp(i·z·kz)` with evanescent suppression (the
/// angular-restriction cutoff when enabled, `k` otherwise), then cropped to
/// the measurement footprint.
///
/// # Errors
///
/// Returns `KwaversError::Config` when a scalar input is non-positive or
/// non-finite, `dx` violates Nyquist, `z_m` is non-finite, or the
/// measurement plane is empty.
pub fn backproject_cw(
    measurement: &Array2<Complex64>,
    dx: f64,
    z_m: f64,
    f0: f64,
    sound_speed: f64,
    angular_restriction: bool,
) -> KwaversResult<Array2<Complex64>> {
    let k = check_scalars(dx, f0, sound_speed)?;
    if !z_m.is_finite() {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "z_m".to_owned(),
            value: z_m.to_string(),
            constraint: "propagation distance must be finite".to_owned(),
        }));
    }
    let [nx, ny] = measurement.shape();
    if nx == 0 || ny == 0 {
        return Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "measurement".to_owned(),
            value: format!("shape = ({nx}, {ny})"),
            constraint: "measurement plane must be non-empty".to_owned(),
        }));
    }

    let n = fft_len(nx.max(ny));
    let propagator = Propagator::new(n, dx, k);
    let mut grid = Array2::zeros((n, n));
    for i in 0..nx {
        for j in 0..ny {
            grid[[i, j]] = measurement[[i, j]];
        }
    }
    let spectrum = fft_2d_complex(&grid);
    let h = propagator.backward(z_m, angular_restriction, dx);
    let mut product = spectrum.clone();
    for i in 0..n {
        for j in 0..n {
            product[[i, j]] *= h[[i, j]];
        }
    }
    let full = ifft_2d_complex(&product);
    let mut out = Array2::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            out[[i, j]] = full[[i, j]];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
