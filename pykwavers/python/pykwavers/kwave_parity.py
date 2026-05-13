"""Pure Python k-Wave parity utilities exported by :mod:`pykwavers`."""

# ============================================================================
# Pure-Python k-Wave parity utilities
# ============================================================================
# These functions match the k-Wave/k-wave-python API and are implemented here
# in pure NumPy/SciPy — no Rust binding required since they are post-processing
# utilities that operate on NumPy arrays.

import numpy as _np


def gaussian(N: int, var: float, magnitude: float = 1.0) -> "_np.ndarray":
    """Create a Gaussian distribution with unit area.

    Matches k-Wave ``makeGaussian`` semantics.

    Parameters
    ----------
    N : int
        Number of samples (should be odd for symmetric result).
    var : float
        Variance (width²) of the Gaussian.  A larger value gives a wider pulse.
    magnitude : float, optional
        Peak amplitude.  Default is 1.0.

    Returns
    -------
    numpy.ndarray
        1-D array of length *N* containing the Gaussian samples.

    References
    ----------
    Treeby & Cox (2010), k-Wave MATLAB toolbox, ``makeGaussian``.
    """
    t = _np.arange(-(N - 1) / 2.0, (N - 1) / 2.0 + 1)
    return magnitude * _np.exp(-(t ** 2) / (2.0 * var))


def spect(
    x: "_np.ndarray",
    fs: float,
    *,
    unwrap_phase: bool = False,
) -> "tuple[_np.ndarray, _np.ndarray, _np.ndarray]":
    """Single-sided amplitude spectrum via FFT.

    Matches k-Wave ``spect`` semantics.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal (1-D time series).
    fs : float
        Sampling frequency in Hz.
    unwrap_phase : bool, optional
        If True, unwrap the phase spectrum.  Default False.

    Returns
    -------
    f : numpy.ndarray
        Frequency axis in Hz (single-sided, 0 to fs/2).
    amp : numpy.ndarray
        Single-sided amplitude spectrum (peak amplitude, not RMS).
    phase : numpy.ndarray
        Phase spectrum in radians.

    References
    ----------
    Treeby & Cox (2010), k-Wave MATLAB toolbox, ``spect``.
    """
    x = _np.asarray(x, dtype=float)
    n = len(x)
    fft_x = _np.fft.rfft(x)
    # Single-sided amplitude: double non-DC / non-Nyquist bins
    amp = _np.abs(fft_x) / n
    amp[1:-1] *= 2.0
    phase = _np.angle(fft_x)
    if unwrap_phase:
        phase = _np.unwrap(phase)
    f = _np.fft.rfftfreq(n, d=1.0 / fs)
    return f, amp, phase


def extract_amp_phase(
    x: "_np.ndarray",
    f: float,
    fs: float,
    *,
    dim: int = 0,
) -> "tuple[float, float]":
    """Extract amplitude and phase of a signal at a given frequency.

    Matches k-Wave ``extractAmpPhase`` semantics (scalar output for 1-D input).

    Parameters
    ----------
    x : numpy.ndarray
        Input signal (1-D time series or multi-dimensional array).
    f : float
        Target frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    dim : int, optional
        Axis along which to compute the FFT for multi-dimensional input.
        Default 0.

    Returns
    -------
    amp : float
        Peak amplitude at frequency *f*.
    phase : float
        Phase in radians at frequency *f*.

    References
    ----------
    Treeby & Cox (2010), k-Wave MATLAB toolbox, ``extractAmpPhase``.
    """
    x = _np.asarray(x, dtype=float)
    n = x.shape[dim]
    freqs = _np.fft.rfftfreq(n, d=1.0 / fs)
    # Find nearest frequency bin
    idx = int(_np.argmin(_np.abs(freqs - f)))
    fft_x = _np.fft.rfft(x, axis=dim)
    # Extract element along the transform axis
    idx_tuple = [slice(None)] * fft_x.ndim
    idx_tuple[dim] = idx
    coeff = fft_x[tuple(idx_tuple)]
    amp = float(_np.abs(coeff)) * 2.0 / n
    phase = float(_np.angle(coeff))
    return amp, phase


def cart2grid(
    kgrid,
    cart_data: "_np.ndarray",
) -> "_np.ndarray":
    """Map Cartesian point data onto the nearest simulation grid points.

    Matches k-Wave ``cart2grid`` semantics.

    Parameters
    ----------
    kgrid : Grid
        kwavers ``Grid`` object defining the simulation domain.
    cart_data : numpy.ndarray
        Array of shape ``(3, N)`` (or ``(2, N)`` for 2-D) containing
        Cartesian (x, y, z) coordinates of the N points.

    Returns
    -------
    numpy.ndarray
        Boolean or integer mask array with shape ``(nx, ny, nz)`` where
        ``1`` marks the nearest grid voxel for each input point.

    References
    ----------
    Treeby & Cox (2010), k-Wave MATLAB toolbox, ``cart2grid``.
    """
    cart_data = _np.asarray(cart_data, dtype=float)
    nx, ny, nz = kgrid.nx, kgrid.ny, kgrid.nz
    dx, dy, dz = kgrid.dx, kgrid.dy, kgrid.dz
    mask = _np.zeros((nx, ny, nz), dtype=_np.int8)

    # Grid origin is at the centre of the domain
    x_vec = ((_np.arange(nx) - nx / 2.0) * dx)
    y_vec = ((_np.arange(ny) - ny / 2.0) * dy)
    z_vec = ((_np.arange(nz) - nz / 2.0) * dz)

    for pt_idx in range(cart_data.shape[1]):
        xi = int(_np.argmin(_np.abs(x_vec - cart_data[0, pt_idx])))
        yi = int(_np.argmin(_np.abs(y_vec - cart_data[1, pt_idx])))
        zi = int(_np.argmin(_np.abs(z_vec - cart_data[2, pt_idx]))) if cart_data.shape[0] > 2 else 0
        mask[xi, yi, zi] = 1

    return mask


def grid2cart(
    kgrid,
    grid_data: "_np.ndarray",
    cart_data: "_np.ndarray",
) -> "_np.ndarray":
    """Extract grid field values at Cartesian point positions.

    Matches k-Wave ``grid2cart`` semantics.  Uses nearest-grid-point lookup.

    Parameters
    ----------
    kgrid : Grid
        kwavers ``Grid`` object defining the simulation domain.
    grid_data : numpy.ndarray
        3-D field array of shape ``(nx, ny, nz)``.
    cart_data : numpy.ndarray
        Array of shape ``(3, N)`` containing Cartesian (x, y, z) coordinates.

    Returns
    -------
    numpy.ndarray
        1-D array of length N containing the field value at each Cartesian point.

    References
    ----------
    Treeby & Cox (2010), k-Wave MATLAB toolbox, ``grid2cart``.
    """
    cart_data = _np.asarray(cart_data, dtype=float)
    grid_data = _np.asarray(grid_data, dtype=float)
    nx, ny, nz = kgrid.nx, kgrid.ny, kgrid.nz
    dx, dy, dz = kgrid.dx, kgrid.dy, kgrid.dz

    x_vec = (_np.arange(nx) - nx / 2.0) * dx
    y_vec = (_np.arange(ny) - ny / 2.0) * dy
    z_vec = (_np.arange(nz) - nz / 2.0) * dz

    n_pts = cart_data.shape[1]
    values = _np.zeros(n_pts)
    for pt_idx in range(n_pts):
        xi = int(_np.argmin(_np.abs(x_vec - cart_data[0, pt_idx])))
        yi = int(_np.argmin(_np.abs(y_vec - cart_data[1, pt_idx])))
        zi = int(_np.argmin(_np.abs(z_vec - cart_data[2, pt_idx]))) if cart_data.shape[0] > 2 else 0
        values[pt_idx] = grid_data[xi, yi, zi]

    return values


def angular_spectrum_cw(
    input_plane: "_np.ndarray",
    dx: float,
    z_pos,
    f0: float,
    medium,
    *,
    angular_restriction: bool = True,
    grid_expansion: int = 0,
) -> "_np.ndarray":
    """Project a 2-D CW pressure plane using the angular spectrum method.

    Matches k-Wave ``angularSpectrumCW`` / k-wave-python ``angular_spectrum_cw``
    semantics. Input is the complex pressure amplitude on the source plane;
    output is the complex pressure at each requested propagation plane.

    Parameters
    ----------
    input_plane : ndarray, shape (Nx, Ny), complex
        Complex pressure amplitude on the source plane at frequency *f0* [Pa].
    dx : float
        Isotropic grid spacing of the input plane [m].
    z_pos : float or array-like of float
        Propagation distance(s) from the source plane [m].
    f0 : float
        Source frequency [Hz].
    medium : float or dict
        Sound speed [m/s] (scalar), or dict with key ``sound_speed`` (m/s) and
        optional keys ``alpha_coeff`` (dB/MHz^y/cm), ``alpha_power`` (y).
    angular_restriction : bool, optional
        Apply the angular restriction filter described in Zeng & McGough (2008).
        Default True.
    grid_expansion : int, optional
        Number of grid points to pad around the input plane (zero-padding).
        Default 0.

    Returns
    -------
    pressure : ndarray, shape (Nx, Ny, Nz), complex
        Complex pressure at each (x, y) position for each z plane in *z_pos*.
        Slice ``[:, :, 0]`` is the source plane (z=0 or z=z_pos[0]).

    References
    ----------
    Zeng & McGough (2008). "Evaluation of the angular spectrum approach for
    simulations of near-field pressures." JASA, 123(1), 68-76.
    """
    # Parse medium
    if isinstance(medium, dict):
        c0 = float(medium["sound_speed"])
        absorbing = "alpha_coeff" in medium and "alpha_power" in medium
        if absorbing:
            # Convert dB·MHz^{-y}·cm^{-1} to Np/m at f0
            alpha_coeff = medium["alpha_coeff"]
            alpha_power = medium["alpha_power"]
            # db2neper converts dB/m at 1 Hz^y; multiply by f^y to get Np/m
            alpha_neper_per_m = (
                alpha_coeff
                * (100.0)  # cm^{-1} → m^{-1}: ×100
                / (8.686)  # dB→Np
                * (1e-6 * f0) ** alpha_power  # scale by (MHz)^y
            )
        else:
            alpha_neper_per_m = 0.0
    else:
        c0 = float(medium)
        absorbing = False
        alpha_neper_per_m = 0.0

    if dx > c0 / (2.0 * f0):
        raise ValueError(
            f"dx={dx} m exceeds Nyquist limit {c0 / (2 * f0):.4g} m at f0={f0} Hz."
        )

    input_plane = _np.asarray(input_plane, dtype=complex)
    Nx, Ny = input_plane.shape
    z_pos_arr = _np.atleast_1d(_np.asarray(z_pos, dtype=float))
    Nz = len(z_pos_arr)

    # Optional zero-pad
    if grid_expansion > 0:
        pad = grid_expansion
        input_plane = _np.pad(input_plane, ((pad, pad), (pad, pad)), mode="constant")
        Nx, Ny = input_plane.shape

    # FFT length: next power of 2 above max(Nx, Ny), then doubled
    N = int(2 ** (_np.ceil(_np.log2(max(Nx, Ny))) + 1))

    # Wavenumber vector (centred, then shifted to FFT order)
    if N % 2 == 0:
        k_vec = _np.arange(-N // 2, N // 2) * (2.0 * _np.pi / (N * dx))
    else:
        k_vec = _np.arange(-(N - 1) // 2, (N - 1) // 2 + 1) * (2.0 * _np.pi / (N * dx))
    k_vec[N // 2] = 0.0  # remove floating-point round-off at DC
    k_vec = _np.fft.ifftshift(k_vec)

    k = 2.0 * _np.pi * f0 / c0

    # 2-D wavenumber grids (indexing='ij' matches Nx×Ny layout)
    ky, kx = _np.meshgrid(k_vec, k_vec, indexing="ij")
    kz = _np.sqrt((k**2 - kx**2 - ky**2).astype(complex))
    sqrt_kx2_ky2 = _np.sqrt(kx**2 + ky**2)

    pressure = _np.zeros((Nx, Ny, Nz), dtype=complex)

    input_plane_fft = _np.fft.fft2(input_plane, (N, N))

    for z_idx in range(Nz):
        z = z_pos_arr[z_idx]
        if z == 0.0:
            pressure[:, :, z_idx] = input_plane
        else:
            # Spectral propagator — conjugate matches k-wave-python (Eq. 6)
            H = _np.conj(_np.exp(1j * z * kz))

            if absorbing:
                # Eq. 11 of Zeng & McGough: H *= exp(-alpha * z * k / kz)
                H = H * _np.exp(-alpha_neper_per_m * z * k / kz)

            if angular_restriction:
                D = (N - 1) * dx
                kc = k * _np.sqrt(0.5 * D**2 / (0.5 * D**2 + z**2))
                H[sqrt_kx2_ky2 > kc] = 0.0

            projected = _np.fft.ifft2(input_plane_fft * H, (N, N))
            pressure[:, :, z_idx] = projected[:Nx, :Ny]

    if grid_expansion > 0:
        pad = grid_expansion
        pressure = pressure[pad:-pad, pad:-pad, :]

    return pressure


def backward_angular_spectrum_cw(
    measurement_plane: "_np.ndarray",
    dx: float,
    z_m: float,
    f0: float,
    medium,
    *,
    angular_restriction: bool = True,
) -> "_np.ndarray":
    """Reconstruct a source plane from a CW measurement via backward angular spectrum.

    Applies the conjugate (time-reversed) propagator to the spectral
    decomposition of the measured pressure field:

        H_fwd(kx, ky) = conj(exp(j·kz·z_m)) = exp(-j·kz·z_m)
        H_back(kx, ky) = exp(+j·kz·z_m)

    so H_back · H_fwd = 1 for all propagating (real kz) spatial frequencies.
    Evanescent components (kx²+ky² > k²) are suppressed rather than
    amplified by the angular restriction filter (Zeng & McGough 2008, Eq. 7).

    Parameters
    ----------
    measurement_plane : ndarray, shape (Nx, Ny), complex
        Complex pressure amplitude at the measurement plane [Pa].
    dx : float
        Isotropic grid spacing [m].
    z_m : float
        Propagation distance from source to measurement plane [m].
    f0 : float
        Source frequency [Hz].
    medium : float or dict
        Sound speed [m/s] (scalar), or dict with key ``sound_speed`` (m/s).
    angular_restriction : bool, optional
        Apply the angular restriction filter to suppress evanescent components.
        Default True.

    Returns
    -------
    source_plane : ndarray, shape (Nx, Ny), complex
        Reconstructed complex pressure amplitude at the source plane.

    References
    ----------
    Zeng & McGough (2008). "Evaluation of the angular spectrum approach for
    simulations of near-field pressures." JASA, 123(1), 68-76.
    """
    c0 = float(medium["sound_speed"]) if isinstance(medium, dict) else float(medium)

    measurement_plane = _np.asarray(measurement_plane, dtype=complex)
    Nx, Ny = measurement_plane.shape

    N = int(2 ** (_np.ceil(_np.log2(max(Nx, Ny))) + 1))

    if N % 2 == 0:
        k_vec = _np.arange(-N // 2, N // 2) * (2.0 * _np.pi / (N * dx))
    else:
        k_vec = _np.arange(-(N - 1) // 2, (N - 1) // 2 + 1) * (2.0 * _np.pi / (N * dx))
    k_vec[N // 2] = 0.0
    k_vec = _np.fft.ifftshift(k_vec)

    k = 2.0 * _np.pi * f0 / c0

    ky, kx = _np.meshgrid(k_vec, k_vec, indexing="ij")
    kz = _np.sqrt((k**2 - kx**2 - ky**2).astype(complex))
    sqrt_kx2_ky2 = _np.sqrt(kx**2 + ky**2)

    P_meas_fft = _np.fft.fft2(measurement_plane, (N, N))

    # Backward propagator: conjugate of H_fwd = exp(-j·kz·z_m)
    H_back = _np.exp(1j * z_m * kz)

    if angular_restriction:
        D = (N - 1) * dx
        kc = k * _np.sqrt(0.5 * D**2 / (0.5 * D**2 + z_m**2))
        H_back[sqrt_kx2_ky2 > kc] = 0.0
    else:
        H_back[sqrt_kx2_ky2 > k] = 0.0

    p_recon_full = _np.fft.ifft2(P_meas_fft * H_back, (N, N))
    return p_recon_full[:Nx, :Ny]


def gaussian_source_2d(
    nx: int,
    ny: int,
    dx: float,
    sigma: float,
    *,
    amplitude: float = 1.0,
    center_x=None,
    center_y=None,
) -> "_np.ndarray":
    """Generate a 2-D Gaussian complex pressure amplitude on a source plane.

    The Gaussian spatial bandwidth σ_k ≈ 1/σ. For the backward angular
    spectrum reconstruction identity to hold (round-trip error ≈ 0), σ must
    satisfy σ_k ≪ k so that all significant energy lies in the propagating
    band (kx²+ky² < k²).

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    dx : float
        Isotropic grid spacing [m].
    sigma : float
        Gaussian half-width (standard deviation) [m].
    amplitude : float, optional
        Peak pressure amplitude [Pa]. Default 1.0.
    center_x : float, optional
        x-coordinate of the Gaussian centre [m]. Default nx/2 * dx.
    center_y : float, optional
        y-coordinate of the Gaussian centre [m]. Default ny/2 * dx.

    Returns
    -------
    source : ndarray, shape (nx, ny), complex
        Complex Gaussian pressure amplitude.
    """
    x = (_np.arange(nx) - nx / 2.0) * dx if center_x is None else _np.arange(nx) * dx - center_x
    y = (_np.arange(ny) - ny / 2.0) * dx if center_y is None else _np.arange(ny) * dx - center_y
    xx, yy = _np.meshgrid(x, y, indexing="ij")
    return (amplitude * _np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))).astype(complex)
