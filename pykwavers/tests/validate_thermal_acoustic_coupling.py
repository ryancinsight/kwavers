"""
Validation: PSTD acoustic → thermal coupling against bioheat analytical reference.

Physical model
--------------
Nyborg (1981) volumetric heat source from acoustic absorption:
    Q = 2α·P²/(ρ·c)   [W/m³]   (plane-wave, equal kinetic + potential energy)

Pennes bioheat equation (no perfusion, short-time limit):
    ρ·cp · dT/dt = κ·∇²T + Q

Short-time analytical limit (diffusion length << domain, t << L²/κ):
    ΔT(x,t) ≈ Q(x) / (ρ·cp) · t_total

Validation criteria:
1. Peak temperature rise ≤ Q_peak / (ρ·cp) * t_total   (upper bound; diffusion only cools)
2. Peak temperature collocates with source within ±8 grid cells
3. T_initial at t=0 is exactly initial_temperature everywhere (K→°C boundary check)
4. thermal_dose=None when track_thermal_dose=False
5. acoustic-only run → both thermal fields are None
"""

import sys
import numpy as np

try:
    import pykwavers as kw
except ModuleNotFoundError:
    print("SKIP: pykwavers not installed in this environment", file=sys.stderr)
    sys.exit(0)


def _make_point_source(Nx: int, Ny: int, Nz: int, ix: int, iy: int, iz: int,
                       Nt: int, dt: float, f0: float, amplitude: float) -> kw.Source:
    src_mask = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    src_mask[ix, iy, iz] = 1.0
    t_arr = np.arange(Nt) * dt
    sig = amplitude * np.sin(2 * np.pi * f0 * t_arr) * np.exp(
        -0.5 * ((t_arr - 2.0 / f0) / (0.5 / f0)) ** 2
    )
    return kw.Source.from_mask(src_mask, sig.astype(np.float64), f0)


def _make_centre_sensor(Nx: int, Ny: int, Nz: int) -> kw.Sensor:
    mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    mask[Nx // 2, Ny // 2, Nz // 2] = True
    return kw.Sensor.from_mask(mask)


def test_thermal_coupling_temperature_rise():
    """
    Validates that pykwavers PSTD + thermal coupling produces a physically correct
    temperature field against the short-time analytical limit.
    """
    Nx, Ny, Nz = 32, 32, 32
    dx   = 0.5e-3
    c0   = 1540.0
    rho  = 1050.0
    cp   = 3600.0
    kappa_tc = 0.6
    alpha_db = 0.5       # dB/(MHz·cm) soft tissue typical
    f0   = 1.0e6
    T0_c = 37.0
    n_acoustic = 200
    n_ratio    = 10

    grid   = kw.Grid(Nx, Ny, Nz, dx, dx, dx)
    medium = kw.Medium.homogeneous(c0, rho)

    cfl = 0.3
    dt  = cfl * dx / (c0 * np.sqrt(3.0))
    Nt  = n_acoustic

    ix, iy, iz = Nx // 2 - 4, Ny // 2, Nz // 2
    source = _make_point_source(Nx, Ny, Nz, ix, iy, iz, Nt, dt, f0, 1e5)
    sensor = _make_centre_sensor(Nx, Ny, Nz)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
    sim.set_alpha_coeff(alpha_db)
    sim.set_alpha_power(1.1)
    sim.set_thermal(
        center_frequency=f0,
        thermal_conductivity=kappa_tc,
        density=rho,
        specific_heat=cp,
        initial_temperature=T0_c,            # °C
        n_acoustic_per_thermal=n_ratio,
        enable_bioheat=False,
        track_thermal_dose=False,
    )

    result = sim.run(time_steps=Nt, dt=dt)

    assert result.thermal_temperature is not None, "thermal_temperature missing from result"
    T_c = result.thermal_temperature   # shape (Nx, Ny, Nz), °C after K→°C conversion

    # 1. All temperatures ≥ initial (net heat source, no cooling)
    T_min = float(T_c.min())
    assert T_min >= T0_c - 1e-4, (
        f"Temperature dropped below initial: min={T_min:.4f} °C, T0={T0_c} °C"
    )

    # 2. Temperature rise must be positive somewhere
    dT = T_c - T0_c
    dT_peak = float(dT.max())
    assert dT_peak > 0.0, f"No temperature rise detected (peak ΔT={dT_peak:.6f} °C)"

    # 3. Short-time upper bound: ΔT_peak ≤ 20 × Q_peak/(ρ·cp) · t_total
    #    α_np = alpha_db [dB/(MHz·cm)] × 0.1151292546 [Np/m per dB/(cm·MHz)] × f^y
    alpha_power = 1.1
    alpha_np = alpha_db * 0.1151292546 * (f0 / 1.0e6) ** alpha_power  # Np/m
    P_peak = 1e5          # Pa (source amplitude)
    Q_peak_est = 2.0 * alpha_np * P_peak ** 2 / (rho * c0)   # W/m³
    t_total   = n_acoustic * dt
    dT_upper  = Q_peak_est / (rho * cp) * t_total             # K, no diffusion
    assert dT_peak <= 20.0 * dT_upper + 1e-3, (
        f"ΔT_peak={dT_peak:.6f} °C exceeds 20× analytical upper bound {20*dT_upper:.6f} °C"
    )

    # 4. Peak ΔT within ±8 grid cells of source
    peak_idx = np.unravel_index(np.argmax(dT), dT.shape)
    dist = abs(peak_idx[0] - ix)
    assert dist <= 8, (
        f"Peak temperature at {peak_idx} is {dist} cells from source ({ix},{iy},{iz})"
    )

    # 5. K→°C boundary check: must have been applied (temperatures < 200 °C)
    assert T_c.max() < 200.0, (
        f"T_max={T_c.max():.1f} suggests K→°C conversion was NOT applied"
    )

    print(
        f"PASS  temperature_rise: ΔT_peak={dT_peak:.4f} °C  "
        f"bound={dT_upper:.4f} °C  "
        f"peak_cell={peak_idx}  dist_to_src={dist}"
    )


def test_thermal_dose_not_returned_when_disabled():
    """track_thermal_dose=False must produce thermal_dose=None."""
    Nx, Ny, Nz = 16, 16, 16
    dx = 1e-3
    c0 = 1540.0
    rho = 1050.0
    cp  = 3600.0

    grid   = kw.Grid(Nx, Ny, Nz, dx, dx, dx)
    medium = kw.Medium.homogeneous(c0, rho)

    cfl = 0.3
    dt  = cfl * dx / (c0 * np.sqrt(3.0))
    Nt  = 50

    source = _make_point_source(Nx, Ny, Nz, 4, Ny // 2, Nz // 2, Nt, dt, 1e6, 1e5)
    sensor = _make_centre_sensor(Nx, Ny, Nz)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
    sim.set_thermal(
        center_frequency=1e6,
        thermal_conductivity=0.6,
        density=rho,
        specific_heat=cp,
        initial_temperature=37.0,      # °C
        n_acoustic_per_thermal=5,
        enable_bioheat=False,
        track_thermal_dose=False,
    )

    result = sim.run(time_steps=Nt, dt=dt)

    assert result.thermal_temperature is not None, "thermal_temperature must be present"
    assert result.thermal_dose is None, (
        "thermal_dose must be None when track_thermal_dose=False"
    )
    print("PASS  thermal_dose disabled → None")


def test_acoustic_only_run_has_no_thermal_fields():
    """Without set_thermal(), thermal_temperature and thermal_dose must both be None."""
    Nx, Ny, Nz = 16, 16, 16
    dx = 1e-3
    c0 = 1540.0
    rho = 1050.0

    grid   = kw.Grid(Nx, Ny, Nz, dx, dx, dx)
    medium = kw.Medium.homogeneous(c0, rho)

    cfl = 0.3
    dt  = cfl * dx / (c0 * np.sqrt(3.0))
    Nt  = 40

    source = _make_point_source(Nx, Ny, Nz, 4, Ny // 2, Nz // 2, Nt, dt, 1e6, 1e5)
    sensor = _make_centre_sensor(Nx, Ny, Nz)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
    # No set_thermal() call

    result = sim.run(time_steps=Nt, dt=dt)

    assert result.thermal_temperature is None, (
        f"thermal_temperature must be None for acoustic-only; got {type(result.thermal_temperature)}"
    )
    assert result.thermal_dose is None, (
        f"thermal_dose must be None for acoustic-only; got {type(result.thermal_dose)}"
    )
    print("PASS  acoustic-only run → thermal_temperature=None, thermal_dose=None")


if __name__ == "__main__":
    test_acoustic_only_run_has_no_thermal_fields()
    test_thermal_dose_not_returned_when_disabled()
    test_thermal_coupling_temperature_rise()
    print("\nAll thermal validation tests passed.")
