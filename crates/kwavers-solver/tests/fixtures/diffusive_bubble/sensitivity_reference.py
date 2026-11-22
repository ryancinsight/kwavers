"""Independent, nondimensional Rayleigh–Plesset / Henry / diffusion reference.

Run with uv run --no-project --with scipy==1.18.1 --with numpy==2.5.5 python independent_reference.py.
The liquid is discretized by nodal differences on z=(physical radius-R)/R_initial.
SciPy BDF integrates the semi-discrete equations and two boundary ledgers together.
All inputs below specify a synthetic verification problem.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
from itertools import repeat
import sys
from math import pi, sqrt

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.optimize import brentq
from scipy.sparse import lil_matrix

R0, P0, TEMPERATURE = 30e-6, 101325.0, 293.15
FREQUENCY, DIFFUSIVITY = 26600.0, 2e-9
DENSITY, VISCOSITY, TENSION = 998.0, 1e-3, 0.072
HENRY, GAS_CONSTANT = 0.26 / P0, 8.31446261815324
DRIVE = 0.03  # Fraction of P0
BETA = 2 * TENSION / (P0 * R0)
DIFF = DIFFUSIVITY / (FREQUENCY * R0**2)
INERTIA = P0 / (DENSITY * (R0 * FREQUENCY)**2)
VISCOUS = 4 * VISCOSITY * FREQUENCY / P0
CAPACITY = 3 * HENRY * GAS_CONSTANT * TEMPERATURE
N0 = 4 * pi / 3 * R0**3 * P0 * (1 + BETA) / (GAS_CONSTANT * TEMPERATURE)


@dataclass(frozen=True)
class Case:
    name: str
    on: int = 8
    off: int = 8
    bulk: float = 1.0
    amplitude: float = DRIVE
    reset_at_second_burst: bool = False


def derivatives(left, right):
    return (
        -right / (left * (left + right)),
        (right - left) / (left * right),
        left / (right * (left + right)),
    ), (
        2 / (left * (left + right)),
        -2 / (left * right),
        2 / (right * (left + right)),
    )


def equilibrium_radius(amount):
    return brentq(
        lambda radius: radius**2 * (radius + BETA) - (1 + BETA) * amount,
        0,
        np.cbrt((1 + BETA) * amount),
        xtol=1e-15,
    )


def run(case, intervals, tolerance, extent=4.0, final_time=96):
    z = extent * np.expm1(8 * np.linspace(0, 1, intervals + 1)) / np.expm1(8)
    interior = z[1:-1]
    widths = np.diff(z)
    first, second = derivatives(widths[:-1], widths[1:])
    left = z[2] / (z[1] * (z[2] - z[1]))
    right = -z[1] / (z[2] * (z[2] - z[1]))
    outer_left = widths[-1] / (widths[-2] * (widths[-2] + widths[-1]))
    outer_right = -(widths[-2] + widths[-1]) / (widths[-2] * widths[-1])
    size = intervals + 3
    jacobian = lil_matrix((size, size), dtype=bool)
    jacobian[:3, :5] = True
    jacobian[3:-1, :3] = True
    for row in range(3, size - 1):
        jacobian[row, max(3, row - 1):min(size - 1, row + 2)] = True
    jacobian[-1, 0] = True
    jacobian[-1, -3:-1] = True
    jacobian = jacobian.tocsr()

    state = np.zeros(size)
    state[[0, 2]] = 1
    absolute = np.full(size, tolerance * 1e-3)
    balance = []
    phases = []
    removed = 0.0
    peak_speed = 0.0
    minimum_radius = 1.0
    maximum_radius = 1.0
    start = 0
    while start < final_time:
        is_on = len(phases) % 2 == 0
        if len(phases) == 2 and case.reset_at_second_burst:
            profile = np.concatenate(([0.0], state[3:-1], [0.0]))
            removed = CAPACITY * simpson((state[0] + z)**2 * profile, x=z)
            state[3:-1] = 0
        phase_end = start + (case.on if is_on else case.off)
        stop = min(final_time, phase_end)

        def rhs(time, current):
            radius, velocity, amount = current[:3]
            pressure = amount / radius**3  # Dry partial pressure / initial dry pressure
            wall = pressure - case.bulk
            profile = np.concatenate(([wall], current[3:-1], [0.0]))
            before, center, after = profile[:-2], profile[1:-1], profile[2:]
            d_z = first[0] * before + first[1] * center + first[2] * after
            d_zz = second[0] * before + second[1] * center + second[2] * after
            position = radius + interior
            coefficient = 2 * DIFF / position + velocity * (1 - (radius / position)**2)
            elapsed = time - start
            ramp = min(1.0, elapsed, phase_end - time)
            drive = (case.amplitude * np.sin(pi * ramp / 2)**2
                     * np.sin(2 * pi * elapsed)) if is_on else 0.0
            out = np.empty_like(current)
            out[0] = velocity
            net = (1 + BETA) * pressure - 1 - drive - BETA / radius - VISCOUS * velocity / radius
            out[1] = (INERTIA * net - 1.5 * velocity**2) / radius
            out[2] = DIFF * CAPACITY * radius**2 * (
                left * (profile[1] - wall) + right * (profile[2] - wall)
            )
            out[3:-1] = DIFF * d_zz + coefficient * d_z
            out[-1] = -DIFF * CAPACITY * (radius + extent)**2 * (
                outer_left * profile[-3] + outer_right * profile[-2]
            )
            return out

        result = solve_ivp(
            rhs, (start, stop), state, method="BDF", jac_sparsity=jacobian,
            atol=absolute, rtol=tolerance, max_step=1 / 48,
        )
        if not result.success:
            raise RuntimeError(result.message)
        minimum_radius = min(minimum_radius, np.min(result.y[0]))
        maximum_radius = max(maximum_radius, np.max(result.y[0]))
        peak_speed = max(peak_speed, np.max(np.abs(result.y[1])) * R0 * FREQUENCY)
        end = result.y[:, -1]
        phases.append((
            case.name, case.on, case.off, case.bulk, int(case.reset_at_second_burst),
            stop, "on" if is_on else "off", 1e6 * (end[2] - state[2]),
            1e6 * (end[2] - 1), R0 * end[0], R0 * FREQUENCY * end[1],
            1e9 * R0 * (equilibrium_radius(end[2]) - 1),
            minimum_radius, maximum_radius, peak_speed,
        ))
        state = end.copy()
        profile = np.concatenate(([state[2] / state[0]**3 - case.bulk], state[3:-1], [0.0]))
        liquid = CAPACITY * simpson((state[0] + z)**2 * profile, x=z)
        balance.append(state[2] - 1 + liquid + state[-1] + removed)
        start = stop
    return phases, max(map(abs, balance))


def wall_step_integral(cycles):
    # Integral of fixed-radius wall uptake for a unit normalized wall step.
    return CAPACITY * (DIFF * cycles + 2 * sqrt(DIFF * cycles / pi))


def check_closed_forms():
    elapsed = 8.0
    ratios = []
    for quiet in (4, 8, 16, 32):
        # One unit wall deficit on for 'elapsed' cycles followed by return to bulk.
        prior = wall_step_integral(elapsed)
        return_amount = prior - wall_step_integral(elapsed + quiet) + wall_step_integral(quiet)
        ratios.append(return_amount / prior)
    assert 0 < ratios[0] < ratios[1] < ratios[2] < ratios[3] < 1
    print("fixed-sphere off return fraction over 4,8,16,32 cycles", *(f"{x:.8f}" for x in ratios), file=sys.stderr)
    saturated, residual = run(Case("equilibrium", amplitude=0), 128, 1e-8, final_time=8)
    assert abs(saturated[-1][8]) < 1e-6 and residual < 1e-12


CASES = (
    Case("baseline"), Case("reset_control", reset_at_second_burst=True),
    Case("short_quiet", off=4), Case("long_quiet", off=16),
    Case("short_pulse", on=4), Case("long_pulse", on=16),
    Case("50pct_short", on=4, off=4), Case("50pct_long", on=16, off=16),
    Case("undersaturated", bulk=0.98), Case("supersaturated", bulk=1.02),
)
HEADER = (
    "case", "on_cycles", "off_cycles", "bulk_fraction", "reset_liquid",
    "end_cycle", "phase", "amount_ppm", "total_ppm", "radius_m", "velocity_m_s",
    "equilibrium_change_nm", "min_relative_radius", "max_relative_radius", "peak_speed_m_s",
)


def report():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", type=int, default=1280)
    parser.add_argument("--rtol", type=float, default=1.25e-9)
    parser.add_argument("--extent", type=float, default=4.0, help="in initial radii")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    check_closed_forms()
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(HEADER)
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        results = workers.map(run, CASES, repeat(args.intervals), repeat(args.rtol), repeat(args.extent))
        for phases, residual in results:
            print(f"{phases[0][0]} inventory residual: {1e6 * residual:.8g} ppm", file=sys.stderr)
            for row in phases:
                writer.writerow((*row[:7], *(f"{value:.12g}" for value in row[7:])))


if __name__ == "__main__":
    report()
