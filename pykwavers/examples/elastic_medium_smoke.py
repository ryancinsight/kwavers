#!/usr/bin/env python3
"""
elastic_medium_smoke.py
========================
Smoke test for the new pykwavers elastic-medium bindings (Phase A of the
k-Wave EWP example replication roadmap).

Exercises the bridging surface added by the kwavers commit that introduced
``HomogeneousMedium::elastic_homogeneous`` and the corresponding
``Medium.elastic`` PyO3 static method:

1. **Round-trip dispersion-relation invariants** — for the canonical
   k-Wave ``example_ewp_layered_medium`` lower-layer values
   ``(c_p = 2000, c_s = 800, ρ = 1200)``, verify that
   ``Medium.elastic`` produces a medium whose Lamé parameters round-trip
   back to the input speeds within float epsilon:
       μ = ρ · c_s²
       λ = ρ · (c_p² − 2 · c_s²)
       c_p_back = sqrt((λ + 2μ) / ρ)   ≈ c_p
       c_s_back = sqrt(μ / ρ)          ≈ c_s

2. **Fluid limit** (c_s = 0) — the canonical k-Wave ``example_ewp_layered_medium``
   upper-layer water values ``(c_p = 1500, c_s = 0, ρ = 1000)``: μ must be
   exactly zero, λ must equal the bulk modulus K = ρ·c_p².

3. **Stability bound rejection** — ``c_s > c_p / √2`` violates λ ≥ 0
   (auxetic / negative-Poisson regime), and the constructor must raise.

4. **Positivity / finiteness rejection** — non-finite or non-positive
   inputs must raise.

This smoke test does NOT yet run an elastic simulation end-to-end; that
requires the next sub-cycle (SolverType.Elastic + Source/Sensor for
displacement / velocity / stress). What it does verify is that the medium
construction surface is sound, mathematically consistent with k-Wave's
``medium.sound_speed_compression`` / ``sound_speed_shear`` API, and ready
to be consumed by an elastic-solver dispatch path.

Usage
-----
    python examples/elastic_medium_smoke.py
"""

from __future__ import annotations

import math
import sys
from typing import Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from example_parity_utils import bootstrap_example_paths

bootstrap_example_paths()

import pykwavers as pkw

# ---------------------------------------------------------------------------
# k-Wave canonical elastic test cases
# (matching example_ewp_layered_medium.m and example_ewp_plane_wave_absorption.m)
# ---------------------------------------------------------------------------
LAYERED_UPPER = (1500.0, 0.0, 1000.0)  # water
LAYERED_LOWER = (2000.0, 800.0, 1200.0)  # bone-like
PLANE_WAVE = (1800.0, 1200.0, 1000.0)  # plane-wave absorption test
SHEAR_BONE = (3000.0, 1400.0, 1850.0)  # snells_law lower-half (bone)


def _check_round_trip(
    case: Tuple[float, float, float], label: str, tol: float = 1e-9
) -> None:
    """Assert dispersion-relation round trip and Lamé-from-speeds inversion."""
    cp, cs, rho = case
    med = pkw.Medium.elastic(cp, cs, rho)

    expected_mu = rho * cs * cs
    expected_lambda = rho * (cp * cp - 2.0 * cs * cs)

    assert (
        abs(med.lame_mu - expected_mu) < tol * max(expected_mu, 1.0)
    ), f"[{label}] μ mismatch: got {med.lame_mu}, expected {expected_mu}"
    assert (
        abs(med.lame_lambda - expected_lambda) < tol * max(abs(expected_lambda), 1.0)
    ), f"[{label}] λ mismatch: got {med.lame_lambda}, expected {expected_lambda}"

    # Round-trip via dispersion relations
    cp_back = med.c_compression
    cs_back = med.c_shear
    assert (
        abs(cp_back - cp) < tol * cp
    ), f"[{label}] c_p round-trip drift: {cp_back} vs {cp}"
    assert (
        abs(cs_back - cs) < tol * max(cs, 1.0)
    ), f"[{label}] c_s round-trip drift: {cs_back} vs {cs}"

    # Density getter must reflect input
    assert (
        abs(med.density - rho) < tol * rho
    ), f"[{label}] density drift: {med.density} vs {rho}"

    # Sanity: max_sound_speed (from acoustic API) returns c_p for a medium
    # constructed via elastic_homogeneous; the underlying sound_speed cache
    # was set to c_compression in the Rust constructor.
    assert (
        abs(med.sound_speed - cp) < tol * cp
    ), f"[{label}] sound_speed accessor must equal c_p; got {med.sound_speed} vs {cp}"

    print(
        f"  [PASS] {label:20s}  cp={cp:7.1f}  cs={cs:7.1f}  rho={rho:6.1f}  "
        f"λ={med.lame_lambda:.3e} Pa  μ={med.lame_mu:.3e} Pa"
    )


def main() -> int:
    print("=" * 78)
    print("elastic_medium_smoke: pykwavers Medium.elastic bridging surface")
    print("=" * 78)

    print("\n[1/4] Dispersion-relation round trip on canonical k-Wave EWP cases")
    print("-" * 78)
    _check_round_trip(LAYERED_LOWER, "ewp_layered_lower")
    _check_round_trip(PLANE_WAVE, "ewp_plane_wave")
    _check_round_trip(SHEAR_BONE, "ewp_shear_bone")

    print("\n[2/4] Fluid limit (c_shear = 0): μ must be 0, λ must equal K = ρ·c_p²")
    print("-" * 78)
    cp, _, rho = LAYERED_UPPER
    fluid = pkw.Medium.elastic(cp, 0.0, rho)
    assert fluid.lame_mu == 0.0, f"μ must be 0 in fluid limit; got {fluid.lame_mu}"
    K_expected = rho * cp * cp
    assert (
        abs(fluid.lame_lambda - K_expected) < 1e-6
    ), f"λ should equal bulk modulus K=ρc²={K_expected}; got {fluid.lame_lambda}"
    assert fluid.c_shear == 0.0, f"c_shear getter must read 0 in fluid limit"
    assert (
        abs(fluid.c_compression - cp) < 1e-9 * cp
    ), f"c_compression getter must equal input c_p"
    print(
        f"  [PASS] ewp_layered_upper (water)  cp={cp:7.1f}  cs=0   "
        f"λ=K=ρc²={K_expected:.3e} Pa  μ=0"
    )

    print("\n[3/4] Stability rejection: c_s > c_p / √2 ⇒ λ < 0 (auxetic) — ValueError")
    print("-" * 78)
    unstable_cases = [
        ("c_s > c_p (gross)", (1500.0, 1600.0, 1000.0)),
        ("c_s = c_p (boundary)", (1500.0, 1500.0, 1000.0)),
        ("c_s = c_p / √2 + ε", (1500.0, 1500.0 / math.sqrt(2.0) + 1.0, 1000.0)),
    ]
    for label, (cp, cs, rho) in unstable_cases:
        try:
            pkw.Medium.elastic(cp, cs, rho)
        except ValueError as e:
            print(f"  [PASS] rejected {label} → ValueError: {str(e)[:60]}...")
        else:
            raise AssertionError(
                f"[{label}] (cp={cp}, cs={cs}, rho={rho}) should have raised"
            )

    # And the just-barely-stable case must still succeed
    cp = 1500.0
    cs_max = cp / math.sqrt(2.0) - 1.0  # safely below the bound
    rho = 1000.0
    med = pkw.Medium.elastic(cp, cs_max, rho)
    assert med.lame_lambda >= 0.0, f"just-stable case must have λ ≥ 0"
    print(
        f"  [PASS] just-below-bound (cp={cp}, cs={cs_max:.3f}) accepted with "
        f"λ={med.lame_lambda:.3e}"
    )

    print("\n[4/4] Input validation: positivity + finiteness — ValueError")
    print("-" * 78)
    invalid_cases = [
        ("density = 0", (1500.0, 800.0, 0.0)),
        ("density < 0", (1500.0, 800.0, -1.0)),
        ("c_p = 0", (0.0, 800.0, 1000.0)),
        ("c_s < 0", (1500.0, -1.0, 1000.0)),
        ("c_p NaN", (float("nan"), 800.0, 1000.0)),
        ("c_s Inf", (1500.0, float("inf"), 1000.0)),
    ]
    for label, (cp, cs, rho) in invalid_cases:
        try:
            pkw.Medium.elastic(cp, cs, rho)
        except ValueError:
            print(f"  [PASS] rejected {label}")
        else:
            raise AssertionError(f"[{label}] should have raised")

    print("\n" + "=" * 78)
    print("All elastic-medium bridging tests passed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
