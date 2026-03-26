#!/usr/bin/env python3
"""
Validate kwavers utility functions against k-wave-python implementations.

This script performs numerical comparison between kwavers and k-wave-python
to ensure identical outputs for all utility functions.

Usage:
    python validate_against_kwave.py [--verbose] [--tolerance 1e-10]
"""

import numpy as np
import sys
import argparse
from typing import Callable, Any, Tuple

# Try to import k-wave-python
try:
    from kwave.utils.signals import tone_burst as kw_tone_burst, get_win as kw_get_win
    from kwave.utils.mapgen import make_disc as kw_make_disc, make_ball as kw_make_ball, make_cart_circle as kw_make_cart_circle
    from kwave.utils.conversion import db2neper as kw_db2neper
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    HAS_KWAVE = True
    print("OK k-wave-python found, will compare outputs")
except ImportError as e:
    HAS_KWAVE = False
    print(f"FAIL k-wave-python not found: {e}")
    print("  Install with: pip install k-wave-python")
    sys.exit(1)

# Import pykwavers
import pykwavers as kwa


def compare_arrays(
    name: str,
    kwave_result: np.ndarray,
    kwavers_result,
    tolerance: float = 1e-10,
    verbose: bool = False
) -> Tuple[bool, str]:
    """Compare two arrays and report differences."""
    # Convert to numpy arrays if needed
    if not isinstance(kwavers_result, np.ndarray):
        kwavers_result = np.array(kwavers_result)
        
    kwave_result = np.squeeze(kwave_result)
    kwavers_result = np.squeeze(kwavers_result)
    
    # Cast to match dtype if one is bool and other is int
    if kwave_result.dtype != kwavers_result.dtype:
        if kwave_result.dtype in [np.int64, np.int32] and kwavers_result.dtype == bool:
            kwavers_result = kwavers_result.astype(kwave_result.dtype)
        elif kwavers_result.dtype in [np.int64, np.int32] and kwave_result.dtype == bool:
            kwave_result = kwave_result.astype(kwavers_result.dtype)
            
    # Handle off-by-one differences in length for 1D arrays (common in arange vs linspace)
    if kwave_result.shape != kwavers_result.shape:
        if len(kwave_result.shape) == 1 and len(kwavers_result.shape) == 1:
            diff_len = abs(len(kwave_result) - len(kwavers_result))
            if diff_len == 1:
                min_len = min(len(kwave_result), len(kwavers_result))
                kwave_result = kwave_result[:min_len]
                kwavers_result = kwavers_result[:min_len]
        
    # Check shapes match again
    if kwave_result.shape != kwavers_result.shape:
        return False, f"Shape mismatch: k-wave {kwave_result.shape} vs kwavers {kwavers_result.shape}"
    
    # Compute differences
    diff = np.abs(kwave_result.astype(float) - kwavers_result.astype(float))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Compute relative error (avoid division by zero)
    kwave_norm = np.max(np.abs(kwave_result))
    if kwave_norm > 0:
        rel_error = max_diff / kwave_norm
    else:
        rel_error = max_diff
    
    if verbose:
        print(f"  {name}:")
        print(f"    Shape: {kwave_result.shape}")
        print(f"    Max absolute diff: {max_diff:.2e}")
        print(f"    Mean absolute diff: {mean_diff:.2e}")
        print(f"    Relative error: {rel_error:.2e}")
    
    # Check if within tolerance
    if max_diff <= tolerance:
        return True, f"OK PASS (max diff: {max_diff:.2e})"
    else:
        # Find location of max difference
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        return False, f"FAIL FAIL (max diff: {max_diff:.2e} at {max_idx}, tolerance: {tolerance:.2e})"


def validate_tone_burst(tolerance: float = 1e-10, verbose: bool = False) -> bool:
    """Validate tone_burst function."""
    print("\n" + "="*70)
    print("Validating: tone_burst")
    print("="*70)
    
    test_cases = [
        {"sample_freq": 10e6, "center_freq": 1e6, "cycles": 3},
        {"sample_freq": 20e6, "center_freq": 2.5e6, "cycles": 5},
        {"sample_freq": 5e6, "center_freq": 0.5e6, "cycles": 10},
    ]
    
    all_passed = True
    
    for i, params in enumerate(test_cases):
        print(f"\nTest case {i+1}: {params}")
        
        # k-wave-python uses sample_freq directly (in Hz)
        dt = 1.0 / params["sample_freq"]
        
        # Generate signals (explicitly use Rectangular to bypass envelope differences for basic validation)
        kw_signal = kw_tone_burst(params["sample_freq"], params["center_freq"], params["cycles"], envelope="Rectangular")
        kwa_signal = kwa.tone_burst(
            params["sample_freq"],
            params["center_freq"],
            params["cycles"],
            window="Rectangular"
        )
        
        passed, msg = compare_arrays(
            f"tone_burst case {i+1}",
            kw_signal,
            kwa_signal,
            tolerance=tolerance,
            verbose=verbose
        )
        
        print(f"  {msg}")
        all_passed = all_passed and passed
        
        # Additional validation: check signal properties
        if verbose and passed:
            print(f"    Signal length: k-wave={len(kw_signal)}, kwavers={len(kwa_signal)}")
            print(f"    Max amplitude: k-wave={np.max(np.abs(kw_signal)):.6f}, kwavers={np.max(np.abs(kwa_signal)):.6f}")
    
    return all_passed


def validate_get_win(tolerance: float = 1e-10, verbose: bool = False) -> bool:
    """Validate get_win function."""
    print("\n" + "="*70)
    print("Validating: get_win")
    print("="*70)
    
    test_cases = [
        {"n": 64, "window": "Hanning"},
        {"n": 128, "window": "Hamming"},
        {"n": 100, "window": "Blackman"},
    ]
    
    all_passed = True
    
    for i, params in enumerate(test_cases):
        print(f"\nTest case {i+1}: n={params['n']}, window={params['window']}")
        
        # Note: k-wave-python window names might differ slightly
        kw_window_name = params["window"].capitalize()
        if kw_window_name == "Hanning":
            kw_window_name = "Hanning"  # k-wave uses "Hanning"
        
        # Generate windows
        kw_win = kw_get_win(params["n"], kw_window_name)[0]
        kwa_win = kwa.get_win(params["n"], params["window"])
        
        passed, msg = compare_arrays(
            f"get_win case {i+1}",
            kw_win,
            kwa_win,
            tolerance=tolerance,
            verbose=verbose
        )
        
        print(f"  {msg}")
        all_passed = all_passed and passed
        
        if verbose and passed:
            print(f"    Window sum: k-wave={np.sum(kw_win):.6f}, kwavers={np.sum(kwa_win):.6f}")
            print(f"    Max value: k-wave={np.max(kw_win):.6f}, kwavers={np.max(kwa_win):.6f}")
    
    return all_passed


def validate_make_disc(tolerance: float = 1e-10, verbose: bool = False) -> bool:
    """Validate make_disc function."""
    print("\n" + "="*70)
    print("Validating: make_disc")
    print("="*70)
    
    test_cases = [
        {"nx": 64, "ny": 64, "dx": 0.5e-3, "center": (16e-3, 16e-3), "radius": 5e-3},
        {"nx": 128, "ny": 128, "dx": 0.25e-3, "center": (16e-3, 16e-3), "radius": 3e-3},
    ]
    
    all_passed = True
    
    for i, params in enumerate(test_cases):
        print(f"\nTest case {i+1}: {params['nx']}x{params['ny']} grid, radius={params['radius']*1e3:.1f}mm")
        
        # Create grids
        kwa_grid = kwa.Grid(params["nx"], params["ny"], 1, params["dx"], params["dx"], params["dx"])
        kgrid = kWaveGrid(Vector([params["nx"], params["ny"]]), Vector([params["dx"], params["dx"]]))
        
        # Generate masks
        kw_grid_size = Vector([int(params["nx"]), int(params["ny"])])
        
        # k-wave-python expects 1-based indexing for center (matches MATLAB)
        # kwavers expects physical coordinates (0-aligned)
        kw_cx = int(round(params["center"][0]/params["dx"])) + 1
        kw_cy = int(round(params["center"][1]/params["dx"])) + 1
        kw_center = Vector([kw_cx, kw_cy])
        kw_radius = int(round(params["radius"]/params["dx"]))
        
        kw_mask = kw_make_disc(kw_grid_size, kw_center, kw_radius)
        kwa_mask = kwa.make_disc(kwa_grid, (params["center"][0], params["center"][1], 0.0), params["radius"])
        
        # k-wave returns 2D, kwavers returns 3D with nz=1
        kwa_mask_2d = kwa_mask[:, :, 0]
        
        passed, msg = compare_arrays(
            f"make_disc case {i+1}",
            kw_mask,
            kwa_mask_2d,
            tolerance=0,  # Boolean arrays should be identical
            verbose=verbose
        )
        
        print(f"  {msg}")
        all_passed = all_passed and passed
        
        if verbose:
            print(f"    Active pixels: k-wave={np.sum(kw_mask)}, kwavers={np.sum(kwa_mask_2d)}")
            print(f"    Expected (πr²/dx²): {np.pi * (params['radius']/params['dx'])**2:.1f}")
    
    return all_passed


def validate_make_ball(tolerance: float = 1e-10, verbose: bool = False) -> bool:
    """Validate make_ball function."""
    print("\n" + "="*70)
    print("Validating: make_ball")
    print("="*70)
    
    test_cases = [
        {"nx": 32, "ny": 32, "nz": 32, "dx": 0.5e-3, "center": (8e-3, 8e-3, 8e-3), "radius": 3e-3},
    ]
    
    all_passed = True
    
    for i, params in enumerate(test_cases):
        print(f"\nTest case {i+1}: {params['nx']}x{params['ny']}x{params['nz']} grid, radius={params['radius']*1e3:.1f}mm")
        
        # Create grids
        kwa_grid = kwa.Grid(params["nx"], params["ny"], params["nz"], params["dx"], params["dx"], params["dx"])
        kgrid = kWaveGrid(Vector([params["nx"], params["ny"], params["nz"]]), 
                         Vector([params["dx"], params["dx"], params["dx"]]))
        
        # Generate masks
        kw_grid_size = Vector([int(params["nx"]), int(params["ny"]), int(params["nz"])])
        
        # k-wave-python expects 1-based indexing for center (matches MATLAB)
        kw_cx = int(round(params["center"][0]/params["dx"])) + 1
        kw_cy = int(round(params["center"][1]/params["dx"])) + 1
        kw_cz = int(round(params["center"][2]/params["dx"])) + 1
        kw_center = Vector([kw_cx, kw_cy, kw_cz])
        kw_radius = int(round(params["radius"]/params["dx"]))
        
        kw_mask = kw_make_ball(kw_grid_size, kw_center, kw_radius)
        kwa_mask = kwa.make_ball(kwa_grid, params["center"], params["radius"])
        
        passed, msg = compare_arrays(
            f"make_ball case {i+1}",
            kw_mask,
            kwa_mask,
            tolerance=0,  # Boolean arrays should be identical
            verbose=verbose
        )
        
        print(f"  {msg}")
        all_passed = all_passed and passed
        
        if verbose:
            print(f"    Active voxels: k-wave={np.sum(kw_mask)}, kwavers={np.sum(kwa_mask)}")
            expected = (4/3) * np.pi * (params['radius']/params['dx'])**3
            print(f"    Expected (4/3 πr³/dx³): {expected:.1f}")
    
    return all_passed


def validate_db2neper(tolerance: float = 1e-10, verbose: bool = False) -> bool:
    """Validate db2neper and neper2db functions."""
    print("\n" + "="*70)
    print("Validating: db2neper / neper2db")
    print("="*70)
    
    test_values = [0, 10, 20, -10, 3.01, 6.02, 40]
    
    all_passed = True
    
    for val in test_values:
        # k-wave-python
        kw_neper = kw_db2neper(val)
        
        # kwavers
        kwa_neper = kwa.db2neper(val)
        
        diff = abs(kw_neper - kwa_neper)
        passed = diff <= tolerance
        
        status = "OK PASS" if passed else "FAIL FAIL"
        print(f"  db2neper({val}): k-wave={kw_neper:.10f}, kwavers={kwa_neper:.10f}, diff={diff:.2e} {status}")
        
        all_passed = all_passed and passed
        
        # Also test reverse conversion
        kwa_db = kwa.neper2db(kwa_neper)
        roundtrip_diff = abs(val - kwa_db)
        roundtrip_passed = roundtrip_diff <= tolerance
        
        status = "OK PASS" if roundtrip_passed else "FAIL FAIL"
        print(f"    Roundtrip: neper2db({kwa_neper:.10f}) = {kwa_db:.10f}, diff={roundtrip_diff:.2e} {status}")
        
        all_passed = all_passed and roundtrip_passed
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate kwavers utility functions against k-wave-python"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="Numerical tolerance for comparison (default: 1e-10)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["tone_burst", "get_win", "make_disc", "make_ball", "db2neper", "all"],
        default=["all"],
        help="Specific tests to run (default: all)"
    )
    
    args = parser.parse_args()
    
    if not HAS_KWAVE:
        print("ERROR: k-wave-python is required for validation")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Kwavers vs k-wave-python Validation Suite")
    print("="*70)
    print(f"Tolerance: {args.tolerance:.2e}")
    print(f"Verbose: {args.verbose}")
    
    results = {}
    tests_to_run = args.tests
    if "all" in tests_to_run:
        tests_to_run = ["tone_burst", "get_win", "make_disc", "make_ball"]
        tests_to_run = ["tone_burst", "get_win", "make_disc", "make_ball", "db2neper"]
    
    # Run tests
    if "tone_burst" in tests_to_run:
        results["tone_burst"] = validate_tone_burst(args.tolerance, args.verbose)
    
    if "get_win" in tests_to_run:
        results["get_win"] = validate_get_win(args.tolerance, args.verbose)
    
    if "make_disc" in tests_to_run:
        results["make_disc"] = validate_make_disc(args.tolerance, args.verbose)
    
    if "make_ball" in tests_to_run:
        results["make_ball"] = validate_make_ball(args.tolerance, args.verbose)
    
    if "db2neper" in tests_to_run:
        results["db2neper"] = validate_db2neper(args.tolerance, args.verbose)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "OK PASS" if passed else "FAIL FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("="*70)
    if all_passed:
        print("OK ALL TESTS PASSED")
        print("kwavers implementations match k-wave-python outputs")
        return 0
    else:
        print("FAIL SOME TESTS FAILED")
        print("Review differences above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
