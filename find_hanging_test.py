#!/usr/bin/env python3
"""Script to identify hanging tests by running them individually with timeout."""

import subprocess
import sys
import time

def get_test_list():
    """Get list of all tests."""
    result = subprocess.run(
        ["cargo", "test", "--lib", "--", "--list"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    tests = []
    for line in result.stdout.splitlines():
        if ": test" in line:
            test_name = line.split(": test")[0].strip()
            tests.append(test_name)
    
    return tests

def run_test(test_name, timeout_sec=5):
    """Run a single test with timeout."""
    try:
        result = subprocess.run(
            ["cargo", "test", "--lib", test_name, "--", "--exact", "--nocapture"],
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        return "PASS" if result.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

def main():
    print("Finding hanging tests...")
    tests = get_test_list()
    print(f"Found {len(tests)} tests")
    
    hanging_tests = []
    failed_tests = []
    passed_tests = []
    
    for i, test in enumerate(tests):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(tests)}")
        
        result = run_test(test, timeout_sec=2)
        
        if result == "TIMEOUT":
            hanging_tests.append(test)
            print(f"  HANGING: {test}")
        elif result == "FAIL":
            failed_tests.append(test)
        else:
            passed_tests.append(test)
    
    print("\n=== SUMMARY ===")
    print(f"Passed:  {len(passed_tests)}")
    print(f"Failed:  {len(failed_tests)}")
    print(f"Hanging: {len(hanging_tests)}")
    
    if hanging_tests:
        print("\n=== HANGING TESTS ===")
        for test in hanging_tests:
            print(f"  - {test}")

if __name__ == "__main__":
    main()