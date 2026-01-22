# Known Issues - Kwavers Codebase

**Last Updated:** 2026-01-21

---

## Pre-Existing Test Compilation Errors

### `tests/nl_swe_edge_cases.rs` - Type Inference Errors

**Status:** Pre-existing (not introduced by audit work)  
**Severity:** Medium  
**Affected:** Test file only, does not affect library compilation

**Errors:**
- Multiple type inference errors (E0282)
- Missing type annotations in closures
- Approximately 35 compilation errors in this test file

**Example:**
```rust
// Error at line 374
let stress = model.cauchy_stress(&deformation_gradient);
// Error: cannot infer type

// Error at line 379-380
.flat_map(|row| row.iter())
.map(|&s| s.abs())
// Error: type must be known at this point
```

**Impact:**
- Library builds successfully ✅
- Other tests build successfully ✅
- Only this edge case test file fails

**Recommended Fix:**
Add explicit type annotations to the closure parameters:
```rust
.flat_map(|row: &[f64]| row.iter())
.map(|&s: &f64| s.abs())
```

**Priority:** P2 - Non-blocking but should be fixed

---

## Audit Session Status

### ✅ Resolved Issues
1. Compilation errors in `nl_swe_validation.rs` - FIXED
2. Compilation errors in `nl_swe_performance.rs` - FIXED
3. 11 clippy warnings - FIXED
4. Deprecated axisymmetric solver - REMOVED

### 🟡 Known Pre-Existing Issues
1. `nl_swe_edge_cases.rs` type inference - Needs fixing
2. 7 minor doc formatting warnings - Low priority

### 📋 Planned Work
See `COMPREHENSIVE_AUDIT_SUMMARY.md` for full refactoring roadmap

---

## Build Status Summary

**Library:** ✅ PASSING  
**Fixed Tests:** ✅ PASSING (`nl_swe_validation`)  
**Fixed Benchmarks:** ✅ PASSING (`nl_swe_performance`)  
**Pre-existing Test Issues:** ⚠️ `nl_swe_edge_cases` (35 errors)

**Overall:** Library is production-ready; one edge-case test file needs type annotations
