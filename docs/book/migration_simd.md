# Chapter 38 — SIMD: Hermes for Vectorized Operations

`kwavers-math` replaces hand-rolled SIMD with `hermes-simd`, the Atlas SIMD
abstraction that dispatches at runtime (AVX2, NEON, or scalar fallback).

## What Changed

| Before (hand-rolled) | After (hermes) |
|---|---|
| `unsafe { _mm256_add_ps(…) }` | `hermes_simd::elementwise_add(a, b, out)` |
| `unsafe { _mm256_mul_ps(…) }` | `hermes_simd::scale(values, scalar)` |
| `#[target_feature(enable="avx2")]` | `CfdSimdOps::<f64>::new()` (runtime dispatch) |

## Architecture

```
kwavers-math::simd_safe::auto_detect::dispatcher
    ├── AVX2 backend  (hermes_simd on x86_64 with AVX2)
    ├── NEON backend  (hermes_simd on AArch64)
    └── SWAR fallback (hermes_simd scalar path)
```

The `CfdSimdOps` wrapper in `kwavers-math::simd_safe` provides the same API
regardless of the chosen backend — the `hermes_simd` runtime-dispatch model.

## Example: element-wise add via hermes

```rust
use hermes_simd;

let a = vec![1.0_f64; 1024];
let b = vec![2.0_f64; 1024];
let mut out = vec![0.0_f64; 1024];
hermes_simd::elementwise_add(&a, &b, &mut out);
// out[i] = 3.0 for all i, dispatched to widest available SIMD width
```

## Why Not Direct SIMD?

- **Safety**: hermes uses `unsafe` once at the dispatch boundary; kwavers stays `#[forbid(unsafe_code)]`
- **Portability**: same code runs on x86, AArch64, and WASM
- **Maintenance**: vector width changes tracked in one place (hermes), not scattered across kwavers
