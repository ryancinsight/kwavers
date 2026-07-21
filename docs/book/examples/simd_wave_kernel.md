# Example: SIMD Wave Kernel

**Source**: `crates/kwavers/examples/simd_wave_kernel.rs`

## Overview

Demonstrates `hermes_simd` runtime-dispatch functions on a 1-D acoustic
pressure field, replacing hand-rolled SIMD loops in the kwavers wave solver.

## Operations covered

| Function | Kernel | kwavers use |
|---|---|---|
| `elementwise_add` | `p_out[i] = p_a[i] + p_b[i]` | multi-source IVP superposition |
| `scale` | `p[i] *= α` | PML exponential damping step |
| `axpy` | `u[i] += α·∇p[i]` | velocity field update |
| `dot` | `Σ p[i]²` | acoustic energy diagnostic |
| `sum` | `Σ p[i]` | pressure field integral |
| `argmax` | `max_i p[i]` | peak pressure detection |

## Running

```bash
cargo run --example simd_wave_kernel
```

## Expected output (excerpt)

```
SIMD wave kernel demo (hermes-simd runtime dispatch)
field size: 1024, dt: 1.0e-8 s, ρ₀: 1000 kg/m³
backend: AVX2 (4×f64)

1. elementwise_add — pressure superposition
   ||p_a||² = …  ||p_b||² = …
   Σ(p_a+p_b) = …  (err: ~1e-16)
   → PASS

2. scale — PML attenuation
   energy ratio = 0.99998  (expected 0.99998)
   → PASS

3. axpy — velocity update
   ||u||² = α²·||∇p||²  → PASS

4. sum + dot — field diagnostics
   → PASS

All SIMD correctness checks PASS
```

## Key API

```rust
use hermes_simd::{axpy, dot, elementwise_add, scale, sum};

// elementwise add: two pressure contributions
elementwise_add(&p_a, &p_b, &mut p_out).unwrap();

// PML damping: p *= exp(-σ·dt)
let damping = (-SIGMA_PML * DT).exp();
scale(&mut p, damping);

// velocity update: u += (-dt/rho) * grad_p
axpy(-DT / RHO0, &grad_p, &mut u).unwrap();

// energy: <p, p>
let energy = dot(&p, &p).unwrap();
```

## Backend selection

`hermes_simd` selects the widest available SIMD backend at **runtime**:
- AVX-512: 8 × f64 per cycle (Intel Ice Lake+)  
- AVX2: 4 × f64 per cycle (Intel Haswell+)  
- NEON: 2 × f64 per cycle (AArch64)  
- Scalar: portable fallback (identical arithmetic, no `unsafe` at call site)

`kwavers` itself remains `#[forbid(unsafe_code)]`.
