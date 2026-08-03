# Kwavers Architecture

Kwavers is a high-performance ultrasound simulation and imaging platform in the Atlas multi-repo stack. It provides physics-based acoustic propagation, beamforming, and image reconstruction for medical ultrasound applications.

## Layering (unidirectional; dependencies point inward/upward only)

```
kwavers-python     (PyO3 bindings; thin, no domain logic)
      |
  kwavers-therapy  (therapeutic ultrasound simulation)
      |
kwavers-diagnostics  (diagnostic imaging pipeline)
      |
   kwavers          (core physics engine)
      |
   Atlas Foundation (aequitas, asclepius, hermes, moirai, leto, hephaestus, coeus, ...)
```

## Workspace Structure (11 crates)

| Crate | Purpose |
|-------|---------|
| `kwavers` | Core physics engine: acoustic propagation, attenuation, nonlinear effects |
| `kwavers-diagnostics` | Diagnostic imaging pipeline: beamforming, image formation |
| `kwavers-therapy` | Therapeutic ultrasound: thermal dose, cavitation, microbubble dynamics |
| `kwavers-python` | PyO3 bindings for Python interop |

## Dependency Strategy

- **Local development**: Path dependencies to Atlas foundation crates (`../../repos/{crate}`)
- **Production releases**: Git dependencies from crates.io with version pins
- Atlas crates consumed: aequitas, asclepius, hyperion, proteus, tyche-core, ritk-*, coeus-*, apollo, hephaestus-*, hermes-simd, moirai-parallel, leto, leto-ops, eunomia, consus-*

## Design Principles

1. **SSOT**: Single source of truth for physics constants (tissue properties, material parameters)
2. **Dry**: No duplication of tensor operations; use `coeus`/`leto`/`gaia` for linear algebra
3. **Zero-copy**: Image data flows through typed views, not owned copies
4. **ZST markers**: Phantom types for units, modalities, and safety boundaries
5. **GATs**: Generic kernels parameterized over scalar type, execution policy, and backend
