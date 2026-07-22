# Appendix A — Migration Quick Reference

## One-line Cheat-sheet

| Old dependency | Atlas replacement | Import path |
|---|---|---|
| `ndarray::Array1<T>` | `leto::Array1<T>` | `use leto::Array1` |
| `ndarray::Array2<T>` | `leto::Array2<T>` | `use leto::Array2` |
| `ndarray::Array3<T>` | `leto::Array3<T>` | `use leto::Array3` |
| `ndarray::ArrayView3<T>` | `leto::ArrayView3<T>` | `use leto::ArrayView3` |
| `nalgebra::Vector3<T>` | `leto::Vector3<T>` | `use leto::Vector3` |
| `nalgebra::Point3<T>` | `leto::Point3<T>` | `use leto::Point3` |
| `nalgebra::Isometry3<T>` | `leto::Isometry3<T>` | `use leto::Isometry3` |
| `nalgebra::solve(A, b)` | `leto_ops::solve(A, b)` | `use leto_ops::solve` |
| `nalgebra::inv(A)` | `leto_ops::inv(A)` | `use leto_ops::inv` |
| `num_complex::Complex64` | `eunomia::Complex64` | `use eunomia::Complex64` |
| `num_traits::Float` | `eunomia::RealField` | `use eunomia::RealField` |
| `rustfft::FftPlanner` | `apollo::FftPlan3D` | `use apollo::FftPlan3D` |
| `rayon::par_iter()` | `moirai_parallel::*` | `use moirai_parallel::*` |
| `tokio::spawn(…)` | `moirai` async executor | `use moirai::*` |
| `rsparse::lusol(…)` | `leto_ops::SparseLuSolver` | `use leto_ops::SparseLuSolver` |
| `burn::tensor::Tensor` | `coeus_tensor::Tensor` | `use coeus_tensor::Tensor` |
| `ndarray_rand::rand_distr` | `leto_ops::uniform_with_seed` | `use leto_ops::uniform_with_seed` |

## Cargo.toml Pattern

```toml
# Remove:
# ndarray = "0.15"
# nalgebra = "0.32"
# rustfft = "6"
# num-complex = "0.4"
# num-traits = "0.2"
# rayon = "1"

# Add (workspace = true if in a member crate):
leto = { workspace = true }
leto-ops = { workspace = true }
eunomia = { workspace = true }
apollo = { workspace = true }
moirai-parallel = { workspace = true }
hermes-simd = { workspace = true }
```

## Verification Checklist

- [ ] `cargo check --workspace` — zero errors
- [ ] `grep -r 'use ndarray\|use nalgebra\|use rustfft\|use rayon\|use tokio' crates/` — zero matches
- [ ] `cargo test --workspace --lib` — all tests pass
- [ ] `cargo build --examples` — all examples build
