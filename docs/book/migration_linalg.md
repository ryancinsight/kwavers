# Chapter 36 — Linear Algebra: Leto and Leto-Ops

This chapter covers replacing `nalgebra` dense and sparse linear algebra with
`leto` arrays and `leto-ops` kernels.

## Dense Operations

| nalgebra | leto-ops |
|---|---|
| `DMatrix::new_random(m, n)` | `Array2::from_shape_fn([m,n], \|_\| rng.gen())` |
| `matrix.solve(&vector)` | `leto_ops::solve(&matrix.view(), &vector.view())?` |
| `matrix.try_inverse()` | `leto_ops::inv(&matrix.view())?` |
| `DMatrix::from_fn(m,n, f)` | `Array2::from_shape_fn([m,n], \|[i,j]\| f(i,j))` |
| `matrix.eigenvalues()` | `leto_ops::eigenvalues(&matrix.view())?` |
| `matrix.svd(…)` | `leto_ops::svd_decompose(&matrix.view())?` |
| `matrix.lu()` | `leto_ops::lu_decompose(&matrix.view())?` |
| `matrix.qr()` | `leto_ops::qr_decompose(&matrix.view())?` |
| `matrix.cholesky()` | `leto_ops::cholesky_decompose(&matrix.view())?` |
| `a * b` (matrix product) | `leto_ops::matmul(&a.view(), &b.view())` |
| `matrix.norm()` | `leto_ops::norm(&matrix.view(), NormKind::Frobenius)` |

## Sparse Operations

| nalgebra-sparse | leto-ops |
|---|---|
| `CsrMatrix::new_csc(…)` | `CooMatrix::new(m,n); .to_csr()` |
| `csr * dense_vec` | `leto_ops::spmv(&csr, &x.view(), …)` |
| `csr * dense_mat` | `leto_ops::spmm(&csr, &b.view(), …)` |
| `DirectSparseSolver` | `leto_ops::SparseLuSolver::default().solve(&csr, &b)?` |

## Example: solve a 3×3 system

```rust
use leto::Array2;
use leto_ops::solve;

let a: Array2<f64> = Array2::from_shape_vec([3,3],
    vec![2.,1.,0.,  1.,3.,1.,  0.,1.,4.]).unwrap();
let b = leto::Array1::from_shape_vec([3], vec![5.,10.,11.]).unwrap();
let x = solve(&a.view(), &b.view()).expect("non-singular");
// x ≈ [13/9, 19/9, 20/9]
```
