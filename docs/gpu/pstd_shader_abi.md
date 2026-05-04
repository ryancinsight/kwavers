# PSTD Shader ABI

This note records the storage-buffer contract for `kwavers/src/gpu/shaders/pstd.wgsl`.

## Bind Groups

- `group(0)`: acoustic fields: `p`, `ux`, `uy`, `uz`, `rhox`, `rhoy`, `rhoz`, and `precomp_source_kappa`.
- `group(1)`: k-space and medium data: `kspace_re`, `kspace_im`, `kappa`, `rho0_inv`, `c0_sq`, `rho0`, `bon_a`, and `alpha_decay`.
- `group(2)`: PML, shifts, sensors, and sources: `pml_sgx`, `pml_sgy`, `pml_sgz`, packed `pml_xyz`, packed `shifts_all`, sensor indices, sensor output, and packed source data.
- `group(3)`: fractional-Laplacian absorption buffers, used only by absorption pipelines.

## Shift Packing

`shifts_all` stores all staggered shift operators as contiguous `f32` slices:

| Offset | Slice |
|---:|---|
| `0` | `x_pos_re[nx]` |
| `nx` | `x_pos_im[nx]` |
| `2*nx` | `x_neg_re[nx]` |
| `3*nx` | `x_neg_im[nx]` |
| `4*nx` | `y_pos_re[ny]` |
| `4*nx + ny` | `y_pos_im[ny]` |
| `4*nx + 2*ny` | `y_neg_re[ny]` |
| `4*nx + 3*ny` | `y_neg_im[ny]` |
| `4*(nx + ny)` | `z_pos_re[nz]` |
| `4*(nx + ny) + nz` | `z_pos_im[nz]` |
| `4*(nx + ny) + 2*nz` | `z_neg_re[nz]` |
| `4*(nx + ny) + 3*nz` | `z_neg_im[nz]` |

## Twiddle Packing

`precomp_alpha_decay` is repurposed for FFT twiddles when the absorption shader is not dispatched:

| Offset | Slice |
|---:|---|
| `0..128` | `cos(-2*pi*k/256)` |
| `128..256` | `sin(-2*pi*k/256)` |
| `256..320` | `cos(-2*pi*k/128)` |
| `320..384` | `sin(-2*pi*k/128)` |
| `384..416` | `cos(-2*pi*k/64)` |
| `416..448` | `sin(-2*pi*k/64)` |
| `448..464` | `cos(-2*pi*k/32)` |
| `464..480` | `sin(-2*pi*k/32)` |

Inverse FFT dispatches use the same table with the imaginary component negated.

## Source Packing

`source_data` stores `[source_mask_indices | source_signals]` in one `f32` buffer. The indices are written as `bitcast<f32>(u32)` and read by the shader as `bitcast<u32>(source_data[src])`. The signal slice starts at `n_src` and is indexed as `source_data[n_src + src * nt + step]`.

## Invariants

- The WGSL `PstdParams` push-constant layout must match the Rust `PstdParams` struct.
- `pml_xyz` stores `[pml_x | pml_y | pml_z]`, each of length `nx * ny * nz`.
- `field_p` may be used as temporary storage only between sensor recording and `pressure_from_density`, which overwrites pressure before the next sensor read.
