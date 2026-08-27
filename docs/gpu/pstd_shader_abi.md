# PSTD Shader ABI

This note records the storage-buffer contract for
`crates/kwavers-gpu/src/pstd_gpu/shaders/pstd.wgsl`.

## Bind Groups

- `group(0)`: acoustic fields: `p`, `ux`, `uy`, `uz`, `rhox`, `rhoy`, `rhoz`, and `precomp_source_kappa`.
- `group(1)`: k-space and medium data: `kspace_re`, `kspace_im`, `kappa`, `rho0_inv`, `c0_sq`, `rho0`, and `bon_a`.
- `group(2)`: PML, shifts, sensors, and sources: `pml_sgx`, `pml_sgy`, `pml_sgz`, packed `pml_xyz`, packed `shifts_all`, sensor indices, sensor output, and packed source data.
- `group(3)`: fractional-Laplacian absorption constants and scratch buffers, used only by absorption pipelines.

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

## FFT buffers

`kspace_re` and `kspace_im` are typed Hephaestus device buffers. Kwavers
prepares one forward and one inverse rank-3 transform over their Leto C-order
layout and encodes those plans into the same grouped command sequence as the
PSTD kernels. One- and two-dimensional grids use singleton axes. Root tables,
Bluestein workspace, and axis dispatch are provider-owned resources and are not
part of the Kwavers shader ABI.

## Source Packing

`source_data` stores `[source_mask_indices | source_signals]` in one `f32` buffer. The indices are written as `bitcast<f32>(u32)` and read by the shader as `bitcast<u32>(source_data[src])`. The signal slice starts at `n_src` and is indexed as `source_data[n_src + src * nt + step]`.

## Output Packing

`sensor_data` reserves `max(n_sensors, 1) * nt` scalars for pressure traces;
the minimum preserves a valid WGPU storage binding for a zero-sensor run. A
peak-pressure request appends one `nx * ny * nz` region at
`params.peak_offset`. After every pressure update, `accumulate_peak_pressure` writes
`max(previous_peak, abs(field_p[idx]))` into that region. The GPU command path
downloads only the appended region for a peak-only request; it never treats the
final pressure frame as the envelope.

## Invariants

- The WGSL `PstdParams` push-constant layout must match the Rust `PstdParams` struct.
- `pml_xyz` stores `[pml_x | pml_y | pml_z]`, each of length `nx * ny * nz`.
- `field_p` may be used as temporary storage only between sensor recording and `pressure_from_density`, which overwrites pressure before the next sensor read.
- `PstdParams` contains 12 scalar fields (48 bytes) in identical Rust and WGSL
  order, including `peak_offset` and `record_peak_pressure`.
