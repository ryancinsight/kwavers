// PSTD (Pseudospectral Time Domain) GPU Compute Shaders
//
// Implements a GPU-resident acoustic PSTD solver with split-field PML.
//
// Bind group layout (≤8 storage buffers per group, ≤4 groups):
//
//   group(0): field buffers (7 storage: p, ux, uy, uz, rhox, rhoy, rhoz)
//   group(1): k-space + medium scalars (8 storage: kspace_re, kspace_im,
//             kappa, rho0_inv, c0_sq, rho0, bon_a, alpha_decay)
//   group(2): PML + shift + sensor (8 storage: pml_sgx, pml_sgy, pml_sgz,
//             pml_xyz (packed), shifts_all (packed), sensor_indices,
//             sensor_data, source_data (packed))
//
// The shift operators for all 6 cases (x/y/z × pos/neg) are packed into a
// single "shifts_all" buffer:
//   Offset 0           .. nx      : x_pos_re[nx]
//   Offset nx          .. 2*nx    : x_pos_im[nx]
//   Offset 2*nx        .. 3*nx    : x_neg_re[nx]
//   Offset 3*nx        .. 4*nx    : x_neg_im[nx]
//   Offset 4*nx        .. 4*nx+ny : y_pos_re[ny]
//   Offset 4*nx+ny     .. 4*nx+2*ny: y_pos_im[ny]
//   Offset 4*nx+2*ny   .. 4*nx+3*ny: y_neg_re[ny]
//   Offset 4*nx+3*ny   .. 4*nx+4*ny: y_neg_im[ny]
//   Offset 4*nx+4*ny   .. 4*(nx+ny)+nz      : z_pos_re[nz]
//   Offset 4*(nx+ny)+nz.. 4*(nx+ny)+2*nz    : z_pos_im[nz]
//   Offset 4*(nx+ny)+2*nz..4*(nx+ny)+3*nz   : z_neg_re[nz]
//   Offset 4*(nx+ny)+3*nz..4*(nx+ny)+4*nz   : z_neg_im[nz]
//
// The pml_xyz buffer packs three 3D arrays: [pml_x | pml_y | pml_z],
// each of size nx*ny*nz. Total = 3*nx*ny*nz f32 values.
//
// References:
//   Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.
//   Liu (1998). Microwave Opt. Technol. Lett. 15(3), 158-165.

// ─── Constants ───────────────────────────────────────────────────────────────

const TWO_PI: f32 = 6.28318530717958647692;

// ─── Bind Group 0: Field buffers (7 storage) ─────────────────────────────────
// field_scratch removed: velocity/density updates now read kspace_re directly.

@group(0) @binding(0)
var<storage, read_write> field_p: array<f32>;

@group(0) @binding(1)
var<storage, read_write> field_ux: array<f32>;

@group(0) @binding(2)
var<storage, read_write> field_uy: array<f32>;

@group(0) @binding(3)
var<storage, read_write> field_uz: array<f32>;

@group(0) @binding(4)
var<storage, read_write> field_rhox: array<f32>;

@group(0) @binding(5)
var<storage, read_write> field_rhoy: array<f32>;

@group(0) @binding(6)
var<storage, read_write> field_rhoz: array<f32>;

// Spectral source correction used by the validated phased-array velocity-source path.
@group(0) @binding(7)
var<storage, read> precomp_source_kappa: array<f32>;

// ─── Push constants: per-dispatch params ─────────────────────────────────────
// Replaces the group(1) uniform buffer. Push constants are embedded directly
// in the command buffer — no PCIe transfer, no write_buffer() overhead.

struct PstdParams {
    nx:        u32,
    ny:        u32,
    nz:        u32,
    axis:      u32,  // 0-2 = x/y/z-positive; 3-5 = x/y/z-negative; also field select
    n_fft:     u32,
    n_batches: u32,
    log2n:     u32,
    inverse:   u32,
    step:      u32,
    dt:        f32,
    n_sensors: u32,
    nt:        u32,
    nonlinear: u32,  // 1 = apply BonA EOS correction in pressure step
    absorbing:  u32, // 1 = apply frequency-centred absorption decay to density
}

var<push_constant> params: PstdParams;

// ─── Bind Group 1: K-space + medium scalars (8 storage) ──────────────────────
// kspace2_re/im removed: kspace_shift_apply now writes in-place to kspace_re/im.

@group(1) @binding(0)
var<storage, read_write> kspace_re: array<f32>;

@group(1) @binding(1)
var<storage, read_write> kspace_im: array<f32>;

@group(1) @binding(2)
var<storage, read> precomp_kappa: array<f32>;

@group(1) @binding(3)
var<storage, read> precomp_rho0_inv: array<f32>;

@group(1) @binding(4)
var<storage, read> precomp_c0_sq: array<f32>;

@group(1) @binding(5)
var<storage, read> precomp_rho0: array<f32>;

// binding(6): B/(2A) nonlinearity parameter per voxel (0.0 when linear)
@group(1) @binding(6)
var<storage, read> precomp_bon_a: array<f32>;

// binding(7): per-voxel absorption decay factor = exp(-alpha_Np_m * c0 * dt)
// Precomputed at the transmit centre frequency. 1.0 everywhere when absorbing=0.
@group(1) @binding(7)
var<storage, read> precomp_alpha_decay: array<f32>;

// ─── Bind Group 3: PML + shift operators + sensor/source (8 storage) ─────────

// pml_sgx/sgy/sgz: staggered PML for velocity (3 separate 3D arrays)
@group(2) @binding(0)
var<storage, read> pml_sgx: array<f32>;

@group(2) @binding(1)
var<storage, read> pml_sgy: array<f32>;

@group(2) @binding(2)
var<storage, read> pml_sgz: array<f32>;

// pml_xyz: packed collocated PML for density: [pml_x(N) | pml_y(N) | pml_z(N)]
@group(2) @binding(3)
var<storage, read> pml_xyz: array<f32>;

// shifts_all: packed shift operators for all 6 cases
// Layout (offsets in f32 elements, all 1D arrays):
//   [0 .. nx)        x_pos_re
//   [nx .. 2nx)      x_pos_im
//   [2nx .. 3nx)     x_neg_re
//   [3nx .. 4nx)     x_neg_im
//   [4nx .. 4nx+ny)  y_pos_re
//   [4nx+ny .. +2ny) y_pos_im
//   [4nx+2ny..+3ny)  y_neg_re
//   [4nx+3ny..+4ny)  y_neg_im
//   [4(nx+ny)..+nz)  z_pos_re
//   [4(nx+ny)+nz..+2nz) z_pos_im
//   [4(nx+ny)+2nz..+3nz) z_neg_re
//   [4(nx+ny)+3nz..+4nz) z_neg_im
@group(2) @binding(4)
var<storage, read> shifts_all: array<f32>;

// sensor_flat_indices[s] = flat index of sensor s in 3D grid
@group(2) @binding(5)
var<storage, read> sensor_flat_indices: array<u32>;

// sensor_data[s * Nt + step]
@group(2) @binding(6)
var<storage, read_write> sensor_data: array<f32>;

// source_data: packed [source_mask_indices(u32_as_f32, n_src) | source_signals(f32, n_src*Nt)]
// We use a single f32 buffer; mask indices are cast u32->f32 (bit-cast on read).
// Layout:
//   [0 .. n_src)           : source_mask_indices as bitcast<f32>(u32)
//   [n_src .. n_src + n_src*Nt) : source_signals[src * Nt + step]
// n_src is encoded in params.n_sensors as upper 16 bits (n_sensors | (n_src << 16))
// BUT for simplicity we pack n_src separately: use a source_count uniform is simpler.
// Instead: store all source data in a plain source_data array; n_src is params.n_sensors >> 16.
@group(2) @binding(7)
var<storage, read> source_data: array<f32>;

// ─── Shared memory for FFT ───────────────────────────────────────────────────

var<workgroup> sm_re: array<f32, 256>;
var<workgroup> sm_im: array<f32, 256>;

// ─── Helpers ──────────────────────────────────────────────────────────────────

// Read shift operator (re or im) for axis=ax, positive/negative, from shifts_all.
// ax_code: 0=x-pos, 1=y-pos, 2=z-pos, 3=x-neg, 4=y-neg, 5=z-neg
fn shift_re(ax_code: u32, idx: u32, nx: u32, ny: u32) -> f32 {
    var base: u32;
    if ax_code == 0u {         // x_pos_re at offset 0
        base = 0u;
    } else if ax_code == 1u {  // y_pos_re at offset 4*nx
        base = 4u * nx;
    } else if ax_code == 2u {  // z_pos_re at offset 4*(nx+ny)
        base = 4u * (nx + ny);
    } else if ax_code == 3u {  // x_neg_re at offset 2*nx
        base = 2u * nx;
    } else if ax_code == 4u {  // y_neg_re at offset 4*nx+2*ny
        base = 4u * nx + 2u * ny;
    } else {                   // z_neg_re at offset 4*(nx+ny)+2*nz
        base = 4u * (nx + ny) + 2u * (params.nz);
    }
    return shifts_all[base + idx];
}

fn shift_im(ax_code: u32, idx: u32, nx: u32, ny: u32) -> f32 {
    var base: u32;
    if ax_code == 0u {         // x_pos_im at offset nx
        base = nx;
    } else if ax_code == 1u {  // y_pos_im at offset 4*nx+ny
        base = 4u * nx + ny;
    } else if ax_code == 2u {  // z_pos_im at offset 4*(nx+ny)+nz
        base = 4u * (nx + ny) + params.nz;
    } else if ax_code == 3u {  // x_neg_im at offset 3*nx
        base = 3u * nx;
    } else if ax_code == 4u {  // y_neg_im at offset 4*nx+3*ny
        base = 4u * nx + 3u * ny;
    } else {                   // z_neg_im at offset 4*(nx+ny)+3*nz
        base = 4u * (nx + ny) + 3u * params.nz;
    }
    return shifts_all[base + idx];
}

// ─── Entry point: fft_1d_smem ─────────────────────────────────────────────────
//
// Shared-memory batched 1D Cooley-Tukey FFT.
// One workgroup per batch (params.n_batches total workgroups).
// Workgroup size: 64 threads.
//   - For n=128 (Y/Z axes, half_n=64): each thread handles exactly 1 butterfly
//     per stage → 100% active thread utilization.
//   - For n=256 (X axis, half_n=128): each thread handles 2 butterflies per
//     stage via the stride-64 inner loop → 2× warps in flight per stage.
// The 64-thread workgroup also doubles SM occupancy vs 128 threads for n=128
// cases, improving latency hiding on shared-memory bank conflicts.
//
// axis selects the transform dimension:
//   axis=0 (X): stride=ny*nz, batch_id indexes (iy, iz) pairs
//   axis=1 (Y): stride=nz,    batch_id indexes (ix, iz) pairs
//   axis=2 (Z): stride=1,     batch_id indexes (ix, iy) pairs
@compute @workgroup_size(64, 1, 1)
fn fft_1d_smem(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let batch_id  = wid.x + wid.y * 65535u;
    let local_tid = lid.x;
    let n         = params.n_fft;
    let log2n     = params.log2n;
    let nx        = params.nx;
    let ny        = params.ny;
    let nz        = params.nz;
    let ax        = params.axis;
    let inv       = params.inverse;

    if batch_id >= params.n_batches { return; }

    var stride: u32;
    var batch_base: u32;

    if ax == 0u {
        let iy = batch_id / nz;
        let iz = batch_id % nz;
        stride = ny * nz;
        batch_base = iy * nz + iz;
    } else if ax == 1u {
        let ix = batch_id / nz;
        let iz = batch_id % nz;
        stride = nz;
        batch_base = ix * ny * nz + iz;
    } else {
        let ix = batch_id / ny;
        let iy = batch_id % ny;
        stride = 1u;
        batch_base = ix * ny * nz + iy * nz;
    }

    let half_n = n >> 1u;

    // Load into shared memory with bit-reversal permutation
    var load_idx = local_tid;
    loop {
        if load_idx >= n { break; }
        let rev_idx = reverseBits(load_idx) >> (32u - log2n);
        let flat = batch_base + rev_idx * stride;
        sm_re[load_idx] = kspace_re[flat];
        sm_im[load_idx] = kspace_im[flat];
        load_idx += 64u;
    }

    workgroupBarrier();

    // Iterative Cooley-Tukey butterfly stages
    var s = 0u;
    loop {
        if s >= log2n { break; }
        let h = 1u << s;
        let group_sz = h << 1u;

        var tid = local_tid;
        loop {
            if tid >= half_n { break; }
            let group_idx = tid / h;
            let local_pos = tid % h;
            let even = group_idx * group_sz + local_pos;
            let odd  = even + h;

            var angle = -TWO_PI * f32(local_pos) / f32(group_sz);
            if inv != 0u { angle = -angle; }

            let w_re = cos(angle);
            let w_im = sin(angle);

            let e_re = sm_re[even];
            let e_im = sm_im[even];
            let o_re = sm_re[odd];
            let o_im = sm_im[odd];

            let wo_re = w_re * o_re - w_im * o_im;
            let wo_im = w_re * o_im + w_im * o_re;

            sm_re[even] = e_re + wo_re;
            sm_im[even] = e_im + wo_im;
            sm_re[odd]  = e_re - wo_re;
            sm_im[odd]  = e_im - wo_im;

            tid += 64u;
        }
        s += 1u;
        workgroupBarrier();
    }

    // IFFT normalization
    if inv != 0u {
        let inv_n = 1.0 / f32(n);
        var tid2 = local_tid;
        loop {
            if tid2 >= n { break; }
            sm_re[tid2] *= inv_n;
            sm_im[tid2] *= inv_n;
            tid2 += 64u;
        }
        workgroupBarrier();
    }

    // Write back
    var write_idx = local_tid;
    loop {
        if write_idx >= n { break; }
        let flat = batch_base + write_idx * stride;
        kspace_re[flat] = sm_re[write_idx];
        kspace_im[flat] = sm_im[write_idx];
        write_idx += 64u;
    }
}

// ─── Entry point: kspace_shift_apply ─────────────────────────────────────────
//
// Apply 1D staggered shift operator × kappa to k-space field in-place.
// Reads kspace_re/kspace_im, writes result back to kspace_re/kspace_im.
// (Each thread reads/writes its own index — no cross-thread hazard.)
//
// params.axis encoding (same as shift operator table):
//   0 = x-positive (dp/dx for velocity)
//   1 = y-positive
//   2 = z-positive
//   3 = x-negative (dux/dx for density)
//   4 = y-negative
//   5 = z-negative
@compute @workgroup_size(256, 1, 1)
fn kspace_shift_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nx  = params.nx;
    let ny  = params.ny;
    let nz  = params.nz;
    let total = nx * ny * nz;
    if idx >= total { return; }

    let ax = params.axis;  // 0..5

    // Determine 1D shift index along the relevant axis
    var shift_idx: u32;
    if ax == 0u || ax == 3u {
        shift_idx = idx / (ny * nz);   // ix
    } else if ax == 1u || ax == 4u {
        shift_idx = (idx % (ny * nz)) / nz; // iy
    } else {
        shift_idx = idx % nz;           // iz
    }

    let s_re = shift_re(ax, shift_idx, nx, ny);
    let s_im = shift_im(ax, shift_idx, nx, ny);
    let kap  = precomp_kappa[idx];

    let in_re = kspace_re[idx];
    let in_im = kspace_im[idx];

    let kap_re = kap * in_re;
    let kap_im = kap * in_im;

    // Write in-place — eliminates copy_kspace2_to_kspace dispatch
    kspace_re[idx] = kap_re * s_re - kap_im * s_im;
    kspace_im[idx] = kap_re * s_im + kap_im * s_re;
}

// ─── Entry point: velocity_update ────────────────────────────────────────────
//
// Split-field PML velocity update:
//   u_new = pml_sg^2 * u_old - dt * rho0_inv * pml_sg * grad_p
//
// grad_p is in kspace_re (IFFT result of in-place kspace_shift).
// Eliminates copy_kspace_to_scratch dispatch.
// params.axis: 0=update ux, 1=uy, 2=uz
@compute @workgroup_size(256, 1, 1)
fn velocity_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }

    let ax      = params.axis;
    let dt      = params.dt;
    let rho_inv = precomp_rho0_inv[idx];
    let grad    = kspace_re[idx];

    if ax == 0u {
        let pml   = pml_sgx[idx];
        field_ux[idx] = pml * pml * field_ux[idx] - dt * rho_inv * pml * grad;
    } else if ax == 1u {
        let pml   = pml_sgy[idx];
        field_uy[idx] = pml * pml * field_uy[idx] - dt * rho_inv * pml * grad;
    } else {
        let pml   = pml_sgz[idx];
        field_uz[idx] = pml * pml * field_uz[idx] - dt * rho_inv * pml * grad;
    }
}

// ─── Entry point: density_update ─────────────────────────────────────────────
//
// Split-field PML density update:
//   rho_new = pml^2 * rho_old - dt * rho0 * pml * div_u
//
// div_u is in kspace_re (IFFT result of in-place kspace_shift).
// Eliminates copy_kspace_to_scratch dispatch.
// params.axis: 0=update rhox, 1=rhoy, 2=rhoz
// pml_xyz packs [pml_x | pml_y | pml_z], each of length nx*ny*nz.
@compute @workgroup_size(256, 1, 1)
fn density_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }

    let ax    = params.axis;
    let dt    = params.dt;
    let rho0  = precomp_rho0[idx];
    let div_u = kspace_re[idx];
    let pml   = pml_xyz[ax * total + idx];  // axis selects which pml array

    if ax == 0u {
        field_rhox[idx] = pml * pml * field_rhox[idx] - dt * rho0 * pml * div_u;
    } else if ax == 1u {
        field_rhoy[idx] = pml * pml * field_rhoy[idx] - dt * rho0 * pml * div_u;
    } else {
        field_rhoz[idx] = pml * pml * field_rhoz[idx] - dt * rho0 * pml * div_u;
    }
}

// ─── Entry point: pressure_from_density ──────────────────────────────────────
//
// Linear:    p = c0² · (rhox + rhoy + rhoz)
// Nonlinear: p = c0² · (rho_total + B/(2A) · rho_total² / rho0)   [Westervelt EOS]
//
// Reference: Treeby & Cox (2010), Eq. (2.14).
@compute @workgroup_size(256, 1, 1)
fn pressure_from_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }

    let rho_total = field_rhox[idx] + field_rhoy[idx] + field_rhoz[idx];
    let c2 = precomp_c0_sq[idx];
    var corrected = rho_total;
    if params.nonlinear != 0u {
        let bon_a = precomp_bon_a[idx];
        let rho0  = precomp_rho0[idx];
        corrected = rho_total + bon_a * rho_total * rho_total / rho0;
    }
    field_p[idx] = c2 * corrected;
}


// ─── Entry point: record_sensors ─────────────────────────────────────────────
//
// sensor_data[s * Nt + step] = p[sensor_flat_indices[s]]
@compute @workgroup_size(256, 1, 1)
fn record_sensors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.n_sensors { return; }
    let flat = sensor_flat_indices[s];
    sensor_data[s * params.nt + params.step] = field_p[flat];
}

// ─── Entry point: apply_absorption ───────────────────────────────────────────
//
// Multiplicative decay applied to all three split density components per step.
// decay[i] = exp(-alpha_Np_m * c0[i] * dt) precomputed at the transmit centre
// frequency f0 using the power-law: alpha_Np_m = alpha_dB_cm * f0_MHz^y / 868.6.
//
// Applied once per time step after all three density_update dispatches,
// before pressure_from_density. First-order time-domain absorption approximation.
//
// Reference: Treeby & Cox (2010), Eq. (A.12); O'Brien et al. (2010) JASA.
@compute @workgroup_size(256, 1, 1)
fn apply_absorption(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }
    let decay = precomp_alpha_decay[idx];
    field_rhox[idx] *= decay;
    field_rhoy[idx] *= decay;
    field_rhoz[idx] *= decay;
}

// ─── Entry point: zero_acoustic_fields ───────────────────────────────────────
//
// Zeros all 7 acoustic field buffers (p, ux, uy, uz, rhox, rhoy, rhoz) in one
// GPU pass. Replaces 7 CPU write_buffer() calls (~112 MB PCIe upload) with
// a sub-millisecond GPU memory-fill (GPU VRAM bandwidth ~500 GB/s).
//
@compute @workgroup_size(256, 1, 1)
fn zero_acoustic_fields(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }
    field_p[idx]    = 0.0;
    field_ux[idx]   = 0.0;
    field_uy[idx]   = 0.0;
    field_uz[idx]   = 0.0;
    field_rhox[idx] = 0.0;
    field_rhoy[idx] = 0.0;
    field_rhoz[idx] = 0.0;
}

// ─── Entry point: inject_pressure_source ─────────────────────────────────────
//
// Additive source injection into all three split density components.
//
// source_data layout (all f32):
//   [0 .. n_src)           : bitcast<f32>(source_mask_index[i]) — flat grid indices
//   [n_src .. n_src*(1+Nt)): source_signals[src_pt * Nt + step]
//
// n_src is encoded in params.axis for this shader (axis field repurposed here).
@compute @workgroup_size(256, 1, 1)
fn inject_pressure_source(@builtin(global_invocation_id) gid: vec3<u32>) {
    let src_pt = gid.x;
    let n_src  = params.axis;  // repurposed: n_src passed via axis field
    if src_pt >= n_src { return; }

    // Read flat index (stored as bitcast f32 of u32)
    let flat = bitcast<u32>(source_data[src_pt]);
    let amp  = source_data[n_src + src_pt * params.nt + params.step];

    field_rhox[flat] += amp;
    field_rhoy[flat] += amp;
    field_rhoz[flat] += amp;
}

// ─── Entry point: inject_velocity_x_source ───────────────────────────────────
//
// Additive velocity-x source injection (for transducer ux-driven simulations).
// Injects directly into field_ux at each source grid point.
//
// source_data layout (identical to inject_pressure_source):
//   [0 .. n_src)           : bitcast<f32>(source_mask_index[i]) — flat grid indices
//   [n_src .. n_src*(1+Nt)): source_signals[src_pt * Nt + step]
//
// n_src is encoded in params.axis (same convention as inject_pressure_source).
@compute @workgroup_size(256, 1, 1)
fn inject_velocity_x_source(@builtin(global_invocation_id) gid: vec3<u32>) {
    let src_pt = gid.x;
    let n_src  = params.axis;
    if src_pt >= n_src { return; }

    let flat = bitcast<u32>(source_data[src_pt]);
    let amp  = source_data[n_src + src_pt * params.nt + params.step];

    kspace_re[flat] += amp;
}

// ─── Entry point: apply_source_kappa ─────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn apply_source_kappa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }

    let kap = precomp_source_kappa[idx];
    kspace_re[idx] *= kap;
    kspace_im[idx] *= kap;
}

// ─── Entry point: add_kspace_to_field_ux ─────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn add_kspace_to_field_ux(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }

    field_ux[idx] += kspace_re[idx];
}

// ─── Helper: copy real field to kspace (real part), zero imaginary ────────────
//
// params.axis selects which field: 0=p, 1=ux, 2=uy, 3=uz
@compute @workgroup_size(256, 1, 1)
fn copy_field_to_kspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }

    let ax  = params.axis;
    var val: f32;
    if ax == 0u {
        val = field_p[idx];
    } else if ax == 1u {
        val = field_ux[idx];
    } else if ax == 2u {
        val = field_uy[idx];
    } else {
        val = field_uz[idx];
    }
    kspace_re[idx] = val;
    kspace_im[idx] = 0.0;
}
