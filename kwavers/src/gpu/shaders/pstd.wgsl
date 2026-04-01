// PSTD (Pseudospectral Time Domain) GPU Compute Shaders
//
// Implements a GPU-resident acoustic PSTD solver with split-field PML.
//
// Bind group layout (≤8 storage buffers per group, ≤4 groups):
//
//   group(0): field buffers (8 storage: p, ux, uy, uz, rhox, rhoy, rhoz, scratch)
//   group(1): uniform params
//   group(2): k-space + medium scalars (8 storage: kspace_re, kspace_im,
//             kspace2_re, kspace2_im, kappa, rho0_inv, c0_sq, rho0)
//   group(3): PML + shift + sensor (8 storage: pml_sgx, pml_sgy, pml_sgz,
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

// ─── Bind Group 0: Field buffers (8 storage) ─────────────────────────────────

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

@group(0) @binding(7)
var<storage, read_write> field_scratch: array<f32>;

// ─── Bind Group 1: Uniform params ────────────────────────────────────────────

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
}

@group(1) @binding(0)
var<uniform> params: PstdParams;

// ─── Bind Group 2: K-space + medium scalars (8 storage) ──────────────────────

@group(2) @binding(0)
var<storage, read_write> kspace_re: array<f32>;

@group(2) @binding(1)
var<storage, read_write> kspace_im: array<f32>;

@group(2) @binding(2)
var<storage, read_write> kspace2_re: array<f32>;

@group(2) @binding(3)
var<storage, read_write> kspace2_im: array<f32>;

@group(2) @binding(4)
var<storage, read> precomp_kappa: array<f32>;

@group(2) @binding(5)
var<storage, read> precomp_rho0_inv: array<f32>;

@group(2) @binding(6)
var<storage, read> precomp_c0_sq: array<f32>;

@group(2) @binding(7)
var<storage, read> precomp_rho0: array<f32>;

// ─── Bind Group 3: PML + shift operators + sensor/source (8 storage) ─────────

// pml_sgx/sgy/sgz: staggered PML for velocity (3 separate 3D arrays)
@group(3) @binding(0)
var<storage, read> pml_sgx: array<f32>;

@group(3) @binding(1)
var<storage, read> pml_sgy: array<f32>;

@group(3) @binding(2)
var<storage, read> pml_sgz: array<f32>;

// pml_xyz: packed collocated PML for density: [pml_x(N) | pml_y(N) | pml_z(N)]
@group(3) @binding(3)
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
@group(3) @binding(4)
var<storage, read> shifts_all: array<f32>;

// sensor_flat_indices[s] = flat index of sensor s in 3D grid
@group(3) @binding(5)
var<storage, read> sensor_flat_indices: array<u32>;

// sensor_data[s * Nt + step]
@group(3) @binding(6)
var<storage, read_write> sensor_data: array<f32>;

// source_data: packed [source_mask_indices(u32_as_f32, n_src) | source_signals(f32, n_src*Nt)]
// We use a single f32 buffer; mask indices are cast u32->f32 (bit-cast on read).
// Layout:
//   [0 .. n_src)           : source_mask_indices as bitcast<f32>(u32)
//   [n_src .. n_src + n_src*Nt) : source_signals[src * Nt + step]
// n_src is encoded in params.n_sensors as upper 16 bits (n_sensors | (n_src << 16))
// BUT for simplicity we pack n_src separately: use a source_count uniform is simpler.
// Instead: store all source data in a plain source_data array; n_src is params.n_sensors >> 16.
@group(3) @binding(7)
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
// Workgroup size: 128 threads.
// Operates on kspace_re / kspace_im in-place.
//
// axis selects the transform dimension:
//   axis=0 (X): stride=ny*nz, batch_id indexes (iy, iz) pairs
//   axis=1 (Y): stride=nz,    batch_id indexes (ix, iz) pairs
//   axis=2 (Z): stride=1,     batch_id indexes (ix, iy) pairs
@compute @workgroup_size(128, 1, 1)
fn fft_1d_smem(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let batch_id  = wid.x;
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
        load_idx += 128u;
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

            tid += 128u;
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
            tid2 += 128u;
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
        write_idx += 128u;
    }
}

// ─── Entry point: kspace_shift_apply ─────────────────────────────────────────
//
// Apply 1D staggered shift operator × kappa to k-space field.
// Reads kspace_re/kspace_im, writes to kspace2_re/kspace2_im.
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

    kspace2_re[idx] = kap_re * s_re - kap_im * s_im;
    kspace2_im[idx] = kap_re * s_im + kap_im * s_re;
}

// ─── Entry point: velocity_update ────────────────────────────────────────────
//
// Split-field PML velocity update:
//   u_new = pml_sg^2 * u_old - dt * rho0_inv * pml_sg * grad_p
//
// grad_p is in field_scratch (IFFT result of kspace_shift → kspace → scratch).
// params.axis: 0=update ux, 1=uy, 2=uz
@compute @workgroup_size(256, 1, 1)
fn velocity_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }

    let ax      = params.axis;
    let dt      = params.dt;
    let rho_inv = precomp_rho0_inv[idx];
    let grad    = field_scratch[idx];

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
// div_u is in field_scratch.
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
    let div_u = field_scratch[idx];
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
// p = c0_sq * (rhox + rhoy + rhoz)
@compute @workgroup_size(256, 1, 1)
fn pressure_from_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }

    let c2 = precomp_c0_sq[idx];
    field_p[idx] = c2 * (field_rhox[idx] + field_rhoy[idx] + field_rhoz[idx]);
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

// ─── Helper: copy kspace2 -> kspace ──────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn copy_kspace2_to_kspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }
    kspace_re[idx] = kspace2_re[idx];
    kspace_im[idx] = kspace2_im[idx];
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

// ─── Helper: copy kspace_re to scratch ────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn copy_kspace_to_scratch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }
    field_scratch[idx] = kspace_re[idx];
}
