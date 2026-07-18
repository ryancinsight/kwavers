// PSTD GPU compute shader for split-field acoustic propagation.
// Detailed buffer ABI: docs/gpu/pstd_shader_abi.md.
// References: Treeby & Cox (2010) JBO 15(2), 021314; Liu (1998) MOTL 15(3).

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

// ─── Immediate data: per-dispatch params ─────────────────────────────────────
// Replaces the group(1) uniform buffer. Immediate data is embedded directly
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
    peak_offset: u32,
    record_peak_pressure: u32,
}

var<immediate> params: PstdParams;

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

// binding(7): packed real/imaginary roots for the shared-memory FFT.
@group(1) @binding(7)
var<storage, read> precomp_twiddle_fft: array<f32>;

// ─── Bind Group 2: PML + shift operators + sensor/source (8 storage) ─────────

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

// shifts_all packs x/y/z positive and negative staggered phase shifts.
@group(2) @binding(4)
var<storage, read> shifts_all: array<f32>;

// sensor_flat_indices[s] = flat index of sensor s in 3D grid
@group(2) @binding(5)
var<storage, read> sensor_flat_indices: array<u32>;

// sensor_data[s * Nt + step]
@group(2) @binding(6)
var<storage, read_write> sensor_data: array<f32>;

// source_data packs bitcast source indices followed by source_signals[src, t].
// Source dispatches pass n_src through params.axis.
@group(2) @binding(7)
var<storage, read> source_data: array<f32>;

// ─── Shared memory for FFT ───────────────────────────────────────────────────

const MAX_FFT_LENGTH: u32 = 1024u;
const MAX_FFT_HALF_LENGTH: u32 = MAX_FFT_LENGTH >> 1u;

var<workgroup> sm_re: array<f32, 1024>;
var<workgroup> sm_im: array<f32, 1024>;
// A single 1,024-point root table serves every smaller power-of-two transform
// by striding its entries. IFFT uses conjugates by negating the loaded
// imaginary component.
var<workgroup> sm_tw_re: array<f32, 512>;
var<workgroup> sm_tw_im: array<f32, 512>;

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
// Shared-memory batched 1D Cooley-Tukey FFT with precomputed twiddles.
// axis=0/1/2 selects x/y/z; one workgroup owns one transform lane.
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

    // ── Load twiddle factors into shared memory ───────────────────────────────
    let n_half = n >> 1u;
    let root_stride = MAX_FFT_LENGTH / n;
    var tw_tid = local_tid;
    loop {
        if tw_tid >= n_half { break; }
        let root_index = tw_tid * root_stride;
        sm_tw_re[tw_tid] = precomp_twiddle_fft[root_index];
        sm_tw_im[tw_tid] = precomp_twiddle_fft[MAX_FFT_HALF_LENGTH + root_index];
        tw_tid += 64u;
    }

    // ── Compute stride and batch base ─────────────────────────────────────────
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

    // ── Load data into shared memory with bit-reversal permutation ────────────
    var load_idx = local_tid;
    loop {
        if load_idx >= n { break; }
        let rev_idx = reverseBits(load_idx) >> (32u - log2n);
        let flat = batch_base + rev_idx * stride;
        sm_re[load_idx] = kspace_re[flat];
        sm_im[load_idx] = kspace_im[flat];
        load_idx += 64u;
    }

    // Single barrier synchronizes both twiddle and data loads before butterflies.
    workgroupBarrier();

    // ── Iterative Cooley-Tukey butterfly stages ───────────────────────────────
    // Twiddle for stage s at butterfly position local_pos:
    //   tw_idx = local_pos * (n >> (s+1))   [stride within n-point twiddle table]
    //   w = (sm_tw_re[tw_idx], ±sm_tw_im[tw_idx])  (+im for IFFT, −im for forward)
    var s = 0u;
    loop {
        if s >= log2n { break; }
        let h = 1u << s;
        let group_sz = h << 1u;
        let tw_stride = n >> (s + 1u);  // = n / group_sz

        var tid = local_tid;
        loop {
            if tid >= n_half { break; }
            let group_idx = tid / h;
            let local_pos = tid % h;
            let even = group_idx * group_sz + local_pos;
            let odd  = even + h;

            let tw_idx = local_pos * tw_stride;
            let w_re = sm_tw_re[tw_idx];
            var w_im = sm_tw_im[tw_idx];
            if inv != 0u { w_im = -w_im; }   // IFFT: conjugate twiddle

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

    // ── IFFT normalization ────────────────────────────────────────────────────
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

    // ── Write back ────────────────────────────────────────────────────────────
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

// ─── Entry point: restore_and_shift_apply ────────────────────────────────────
//
// Fused restore-kspace + kspace-shift for velocity axes 1 and 2.
//
// ## Theorem (global-memory traffic reduction)
//
// For velocity axes 1 and 2, the naive sequence is:
//   1. restore_kspace  : reads absorb_scratch_kre/kim (2N f32),
//                        writes kspace_re/im            (2N f32) → 4N f32
//   2. kspace_shift    : reads kspace_re/im + precomp   (2N+3N f32),
//                        writes kspace_re/im in-place   (2N f32) → 7N f32
//   Total: 11N f32 traffic (4N intermediate kspace is pure overhead).
//
// The fused kernel reads absorb_scratch_kre/kim directly, applies kappa × shift,
// and writes the result into kspace_re/im without the intermediate kspace store:
//   reads: absorb_scratch_kre/im (2N) + precomp_kappa/shift (3N) = 5N f32
//   writes: kspace_re/im (2N f32)
//   Total: 7N f32 — saves 4N f32 per axis and eliminates 1 dispatch per axis.
//
// Called for axes 1 and 2 in velocity_update (axis 0 keeps kspace from FFT(p)
// and uses plain kspace_shift_apply since no restore is needed).
//
// params.axis: same encoding as kspace_shift_apply (0=x-pos, 1=y-pos, 2=z-pos)
//
// Requires bind group 3 (absorb) for absorb_scratch_kre/kim (bindings 4/5).
@compute @workgroup_size(256, 1, 1)
fn restore_and_shift_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let nx    = params.nx;
    let ny    = params.ny;
    let nz    = params.nz;
    let total = nx * ny * nz;
    if idx >= total { return; }

    let ax = params.axis;

    var shift_idx: u32;
    if ax == 0u || ax == 3u {
        shift_idx = idx / (ny * nz);
    } else if ax == 1u || ax == 4u {
        shift_idx = (idx % (ny * nz)) / nz;
    } else {
        shift_idx = idx % nz;
    }

    let s_re = shift_re(ax, shift_idx, nx, ny);
    let s_im = shift_im(ax, shift_idx, nx, ny);
    let kap  = precomp_kappa[idx];

    // Read from absorb_scratch (saved FFT(p)) instead of the intermediate kspace.
    let in_re = absorb_scratch_kre[idx];
    let in_im = absorb_scratch_kim[idx];

    let kap_re = kap * in_re;
    let kap_im = kap * in_im;

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

// ─── Entry point: snapshot_rho0_plus_rho ─────────────────────────────────────
//
// Precomputes the nonlinear mass-conservation coefficient
//   rho0_plus_rho = 2 * (rhox + rhoy + rhoz) + rho0
// into field_p, to be consumed by the three density_update dispatches that
// follow. field_p is free at this point in the step (it still holds the
// previous step's pressure, already recorded by sensors; pressure_from_density
// overwrites it after the density updates complete).
//
// Matches k-Wave MATLAB kspaceFirstOrder3D.m nonlinear mass conservation
// (lines 919–924). Must be dispatched only when params.nonlinear != 0 and
// must run BEFORE the three density_update dispatches so all three axes see
// the same pre-update rho_total.
//
// Reference: Treeby & Cox (2010), Eq. (A.6); k-Wave MATLAB commit b0ee57c.
@compute @workgroup_size(256, 1, 1)
fn snapshot_rho0_plus_rho(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }
    let rho_total = field_rhox[idx] + field_rhoy[idx] + field_rhoz[idx];
    field_p[idx] = 2.0 * rho_total + precomp_rho0[idx];
}

// ─── Entry point: density_update ─────────────────────────────────────────────
//
// Split-field PML density update:
//   Linear:    rho_new = pml^2 * rho_old - dt * rho0                   * pml * div_u
//   Nonlinear: rho_new = pml^2 * rho_old - dt * (2*(rhox+rhoy+rhoz)+rho0) * pml * div_u
//
// In the nonlinear branch the mass-conservation coefficient uses the
// TOTAL pre-update density sum (rhox+rhoy+rhoz) snapshotted into field_p by
// snapshot_rho0_plus_rho before the three density_update dispatches.  This
// matches k-Wave C++ OMP computeDensityNonliner exactly:
//
//   sumRhos = rhoX + rhoY + rhoZ   (pre-update, same for all axes)
//   rhoX -= dt * (2*sumRhos + rho0) * duxdx
//   rhoY -= dt * (2*sumRhos + rho0) * duydy
//   rhoZ -= dt * (2*sumRhos + rho0) * duzdz
//
// This is the physically correct nonlinear continuity formulation from
// Treeby & Cox (2010) Eq. (A.3):
//   d(rho_i)/dt = -(rho0 + sum_j rho_j) * du_i/dx_i
//
// field_p holds snapshot_rho0_plus_rho = 2*(rhox+rhoy+rhoz)+rho0 at this
// point in the step (overwritten by pressure_from_density later).
//
// div_u is in kspace_re (IFFT result of in-place kspace_shift).
// params.axis: 0=update rhox, 1=rhoy, 2=rhoz
// pml_xyz packs [pml_x | pml_y | pml_z], each of length nx*ny*nz.
@compute @workgroup_size(256, 1, 1)
fn density_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }

    let ax    = params.axis;
    let dt    = params.dt;
    let div_u = kspace_re[idx];
    let pml   = pml_xyz[ax * total + idx];  // axis selects which pml array

    // Nonlinear mass conservation: use total pre-update density sum from
    // snapshot_rho0_plus_rho (stored in field_p before density loop begins).
    // This matches k-Wave C++ OMP and Treeby & Cox (2010) Eq. (A.3).
    // Linear path uses unperturbed rho0 only.
    var mass_coef: f32;
    if params.nonlinear != 0u {
        // field_p holds 2*(rhox+rhoy+rhoz)+rho0 (snapshotted before density loop)
        mass_coef = field_p[idx];
    } else {
        mass_coef = precomp_rho0[idx];
    }

    if ax == 0u {
        field_rhox[idx] = pml * pml * field_rhox[idx] - dt * mass_coef * pml * div_u;
    } else if ax == 1u {
        field_rhoy[idx] = pml * pml * field_rhoy[idx] - dt * mass_coef * pml * div_u;
    } else {
        field_rhoz[idx] = pml * pml * field_rhoz[idx] - dt * mass_coef * pml * div_u;
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

// ─── Entry point: accumulate_peak_pressure ───────────────────────────────────
//
// The requested peak envelope is `max_t |p|`. It occupies the run-local output
// buffer after the sensor-trace region so sensor-only runs retain their former
// allocation and transfer contract.
@compute @workgroup_size(256, 1, 1)
fn accumulate_peak_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total || params.record_peak_pressure == 0u { return; }
    let output_idx = params.peak_offset + idx;
    sensor_data[output_idx] = max(sensor_data[output_idx], abs(field_p[idx]));
}

// ─── Entry point: zero_kspace ────────────────────────────────────────────────
//
// Zeroes kspace_re and kspace_im in one GPU pass.
// Used before velocity-x source injection so that inject_velocity_x_source
// can accumulate into a clean kspace buffer, without requiring a CPU-side
// encoder.clear_buffer() call that would break single-pass-per-step dispatch.
@compute @workgroup_size(256, 1, 1)
fn zero_kspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }
    kspace_re[idx] = 0.0;
    kspace_im[idx] = 0.0;
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
// Additive pressure-source injection into the k-space source work buffer.
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

    kspace_re[flat] += amp;
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

// ─── Entry point: add_kspace_to_density ─────────────────────────────────────
//
// The source work buffer holds the k-space-corrected additive mass source.
// Each active split-density component receives the same increment, matching
// the CPU PSTD source contract and preserving the total EOS contribution.
@compute @workgroup_size(256, 1, 1)
fn add_kspace_to_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.nx * params.ny * params.nz;
    if idx >= total { return; }

    let source = kspace_re[idx];
    field_rhox[idx] += source;
    if params.ny > 1u {
        field_rhoy[idx] += source;
    }
    if params.nz > 1u {
        field_rhoz[idx] += source;
    }
}

// ─── Bind Group 3: Fractional-Laplacian absorption operators ─────────────────
//
// Used when `params.absorbing != 0`. The pipeline layout for absorption
// shaders declares this group at index 3. Non-absorption pipelines use the
// 3-group layout and never access these bindings.
//
// Layout:
//   binding 0: absorb_nabla1    — |k|^(y−2) in FFT order (read-only)
//   binding 1: absorb_nabla2    — |k|^(y−1) in FFT order (read-only)
//   binding 2: absorb_tau       — −2α₀c₀^(y−1) per voxel (read-only)
//   binding 3: absorb_eta       — 2α₀c₀^y·tan(πy/2) per voxel (read-only)
//   binding 4: absorb_scratch_kre — temp complex re save (read_write)
//   binding 5: absorb_scratch_kim — temp complex im save (read_write)
//   binding 6: absorb_scratch_l1  — L1 = IFFT(nabla1·FFT(div_u)) (read_write)
//   binding 7: absorb_scratch_l2  — L2 = IFFT(nabla2·FFT(div_u)) (read_write)
//
// References: Treeby & Cox (2010) Eqs. 9–10, 19–21.

@group(3) @binding(0)
var<storage, read> absorb_nabla1: array<f32>;

@group(3) @binding(1)
var<storage, read> absorb_nabla2: array<f32>;

@group(3) @binding(2)
var<storage, read> absorb_tau: array<f32>;

@group(3) @binding(3)
var<storage, read> absorb_eta: array<f32>;

@group(3) @binding(4)
var<storage, read_write> absorb_scratch_kre: array<f32>;

@group(3) @binding(5)
var<storage, read_write> absorb_scratch_kim: array<f32>;

@group(3) @binding(6)
var<storage, read_write> absorb_scratch_l1: array<f32>;

@group(3) @binding(7)
var<storage, read_write> absorb_scratch_l2: array<f32>;

// ─── Entry point: absorb_save_kspace ─────────────────────────────────────────
//
// Save kspace_re/im → absorb_scratch_kre/kim.
// Called after kspace_shift_apply to preserve FFT(div_u) before nabla
// multiplication overwrites kspace_re/im.
@compute @workgroup_size(256, 1, 1)
fn absorb_save_kspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.nx * params.ny * params.nz { return; }
    absorb_scratch_kre[idx] = kspace_re[idx];
    absorb_scratch_kim[idx] = kspace_im[idx];
}

// ─── Entry point: absorb_mul_nabla ───────────────────────────────────────────
//
// Multiply kspace_re/im by a real-valued nabla operator in-place.
// params.axis: 0 → multiply by nabla1 = |k|^(y−2)
//              1 → multiply by nabla2 = |k|^(y−1)
@compute @workgroup_size(256, 1, 1)
fn absorb_mul_nabla(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.nx * params.ny * params.nz { return; }
    var n: f32;
    if params.axis == 0u {
        n = absorb_nabla1[idx];
    } else {
        n = absorb_nabla2[idx];
    }
    kspace_re[idx] *= n;
    kspace_im[idx] *= n;
}

// ─── Entry point: absorb_copy_to_scratch ─────────────────────────────────────
//
// Copy kspace_re (= IFFT result = Lα) to a scratch buffer.
// params.axis: 0 → copy to absorb_scratch_l1 (L1 = IFFT(nabla1·FFT(div_u)))
//              1 → copy to absorb_scratch_l2 (L2 = IFFT(nabla2·FFT(div_u)))
@compute @workgroup_size(256, 1, 1)
fn absorb_copy_to_scratch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.nx * params.ny * params.nz { return; }
    if params.axis == 0u {
        absorb_scratch_l1[idx] = kspace_re[idx];
    } else {
        absorb_scratch_l2[idx] = kspace_re[idx];
    }
}

// ─── Entry point: absorb_restore_kspace ──────────────────────────────────────
//
// Restore kspace_re/im from absorb_scratch_kre/kim.
// Called after each IFFT to reset kspace to FFT(div_u) for the next nabla pass.
@compute @workgroup_size(256, 1, 1)
fn absorb_restore_kspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.nx * params.ny * params.nz { return; }
    kspace_re[idx] = absorb_scratch_kre[idx];
    kspace_im[idx] = absorb_scratch_kim[idx];
}

// ─── Entry point: absorb_accum_div_u ─────────────────────────────────────────
//
// Accumulate per-axis velocity divergence into absorb_scratch_kre.
//
// After each density-loop IFFT, kspace_re holds the kappa-corrected velocity
// divergence ∂u_α/∂α for axis α. This shader accumulates it so that after all
// 3 axes, absorb_scratch_kre = div_u_total = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z.
//
// params.axis encodes the current density-loop axis (0, 1, or 2):
//   axis=0: WRITE  scratch_kre = kspace_re       (initialize, avoids a GPU buffer-clear call)
//   axis=1: ADD    scratch_kre += kspace_re
//   axis=2: ADD    scratch_kre += kspace_re
//
// Using axis=0 to initialize eliminates the need for a CommandEncoder::clear_buffer()
// call before the density loop, which can cause implicit pipeline barriers in some
// wgpu backends (D3D12) and significantly degrade GPU throughput.
@compute @workgroup_size(256, 1, 1)
fn absorb_accum_div_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.nx * params.ny * params.nz { return; }
    if params.axis == 0u {
        absorb_scratch_kre[idx] = kspace_re[idx];   // initialize on first axis
    } else {
        absorb_scratch_kre[idx] += kspace_re[idx];  // accumulate for axes 1 and 2
    }
}

// ─── Entry point: absorb_prep_l1_kspace ──────────────────────────────────────
//
// Prepare kspace for the L1 = IFFT(nabla1 · FFT(ρ₀ · div_u_total)) computation.
//
// Sets kspace_re = ρ₀ × div_u_total  (from absorb_scratch_kre),  kspace_im = 0.
// This is the first operand needed by k-Wave's power-law pressure formula
// (computePressureTermsLinearPowerLaw: pVelocityGradientSum = rho0 * div_u).
@compute @workgroup_size(256, 1, 1)
fn absorb_prep_l1_kspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.nx * params.ny * params.nz { return; }
    kspace_re[idx] = precomp_rho0[idx] * absorb_scratch_kre[idx];
    kspace_im[idx] = 0.0;
}

// ─── Entry point: absorb_prep_l2_kspace ──────────────────────────────────────
//
// Prepare kspace for the L2 = IFFT(nabla2 · FFT(ρ_total)) computation.
//
// Sets kspace_re = ρx + ρy + ρz (density total, post-update),  kspace_im = 0.
// This is the second operand needed by k-Wave's power-law pressure formula
// (computePressureTermsLinearPowerLaw: pDensitySum = rhoX + rhoY + rhoZ).
@compute @workgroup_size(256, 1, 1)
fn absorb_prep_l2_kspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.nx * params.ny * params.nz { return; }
    kspace_re[idx] = field_rhox[idx] + field_rhoy[idx] + field_rhoz[idx];
    kspace_im[idx] = 0.0;
}

// ─── Entry point: absorb_pressure_correction ─────────────────────────────────
//
// Add fractional-Laplacian absorption correction to pressure.
//
// Implements k-Wave C++ sumPressureTermsLinear:
//   p += c₀² · (τ · L1  −  η · L2)
//
// where L1 = absorb_scratch_l1 = IFFT(nabla1 · FFT(ρ₀ · div_u_total))
//       L2 = absorb_scratch_l2 = IFFT(nabla2 · FFT(ρ_total))
//
// This formulation applies absorption instantaneously to pressure (not density),
// exactly matching k-Wave C++ OMP `computePressureLinearPowerLaw`.
// No Δt factor — unlike the per-density approach, this is not a running integral.
//
// Reference: Treeby & Cox (2010) Eq. 19–21; k-Wave C++ OMP KSpaceFirstOrderSolver.cpp
//   computePressureTermsLinearPowerLaw + computePowerLawAbsorbtionTerm + sumPressureTermsLinear.
@compute @workgroup_size(256, 1, 1)
fn absorb_pressure_correction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.nx * params.ny * params.nz { return; }
    let c2 = precomp_c0_sq[idx];
    field_p[idx] += c2 * (absorb_tau[idx] * absorb_scratch_l1[idx]
                        - absorb_eta[idx] * absorb_scratch_l2[idx]);
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
