// 3D MVDR (Minimum Variance Distortionless Response / Capon) Beamforming Compute Shader
//
// ## Algorithm (Capon 1969, Synnevåg et al. 2007)
//
// For each voxel v (one GPU invocation):
//
//   1. τᵢ = ‖pᵥ − eᵢ‖ / (c · Δt)       receive delay (samples) for element i
//
//   2. Spatially-smoothed covariance (Shan & Kailath 1985):
//        R̂ = (1/Q) Σ_q R_q,  R_q = X_q · X_qᵀ / N
//        X_q[i,n] = (1/F) Σ_f x_i^f(τ_i + n)
//
//   3. Diagonal loading:   R_δ = R̂ + δ · (tr(R̂)/L) · I_L
//
//   4. Cholesky-Banachiewicz:  R_δ = L · Lᵀ
//
//   5. Solve L y = 1 (forward), then Lᵀ u = y (backward).
//
//   6. P = 1 / (1ᵀ u);   output = |P · uᵀ x̄₀|
//      where x̄₀[i] = (1/F) Σ_f x_i^f(τᵢ) at sub-aperture 0.
//
// MAX_L = 32: sub_x * sub_y * sub_z must not exceed 32.
// workgroup_size = (1, 1, 1): sequential per-voxel work; parallelism is across voxels.
//
// ## References
// - Capon J. (1969) High-resolution frequency-wavenumber spectrum analysis. IEEE Proc. 57(8).
// - Synnevåg J.F., Austeng A., Holm S. (2007) IEEE TUFFC 54(8) 1606–1613.
// - Shan T.J., Kailath T. (1985) IEEE TASSP 33(3) 527–536.

// Uniform parameters — must match `MvdrGpuParams` in `mvdr/params.rs` exactly.
// 24 scalar fields × 4 bytes = 96 bytes (6 × 16-byte uniform rows).
struct MvdrParams {
    vol_x: u32, vol_y: u32, vol_z: u32, nel_x: u32,
    nel_y: u32, nel_z: u32, sub_x: u32, sub_y: u32,
    sub_z: u32, num_frames: u32, num_samples: u32, pad0: u32,
    vox_dx: f32, vox_dy: f32, vox_dz: f32, elem_sx: f32,
    elem_sy: f32, elem_sz: f32, sound_speed: f32, sampling_freq: f32,
    diagonal_loading: f32, pad1: f32, pad2: f32, pad3: f32,
}

// RF data: [frames × channels × samples] (trailing dim-1 stripped from the 4-D host array).
@group(0) @binding(0) var<storage, read>       rf_data: array<f32>;
// Reconstructed voxel amplitudes: [vol_x × vol_y × vol_z] row-major.
@group(0) @binding(1) var<storage, read_write> output:  array<f32>;
// Configuration uniform.
@group(0) @binding(2) var<uniform>             params:  MvdrParams;

// Per-invocation private storage (zero-initialised per WGSL spec §7.2.3).
// Sized for MAX_L = 32; host code validates L ≤ 32 before dispatch.
var<private> r_mat: array<f32, 1024>; // 32×32 spatially-smoothed covariance (row-major)
var<private> chol:  array<f32, 1024>; // 32×32 Cholesky factor L (lower triangular)
var<private> u_vec: array<f32, 32>;   // forward-sub intermediate y, then MVDR weights u
var<private> x_bar: array<f32, 32>;   // mean delay-aligned signal for sub-aperture 0

// Linear interpolation into the RF buffer.
// Returns 0 outside the valid sample window [0, num_samples − 2].
fn rf_interp(frame: u32, ch: u32, tau: f32) -> f32 {
    if tau < 0.0 { return 0.0; }
    let n0 = u32(tau);
    if n0 + 1u >= params.num_samples { return 0.0; }
    let alpha = tau - f32(n0);
    let M    = params.nel_x * params.nel_y * params.nel_z;
    let base = frame * M * params.num_samples + ch * params.num_samples;
    return rf_data[base + n0] + alpha * (rf_data[base + n0 + 1u] - rf_data[base + n0]);
}

@compute @workgroup_size(1, 1, 1)
fn mvdr_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let vx = id.x;
    let vy = id.y;
    let vz = id.z;
    if vx >= params.vol_x || vy >= params.vol_y || vz >= params.vol_z { return; }

    let lx    = params.sub_x;
    let ly    = params.sub_y;
    let lz    = params.sub_z;
    let L     = lx * ly * lz;
    let nel_x = params.nel_x;
    let nel_y = params.nel_y;
    let nel_z = params.nel_z;

    // Guard: L must fit in the private MAX_L = 32 allocation.
    if L == 0u || L > 32u { return; }

    let n_sub_x = nel_x - lx + 1u;
    let n_sub_y = nel_y - ly + 1u;
    let n_sub_z = nel_z - lz + 1u;
    let n_sub   = n_sub_x * n_sub_y * n_sub_z;

    // Voxel physical position (centred grid).
    let pv_x = (f32(params.vol_x) - 1.0) * (-0.5) * params.vox_dx + f32(vx) * params.vox_dx;
    let pv_y = (f32(params.vol_y) - 1.0) * (-0.5) * params.vox_dy + f32(vy) * params.vox_dy;
    let pv_z = (f32(params.vol_z) - 1.0) * (-0.5) * params.vox_dz + f32(vz) * params.vox_dz;

    // -----------------------------------------------------------------------
    // 1. Spatially-smoothed covariance accumulation.
    //    r_mat is zero-initialised (var<private>).
    // -----------------------------------------------------------------------
    for (var qx = 0u; qx < n_sub_x; qx++) {
        for (var qy = 0u; qy < n_sub_y; qy++) {
            for (var qz = 0u; qz < n_sub_z; qz++) {
                for (var si = 0u; si < L; si++) {
                    // Element i within sub-aperture (qx, qy, qz).
                    let di_x = si / (ly * lz);
                    let di_y = (si / lz) % ly;
                    let di_z = si % lz;
                    let ei_x = qx + di_x;
                    let ei_y = qy + di_y;
                    let ei_z = qz + di_z;
                    let ch_i = ei_x * nel_y * nel_z + ei_y * nel_z + ei_z;

                    let ep_ix = (f32(nel_x) - 1.0) * (-0.5) * params.elem_sx + f32(ei_x) * params.elem_sx;
                    let ep_iy = (f32(nel_y) - 1.0) * (-0.5) * params.elem_sy + f32(ei_y) * params.elem_sy;
                    let ep_iz = (f32(nel_z) - 1.0) * (-0.5) * params.elem_sz + f32(ei_z) * params.elem_sz;
                    let dxi   = pv_x - ep_ix;
                    let dyi   = pv_y - ep_iy;
                    let dzi   = pv_z - ep_iz;
                    let tau_i = sqrt(dxi * dxi + dyi * dyi + dzi * dzi)
                                / params.sound_speed * params.sampling_freq;

                    // Lower triangle only (sj ≤ si) — R is symmetric.
                    for (var sj = 0u; sj <= si; sj++) {
                        let dj_x = sj / (ly * lz);
                        let dj_y = (sj / lz) % ly;
                        let dj_z = sj % lz;
                        let ej_x = qx + dj_x;
                        let ej_y = qy + dj_y;
                        let ej_z = qz + dj_z;
                        let ch_j = ej_x * nel_y * nel_z + ej_y * nel_z + ej_z;

                        let ep_jx = (f32(nel_x) - 1.0) * (-0.5) * params.elem_sx + f32(ej_x) * params.elem_sx;
                        let ep_jy = (f32(nel_y) - 1.0) * (-0.5) * params.elem_sy + f32(ej_y) * params.elem_sy;
                        let ep_jz = (f32(nel_z) - 1.0) * (-0.5) * params.elem_sz + f32(ej_z) * params.elem_sz;
                        let dxj   = pv_x - ep_jx;
                        let dyj   = pv_y - ep_jy;
                        let dzj   = pv_z - ep_jz;
                        let tau_j = sqrt(dxj * dxj + dyj * dyj + dzj * dzj)
                                    / params.sound_speed * params.sampling_freq;

                        // R_q[si,sj] = (1/N) Σ_n X_i[n] * X_j[n]
                        // X_k[n] = (1/F) Σ_f rf(f, ch_k, τ_k + n)
                        var cov    = 0.0f;
                        let inv_f  = 1.0 / f32(max(params.num_frames, 1u));
                        for (var n = 0u; n < params.num_samples; n++) {
                            var xi = 0.0f;
                            var xj = 0.0f;
                            for (var fr = 0u; fr < params.num_frames; fr++) {
                                xi += rf_interp(fr, ch_i, tau_i + f32(n));
                                xj += rf_interp(fr, ch_j, tau_j + f32(n));
                            }
                            cov += (xi * inv_f) * (xj * inv_f);
                        }
                        cov /= f32(params.num_samples);

                        r_mat[si * 32u + sj] += cov;
                        if si != sj { r_mat[sj * 32u + si] += cov; }
                    }
                }
            }
        }
    }

    // Average over sub-apertures.
    let inv_nsub = 1.0 / f32(n_sub);
    for (var i = 0u; i < L; i++) {
        for (var j = 0u; j < L; j++) {
            r_mat[i * 32u + j] *= inv_nsub;
        }
    }

    // -----------------------------------------------------------------------
    // 2. Diagonal loading: R_δ = R + δ · (tr(R)/L) · I
    // -----------------------------------------------------------------------
    var trace = 0.0f;
    for (var i = 0u; i < L; i++) { trace += r_mat[i * 32u + i]; }
    let loading = params.diagonal_loading * trace / f32(L);
    for (var i = 0u; i < L; i++) { r_mat[i * 32u + i] += loading; }

    // -----------------------------------------------------------------------
    // 3. Cholesky-Banachiewicz: R_δ = L · Lᵀ
    //    chol[] stores lower-triangular L (zero-initialised).
    // -----------------------------------------------------------------------
    var chol_ok = true;
    var j       = 0u;
    loop {
        if j >= L || !chol_ok { break; }
        var s = r_mat[j * 32u + j];
        for (var k = 0u; k < j; k++) {
            s -= chol[j * 32u + k] * chol[j * 32u + k];
        }
        if s <= 0.0 { chol_ok = false; break; }
        chol[j * 32u + j] = sqrt(s);
        let inv_ljj = 1.0 / chol[j * 32u + j];
        for (var i = j + 1u; i < L; i++) {
            var t = r_mat[i * 32u + j];
            for (var k = 0u; k < j; k++) {
                t -= chol[i * 32u + k] * chol[j * 32u + k];
            }
            chol[i * 32u + j] = t * inv_ljj;
        }
        j++;
    }

    let v_idx = vx * params.vol_y * params.vol_z + vy * params.vol_z + vz;
    if !chol_ok { output[v_idx] = 0.0; return; }

    // -----------------------------------------------------------------------
    // 4. Forward substitution: L · y = 1  (y stored in u_vec)
    // -----------------------------------------------------------------------
    for (var i = 0u; i < L; i++) {
        var s = 1.0f;
        for (var k = 0u; k < i; k++) { s -= chol[i * 32u + k] * u_vec[k]; }
        u_vec[i] = s / chol[i * 32u + i];
    }

    // Copy y into x_bar before backward substitution overwrites u_vec.
    for (var i = 0u; i < L; i++) { x_bar[i] = u_vec[i]; }

    // -----------------------------------------------------------------------
    // 5. Backward substitution: Lᵀ · u = y  (x_bar = y; result → u_vec)
    //    Lᵀ[i,j] = chol[j * 32u + i]   (transpose of L)
    //    Process from i = L−1 down to 0.
    // -----------------------------------------------------------------------
    var ii = 0u;
    loop {
        if ii >= L { break; }
        let i = L - 1u - ii;
        var s = x_bar[i];
        for (var k = i + 1u; k < L; k++) {
            s -= chol[k * 32u + i] * u_vec[k];
        }
        u_vec[i] = s / chol[i * 32u + i];
        ii++;
    }

    // -----------------------------------------------------------------------
    // 6. MVDR output power: P = 1 / (1ᵀ u)
    // -----------------------------------------------------------------------
    var denom = 0.0f;
    for (var i = 0u; i < L; i++) { denom += u_vec[i]; }
    if abs(denom) < 1e-30f { output[v_idx] = 0.0; return; }
    let P = 1.0 / denom;

    // -----------------------------------------------------------------------
    // 7. Mean delay-aligned signal x̄₀ for canonical sub-aperture 0 (q=0).
    //    Reuse x_bar (y is no longer needed).
    // -----------------------------------------------------------------------
    for (var si = 0u; si < L; si++) {
        let di_x = si / (ly * lz);
        let di_y = (si / lz) % ly;
        let di_z = si % lz;
        let ch_i = di_x * nel_y * nel_z + di_y * nel_z + di_z;

        let ep_x = (f32(nel_x) - 1.0) * (-0.5) * params.elem_sx + f32(di_x) * params.elem_sx;
        let ep_y = (f32(nel_y) - 1.0) * (-0.5) * params.elem_sy + f32(di_y) * params.elem_sy;
        let ep_z = (f32(nel_z) - 1.0) * (-0.5) * params.elem_sz + f32(di_z) * params.elem_sz;
        let dx0  = pv_x - ep_x;
        let dy0  = pv_y - ep_y;
        let dz0  = pv_z - ep_z;
        let tau0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
                   / params.sound_speed * params.sampling_freq;

        var mean_val = 0.0f;
        for (var fr = 0u; fr < params.num_frames; fr++) {
            mean_val += rf_interp(fr, ch_i, tau0);
        }
        x_bar[si] = mean_val / f32(max(params.num_frames, 1u));
    }

    // MVDR output = |P · uᵀ · x̄₀|
    var dot_val = 0.0f;
    for (var i = 0u; i < L; i++) { dot_val += u_vec[i] * x_bar[i]; }
    output[v_idx] = abs(P * dot_val);
}
