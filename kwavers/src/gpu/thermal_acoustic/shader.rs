//! Three-pass thermal-acoustic compute kernels (WGSL source).
//!
//! ## Race-condition analysis (original single-pass design)
//!
//! The original single `main` entry point had three read-write races:
//!
//! 1. **Pressure vs velocity** (intra-pass): `update_acoustic` wrote `p_curr[idx]`
//!    at line A, then read `p_curr[x-1, y, z]` at line B.  Thread at (x-1, y, z)
//!    wrote `p_curr[x-1, y, z]` concurrently → read-write race.
//! 2. **Velocity input** (intra-pass): pressure divergence read `ux_curr[x+1, y, z]`
//!    while the velocity update was writing `ux_curr[x+1, y, z]` in the thread at
//!    (x+1, y, z) → read-write race.
//! 3. **Thermal Laplacian** (intra-pass): `update_thermal` read `T_curr[x+1, y, z]`
//!    (for the Laplacian) while writing `T_curr[x, y, z]`; the neighboring thread
//!    did the opposite → read-write race.
//!
//! ## Three-pass fix
//!
//! The kernel is split into three entry points, each dispatched as a **separate
//! `ComputePass`** from `GpuThermalAcousticSolver::step()`.  wgpu inserts a
//! full pipeline barrier between any two compute passes in the same encoder,
//! making all writes from pass N visible to pass N+1.
//!
//! | Pass | Entry                  | Reads                         | Writes           |
//! |------|------------------------|-------------------------------|------------------|
//! | 1    | `update_pressure`      | p_prev, ux_prev…uz_prev, T_prev | p_curr, Q_ac   |
//! | 2    | `update_velocity`      | p_curr (fully written), _prev   | ux_curr…uz_curr |
//! | 3    | `update_thermal`       | T_prev (Laplacian), Q_ac        | T_curr          |
//!
//! ## Theorem (race-freedom, each pass)
//!
//! - Pass 1 reads only `_prev` buffers and `T_prev` (all immutable in this pass)
//!   and writes only `p_curr` and `Q_ac`.  No thread reads a cell written by
//!   another thread in this pass → race-free.
//!
//! - Pass 2 reads `p_curr` (completed by Pass 1 barrier), `ux_prev`, `uy_prev`,
//!   `uz_prev`.  Writes `ux_curr`, `uy_curr`, `uz_curr`.  Each write cell `idx`
//!   is written by exactly one thread; reads of `p_curr` are from a fully
//!   committed, immutable snapshot from Pass 1 → race-free.
//!
//! - Pass 3 reads `T_prev` for the Laplacian stencil (immutable `_prev` buffer)
//!   and `Q_ac` (committed by Pass 1 barrier).  Writes `T_curr`.  No thread
//!   reads a cell written by another thread in this pass → race-free.
//!
//! SRP: this file changes when shader physics, workgroup size, or binding layout changes.

pub(super) fn thermal_acoustic_wgsl() -> String {
    r#"
struct Config {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
    c_ref: f32,
    rho_ref: f32,
    dc_dT: f32,
    drho_dT: f32,
    T_ref: f32,
    alpha_thermal: f32,
    alpha_ac: f32,
    T_arterial: f32,
    w_b: f32,
    Q_met: f32,
}

// Bindings — all three passes share the same bind group layout.
@group(0) @binding(0)  var<storage, read_write> p_curr:  array<f32>;
@group(0) @binding(1)  var<storage, read>       p_prev:  array<f32>;
@group(0) @binding(2)  var<storage, read_write> ux_curr: array<f32>;
@group(0) @binding(3)  var<storage, read>       ux_prev: array<f32>;
@group(0) @binding(4)  var<storage, read_write> uy_curr: array<f32>;
@group(0) @binding(5)  var<storage, read>       uy_prev: array<f32>;
@group(0) @binding(6)  var<storage, read_write> uz_curr: array<f32>;
@group(0) @binding(7)  var<storage, read>       uz_prev: array<f32>;
@group(0) @binding(8)  var<storage, read_write> T_curr:  array<f32>;
@group(0) @binding(9)  var<storage, read>       T_prev:  array<f32>;
@group(0) @binding(10) var<storage, read_write> Q_ac:    array<f32>;
@group(0) @binding(11) var<uniform>             cfg:     Config;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * cfg.nx + z * cfg.nx * cfg.ny;
}

fn compute_sound_speed(T: f32) -> f32 {
    return cfg.c_ref + cfg.dc_dT * (T - cfg.T_ref);
}

fn compute_density(T: f32) -> f32 {
    return cfg.rho_ref + cfg.drho_dT * (T - cfg.T_ref);
}

// ── Pass 1: pressure update ───────────────────────────────────────────────────
//
// p_curr[idx] = p_prev[idx] − K · dt · div(u_prev)
// Q_ac[idx]   = α_ac · p_curr[idx]² / (ρ · c)
//
// Reads exclusively from _prev buffers → race-free within this pass.
// T_prev used for temperature-dependent material properties.
@compute @workgroup_size(8, 8, 4)
fn update_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let z = gid.z;
    if x >= cfg.nx || y >= cfg.ny || z >= cfg.nz { return; }
    let idx = index_3d(x, y, z);

    let T   = T_prev[idx];
    let rho = compute_density(T);
    let c   = compute_sound_speed(T);
    let K   = rho * c * c;

    // Forward difference divergence of old velocity (ux_prev, uy_prev, uz_prev).
    // All reads from _prev → no race with velocity update pass.
    let xp1 = min(x + 1u, cfg.nx - 1u);
    let yp1 = min(y + 1u, cfg.ny - 1u);
    let zp1 = min(z + 1u, cfg.nz - 1u);
    let ux_x = (ux_prev[index_3d(xp1, y, z)] - ux_prev[idx]) / cfg.dx;
    let uy_y = (uy_prev[index_3d(x, yp1, z)] - uy_prev[idx]) / cfg.dy;
    let uz_z = (uz_prev[index_3d(x, y, zp1)] - uz_prev[idx]) / cfg.dz;

    let p_new = p_prev[idx] - K * cfg.dt * (ux_x + uy_y + uz_z);
    p_curr[idx] = p_new;
    Q_ac[idx]   = cfg.alpha_ac * p_new * p_new / (rho * c);
}

// ── Pass 2: velocity update ───────────────────────────────────────────────────
//
// ux_curr[idx] = ux_prev[idx] − (1/ρ) · dt · (p_curr[x,y,z] − p_curr[x-1,y,z]) / dx
//
// Reads p_curr (committed by Pass 1 barrier) and _prev velocity buffers.
// Writes ux_curr, uy_curr, uz_curr.  Each idx is written by exactly one thread.
@compute @workgroup_size(8, 8, 4)
fn update_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let z = gid.z;
    if x >= cfg.nx || y >= cfg.ny || z >= cfg.nz { return; }
    let idx = index_3d(x, y, z);

    let T   = T_prev[idx];
    let rho = compute_density(T);
    let p   = p_curr[idx];

    // Backward difference pressure gradient.
    // select(fallback, value, condition) avoids u32 underflow at boundaries.
    let xm1 = select(0u, x - 1u, x > 0u);
    let ym1 = select(0u, y - 1u, y > 0u);
    let zm1 = select(0u, z - 1u, z > 0u);

    let dp_dx = select(0.0, (p - p_curr[index_3d(xm1, y, z)]) / cfg.dx, x > 0u);
    let dp_dy = select(0.0, (p - p_curr[index_3d(x, ym1, z)]) / cfg.dy, y > 0u);
    let dp_dz = select(0.0, (p - p_curr[index_3d(x, y, zm1)]) / cfg.dz, z > 0u);

    let coeff = cfg.dt / rho;
    ux_curr[idx] = ux_prev[idx] - coeff * dp_dx;
    uy_curr[idx] = uy_prev[idx] - coeff * dp_dy;
    uz_curr[idx] = uz_prev[idx] - coeff * dp_dz;
}

// ── Pass 3: thermal update ────────────────────────────────────────────────────
//
// T_curr[idx] = T_prev[idx] + dt · (α∇²T_prev + w_b·(T_art−T_prev) + Q_met + Q_ac)
//
// Laplacian uses T_PREV (immutable in this pass) → no read-write race.
// Q_ac is committed by the Pass 1 barrier.
@compute @workgroup_size(8, 8, 4)
fn update_thermal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y; let z = gid.z;
    if x >= cfg.nx || y >= cfg.ny || z >= cfg.nz { return; }
    let idx = index_3d(x, y, z);

    let T = T_prev[idx];

    // Central-difference Laplacian of T_prev.
    // Clamped stencil: boundary cells use one-sided differences.
    let xp1 = min(x + 1u, cfg.nx - 1u);
    let yp1 = min(y + 1u, cfg.ny - 1u);
    let zp1 = min(z + 1u, cfg.nz - 1u);
    let xm1 = select(0u, x - 1u, x > 0u);
    let ym1 = select(0u, y - 1u, y > 0u);
    let zm1 = select(0u, z - 1u, z > 0u);

    let Txx = (T_prev[index_3d(xp1, y, z)] - 2.0 * T + T_prev[index_3d(xm1, y, z)])
              / (cfg.dx * cfg.dx);
    let Tyy = (T_prev[index_3d(x, yp1, z)] - 2.0 * T + T_prev[index_3d(x, ym1, z)])
              / (cfg.dy * cfg.dy);
    let Tzz = (T_prev[index_3d(x, y, zp1)] - 2.0 * T + T_prev[index_3d(x, y, zm1)])
              / (cfg.dz * cfg.dz);

    let laplacian_T    = Txx + Tyy + Tzz;
    let perfusion_term = cfg.w_b * (cfg.T_arterial - T);
    let dT_dt = cfg.alpha_thermal * laplacian_T + perfusion_term + cfg.Q_met + Q_ac[idx];
    T_curr[idx] = T + cfg.dt * dT_dt;
}
"#
    .to_string()
}
