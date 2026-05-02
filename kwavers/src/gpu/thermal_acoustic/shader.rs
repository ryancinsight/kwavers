//! Fused thermal-acoustic compute kernel (WGSL source).
//!
//! SRP: changes when the shader physics, workgroup size, or binding layout changes.

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

fn clamp_index(i: u32, size: u32) -> u32 {
    return min(i, size - 1u);
}

fn compute_sound_speed(T: f32) -> f32 {
    return cfg.c_ref + cfg.dc_dT * (T - cfg.T_ref);
}

fn compute_density(T: f32) -> f32 {
    return cfg.rho_ref + cfg.drho_dT * (T - cfg.T_ref);
}

fn update_acoustic(idx: u32, x: u32, y: u32, z: u32) {
    let T   = T_curr[idx];
    let rho = compute_density(T);
    let c   = compute_sound_speed(T);
    let K   = rho * c * c;

    let ux_x = (ux_curr[index_3d(clamp_index(x + 1u, cfg.nx), y, z)] -
                ux_curr[index_3d(clamp_index(x, cfg.nx), y, z)]) / cfg.dx;
    let uy_y = (uy_curr[index_3d(x, clamp_index(y + 1u, cfg.ny), z)] -
                uy_curr[index_3d(x, clamp_index(y, cfg.ny), z)]) / cfg.dy;
    let uz_z = (uz_curr[index_3d(x, y, clamp_index(z + 1u, cfg.nz))] -
                uz_curr[index_3d(x, y, clamp_index(z, cfg.nz))]) / cfg.dz;

    let div_u = ux_x + uy_y + uz_z;
    p_curr[idx] = p_prev[idx] - K * cfg.dt * div_u;

    let p_val = p_curr[idx];
    Q_ac[idx] = cfg.alpha_ac * p_val * p_val / (rho * c);

    if x > 0u && x < cfg.nx - 1u {
        let dp_dx = (p_curr[index_3d(x, y, z)] - p_curr[index_3d(x - 1u, y, z)]) / cfg.dx;
        ux_curr[idx] = ux_prev[idx] - (1.0 / rho) * cfg.dt * dp_dx;
    }
    if y > 0u && y < cfg.ny - 1u {
        let dp_dy = (p_curr[index_3d(x, y, z)] - p_curr[index_3d(x, y - 1u, z)]) / cfg.dy;
        uy_curr[idx] = uy_prev[idx] - (1.0 / rho) * cfg.dt * dp_dy;
    }
    if z > 0u && z < cfg.nz - 1u {
        let dp_dz = (p_curr[index_3d(x, y, z)] - p_curr[index_3d(x, y, z - 1u)]) / cfg.dz;
        uz_curr[idx] = uz_prev[idx] - (1.0 / rho) * cfg.dt * dp_dz;
    }
}

fn update_thermal(idx: u32, x: u32, y: u32, z: u32) {
    let T = T_curr[idx];

    let Txx = (T_curr[index_3d(clamp_index(x + 1u, cfg.nx), y, z)] -
               2.0 * T +
               T_curr[index_3d(clamp_index(x, cfg.nx), y, z)]) / (cfg.dx * cfg.dx);
    let Tyy = (T_curr[index_3d(x, clamp_index(y + 1u, cfg.ny), z)] -
               2.0 * T +
               T_curr[index_3d(x, clamp_index(y, cfg.ny), z)]) / (cfg.dy * cfg.dy);
    let Tzz = (T_curr[index_3d(x, y, clamp_index(z + 1u, cfg.nz))] -
               2.0 * T +
               T_curr[index_3d(x, y, clamp_index(z, cfg.nz))]) / (cfg.dz * cfg.dz);

    let laplacian_T = Txx + Tyy + Tzz;
    let perfusion_term = cfg.w_b * (cfg.T_arterial - T);
    let dT_dt = cfg.alpha_thermal * laplacian_T + perfusion_term + cfg.Q_met + Q_ac[idx];
    T_curr[idx] = T_prev[idx] + cfg.dt * dT_dt;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if x >= cfg.nx || y >= cfg.ny || z >= cfg.nz { return; }

    let idx = index_3d(x, y, z);
    update_acoustic(idx, x, y, z);
    update_thermal(idx, x, y, z);
}
"#
    .to_string()
}
