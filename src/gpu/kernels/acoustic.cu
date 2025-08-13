// Acoustic wave kernel for CUDA
// Template placeholders: {{float_type}}, {{block_size_x}}, {{block_size_y}}, {{block_size_z}}

extern "C" __global__ void acoustic_wave_kernel(
    {{float_type}}* pressure,
    {{float_type}}* velocity_x,
    {{float_type}}* velocity_y,
    {{float_type}}* velocity_z,
    const {{float_type}}* density,
    const {{float_type}}* sound_speed,
    int nx, int ny, int nz,
    {{float_type}} dt, {{float_type}} dx, {{float_type}} dy, {{float_type}} dz
) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Check bounds - interior points only
    if (i <= 0 || i >= nx-1 || j <= 0 || j >= ny-1 || k <= 0 || k >= nz-1) {
        return;
    }
    
    int idx = k * nx * ny + j * nx + i;
    
    // Load material properties
    {{float_type}} rho = density[idx];
    {{float_type}} c = sound_speed[idx];
    {{float_type}} rho_c2 = rho * c * c;
    
    // Compute velocity divergence using central differences
    {{float_type}} dvx_dx = (velocity_x[idx + 1] - velocity_x[idx - 1]) / (2.0 * dx);
    {{float_type}} dvy_dy = (velocity_y[idx + nx] - velocity_y[idx - nx]) / (2.0 * dy);
    {{float_type}} dvz_dz = (velocity_z[idx + nx*ny] - velocity_z[idx - nx*ny]) / (2.0 * dz);
    
    {{float_type}} div_v = dvx_dx + dvy_dy + dvz_dz;
    
    // Update pressure
    pressure[idx] -= rho_c2 * div_v * dt;
    
    // Compute pressure gradient
    {{float_type}} dp_dx = (pressure[idx + 1] - pressure[idx - 1]) / (2.0 * dx);
    {{float_type}} dp_dy = (pressure[idx + nx] - pressure[idx - nx]) / (2.0 * dy);
    {{float_type}} dp_dz = (pressure[idx + nx*ny] - pressure[idx - nx*ny]) / (2.0 * dz);
    
    // Update velocities
    velocity_x[idx] -= (dt / rho) * dp_dx;
    velocity_y[idx] -= (dt / rho) * dp_dy;
    velocity_z[idx] -= (dt / rho) * dp_dz;
}