// Acoustic wave kernels for CUDA - Proper Staggered Grid FDTD Implementation
// Template placeholders: {{float_type}}, {{block_size_x}}, {{block_size_y}}, {{block_size_z}}

// Kernel 1: Update pressure at cell centers using velocity divergence
extern "C" __global__ void update_pressure_kernel(
    {{float_type}}* pressure,
    const {{float_type}}* velocity_x,
    const {{float_type}}* velocity_y,
    const {{float_type}}* velocity_z,
    const {{float_type}}* density,
    const {{float_type}}* sound_speed,
    int nx, int ny, int nz,
    {{float_type}} dt, {{float_type}} dx, {{float_type}} dy, {{float_type}} dz
) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Check bounds - pressure points (cell centers)
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }
    
    int idx = k * nx * ny + j * nx + i;
    
    // Load material properties at cell center
    {{float_type}} rho = density[idx];
    {{float_type}} c = sound_speed[idx];
    {{float_type}} rho_c2 = rho * c * c;
    
    // Compute velocity divergence using staggered grid differences
    // Velocity components are defined at face centers, so we difference between faces
    {{float_type}} dvx_dx = 0.0;
    {{float_type}} dvy_dy = 0.0;
    {{float_type}} dvz_dz = 0.0;
    
    // X-direction: velocity_x at i+1/2 and i-1/2 faces
    if (i < nx - 1) {
        dvx_dx = (velocity_x[idx] - velocity_x[idx - 1]) / dx;
    }
    
    // Y-direction: velocity_y at j+1/2 and j-1/2 faces  
    if (j < ny - 1) {
        dvy_dy = (velocity_y[idx] - velocity_y[idx - nx]) / dy;
    }
    
    // Z-direction: velocity_z at k+1/2 and k-1/2 faces
    if (k < nz - 1) {
        dvz_dz = (velocity_z[idx] - velocity_z[idx - nx*ny]) / dz;
    }
    
    {{float_type}} div_v = dvx_dx + dvy_dy + dvz_dz;
    
    // Update pressure at cell center
    pressure[idx] -= rho_c2 * div_v * dt;
}

// Kernel 2: Update velocity X component at face centers
extern "C" __global__ void update_velocity_x_kernel(
    {{float_type}}* velocity_x,
    const {{float_type}}* pressure,
    const {{float_type}}* density,
    int nx, int ny, int nz,
    {{float_type}} dt, {{float_type}} dx
) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Check bounds - velocity_x points (x-face centers)
    if (i >= nx-1 || j >= ny || k >= nz) {
        return;
    }
    
    int idx = k * nx * ny + j * nx + i;
    
    // Interpolate density to face center (average of adjacent cells)
    {{float_type}} rho = 0.5 * (density[idx] + density[idx + 1]);
    
    // Compute pressure gradient at x-face center
    {{float_type}} dp_dx = (pressure[idx + 1] - pressure[idx]) / dx;
    
    // Update velocity_x at face center
    velocity_x[idx] -= (dt / rho) * dp_dx;
}

// Kernel 3: Update velocity Y component at face centers
extern "C" __global__ void update_velocity_y_kernel(
    {{float_type}}* velocity_y,
    const {{float_type}}* pressure,
    const {{float_type}}* density,
    int nx, int ny, int nz,
    {{float_type}} dt, {{float_type}} dy
) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Check bounds - velocity_y points (y-face centers)
    if (i >= nx || j >= ny-1 || k >= nz) {
        return;
    }
    
    int idx = k * nx * ny + j * nx + i;
    
    // Interpolate density to face center (average of adjacent cells)
    {{float_type}} rho = 0.5 * (density[idx] + density[idx + nx]);
    
    // Compute pressure gradient at y-face center
    {{float_type}} dp_dy = (pressure[idx + nx] - pressure[idx]) / dy;
    
    // Update velocity_y at face center
    velocity_y[idx] -= (dt / rho) * dp_dy;
}

// Kernel 4: Update velocity Z component at face centers
extern "C" __global__ void update_velocity_z_kernel(
    {{float_type}}* velocity_z,
    const {{float_type}}* pressure,
    const {{float_type}}* density,
    int nx, int ny, int nz,
    {{float_type}} dt, {{float_type}} dz
) {
    // Calculate global thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Check bounds - velocity_z points (z-face centers)
    if (i >= nx || j >= ny || k >= nz-1) {
        return;
    }
    
    int idx = k * nx * ny + j * nx + i;
    
    // Interpolate density to face center (average of adjacent cells)
    {{float_type}} rho = 0.5 * (density[idx] + density[idx + nx*ny]);
    
    // Compute pressure gradient at z-face center
    {{float_type}} dp_dz = (pressure[idx + nx*ny] - pressure[idx]) / dz;
    
    // Update velocity_z at face center
    velocity_z[idx] -= (dt / rho) * dp_dz;
}

// Shared memory pressure update kernel
extern "C" __global__ void update_pressure_kernel_shared(
    {{float_type}}* pressure,
    const {{float_type}}* velocity_x,
    const {{float_type}}* velocity_y,
    const {{float_type}}* velocity_z,
    const {{float_type}}* density,
    const {{float_type}}* sound_speed,
    int nx, int ny, int nz,
    {{float_type}} dt, {{float_type}} dx, {{float_type}} dy, {{float_type}} dz
) {
    // Define shared memory tiles for velocity components
    __shared__ {{float_type}} vx_tile[{{block_size_x}}+1][{{block_size_y}}][{{block_size_z}}];
    __shared__ {{float_type}} vy_tile[{{block_size_x}}][{{block_size_y}}+1][{{block_size_z}}];
    __shared__ {{float_type}} vz_tile[{{block_size_x}}][{{block_size_y}}][{{block_size_z}}+1];
    
    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Global indices
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z * blockDim.z + tz;
    
    // Check bounds
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }
    
    int idx = k * nx * ny + j * nx + i;
    
    // Load velocity data into shared memory
    vx_tile[tx][ty][tz] = velocity_x[idx];
    vy_tile[tx][ty][tz] = velocity_y[idx];
    vz_tile[tx][ty][tz] = velocity_z[idx];
    
    // Load halo regions for stencil computation
    if (tx == {{block_size_x}}-1 && i < nx-1) {
        vx_tile[tx+1][ty][tz] = velocity_x[idx + 1];
    }
    if (ty == {{block_size_y}}-1 && j < ny-1) {
        vy_tile[tx][ty+1][tz] = velocity_y[idx + nx];
    }
    if (tz == {{block_size_z}}-1 && k < nz-1) {
        vz_tile[tx][ty][tz+1] = velocity_z[idx + nx*ny];
    }
    
    __syncthreads(); // Synchronize to ensure all data is loaded
    
    // Load material properties
    {{float_type}} rho = density[idx];
    {{float_type}} c = sound_speed[idx];
    {{float_type}} rho_c2 = rho * c * c;
    
    // Compute velocity divergence using shared memory
    {{float_type}} dvx_dx = 0.0;
    {{float_type}} dvy_dy = 0.0;
    {{float_type}} dvz_dz = 0.0;
    
    if (tx > 0 || i > 0) {
        {{float_type}} vx_left = (tx > 0) ? vx_tile[tx-1][ty][tz] : velocity_x[idx - 1];
        dvx_dx = (vx_tile[tx][ty][tz] - vx_left) / dx;
    }
    
    if (ty > 0 || j > 0) {
        {{float_type}} vy_back = (ty > 0) ? vy_tile[tx][ty-1][tz] : velocity_y[idx - nx];
        dvy_dy = (vy_tile[tx][ty][tz] - vy_back) / dy;
    }
    
    if (tz > 0 || k > 0) {
        {{float_type}} vz_down = (tz > 0) ? vz_tile[tx][ty][tz-1] : velocity_z[idx - nx*ny];
        dvz_dz = (vz_tile[tx][ty][tz] - vz_down) / dz;
    }
    
    {{float_type}} div_v = dvx_dx + dvy_dy + dvz_dz;
    
    // Update pressure
    pressure[idx] -= rho_c2 * div_v * dt;
}