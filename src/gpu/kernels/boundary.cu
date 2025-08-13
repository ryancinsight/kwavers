// Boundary condition kernel for CUDA
// This kernel handles boundary conditions separately from the main computation

extern "C" __global__ void apply_boundary_kernel(
    {{float_type}}* field,
    int nx, int ny, int nz,
    int boundary_type  // 0: absorbing, 1: reflecting, 2: periodic
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = k * nx * ny + j * nx + i;
    
    // Check if we're at a boundary
    bool at_boundary = (i == 0 || i == nx-1 || 
                       j == 0 || j == ny-1 || 
                       k == 0 || k == nz-1);
    
    if (!at_boundary) return;
    
    switch (boundary_type) {
        case 0: // Absorbing (simple exponential decay)
            {
                {{float_type}} decay = 0.95;
                field[idx] *= decay;
            }
            break;
            
        case 1: // Reflecting
            // For reflecting boundaries, we'd need to handle each face separately
            // This is a simplified version
            field[idx] = 0.0;
            break;
            
        case 2: // Periodic
            // Periodic boundaries require copying from opposite side
            // This would need more complex logic per face
            break;
    }
}