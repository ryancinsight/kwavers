// K-space propagation shader

struct GridParams {
    nx: u32,
    ny: u32,
    nz: u32,
    dt: f32,
    c0: f32,
}

struct Complex {
    real: f32,
    imag: f32,
}

@group(0) @binding(0)
var<storage, read_write> spectrum: array<Complex>;

@group(0) @binding(1)
var<storage, read> kspace: array<vec3<f32>>; // kx, ky, kz

var<push_constant> params: GridParams;

fn index_3d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * params.nx + z * params.nx * params.ny;
}

fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

fn complex_exp(theta: f32) -> Complex {
    return Complex(cos(theta), sin(theta));
}

@compute @workgroup_size(8, 8, 8)
fn propagate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    // Bounds check
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    let idx = index_3d(x, y, z);
    let k = kspace[idx];
    
    // Calculate k^2
    let k2 = k.x * k.x + k.y * k.y + k.z * k.z;
    
    // Angular frequency from dispersion relation: ω = c * |k|
    let omega = params.c0 * sqrt(k2);
    
    // Propagation phase: exp(-i * ω * dt)
    let phase = -omega * params.dt;
    let propagator = complex_exp(phase);
    
    // Apply propagation
    spectrum[idx] = complex_mul(spectrum[idx], propagator);
}

// Placeholder FFT - in practice would use a proper FFT implementation
@compute @workgroup_size(8, 8, 8)
fn fft_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // This is a placeholder - real FFT would be much more complex
    // For now, just pass through
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    
    if (x >= params.nx || y >= params.ny || z >= params.nz) {
        return;
    }
    
    // In a real implementation, this would perform FFT
    // For now, it's a no-op placeholder
}