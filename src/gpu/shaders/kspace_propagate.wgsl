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

// Cooley-Tukey radix-2 FFT implementation for GPU
// Based on: Frigo & Johnson (2005) "The Design and Implementation of FFTW3"
@compute @workgroup_size(64, 1, 1)
fn fft_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let stride = global_id.y;
    let n = params.nx * params.ny * params.nz;
    
    if (idx >= n) {
        return;
    }
    
    // Bit-reversal permutation
    var rev_idx = 0u;
    var temp_idx = idx;
    for (var i = 0u; i < 32u; i = i + 1u) {
        if ((1u << i) >= n) {
            break;
        }
        rev_idx = (rev_idx << 1u) | (temp_idx & 1u);
        temp_idx = temp_idx >> 1u;
    }
    
    if (idx < rev_idx) {
        // Swap elements at idx and rev_idx
        let temp = pressure_field[idx];
        pressure_field[idx] = pressure_field[rev_idx];
        pressure_field[rev_idx] = temp;
    }
    
    workgroupBarrier();
    
    // Cooley-Tukey decimation-in-time
    var m = 2u;
    while (m <= n) {
        let half_m = m >> 1u;
        let theta = -2.0 * PI / f32(m);
        
        if ((idx & (m - 1u)) < half_m) {
            let k = idx & (half_m - 1u);
            let j = ((idx >> log2(m)) << log2(m)) + k;
            let t_idx = j + half_m;
            
            let angle = theta * f32(k);
            let w_real = cos(angle);
            let w_imag = sin(angle);
            
            let t_real = pressure_field[t_idx].real;
            let t_imag = pressure_field[t_idx].imag;
            
            let temp_real = w_real * t_real - w_imag * t_imag;
            let temp_imag = w_real * t_imag + w_imag * t_real;
            
            pressure_field[t_idx].real = pressure_field[j].real - temp_real;
            pressure_field[t_idx].imag = pressure_field[j].imag - temp_imag;
            
            pressure_field[j].real = pressure_field[j].real + temp_real;
            pressure_field[j].imag = pressure_field[j].imag + temp_imag;
        }
        
        workgroupBarrier();
        m = m << 1u;
    }
}

// Helper function for computing log2
fn log2(n: u32) -> u32 {
    var result = 0u;
    var temp = n >> 1u;
    while (temp > 0u) {
        result = result + 1u;
        temp = temp >> 1u;
    }
    return result;
}