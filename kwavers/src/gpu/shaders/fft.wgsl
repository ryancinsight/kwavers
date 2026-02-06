// FFT compute shader
// This is a simplified radix-2 FFT implementation

struct Complex {
    real: f32,
    imag: f32,
}

@group(0) @binding(0)
var<storage, read_write> data: array<Complex>;

@group(0) @binding(1)
var<storage, read> twiddle_factors: array<Complex>;

var<push_constant> n: u32; // FFT size

fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

fn complex_add(a: Complex, b: Complex) -> Complex {
    return Complex(a.real + b.real, a.imag + b.imag);
}

fn complex_sub(a: Complex, b: Complex) -> Complex {
    return Complex(a.real - b.real, a.imag - b.imag);
}

@compute @workgroup_size(64)
fn fft_radix2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    
    // Cooley-Tukey radix-2 FFT butterfly operation
    // This is simplified - a full implementation would need multiple passes
    
    let stride = 2u;
    let half_stride = 1u;
    
    if (thread_id < n / 2u) {
        let i = thread_id * stride;
        let j = i + half_stride;
        
        if (j < n) {
            let twiddle = twiddle_factors[thread_id];
            
            let a = data[i];
            let b = complex_mul(data[j], twiddle);
            
            data[i] = complex_add(a, b);
            data[j] = complex_sub(a, b);
        }
    }
}