// FFT compute shader — iterative Cooley-Tukey radix-2 DIT (Decimation-In-Time)
//
// ## Algorithm (Cooley & Tukey 1965)
//
// This shader implements one butterfly stage of the iterative radix-2 DIT FFT.
// The host dispatches this shader exactly log₂(N) times, incrementing `stage`
// from 0 to log₂(N)−1 on each dispatch.  A final divide-by-N pass is required
// for the inverse FFT (set `inverse = 1u`).
//
// ### Stage description
//
// At stage s (0-indexed), the butterfly half-width is h = 2^s.
// Each thread handles one butterfly pair (even, odd):
//
// ```
//   group_size = 2·h
//   group_idx  = thread_id / h       (which group)
//   local_idx  = thread_id mod h     (position within group)
//   even = group_idx · group_size + local_idx
//   odd  = even + h
//   twiddle φ = −2π · local_idx / group_size   (forward FFT)
//             = +2π · local_idx / group_size   (inverse FFT; sign flip)
//   W = exp(i·φ) = (cos φ, sin φ)
//   data[even] ← data[even] + W · data[odd]
//   data[odd]  ← data[even] − W · data[odd]
// ```
//
// ### Bit-reversal permutation
//
// The input must be in bit-reversed order before the first butterfly stage.
// The entry point `fft_bitrev` performs the bit-reversal permutation in-place
// and must be dispatched **once** before the first butterfly pass.
//
// ### Inverse FFT normalisation
//
// After all log₂(N) butterfly passes with `inverse = 1u`, dispatch
// `fft_scale` once to divide every element by N.
//
// ## Binding layout
//
// group(0) binding(0): `data_re`  — real parts of complex samples  (f32, read_write)
// group(0) binding(1): `data_im`  — imaginary parts of samples      (f32, read_write)
// group(1) binding(0): `params`   — FftParams uniform
//
// ## References
//
// - Cooley JW, Tukey JW (1965). "An algorithm for the machine calculation of
//   complex Fourier series." Math. Comp. 19(90), 297–301. DOI: 10.2307/2003354
// - Duhamel P, Vetterli M (1990). "Fast Fourier transforms: a tutorial review
//   and a state of the art." Signal Process. 19(4), 259–299.
//   DOI: 10.1016/0165-1684(90)90158-U

// ─── Uniform parameters ───────────────────────────────────────────────────────

struct FftParams {
    n:       u32,   // FFT length (must be a power of 2)
    stage:   u32,   // Current butterfly stage (0 … log₂(n)−1)
    inverse: u32,   // 0 = forward; 1 = inverse (negates twiddle angle)
    _pad:    u32,   // Padding to 16-byte alignment (std140)
}

// ─── Buffers ──────────────────────────────────────────────────────────────────

@group(0) @binding(0)
var<storage, read_write> data_re: array<f32>;

@group(0) @binding(1)
var<storage, read_write> data_im: array<f32>;

@group(1) @binding(0)
var<uniform> params: FftParams;

// ─── Helpers ──────────────────────────────────────────────────────────────────

const TWO_PI: f32 = 6.28318530717958647692;

/// Complex multiply: (a_re + i·a_im) · (b_re + i·b_im)
fn cmul_re(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> f32 {
    return a_re * b_re - a_im * b_im;
}
fn cmul_im(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> f32 {
    return a_re * b_im + a_im * b_re;
}

/// Bit-reverse a value of `bits` significant bits.
fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r: u32 = 0u;
    var b = bits;
    loop {
        if b == 0u { break; }
        r = (r << 1u) | (v & 1u);
        v >>= 1u;
        b -= 1u;
    }
    return r;
}

// ─── Entry point: bit-reversal permutation ────────────────────────────────────

/// Reorder `data_re` and `data_im` into bit-reversed order in-place.
///
/// Dispatch with N/2 threads total (each thread handles one swap).
/// Must be called **before** any butterfly pass.
///
/// Workgroup size: (256, 1, 1).  Dispatch ceil(N/256) workgroups.
@compute @workgroup_size(256, 1, 1)
fn fft_bitrev(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n {
        return;
    }

    // Compute log₂(n) — we'll iterate until the shifted value reaches 0
    var log2n: u32 = 0u;
    var tmp = params.n >> 1u;
    loop {
        if tmp == 0u { break; }
        log2n += 1u;
        tmp >>= 1u;
    }

    let j = bit_reverse(i, log2n);
    if j > i {
        // Swap data[i] ↔ data[j]
        let re_i = data_re[i];
        let im_i = data_im[i];
        data_re[i] = data_re[j];
        data_im[i] = data_im[j];
        data_re[j] = re_i;
        data_im[j] = im_i;
    }
}

// ─── Entry point: one butterfly pass (forward or inverse) ────────────────────

/// Execute one radix-2 DIT butterfly pass at the given `params.stage`.
///
/// ## Algorithm (Cooley & Tukey 1965, §3)
///
/// At stage s, butterfly half-width h = 2^s, full group size = 2h.
/// Thread `id` handles the butterfly pair (even, odd) where:
///
/// ```text
///   h         = 1 << stage
///   group_size = h << 1
///   group_idx  = id / h
///   local_idx  = id % h
///   even = group_idx · group_size + local_idx
///   odd  = even + h
/// ```
///
/// Twiddle factor W = exp(−2πi · local_idx / group_size) for forward FFT,
///                  = exp(+2πi · local_idx / group_size) for inverse FFT.
///
/// ## Dispatch
///
/// Each thread handles one butterfly (2 elements).  Dispatch N/2 threads.
/// Call for stage = 0, 1, …, log₂(N)−1 in order.
///
/// ## References
/// - Cooley JW, Tukey JW (1965). Math. Comp. 19(90), 297–301.
///   DOI: 10.2307/2003354
@compute @workgroup_size(256, 1, 1)
fn fft_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let half_n = params.n >> 1u;
    if id >= half_n {
        return;
    }

    let h = 1u << params.stage;           // butterfly half-width
    let group_size = h << 1u;             // full butterfly group size
    let group_idx = id / h;               // which group
    let local_idx = id % h;               // position within group

    let even = group_idx * group_size + local_idx;
    let odd  = even + h;

    // Twiddle angle: negative for forward FFT, positive for inverse
    var angle = -TWO_PI * f32(local_idx) / f32(group_size);
    if params.inverse != 0u {
        angle = -angle;
    }

    let w_re = cos(angle);
    let w_im = sin(angle);

    let e_re = data_re[even];
    let e_im = data_im[even];
    let o_re = data_re[odd];
    let o_im = data_im[odd];

    // W · data[odd]
    let wo_re = cmul_re(w_re, w_im, o_re, o_im);
    let wo_im = cmul_im(w_re, w_im, o_re, o_im);

    // Butterfly: even' = even + W·odd;  odd' = even − W·odd
    data_re[even] = e_re + wo_re;
    data_im[even] = e_im + wo_im;
    data_re[odd]  = e_re - wo_re;
    data_im[odd]  = e_im - wo_im;
}

// ─── Entry point: IFFT normalisation (divide by N) ───────────────────────────

/// Divide every element by N to complete the inverse FFT normalisation.
///
/// Dispatch once, after all log₂(N) inverse butterfly passes, with N threads.
@compute @workgroup_size(256, 1, 1)
fn fft_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n {
        return;
    }
    let inv_n = 1.0 / f32(params.n);
    data_re[i] *= inv_n;
    data_im[i] *= inv_n;
}
