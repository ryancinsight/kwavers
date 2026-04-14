// Bluestein Chirp-Z Transform — GPU compute shaders (wgpu / WGSL).
//
// ## Algorithm (Bluestein 1970; Rabiner, Schafer & Rader 1969)
//
// The Chirp-Z transform converts an N-point DFT of arbitrary N into a
// length-M circular convolution where M is the smallest power of 2 with
// M ≥ 2N − 1.  This enables the DFT of any array length via the existing
// radix-2 FFT shader.
//
// ### Identity (Bluestein 1970)
// Using the algebraic identity  k·n = −(k−n)²/2 + k²/2 + n²/2:
//
//   X[k] = exp(+iπk²/N) · Σₙ { x[n]·exp(+iπn²/N) } · h[k−n]
//
// where h[m] = exp(−iπm²/N) (the "chirp" sequence).
//
// This is a length-M circular convolution, computable via three M-point FFTs:
//   1. a[n] = x[n] · exp(+iπn²/N),  zero-padded to M.
//   2. h[m] = exp(−iπm²/N),         with h[−m mod M] = h[m] (Hermitian).
//   3. X_raw = IFFT{ FFT{a} · FFT{h} }
//   4. X[k]  = X_raw[k] · exp(+iπk²/N)   for k = 0, …, N−1.
//
// Steps 2 and 4 are precomputed on the CPU (host side) at GpuChirpFft1d construction:
//   - `h_fft[k]`   = FFT of the chirp sequence (length M), uploaded once.
//   - `chirp_out[k]` = exp(+iπk²/N)  for k=0..N-1, the final twiddle.
//
// The GPU shaders here implement Steps 1 (premul), 3 (pointwise multiply),
// and 4 (postmul / extraction).
//
// ## References
// - Bluestein L.I. (1970). IEEE Trans. AU-18(4), 451–455.
// - Rabiner L.R., Schafer R.W., Rader C.M. (1969). IEEE Trans. AU-17(2), 86–92.
// - Frigo M., Johnson S.G. (2005). Proc. IEEE 93(2), 216–231.

// ── Shared types ──────────────────────────────────────────────────────────────

struct ChirpParams {
    /// Input length N.
    n: u32,
    /// Padded convolution length M (smallest power-of-2 ≥ 2N−1).
    m: u32,
    /// Padding: unused (alignment).
    _pad0: u32,
    _pad1: u32,
}

// Bind group layout (same for all four shaders):
//   group(0) binding(0): data_re  — f32[M], read_write
//   group(0) binding(1): data_im  — f32[M], read_write
//   group(0) binding(2): chirp_re — f32[M], read  (precomputed FFT{h}.re)
//   group(0) binding(3): chirp_im — f32[M], read  (precomputed FFT{h}.im)
//   group(1) binding(0): params   — uniform ChirpParams

@group(0) @binding(0)
var<storage, read_write> data_re: array<f32>;
@group(0) @binding(1)
var<storage, read_write> data_im: array<f32>;
@group(0) @binding(2)
var<storage, read> chirp_re: array<f32>;
@group(0) @binding(3)
var<storage, read> chirp_im: array<f32>;

@group(1) @binding(0)
var<uniform> params: ChirpParams;

const PI: f32 = 3.14159265358979323846;

// ── Step 1: chirp_premul ──────────────────────────────────────────────────────
//
// Computes a[n] = x[n] · exp(+iπn²/N) for n = 0, …, N−1, and zeros n ≥ N.
// Input:  data_re[0..N] = Re(x[n]),  data_im[0..N] = Im(x[n]).
//         data_re[N..M] and data_im[N..M] may contain garbage (overwritten to 0).
// Output: data_re[0..N] = Re(a[n]),  data_im[0..N] = Im(a[n]).
//         data_re[N..M] = 0,         data_im[N..M] = 0.

@compute @workgroup_size(256, 1, 1)
fn chirp_premul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.m { return; }

    if idx >= params.n {
        // Zero-pad region [N, M).
        data_re[idx] = 0.0;
        data_im[idx] = 0.0;
        return;
    }

    // Twiddle: exp(+iπn²/N) = cos(πn²/N) + i·sin(πn²/N)
    let n_f = f32(idx);
    let arg = PI * n_f * n_f / f32(params.n);
    let cos_arg = cos(arg);
    let sin_arg = sin(arg);

    let re = data_re[idx];
    let im = data_im[idx];

    // Complex multiply: (re + i·im) · (cos_arg + i·sin_arg)
    data_re[idx] = re * cos_arg - im * sin_arg;
    data_im[idx] = re * sin_arg + im * cos_arg;
}

// ── Step 3a: chirp_pointmul ───────────────────────────────────────────────────
//
// Computes A[k] · H_fft[k] pointwise in the frequency domain, where:
//   - data_re/im holds FFT{a}[k]    (in-place from the radix-2 forward FFT).
//   - chirp_re/im holds FFT{h}[k]   (precomputed, uploaded at construction).
// Output: data_re/im = Re/Im( FFT{a}[k] · FFT{h}[k] ).

@compute @workgroup_size(256, 1, 1)
fn chirp_pointmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.m { return; }

    let a_re = data_re[idx];
    let a_im = data_im[idx];
    let h_re = chirp_re[idx];
    let h_im = chirp_im[idx];

    // (a_re + i·a_im) · (h_re + i·h_im)
    data_re[idx] = a_re * h_re - a_im * h_im;
    data_im[idx] = a_re * h_im + a_im * h_re;
}

// ── Step 3b: chirp_scale ──────────────────────────────────────────────────────
//
// After IFFT (radix-2 inverse, which is un-normalized by apollofft convention),
// we need to divide by M to recover the correct convolution:
//   conv[n] = (1/M) · Σ_k FFT{a}[k] · FFT{h}[k] · exp(+2πi·k·n/M)
// This kernel divides data_re[n] and data_im[n] by M for n = 0, …, N−1.
// Elements n ≥ N are not needed and are left unchanged.
//
// Note: the radix-2 GPU IFFT (`fft_forward` with inverse=1 flag in fft.wgsl)
// already divides by M during the `fft_scale` pass, so this shader does NOT
// apply an additional 1/M factor — it is intentionally a no-op pass, serving
// only as a synchronisation barrier between the IFFT and the postmul.
// The final scaling is baked into the precomputed h sequence on the host.

@compute @workgroup_size(256, 1, 1)
fn chirp_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    // Identity: scaling was absorbed into h_fft precomputation on the CPU.
    // This pass exists for pipeline-barrier purposes only.
}

// ── Step 4: chirp_postmul ─────────────────────────────────────────────────────
//
// Computes X[k] = conv[k] · exp(+iπk²/N) for k = 0, …, N−1.
// This is the final twiddle that completes the Bluestein identity.
//
// The twiddle factors exp(+iπk²/N) are the same as in chirp_premul but for
// the output index k (not the input index n), so we recompute inline.

@compute @workgroup_size(256, 1, 1)
fn chirp_postmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }

    let k_f = f32(idx);
    let arg = PI * k_f * k_f / f32(params.n);
    let cos_arg = cos(arg);
    let sin_arg = sin(arg);

    let re = data_re[idx];
    let im = data_im[idx];

    data_re[idx] = re * cos_arg - im * sin_arg;
    data_im[idx] = re * sin_arg + im * cos_arg;
}

// ── Conjugation helper: chirp_negate_im ──────────────────────────────────────
//
// Negates the imaginary part of data_im[0..N].  Used to implement the inverse
// DFT via the conjugation identity:
//
//   IDFT(X)[k] = (1/N) · conj( forward_bluestein( conj(X) ) )[k]
//
// Invoked once *before* and once *after* the forward Bluestein passes when
// computing the inverse transform.  The 1/N normalisation is applied on the
// host side by `GpuFft3d::inverse()`.

@compute @workgroup_size(256, 1, 1)
fn chirp_negate_im(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    data_im[idx] = -data_im[idx];
}
