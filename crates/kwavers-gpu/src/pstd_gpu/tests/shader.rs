//! Shader ABI and validated k-space path regression tests.

/// Regression guard: the GPU phased-array path injects velocity sources in
/// k-space and must retain the source-kappa reconstruction chain.
/// # Panics
/// - Panics if `inject_velocity_x_source entry point must exist`.
///
#[test]
fn test_velocity_source_shader_uses_validated_kspace_path() {
    let src = include_str!("../shaders/pstd.wgsl");
    let inject_start = src
        .find("fn inject_velocity_x_source")
        .expect("inject_velocity_x_source entry point must exist");
    let inject_block = &src[inject_start..src.len().min(inject_start + 600)];

    assert!(
        inject_block.contains("kspace_re[flat] += amp;"),
        "inject_velocity_x_source must inject into kspace_re for the validated source-kappa path"
    );
    assert!(
        src.contains("fn apply_source_kappa"),
        "velocity-source k-space injection requires apply_source_kappa"
    );
    assert!(
        src.contains("fn add_kspace_to_field_ux"),
        "velocity-source k-space injection requires add_kspace_to_field_ux"
    );
}

/// Regression guard: the WGSL push-constant layout must stay aligned with
/// the Rust-side `PstdParams` struct used for dispatch.
/// # Panics
/// - Panics if `PstdParams push-constant struct must exist in WGSL`.
///
#[test]
fn test_pstd_shader_push_constant_abi_matches_rust() {
    let src = include_str!("../shaders/pstd.wgsl");
    let struct_start = src
        .find("struct PstdParams")
        .expect("PstdParams push-constant struct must exist in WGSL");
    let struct_end = src[struct_start..]
        .find("\n}")
        .map(|offset| struct_start + offset + 2)
        .expect("PstdParams push-constant struct must close in WGSL");
    let struct_block = &src[struct_start..struct_end];

    assert!(
        !struct_block.contains("dx:"),
        "WGSL PstdParams must not add fields absent from Rust immediate data"
    );
    assert!(
        src.contains("precomp_source_kappa"),
        "validated phased-array GPU path requires precomp_source_kappa in the shader ABI"
    );
    assert_eq!(struct_block.matches(':').count(), 16);
    assert!(struct_block.contains("peak_offset: u32"));
    assert!(struct_block.contains("record_peak_pressure: u32"));
}

/// The host root table and shader workgroup arrays must describe the same
/// 1,024-point FFT contract.
#[test]
fn pstd_shader_declares_the_1024_point_shared_fft_contract() {
    let src = include_str!("../shaders/pstd.wgsl");

    assert!(src.contains("const MAX_FFT_LENGTH: u32 = 1024u;"));
    assert!(src.contains("var<workgroup> sm_re: array<f32, 1024>;"));
    assert!(src.contains("var<workgroup> sm_tw_re: array<f32, 512>;"));
    assert!(src.contains("let root_stride = MAX_FFT_LENGTH / n;"));
}

#[test]
fn pstd_shader_accumulates_peak_pressure_in_the_requested_output_region() {
    let src = include_str!("../shaders/pstd.wgsl");
    let peak_start = src
        .find("fn accumulate_peak_pressure(")
        .expect("peak-pressure entry point must exist");
    let peak_block = &src[peak_start..src.len().min(peak_start + 500)];

    assert!(peak_block.contains("params.peak_offset + idx"));
    assert!(peak_block.contains("max(sensor_data[output_idx], abs(field_p[idx]))"));
}
