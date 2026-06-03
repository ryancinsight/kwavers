//! Shader ABI and validated k-space path regression tests.

/// Regression guard: the GPU phased-array path injects velocity sources in
/// k-space and must retain the source-kappa reconstruction chain.
/// # Panics
/// - Panics if `inject_velocity_x_source entry point must exist`.
///
#[test]
fn test_velocity_source_shader_uses_validated_kspace_path() {
    let src = include_str!("../../../../../gpu/shaders/pstd.wgsl");
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
    let src = include_str!("../../../../../gpu/shaders/pstd.wgsl");
    let struct_start = src
        .find("struct PstdParams")
        .expect("PstdParams push-constant struct must exist in WGSL");
    let struct_block = &src[struct_start..src.len().min(struct_start + 500)];

    assert!(
        !struct_block.contains("dx:"),
        "WGSL PstdParams must not add fields absent from Rust push constants"
    );
    assert!(
        src.contains("precomp_source_kappa"),
        "validated phased-array GPU path requires precomp_source_kappa in the shader ABI"
    );
}
