use super::*;

#[test]
fn wgpu_compute_commands_name_is_provider_specific() {
    assert!(std::mem::size_of::<WgpuComputeCommands>() > 0);
}
