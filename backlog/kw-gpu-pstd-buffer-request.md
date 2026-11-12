<a id="kw-gpu-pstd-buffer-request"></a>

## KW-GPU-PSTD-BUFFER-REQUEST — GPU PSTD requests a 128 MiB buffer limit whatever the adapter supports [minor] — todo

priority: feature; needs: none; basis: d1a3d463bfed5532f06d6ca5405ea64075335b27; scope: `crates/kwavers-gpu/src/gpu/device.rs`, `crates/kwavers-gpu/src/pstd_gpu/pipeline/auto_device.rs`, `crates/kwavers-gpu/src/pstd_gpu/pipeline/capacity.rs`

- **Origin.** #1059 refuses a plan whose `pml_xyz` or `shifts_all` exceeds a limit the device enables. The provider asks for that limit itself.
- **Read at the basis.** `minimal_compute_limits` (`crates/kwavers-gpu/src/gpu/device.rs` line 300) requests `max_buffer_size: 128 * 1024 * 1024` (line 302). `pstd_required_limits` (`pipeline/auto_device.rs` line 70) starts from `WgpuDevice::required_limits()`, which starts from `minimal_compute_limits` (`device.rs` lines 81 and 82). At the locked Hephaestus revision (72ffa0577952411a27182782637248a2c8a02ead), `DeviceLimits` (`hephaestus-core/src/domain/device.rs` line 43) has no field for the storage binding size, and `wgpu_limits_from_device_limits` (`hephaestus-wgpu/src/infrastructure/device/features.rs`, from line 69) takes `max_buffer_size` from the request and every limit it does not name from `wgpu::Limits::downlevel_defaults` (`..downlevel`).
- **Relayed** from the second verdict on #1060 (comment 6815670463): in `wgpu-types` 30.0.1 the `max_storage_buffer_binding_size` of `downlevel_defaults` is 128 MiB.
- **Run by the integrator of #1059**, on its development host, which has an adapter: `SimulationRunner::run` on a 256x256x256 request returned `ResourceLimitExceeded` naming `max_buffer_size = 134217728` (`pml_xyz` needs 201326592 bytes). Derived: `pml_xyz` fits up to 11184810 grid points (134217728 / 12, rounded down).
- **Not known:** the limits the adapters in use support.
- **Acceptance:** the provider requests buffer limits from what the adapter supports or the plan needs, so a plan the adapter can hold is not refused by the request; the capacity refusal then names the adapter limit; a device test builds a plan above 11184810 points where the adapter supports it; a device test lowers `max_storage_buffer_binding_size` alone and fails when `StorageBufferLimits::enabled_on` reads `max_buffer_size` into both fields (that mutant passes every test of #1059: its first verdict, comment 6814685775, finding 2).
- **Next step:** read `adapter.limits()` on a host with an adapter; file the binding-size field on the Hephaestus board.
