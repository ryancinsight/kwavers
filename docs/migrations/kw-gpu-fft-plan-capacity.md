<a id="kw-gpu-fft-plan-capacity"></a>

## KW-GPU-FFT-PLAN-CAPACITY — Check the FFT workspace and dispatch before constructing PSTD state [patch] — verified locally

- The WGPU construction boundary uses the Hephaestus requirements query.
  The typed capacity error retains the resource, required quantity, and
  device limit as Kwavers ResourceLimitExceeded.
- The runner holds a checked plan for the selected device, grid, and
  validated medium before it expands the PML and power-law fields.
  A rejected PML-enabled grid avoids allocating ten volume-sized f32
  fields, a total of 40 bytes × grid points. With PML disabled, all six
  identity boundary views share the same immutable host allocation.
- Real-device tests cover field-buffer refusal, a 17×16×16 Bluestein
  plan at 65,535 and 65,536 bytes, and both large cases through
  SimulationRunner::run. The 129×256×264 grid requires 138,412,032 bytes
  against a 134,217,728-byte ceiling. The 200³ grid needs 80,000 against
  65,535 workgroups, with and without a sensor. Tests assert full error
  messages, bitwise parity of the staged values, and identity-buffer aliasing.
- The two runner requests use a lossless medium to select the 23-binding
  pipeline on this 29-binding adapter. The absorbing pipeline needs 31
  bindings and verification on a device that provides them.
- Hosted checks and fresh provider resolution remain integration gates.

### Original evidence

priority: correctness; needs: none; basis: d1a3d463bfed5532f06d6ca5405ea64075335b27; scope: `crates/kwavers-gpu/src/pstd_gpu/pipeline/`, `crates/kwavers-simulation/src/runner.rs`

- **Origin.** The remainder of the capacity item that #1059 closes and deletes; its acceptance asks for the refusal "before the run allocates field buffers". #1059 checks two buffers, `pml_xyz` and `shifts_all`; its first verdict (comment 6814685775, finding 1) shows that the FFT plan Hephaestus prepares is outside that check.
- **Read at the basis.** `build_state` creates the seven field buffers from line 117 of `crates/kwavers-gpu/src/pstd_gpu/pipeline/construction/solver.rs` and the two k-space buffers at lines 137 and 140, and only then prepares the FFT plans (lines 155 and 158); `with_auto_device` wraps the failure as `InvalidInput` (`pipeline/auto_device.rs` lines 62 and 63). At the locked Hephaestus revision (72ffa0577952411a27182782637248a2c8a02ead), `crates/hephaestus-wgpu/src/application/fft/plan.rs` gives a non-power-of-two axis of length n a workspace of m values per line, m the power of two at or above 2n - 1 (line 73). It refuses a volume, workspace or twiddle buffer over `max_buffer_size` (line 228) and a workspace of more than 256 values (`WORKGROUP_SIZE`, `kernel.rs` line 10) times `max_compute_workgroups_per_dimension` (line 236), each as `HephaestusError::InvalidConfiguration`, whose one field is a message.
- **Run by the judge of #1059** at its first judged head (its comment 6814685775): `SimulationRunner::run` on 129x256x264 returned `InvalidInput` ending "FFT workspace requires 138412032 bytes, exceeding device max_buffer_size=134217728"; `GpuPstdSolver::new` on 17x16x16 at a `max_buffer_size` of 61440 returned a message ending "FFT workspace requires 65536 bytes, exceeding device max_buffer_size=61440". **Run by the integrator** on an export of that head: `SimulationRunner::run` on 200x200x200 returned `InvalidInput` ending "FFT dispatch requires 80000 x workgroups, exceeding device max_compute_workgroups_per_dimension=65535", with and without a sensor point. No run panicked. Derived: the #1059 check admits all three: `pml_xyz` needs 104620032, 52224 and 96000000 bytes and `shifts_all` 10384, 784 and 9600, each under the `max_buffer_size` of its run (134217728 bytes, and 61440 for 17x16x16); the binding limit of the judge's device is not reported.
