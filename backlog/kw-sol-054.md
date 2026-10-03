<a id="kw-sol-054"></a>

## KW-SOL-054 — Repair AVX-512 FDTD layout contract [patch] — todo

priority: verification; needs: none; scope: `crates/kwavers-solver` AVX-512 FDTD kernels and their `avx512` tests

- **Cannot be verified here, and is not verified anywhere.** The acceptance needs matching analytical reference fields on an AVX-512 host. The development machine reports `avx512f=false` (hybrid Core Ultra, AVX-512 fused off), so the seven `avx512` tests pass on the scalar fallback, and two of them assert nothing there (`processor_or_skip` returns `None` and `pressure_update_keeps_interior_constant_for_uniform_field` / `velocity_update_matches_linear_pressure_gradient` early-return). A green run here is evidence about the fallback, not the kernels.
- **No CI host runs them either:** no workflow sets an AVX-512 runner or `target-feature`, so the kernels ship unexercised on every available machine class.
- **Acceptance:** the value-semantic cases run on a host reporting `avx512f=true` and the run is recorded; until then the kernels carry no behavioral evidence and the item stays open.
- **Emulator route tried 2026-09-09, both unavailable from this host:** Intel SDE (`downloadmirror.intel.com` resets over IPv6 and the package pages return Akamai `Access Denied` over IPv4, so the current URL cannot be discovered; an unofficial mirror is not acceptable for a tool that executes test binaries) and WSL + `qemu-user` (the registered Ubuntu distro's `ext4.vhdx` is gone).
- **Smallest unblocking action, any one:** place an `sde`/`sde64` on `PATH` (manual click-through download), repair the WSL distro, or register the self-hosted AVX-512 runner.
