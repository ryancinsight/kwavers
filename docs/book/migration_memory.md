# Chapter 39 — Memory: Mnemosyne and Themis

The Atlas memory subsystem replaces ad-hoc `Vec<T>` allocations in hot paths
with `mnemosyne` (allocator) and `themis` (region/arena strategies).

## Why Custom Allocators?

kwavers simulation loops allocate and free many same-size scratch buffers
(FFT plans, FDTD step buffers, sensor data). The system allocator fragments
these over time. `themis` arenas amortize allocation to a single region bump.

## Crate Roles

| Crate | Role |
|---|---|
| `mnemosyne` | Global allocator + thread-local scratch pools |
| `themis` | Region/arena strategies, per-solve lifetime scopes |

## Integration in kwavers

kwavers currently uses `mnemosyne` at the optional `eunomia` integration
boundary. The `mnemosyne::eunomia` feature enables `Complex<T>` scratch
cache allocation through the mnemosyne TLS pool.

```toml
# kwavers/Cargo.toml
mnemosyne = { workspace = true, features = ["std_tls", "eunomia"] }
```

## Pattern: Arena-scoped scratch

```rust
// Future pattern (themis integration in progress):
// use themis::Arena;
// let scratch = Arena::new(1 << 20); // 1 MiB region
// let buf: &mut [f64] = scratch.alloc_slice(1024);
// // buf is valid for the arena's lifetime — zero-copy, zero-heap-alloc
```

## Zero-Copy Arrays

`leto::VecStorage<T>` is backed by `Vec<T>` today; the migration path is to
replace `Vec<T>` with a `MnemosyneStorage<T>` that draws from the TLS pool,
eliminating system allocator round-trips in hot simulation loops.
