# kwavers-alloc-probe

Thread-scoped allocation counting for
[kwavers](https://github.com/ryancinsight/kwavers) allocation-contract tests.

A kwavers hot path is expected to be allocation-free, and that expectation is asserted
with exact counts. A process-global counting allocator cannot support such an assertion
under a parallel test runner: it observes every thread in the process, so a measurement
window occasionally records harness or peer-test allocations and the count flakes.

This probe counts only on threads that have opened a `Window`, confining the measurement
to the test's own effects while forwarding every request to the system allocator
unchanged.

## Usage

In a test binary:

```rust,ignore
#[global_allocator]
static GLOBAL: kwavers_alloc_probe::ThreadScopedAllocator =
    kwavers_alloc_probe::ThreadScopedAllocator;

let window = kwavers_alloc_probe::Window::open();
let value = build_the_thing();
let change = window.change();
assert_eq!(change.allocations, 1);
```

The example is marked `ignore` because installing a `#[global_allocator]` is a
crate-level decision that a doctest cannot make; the crate's own tests exercise the
counting contract directly.

## Documentation

- API reference: <https://docs.rs/kwavers-alloc-probe>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
