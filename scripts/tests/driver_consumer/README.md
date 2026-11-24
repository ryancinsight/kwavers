# External Driver qualification

This standalone Cargo workspace consumes exactly the published `kwavers-driver 0.5.0`. Cargo enables Driver's actual default through the consumer's `io` feature. The `kwavers` feature forwards only the Transducer feature. Each matrix cell is a separate invocation. The test binary imports Driver through the public crate API.

Use Rust 1.97.0, cargo-nextest and Python 3.12 or newer. From this directory run:

```sh
python3 replay.py --output /tmp/driver-published-results --bench
```

The published baseline exits with status 1 after running every cell. Read each cell's log for the assertion and input, and `results.json` for the activated features, direct dependencies, registry identities, archive checksum and exit codes. A release candidate passes when all cells exit successfully. Nextest terminates an individual test at 60 seconds. The benchmark is informational.

To rehearse a packaged candidate, give the same runner an archive. For example:

```sh
python3 replay.py --archive /path/to/kwavers-driver-0.5.1.crate --output /tmp/driver-preview-results --bench
```

The runner copies the consumer to a temporary directory, extracts the candidate, and changes only Driver's dependency to the exact unpacked archive. All transitive dependencies must resolve to crates.io. The published replay requires every dependency to resolve to crates.io and verifies the published SHA-256. A candidate rehearsal cannot supply a registry receipt for an unpublished version.

The pair fixture uses the published 6/2 mm and 6/6.3 mm route lengths, plus the two-layer 6+2 versus 2+6 mm layout. The tests read individual audit counters. Other board rules can flag these small, open-ended layouts. Power and thermal tests model a capacitive channel at 2 MHz and check each heat path against its per-cycle energy. The 96-lane handoff uses four tile profiles and the selected simulator. The KiCad tests save a two-pin interface and inspect PCB tracks, net labels, project settings and errors. The performance harness measures 96-lane PCB and project-string emission.

The matrix can also be run manually with `cargo +1.97.0 nextest run --locked --no-fail-fast` and these four flag sets, in order: `--no-default-features`, no flags, `--no-default-features --features kwavers`, and `--all-features`.
