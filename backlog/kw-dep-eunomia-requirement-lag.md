<a id="kw-dep-eunomia-requirement-lag"></a>

## KW-DEP-EUNOMIA-REQUIREMENT-LAG — The Eunomia requirement admits no version its repository now holds [patch] — todo

priority: architecture; needs: none; basis: 9b2b75c590efb6503f92d01aee2fae79ef73690e; scope: `Cargo.toml`, `Cargo.lock`

- **Read at the basis.** `Cargo.toml` line 63 requires `eunomia = { version = "0.11.0", git = "https://github.com/ryancinsight/eunomia", ... }`, and `Cargo.lock` pins eunomia 0.11.0 at d8ee31abf9e10c3aaa4efd67ed5912a612086115.
- **Observed by the integrator of #1059**, from its lane `worktrees/kw-swefix2` on its development host: a resolution outside the lockfile (the `cargo update` that `cargo semver-checks` 0.51.0 runs) fails with "failed to select a version for the requirement `eunomia = \"^0.11.0\"`; candidate versions found which didn't match: 0.12.0; location searched: Git repository https://github.com/ryancinsight/eunomia". Locked builds are unaffected.
- **Not known:** which other first-party requirements of this workspace lag, and whether Leto and the other providers this workspace takes Eunomia through already admit 0.12 (Hephaestus `master` at 0a4e22f133e2c392554d640fb2a5606f052388fc requires `>=0.10.0, <0.13.0`). `git grep -n -i eunomia origin/main -- backlog.md backlog` in the Atlas repository, filtered for `0.12` and `kwavers`, found no item for this lag; the search is of those patterns only.
- **Acceptance:** the requirement admits the provider's current version, `Cargo.lock` pins a landed Eunomia commit of that version, and the workspace gates pass; `cargo update --dry-run` resolves.
- **Next step:** read the Eunomia requirement of each first-party provider in `Cargo.lock`; advance in layer order, providers first.
