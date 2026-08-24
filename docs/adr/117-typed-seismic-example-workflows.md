# 117. Type and partition the seismic example workflows

- Status: Accepted
- Date: 2026-08-20
- Item: `backlog.md#kw-examples-115--type-and-partition-the-seismic-example-workflows-major-arch--in-progress-2026-08-20`

## Context

The 2-D seismic imaging, 3-D seismic imaging, and transcranial FWI examples are
2,845, 1,870, and 755 lines respectively. They repeat CT loading and selection,
bone-volume-fraction property mapping, Ricker sampling, phantom construction,
quality metrics, raster output, and workflow control. The entry files own
multiple bounded concerns and the 3-D example has no executable value tests.

Their input behavior is also ambiguous. Missing or unreadable CT data changes a
requested real-data workflow into a synthetic workflow; absent MNI or T1 data
silently removes a reconstruction stage; some FWI and RTM failures are skipped
or replaced with zero-valued images. A successful process exit therefore does
not identify which computation ran.

The duplicated physical primitives already have providers:

- Aequitas owns dimensioned `Length`, `Frequency`, `Pressure`, `Velocity`, and
  `MassDensity` quantities with transparent zero-cost storage.
- `kwavers-physics` owns CT-derived heterogeneous skull construction and its
  bone-volume-fraction calibration.
- `kwavers-signal` owns `DomainRickerWavelet`.
- RITK owns DICOM, NIfTI, and PNG-series input unconditionally; the Kwavers
  `dicom` and `ritk` features are empty compatibility aliases.

ADR 115 covers frequency-domain FWI acquisition only. These examples use the
time-domain FWI implementation, so no part of that seam is reused here.

## Decision

Create one vertical `examples/seismic_imaging/` tree. Its common branch owns
only concerns shared by at least two workflows; dimension-specific branches own
their acquisition and reconstruction policy:

```text
seismic_imaging/
  common/
    configuration/
    input/
    medium/
    source/
    metrics/
    artifacts/
  two_dimensional/
    acquisition/
    inversion/
    workflow/
  three_dimensional/
    acquisition/
    inversion/
    workflow/
  transcranial/
    acquisition/
    inversion/
    workflow/
```

Every `mod.rs` is a manifest and every implementation leaf remains at or below
500 lines. Shared code moves to the deepest common ancestor; a function is not
made common solely because it has a similar name.

### Typed physical and shape boundaries

Configuration stores physical values as Aequitas quantities. Conversion to
the solver's current base-unit `f64` contract occurs once at the call boundary.
Grid families are zero-sized strategy types implementing a sealed local trait
with associated dimension and acquisition constants. Generic common code
monomorphizes per shape; there is no runtime shape branch or stored marker.

Const generics parameterize fixed-size acquisition tables where the table shape
is part of the workflow contract. Runtime-selected clinical volume dimensions
remain validated runtime values to avoid unbounded monomorphization.

No GAT is introduced: the workflows do not expose a lending iterator or a type
family varying by borrow lifetime. No `Cow` is introduced: selected paths and
decoded volumes are owned across the workflow, while transient views remain
ordinary borrows. Adding either without those contracts would be decorative
indirection rather than a zero-cost requirement.

### Provider ownership

Synthetic and decoded CT data are converted through the provider-owned
heterogeneous-skull API. The examples do not retain local `bvf`,
`hu_to_sound_speed`, or `hu_to_density` implementations. Source samples are
generated through `DomainRickerWavelet`; local copies of the Ricker equation
are deleted. Provider failures propagate with workflow context.

### Explicit workflow modes

Input is represented by an enum whose variants carry exactly their data:

```rust
enum PhantomSource {
    Synthetic,
    Ct(PathBuf),
}

enum BrainPrior {
    Uniform,
    Mni(PathBuf),
    T1(PathBuf),
}
```

The documented no-argument mode is synthetic and uniform. Selecting a path
makes that path required: parse, validation, or computation failure returns an
error and never changes variants. FWI and RTM stages likewise return their
typed failure instead of being skipped or replaced with a fabricated image.
Outputs default to a gitignored directory below `target/seismic-imaging/` and
may be replaced by an explicit output path.

### Feature deletion

Delete the empty `dicom` and `ritk` features and remove `dicom` from `full`.
Update every in-repository run command in the same change. No deprecated alias
or forwarding feature remains. This is a breaking Cargo-feature change and is
therefore classified `[major]`; the changelog is the migration path.

## Verification

- Parser tests cover the default, every explicit variant, conflicting options,
  missing values, and unknown options.
- Provider differential tests compare the shared CT and source paths with the
  direct provider calls at water, intermediate bone fraction, cortical bone,
  negative HU, and out-of-range HU boundaries.
- Small deterministic workflow tests assert objective/image values and error
  propagation; they do not assert only successful construction.
- Focused Nextest, strict Clippy, doctests, bounded example runs, mdBook tests
  and build, link audits, and feature-residue scans gate the change.
- Exact-head hosted CI and architecture matrices gate merge.

## Consequences

The examples become independently navigable and their observed output names the
selected workflow. Shared physical formulas disappear from consumer code, and
real-data failures can no longer produce plausible synthetic artifacts.

The entry points and Cargo feature names change without compatibility aliases.
External commands using `--features dicom` or `--features ritk` must remove
those flags because RITK I/O is already compiled as a normal dependency.

## Alternatives rejected

**Keep the monoliths and extract a generic utility file.** A generic file would
preserve mixed ownership and create a junk drawer. The vertical tree assigns
each concern one named home.

**Keep environment-variable fallback.** It makes a load error observationally
equivalent to selecting a different experiment and masks integration defects.

**Copy the provider equations into the common module.** Consolidating three
consumer copies into one consumer copy still leaves ownership below the actual
provider and allows future numerical drift.

**Add GAT, `Cow`, or dynamic dispatch to satisfy an abstraction checklist.**
None models a present contract in these workflows. The selected quantities,
associated consts, const generics, and static strategy dispatch encode the real
variation dimensions without allocation or vtable cost.
