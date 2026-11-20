<a id="kw-dep-chrono-advisory"></a>

## KW-DEP-CHRONO-ADVISORY — Clear the Chrono advisory from the audited graph [patch] — done

Kwavers uses `SystemTime` for clinical timestamps, Horae for UTC civil time,
and Iris for HTML plots. RITK's native dataset and writer remove the last
transitive route to Chrono through `dicom-core`.

The locked workspace resolves all features and target platforms without
Chrono across normal, build, and development edges. The lockfile also
excludes the package. The supply-chain policy no longer ignores
RUSTSEC-2026-0543.
