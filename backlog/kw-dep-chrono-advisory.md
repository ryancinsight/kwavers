<a id="kw-dep-chrono-advisory"></a>

## KW-DEP-CHRONO-ADVISORY — Clear the Chrono advisory from the audited graph [patch] — in progress

- The main branch changed the diagnostic examination timestamp to the
  existing `SystemTime` contract. The candidate formats UTC civil-time
  reports with Horae and removes the remaining integration-test dev edge.
- Kwavers uses Iris instead of Plotly, removing the Plotly route to
  `serde_with`'s Chrono feature. `ritk-io` still reaches
  `dicom-core` for the external dataset implementation.
- Keep the advisory hold for that route. Its removal depends on RITK's
  native dataset, writer, and codec parity. Once `cargo tree -i chrono`
  reports an empty audited graph, remove the hold and run
  `cargo deny check advisories`.
