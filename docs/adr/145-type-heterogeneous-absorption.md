# ADR 145: Type heterogeneous absorption fields

- Status: Proposed
- Class: [major]
- Date: 2026-11-21

Heterogeneous media store one reciprocal-length field for reference loss,
one dimensionless exponent field, and an Aequitas reference frequency.
Spatial lookup evaluates the field at its selected coordinate before
applying `α_ref (f/f_ref)^y`. The ratio is dimensionless and the result
retains inverse-length units. The independent tissue-medium cache also
uses the typed loss field.

The field retains one Leto buffer of base-unit samples so existing solver
array views can borrow it directly. Typed indexing and mutation prevent
cross-dimension writes. Input factories convert decibels per centimetre at
the declared reference through the shared absorption conversion. The
legacy write-only duplicate no longer allocates a voxel array.

A nondefault 2 MHz reference test checks that the typed sample and solver
view agree at 6 Np/m and that a three-halves exponent yields 48 Np/m at
8 MHz. A compile-fail case rejects assigning frequency to the loss field.
