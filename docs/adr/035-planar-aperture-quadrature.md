# ADR 035: Planar aperture quadrature

## Status

Accepted for the 4.0.0 transducer API.

- Revision 2026-10-16: rectangles join the same kernel as a third shape,
  because linear-array elements are rectangles. Evidence: the rectangle
  tests under Verification and the `linear_array_neuromodulation`
  example, which compares this oracle with a full-wave PSTD run.

## Context

The Rayleigh provider accepts only complete circular pistons. Hybrid Fresnel
zone pMUT cells require independently driven central-circle and annular
electrode sectors. Approximating those surfaces with point or circular-piston
clusters changes aperture area and diffraction, while consumer-owned
quadrature would duplicate the provider contract.

## Decision

Replace `CircularPiston` with `PlanarAperture`. One aperture carries an
oriented local frame, a complex surface-pressure phasor, and one outline:
a complete disk, an annular sector, or a rectangle whose width runs along
the first axis. Polar outlines use the squared-radius Gauss-Legendre rule,
which maps directly onto arbitrary radial bounds, and periodic angular
samples over the requested sector. A rectangle uses a tensor-product
Gauss-Legendre rule over its width and height. `RayleighIntegralSpec`
carries one primary and one secondary order, read per outline (radius and
angle, or width and height). The kernel builds each aperture's surface rule
once and retains the rigid-baffle and layered straight-ray contracts.
`PlanarApertureShape` is `#[non_exhaustive]`, so a further outline does not
break downstream matches.

Every independently driven electrode sector is one aperture. Electromechanical
coupling and drive-to-surface-pressure calibration remain consumer inputs; the
propagator owns only the prescribed acoustic boundary field.

## Rejected alternatives

- Preserve `CircularPiston` and add a second sector function: duplicates the
  propagation kernel and prevents one heterogeneous aperture list.
- A separate rectangle kernel beside the polar one (first draft for linear
  arrays): the same duplication as a second sector function, and it keeps
  rectangles out of mixed aperture lists.
- Tessellate sectors into circular pistons: does not preserve the commanded
  annular-sector boundary and introduces a resolution-dependent approximation.
- Collapse hybrid electrodes into one disk phasor: removes the independent
  spatial control required by the device.

## Verification

Disk closed-form and far-field regressions remain unchanged. New tests assert
the analytical annular-sector area and coherent equality between a complete
annulus and independently driven sectors carrying the same phasor on axis.
Rectangle tests assert the exact area and enclosing radius, the surface
rule's area, the Fraunhofer `sinc·sinc` pattern and the on-axis far-field
amplitude `kA/(2πR)` within the derived Fresnel bound, coherent equality of
two abutting halves with the whole, and area-conserving rasterization.
