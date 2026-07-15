# ADR-035: Planar aperture quadrature

## Status

Accepted for the 4.0.0 transducer API.

## Context

The Rayleigh provider accepts only complete circular pistons. Hybrid Fresnel
zone pMUT cells require independently driven central-circle and annular
electrode sectors. Approximating those surfaces with point or circular-piston
clusters changes aperture area and diffraction, while consumer-owned
quadrature would duplicate the provider contract.

## Decision

Replace `CircularPiston` with `PlanarAperture`. One aperture carries an
oriented local frame, a complex surface-pressure phasor, and either complete
disk or annular-sector radial/angular bounds. The existing squared-radius
Gauss-Legendre rule maps directly onto arbitrary radial bounds; periodic
angular samples map onto the requested sector. The kernel integrates each
surface once and retains the rigid-baffle and layered straight-ray contracts.

Every independently driven electrode sector is one aperture. Electromechanical
coupling and drive-to-surface-pressure calibration remain consumer inputs; the
propagator owns only the prescribed acoustic boundary field.

## Rejected alternatives

- Preserve `CircularPiston` and add a second sector function: duplicates the
  propagation kernel and prevents one heterogeneous aperture list.
- Tessellate sectors into circular pistons: does not preserve the commanded
  annular-sector boundary and introduces a resolution-dependent approximation.
- Collapse hybrid electrodes into one disk phasor: removes the independent
  spatial control required by the device.

## Verification

Disk closed-form and far-field regressions remain unchanged. New tests assert
the analytical annular-sector area and coherent equality between a complete
annulus and independently driven sectors carrying the same phasor on axis.
