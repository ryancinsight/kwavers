//! Coordinate-conversion and surface-sampling helpers for abdominal array placement.

use std::collections::VecDeque;

use leto::Array3;

use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::geometry::{is_boundary_3d, Point3};

/// Keep only the largest 6-connected component of `mask`, returning a new mask.
///
/// ## Why this exists — CT bed / table removal
///
/// A clinical CT volume thresholded by HU contains the patient *and* the
/// imaging-table material (carbon-fibre / PVC pad, HU ≈ −100 to +200) and
/// any non-anatomical structures inside the FOV (IV bags, EKG leads,
/// positioning cushions). For the abdominal placement task only the patient
/// matters: the table introduces a long inferior shelf that flips the
/// nearest-skin computation to the bed surface and contaminates the
/// rendered body cloud with a planar slab beneath the patient.
///
/// In all clinical CT acquisitions the patient is by definition the largest
/// connected tissue volume in the scan; the bed and ancillary objects form
/// strictly smaller 6-connected components in HU-thresholded space (the
/// bed–patient contact is, at the imaging resolution, an air-gap of one
/// to a few voxels because the body has a curved posterior and the bed is
/// flat). This routine retains only that largest component.
///
/// ## Algorithm
///
/// Standard breadth-first labelling: iterate the volume; for each
/// unvisited foreground voxel start a BFS, count its voxels, record its
/// label. After labelling all components, rebuild the output mask
/// containing only voxels carrying the maximum-count label. Time
/// complexity O(N) for `N = nx · ny · nz`; memory O(N) for the visited
/// + label scratch.
///
/// ## Returns
///
/// The same input mask if it has zero or one connected component;
/// otherwise the input mask with all components except the largest
/// cleared to `false`.
pub(crate) fn keep_largest_connected_component_3d(mask: &Array3<bool>) -> Array3<bool> {
    let [nx, ny, nz] = mask.shape();
    if nx == 0 || ny == 0 || nz == 0 {
        return mask.clone();
    }

    // 0 = unvisited / background; positive integers identify components.
    let mut labels: Array3<u32> = Array3::zeros((nx, ny, nz));
    let mut sizes: Vec<u64> = vec![0]; // sizes[0] is unused so labels start at 1.

    for start_iz in 0..nz {
        for start_iy in 0..ny {
            for start_ix in 0..nx {
                if !mask[[start_ix, start_iy, start_iz]]
                    || labels[[start_ix, start_iy, start_iz]] != 0
                {
                    continue;
                }
                let label = sizes.len() as u32;
                let mut count: u64 = 0;
                let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();
                queue.push_back((start_ix, start_iy, start_iz));
                labels[[start_ix, start_iy, start_iz]] = label;
                while let Some((ix, iy, iz)) = queue.pop_front() {
                    count += 1;
                    macro_rules! visit_neighbour {
                        ($nix:expr, $niy:expr, $niz:expr) => {{
                            let (nix, niy, niz) = ($nix, $niy, $niz);
                            if mask[[nix, niy, niz]] && labels[[nix, niy, niz]] == 0 {
                                labels[[nix, niy, niz]] = label;
                                queue.push_back((nix, niy, niz));
                            }
                        }};
                    }
                    if ix > 0 {
                        visit_neighbour!(ix - 1, iy, iz);
                    }
                    if ix + 1 < nx {
                        visit_neighbour!(ix + 1, iy, iz);
                    }
                    if iy > 0 {
                        visit_neighbour!(ix, iy - 1, iz);
                    }
                    if iy + 1 < ny {
                        visit_neighbour!(ix, iy + 1, iz);
                    }
                    if iz > 0 {
                        visit_neighbour!(ix, iy, iz - 1);
                    }
                    if iz + 1 < nz {
                        visit_neighbour!(ix, iy, iz + 1);
                    }
                }
                sizes.push(count);
            }
        }
    }

    if sizes.len() <= 2 {
        return mask.clone();
    }

    // sizes[0] is the unused sentinel; argmax over [1..].
    let mut best_label: u32 = 1;
    let mut best_size: u64 = sizes[1];
    for (idx, &count) in sizes.iter().enumerate().skip(2) {
        if count > best_size {
            best_size = count;
            best_label = idx as u32;
        }
    }

    let mut output: Array3<bool> = Array3::from_elem((nx, ny, nz), false);
    for ([ix, iy, iz], &label) in labels.indexed_iter() {
        if label == best_label {
            output[[ix, iy, iz]] = true;
        }
    }
    output
}

/// Convert a voxel centroid (index-space float) to physical metres.
pub(crate) fn index_to_point(
    index: [f64; 3],
    spacing_m: [f64; 3],
    center_index: [f64; 3],
) -> Point3 {
    Point3 {
        x_m: (index[0] - center_index[0]) * spacing_m[0],
        y_m: (index[1] - center_index[1]) * spacing_m[1],
        z_m: (index[2] - center_index[2]) * spacing_m[2],
    }
}

/// Euclidean distance between two 3-D points `m`.
pub(super) fn distance_3d(a: Point3, b: Point3) -> f64 {
    let dx = a.x_m - b.x_m;
    let dy = a.y_m - b.y_m;
    let dz = a.z_m - b.z_m;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Convert voxel indices to physical metres relative to `center_index`.
pub(super) fn voxel_to_point(
    ix: usize,
    iy: usize,
    iz: usize,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
) -> Point3 {
    Point3 {
        x_m: (ix as f64 - center_index[0]) * spacing_m[0],
        y_m: (iy as f64 - center_index[1]) * spacing_m[1],
        z_m: (iz as f64 - center_index[2]) * spacing_m[2],
    }
}

/// Flood-fill the exterior air region from the volume boundary.
///
/// ## Algorithm
///
/// Seeds a BFS at every non-body voxel that touches one of the six faces of
/// the bounding box.  The BFS propagates through 6-connected non-body voxels.
/// The result is a mask where `true` means "reachable from the volume boundary
/// without crossing any body-mask voxel" — i.e. it is in the exterior air
/// connected to the outside of the patient.
///
/// ## Why this is necessary
///
/// `is_boundary_3d` flags **any** body voxel with an inactive 6-connected
/// neighbour.  Inside a real CT, intestinal gas, bile ducts, retroperitoneal
/// fat pockets, and vessel lumens all create interior air/low-HU voids.  Their
/// body-mask boundaries are detected as "skin" by the naive boundary test,
/// placing the nearest-skin-contact inside the patient.  The flood-fill
/// correctly restricts skin candidates to the **exterior** skin surface.
pub(crate) fn exterior_air_mask(body_mask: &Array3<bool>) -> Array3<bool> {
    let [nx, ny, nz] = body_mask.shape();
    let mut exterior: Array3<bool> = Array3::from_elem((nx, ny, nz), false);
    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

    // ── Detect CT cut planes ──────────────────────────────────────────────────
    //
    // A CT volume truncated mid-body (e.g. chest cut for an abdominal scan) has
    // a face where the majority of voxels are body tissue.  Seeding exterior air
    // from such a face lets interior voids that touch the cut plane (lungs, pleural
    // cavity, airways) flood through the BFS and be labelled "exterior", placing the
    // nearest-skin-contact on an internal boundary (lung–liver interface, spine, etc.)
    // rather than the true exterior skin surface.
    //
    // Heuristic: if the body-voxel fraction on a face exceeds CUT_BODY_FRACTION,
    // that face is treated as a cut plane and excluded from exterior-air seeding.
    // The true exterior air is still reached via the remaining open faces (lateral
    // sides, anterior surface).  Fallback: if every face exceeds the threshold
    // (degenerate or fully-enclosed volume), seed all six faces.
    const CUT_BODY_FRACTION: f64 = 0.50;

    let mut xmin_body: usize = 0;
    let mut xmax_body: usize = 0;
    let mut ymin_body: usize = 0;
    let mut ymax_body: usize = 0;
    let mut zmin_body: usize = 0;
    let mut zmax_body: usize = 0;

    for iy in 0..ny {
        for iz in 0..nz {
            if body_mask[[0, iy, iz]] {
                xmin_body += 1;
            }
            if body_mask[[nx - 1, iy, iz]] {
                xmax_body += 1;
            }
        }
    }
    for ix in 0..nx {
        for iz in 0..nz {
            if body_mask[[ix, 0, iz]] {
                ymin_body += 1;
            }
            if body_mask[[ix, ny - 1, iz]] {
                ymax_body += 1;
            }
        }
    }
    for ix in 0..nx {
        for iy in 0..ny {
            if body_mask[[ix, iy, 0]] {
                zmin_body += 1;
            }
            if body_mask[[ix, iy, nz - 1]] {
                zmax_body += 1;
            }
        }
    }

    // `as f64 < X` is ambiguous (parsed as `as f64<X>`); parenthesise the division.
    let open_xmin = ((xmin_body as f64) / ((ny * nz) as f64)) < CUT_BODY_FRACTION;
    let open_xmax = ((xmax_body as f64) / ((ny * nz) as f64)) < CUT_BODY_FRACTION;
    let open_ymin = ((ymin_body as f64) / ((nx * nz) as f64)) < CUT_BODY_FRACTION;
    let open_ymax = ((ymax_body as f64) / ((nx * nz) as f64)) < CUT_BODY_FRACTION;
    let open_zmin = ((zmin_body as f64) / ((nx * ny) as f64)) < CUT_BODY_FRACTION;
    let open_zmax = ((zmax_body as f64) / ((nx * ny) as f64)) < CUT_BODY_FRACTION;

    // When no face is "open", fall back to seeding all (degenerate volume).
    let any_open = open_xmin || open_xmax || open_ymin || open_ymax || open_zmin || open_zmax;
    let do_xmin = !any_open || open_xmin;
    let do_xmax = !any_open || open_xmax;
    let do_ymin = !any_open || open_ymin;
    let do_ymax = !any_open || open_ymax;
    let do_zmin = !any_open || open_zmin;
    let do_zmax = !any_open || open_zmax;

    // ── Seed: non-body voxels on all qualifying (non-cut) faces ───────────────
    let mut seed = |ix: usize, iy: usize, iz: usize| {
        if !body_mask[[ix, iy, iz]] && !exterior[[ix, iy, iz]] {
            exterior[[ix, iy, iz]] = true;
            queue.push_back((ix, iy, iz));
        }
    };

    if do_xmin || do_xmax {
        for iy in 0..ny {
            for iz in 0..nz {
                if do_xmin {
                    seed(0, iy, iz);
                }
                if do_xmax {
                    seed(nx - 1, iy, iz);
                }
            }
        }
    }
    if do_ymin || do_ymax {
        for ix in 0..nx {
            for iz in 0..nz {
                if do_ymin {
                    seed(ix, 0, iz);
                }
                if do_ymax {
                    seed(ix, ny - 1, iz);
                }
            }
        }
    }
    if do_zmin || do_zmax {
        for ix in 0..nx {
            for iy in 0..ny {
                if do_zmin {
                    seed(ix, iy, 0);
                }
                if do_zmax {
                    seed(ix, iy, nz - 1);
                }
            }
        }
    }

    // ── BFS through non-body voxels ───────────────────────────────────────────
    while let Some((ix, iy, iz)) = queue.pop_front() {
        macro_rules! expand {
            ($nx2:expr, $ny2:expr, $nz2:expr) => {
                let (nx2, ny2, nz2) = ($nx2, $ny2, $nz2);
                if !body_mask[[nx2, ny2, nz2]] && !exterior[[nx2, ny2, nz2]] {
                    exterior[[nx2, ny2, nz2]] = true;
                    queue.push_back((nx2, ny2, nz2));
                }
            };
        }
        if ix > 0 {
            expand!(ix - 1, iy, iz);
        }
        if ix + 1 < nx {
            expand!(ix + 1, iy, iz);
        }
        if iy > 0 {
            expand!(ix, iy - 1, iz);
        }
        if iy + 1 < ny {
            expand!(ix, iy + 1, iz);
        }
        if iz > 0 {
            expand!(ix, iy, iz - 1);
        }
        if iz + 1 < nz {
            expand!(ix, iy, iz + 1);
        }
    }

    exterior
}

/// Nearest **exterior** skin contact point to `focus_m`.
///
/// Restricts candidates to body voxels that directly adjoin the
/// **pre-computed** exterior air mask (see [`exterior_air_mask`]).  This
/// excludes internal tissue interfaces, intestinal gas walls, and
/// retroperitoneal fat boundaries — all of which `is_boundary_3d` would
/// incorrectly accept.  Pass the result of [`exterior_air_mask`] as
/// `exterior`; it is computed once and reused for body surface extraction.
///
/// ## Approach-angle penalty
///
/// For organs near the diaphragm (liver, adrenal), the geometrically nearest
/// exterior skin can be the chest wall above the organ, requiring a beam path
/// through ribs and pleura — clinically undesirable.  The laterally correct
/// approach is from the anterior or flank skin at the same axial level.
///
/// To prefer horizontal (lateral/anterior) access over vertical
/// (superior/inferior) access, the minimisation objective is
///
/// ```text
/// score(P) = ‖P − F‖ + W · (P_z − F_z)² / ‖P − F‖
/// ```
///
/// where `W = APPROACH_Z_WEIGHT`.  At `W = 0` this reduces to pure Euclidean
/// distance; the penalty scales with the square of the z-deviation of the
/// beam, and is normalised by distance so that only the direction matters for
/// comparable candidates.  The value `W = 4.0` is derived by requiring that
/// an anterior skin contact at twice the chest-wall distance is preferred to
/// the chest-wall contact.
pub(crate) fn nearest_exterior_skin_point(
    body_mask: &Array3<bool>,
    exterior: &Array3<bool>,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
    focus_m: Point3,
) -> KwaversResult<Point3> {
    // ── Approach-angle penalty weights ─────────────────────────────────────────
    //
    // score(P) = ‖P − F‖
    //          + W_z · (P_z − F_z)² / ‖P − F‖       [superior/inferior penalty]
    //          + W_y · y_cross²      / ‖P − F‖       [coronal-midplane penalty]
    //
    // where y_cross = max(0, −sign(F_y) · P_y): positive only when the skin
    // contact P is on the OPPOSITE side of y = 0 from the focus.  For an anterior
    // organ (F_y > 0), approaching from the posterior half (P_y < 0) requires the
    // beam to traverse the spine and IVC — clinically contraindicated.  For a
    // posterior organ (F_y < 0, e.g. retroperitoneal kidney), skin on the
    // posterior half (P_y < 0) shares the sign of F_y, so y_cross = 0 and no
    // penalty is applied; the flank/posterior approach remains unhindered.
    //
    // Derivation of W_z = 4.0:
    //   Chest wall (liver): d_c ≈ 44 mm, dz ≈ 34 mm.  Anterior abdominal: d_h ≈ 88 mm, dz = 0.
    //   Score_c < Score_h  ↔  44 + W_z·34²/44 < 88  ↔  W_z < 2.14; W_z = 4 intentionally
    //   exceeds this so that the chest wall is excluded even when d_h is modestly larger.
    //
    // Derivation of W_y = 6.0:
    //   Posterior spine approach (liver): d ≈ 57 mm, y_cross ≈ 16.5 mm.
    //   Anterior skin: d_h ≈ 88 mm, y_cross = 0.
    //   Score_posterior < Score_anterior  ↔  57 + W_y·16.5²/57 < 88
    //   ↔  W_y·4.78 < 31  ↔  W_y < 6.5; W_y = 6 is the tightest safe value.
    const APPROACH_Z_WEIGHT: f64 = 4.0;
    const APPROACH_Y_WEIGHT: f64 = 6.0;

    let [nx, ny, nz] = body_mask.shape();

    let mut best_score = f64::INFINITY;
    let mut best = focus_m;
    let mut found = false;

    for ([ix, iy, iz], active) in body_mask.indexed_iter() {
        if !active {
            continue;
        }
        // Exterior skin = body voxel adjacent to exterior air.
        let on_skin = (ix > 0 && exterior[[ix - 1, iy, iz]])
            || (ix + 1 < nx && exterior[[ix + 1, iy, iz]])
            || (iy > 0 && exterior[[ix, iy - 1, iz]])
            || (iy + 1 < ny && exterior[[ix, iy + 1, iz]])
            || (iz > 0 && exterior[[ix, iy, iz - 1]])
            || (iz + 1 < nz && exterior[[ix, iy, iz + 1]]);
        if !on_skin {
            continue;
        }
        let p = voxel_to_point(ix, iy, iz, spacing_m, center_index);
        let d = distance_3d(p, focus_m);
        if d < 1e-9 {
            continue;
        }
        // z-approach penalty (superior/inferior tilt).
        let dz = p.z_m - focus_m.z_m;
        // y-approach penalty: penalises skin contacts that are posterior of the
        // focus (for anterior organs, focus_y ≥ 0) or anterior of the focus (for
        // posterior / retroperitoneal organs, focus_y < 0). Comparing against
        // focus_y rather than y = 0 ensures the penalty fires whenever the skin
        // contact requires the beam to traverse the focus in y — i.e. when
        // skin_y < focus_y for an anterior organ. The prior formula compared
        // against y = 0, which failed when skin_y ∈ (0, focus_y) (slightly
        // anterior of centre but still posterior of the focus).
        let y_cross = if focus_m.y_m >= 0.0 {
            (focus_m.y_m - p.y_m).max(0.0)
        } else {
            (p.y_m - focus_m.y_m).max(0.0)
        };
        let score = d + APPROACH_Z_WEIGHT * dz * dz / d + APPROACH_Y_WEIGHT * y_cross * y_cross / d;
        if score < best_score {
            best_score = score;
            best = p;
            found = true;
        }
    }

    found.then_some(best).ok_or_else(|| {
        KwaversError::InvalidInput("body mask has no exterior skin boundary voxels".to_owned())
    })
}

/// Sub-sampled **exterior** skin surface of the body mask.
///
/// Uses the flood-filled exterior air mask so only true skin surface voxels
/// are included.  Internal tissue interfaces are excluded.
pub(super) fn exterior_body_surface_points(
    body_mask: &Array3<bool>,
    exterior: &Array3<bool>,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
    stride: usize,
) -> Vec<Point3> {
    let [nx, ny, nz] = body_mask.shape();
    body_mask
        .indexed_iter()
        .filter(|&([ix, iy, iz], &active)| {
            if !active || (ix + iy + iz) % stride != 0 {
                return false;
            }
            (ix > 0 && exterior[[ix - 1, iy, iz]])
                || (ix + 1 < nx && exterior[[ix + 1, iy, iz]])
                || (iy > 0 && exterior[[ix, iy - 1, iz]])
                || (iy + 1 < ny && exterior[[ix, iy + 1, iz]])
                || (iz > 0 && exterior[[ix, iy, iz - 1]])
                || (iz + 1 < nz && exterior[[ix, iy, iz + 1]])
        })
        .map(|([ix, iy, iz], _)| voxel_to_point(ix, iy, iz, spacing_m, center_index))
        .collect()
}

/// Sub-sampled exterior surface points of a binary 3-D mask using the naive
/// `is_boundary_3d` test.  Use for **organ** surfaces (internal boundaries are
/// correct there); use [`exterior_body_surface_points`] for the body skin.
pub(super) fn surface_points_3d(
    mask: &Array3<bool>,
    spacing_m: [f64; 3],
    center_index: [f64; 3],
    stride: usize,
) -> Vec<Point3> {
    mask.indexed_iter()
        .filter(|&([ix, iy, iz], &active)| {
            active && is_boundary_3d(mask, ix, iy, iz) && (ix + iy + iz) % stride == 0
        })
        .map(|([ix, iy, iz], _)| voxel_to_point(ix, iy, iz, spacing_m, center_index))
        .collect()
}
