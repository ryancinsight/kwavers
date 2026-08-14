//! Filesystem path predicates shared across the domain crates.
//!
//! File extensions reach kwavers from clinical archives, scanner exports, and operator-typed
//! configuration, where `.NII`, `.Dcm`, and `.NPZ` are as common as their lowercase spellings.
//! Extension dispatch therefore compares case-insensitively; a case-sensitive `ends_with`
//! silently routes a valid file to the "unsupported format" arm.

/// Returns `true` when `path` ends with any of `suffixes`, compared ASCII case-insensitively.
///
/// Matches on the raw tail rather than [`std::path::Path::extension`] so multi-part extensions
/// (`.nii.gz`) are expressible as a single suffix.
///
/// # Examples
///
/// ```
/// use kwavers_core::path::has_suffix_ignore_ascii_case;
///
/// assert!(has_suffix_ignore_ascii_case("scan.NII.GZ", &[".nii", ".nii.gz"]));
/// assert!(has_suffix_ignore_ascii_case("series.Dcm", &[".dcm", ".dicom"]));
/// assert!(!has_suffix_ignore_ascii_case("volume.raw", &[".nii", ".nii.gz"]));
/// ```
#[must_use]
pub fn has_suffix_ignore_ascii_case(path: &str, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|suffix| {
        path.len()
            .checked_sub(suffix.len())
            .and_then(|start| path.get(start..))
            .is_some_and(|tail| tail.eq_ignore_ascii_case(suffix))
    })
}

#[cfg(test)]
mod tests {
    use super::has_suffix_ignore_ascii_case;

    #[test]
    fn matches_regardless_of_case() {
        for spelling in ["scan.nii", "scan.NII", "scan.NiI"] {
            assert!(
                has_suffix_ignore_ascii_case(spelling, &[".nii"]),
                "{spelling} must match .nii"
            );
        }
    }

    #[test]
    fn matches_multi_part_extension() {
        assert!(has_suffix_ignore_ascii_case("scan.NII.GZ", &[".nii.gz"]));
        assert!(!has_suffix_ignore_ascii_case("scan.gz", &[".nii.gz"]));
    }

    #[test]
    fn rejects_unlisted_and_bare_suffix() {
        assert!(!has_suffix_ignore_ascii_case("scan.raw", &[".nii", ".dcm"]));
        // The suffix itself is not a filename with that extension, but the tail does match;
        // callers validate non-emptiness separately.
        assert!(!has_suffix_ignore_ascii_case("nii", &[".nii"]));
    }

    #[test]
    fn empty_suffix_list_never_matches() {
        assert!(!has_suffix_ignore_ascii_case("scan.nii", &[]));
    }

    #[test]
    fn suffix_longer_than_path_does_not_panic() {
        assert!(!has_suffix_ignore_ascii_case("a", &[".nii.gz"]));
    }

    #[test]
    fn multibyte_path_does_not_panic_on_boundary() {
        // The byte offset `len - suffix.len()` lands mid-character; slicing must not panic.
        assert!(!has_suffix_ignore_ascii_case("scan_αβγ", &[".gz"]));
        assert!(has_suffix_ignore_ascii_case("scan_αβγ.NII", &[".nii"]));
    }
}
