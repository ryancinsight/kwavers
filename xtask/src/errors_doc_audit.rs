//! Audit for `# Errors` sections on functions that cannot fail.
//!
//! `clippy::missing_errors_doc` fires when a fallible function has no `# Errors`
//! section. It says nothing about whether the section is true, so a template
//! that appends one everywhere silences the lint while making the documentation
//! worse: measured 2026-09-09, 1141 copies of a single sentence across 544
//! files, 350 of them on functions whose signature returns neither `Result` nor
//! `Option`. `pub fn num_sensors(&self) -> usize` promised an `Err`.
//!
//! A reader told that a call can fail writes a match arm that can never run.
//! All 350 are gone; this audit is what keeps the template from coming back.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Sites remaining. Zero, and it stays there.
///
/// The class opened at 350 -- an earlier line-at-a-time estimate said 300
/// because it could not follow a signature wrapping across lines, which this
/// audit does. All 350 were removed in the same change that closed this to
/// zero, so any rise is a new one.
const BASELINE: usize = 0;

/// One documented function that cannot return the error it documents.
struct FalseSection {
    file: String,
    line: usize,
    signature: String,
}

/// Whether a return type can carry an error the caller handles.
fn can_fail(return_type: &str) -> bool {
    return_type.contains("Result") || return_type.contains("Option")
}

/// The return type of the signature starting at `lines[start]`, or `None` when
/// the item is not a function.
///
/// A signature may wrap across lines, so it is accumulated until the body opens
/// or a trait method's semicolon ends it. A unit return is the empty string,
/// which `can_fail` rejects.
fn return_type_of(lines: &[&str], start: usize) -> Option<String> {
    let first = lines[start].trim_start();
    let is_fn = ["", "pub ", "pub(crate) ", "pub(super) "]
        .iter()
        .any(|vis| {
            ["fn ", "const fn ", "async fn ", "unsafe fn "]
                .iter()
                .any(|kind| first.starts_with(&format!("{vis}{kind}")))
        });
    if !is_fn {
        return None;
    }

    let mut signature = String::new();
    for line in lines.iter().skip(start).take(24) {
        signature.push(' ');
        signature.push_str(line.trim());
        if line.contains('{') || line.trim_end().ends_with(';') {
            break;
        }
    }

    // Cut the body and any `where` clause before reading the arrow, so a bound
    // mentioning `Result` cannot be mistaken for the return type.
    let head = signature
        .split(" where ")
        .next()
        .unwrap_or(&signature)
        .split('{')
        .next()
        .unwrap_or(&signature)
        .to_string();

    Some(match head.rsplit_once("->") {
        Some((_, tail)) => tail.trim().trim_end_matches(';').trim().to_string(),
        None => String::new(),
    })
}

/// Every `# Errors` section in `source` sitting on an infallible function.
fn false_sections_in(source: &str, path: &str) -> Vec<FalseSection> {
    let lines: Vec<&str> = source.split('\n').collect();
    let mut found = Vec::new();

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("///") || !trimmed.contains("# Errors") {
            continue;
        }

        // Walk past the rest of the doc block and any attributes to the item.
        let mut cursor = index + 1;
        while cursor < lines.len() {
            let candidate = lines[cursor].trim_start();
            if candidate.starts_with("///") || candidate.starts_with("#[") || candidate.is_empty() {
                cursor += 1;
                continue;
            }
            break;
        }
        if cursor >= lines.len() {
            continue;
        }

        if let Some(return_type) = return_type_of(&lines, cursor) {
            if !can_fail(&return_type) {
                found.push(FalseSection {
                    file: path.to_string(),
                    line: cursor + 1,
                    signature: lines[cursor].trim().to_string(),
                });
            }
        }
    }

    found
}

/// Fail when any function documents an error it cannot return.
///
/// # Errors
///
/// Returns an error when a source file cannot be read, or when the number of
/// false sections exceeds [`BASELINE`].
pub fn audit_errors_docs(workspace_root: &Path) -> Result<()> {
    println!("Auditing `# Errors` sections against their signatures...");

    let crates_dir = workspace_root.join("crates");
    let mut found = Vec::new();

    for entry in WalkDir::new(&crates_dir)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let source = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let relative = path
            .strip_prefix(workspace_root)
            .unwrap_or(path)
            .display()
            .to_string();
        found.extend(false_sections_in(&source, &relative));
    }

    let total = found.len();
    println!("  functions documenting an error they cannot return: {total}");

    for site in found.iter().take(10) {
        println!("    {}:{}: {}", site.file, site.line, site.signature);
    }
    if total > 10 {
        println!("    ... and {} more", total - 10);
    }

    if total > BASELINE {
        anyhow::bail!(
            "{total} `# Errors` sections sit on functions that cannot fail, above the \
             baseline of {BASELINE}. An `# Errors` section names the condition that \
             produces the error, or it does not exist."
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{can_fail, false_sections_in};

    #[test]
    fn a_section_on_an_infallible_function_is_reported() {
        let source =
            "/// Count them.\n/// # Errors\n/// - Returns Err.\npub fn n(&self) -> usize {\n";
        let found = false_sections_in(source, "x.rs");
        assert_eq!(
            found.len(),
            1,
            "an infallible fn documenting Err must be reported"
        );
        assert_eq!(found[0].line, 4);
    }

    #[test]
    fn a_section_on_a_fallible_function_is_left_alone() {
        let source =
            "/// Do it.\n/// # Errors\n/// - When empty.\npub fn go(&self) -> Result<(), E> {\n";
        assert!(false_sections_in(source, "x.rs").is_empty());
    }

    #[test]
    fn a_wrapped_signature_is_read_to_its_arrow() {
        // The arrow is three lines below `fn`; a line-at-a-time reader misses it
        // and would report this as infallible.
        let source = "/// Do it.\n/// # Errors\n/// - When it fails.\npub fn go(\n    a: u8,\n    b: u8,\n) -> Result<(), E> {\n";
        assert!(false_sections_in(source, "x.rs").is_empty());
    }

    #[test]
    fn a_unit_return_counts_as_infallible() {
        let source = "/// Do it.\n/// # Errors\n/// - Never.\npub fn go(&self) {\n";
        assert_eq!(false_sections_in(source, "x.rs").len(), 1);
    }

    #[test]
    fn a_where_clause_mentioning_result_does_not_excuse_the_signature() {
        let source = "/// Do it.\n/// # Errors\n/// - Never.\npub fn go<F>(f: F) -> usize\nwhere\n    F: Fn() -> Result<(), E>,\n{\n";
        assert_eq!(
            false_sections_in(source, "x.rs").len(),
            1,
            "the bound's Result must not be read as the return type"
        );
    }

    #[test]
    fn aliases_carrying_result_in_their_name_can_fail() {
        assert!(can_fail("KwaversResult<Grid>"));
        assert!(can_fail("Option<&Bounds>"));
        assert!(!can_fail("usize"));
        assert!(!can_fail(""));
    }
}
