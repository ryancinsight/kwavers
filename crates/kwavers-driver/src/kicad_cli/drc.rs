//! The DRC result model ([`DrcReport`], [`DrcDefectCount`]) and the permissive, version-tolerant
//! parser for kicad-cli's DRC JSON. kicad-cli's layout has shifted across major versions
//! (KiCad 7/8/9/10), so the parser hand-scans the field names it cares about instead of pinning to a
//! serde schema — it survives future format changes and degrades to zero counts rather than failing.

/// Result of an external Kicad DRC run. The parsed counts are best-effort; the raw JSON is kept so a
/// caller can drill into specific violations from the example log.
#[derive(Debug, Clone, Default)]
pub struct DrcReport {
    /// Number of `violations[]` (hard errors) in the DRC report.
    pub violations: usize,
    /// Number of `unconnected_items[]` (open pins / dangling copper).
    pub unconnected_items: usize,
    /// Number of `warnings[]` (best-practice hits, not hard rejects).
    pub warnings: usize,
    /// Per-KiCad defect-type counts from `type` fields in `violations[]` and `unconnected_items[]`.
    pub defect_counts: Vec<DrcDefectCount>,
    /// Full DRC JSON, preserved for diagnostic pretty-printing.
    pub raw_json: String,
}

/// Count of one KiCad DRC defect `type`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrcDefectCount {
    /// The string identifier or category of the DRC defect type.
    pub kind: String,
    /// The number of occurrences of this defect type.
    pub count: usize,
}

impl DrcReport {
    /// True iff every hard-check counter is zero. So called by the example gate.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.violations == 0 && self.unconnected_items == 0
    }
}

/// Permissive parser for the KiCad-DRC JSON. kicad-cli's JSON layout has shifted a few times between
/// major versions (KiCad 7 vs 8 vs 9 vs 10); we hand-scan the field names we care about instead of
/// pinning to a serde-derived schema, so the parser survives future format changes.
pub(super) fn parse_drc_json(text: &str) -> DrcReport {
    fn array_body_after<'a>(text: &'a str, needles: &[&str]) -> Option<&'a str> {
        for n in needles {
            let Some(idx) = text.find(n) else {
                continue;
            };
            let after = text[idx + n.len()..].trim_start();
            let Some(array) = after.strip_prefix('[') else {
                continue;
            };
            let mut depth = 1;
            let mut in_string = false;
            let mut escaped = false;
            for (i, c) in array.char_indices() {
                if escaped {
                    escaped = false;
                    continue;
                }
                if c == '\\' {
                    escaped = true;
                    continue;
                }
                if c == '"' {
                    in_string = !in_string;
                    continue;
                }
                if !in_string {
                    if c == '[' {
                        depth += 1;
                    } else if c == ']' {
                        depth -= 1;
                        if depth == 0 {
                            return Some(&array[..i]);
                        }
                    }
                }
            }
        }
        None
    }

    fn count_top_level_objects_matching(body: &str, needle: &str) -> usize {
        let mut count = 0;
        let mut depth = 0;
        let mut in_string = false;
        let mut escaped = false;
        let mut start = None;
        for (i, c) in body.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            if c == '\\' {
                escaped = true;
                continue;
            }
            if c == '"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            if c == '{' {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start.take() {
                        if body[s..=i].contains(needle) {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    fn count_after(text: &str, needles: &[&str]) -> usize {
        for n in needles {
            if let Some(idx) = text.find(n) {
                let after = text[idx + n.len()..].trim_start();
                if after.starts_with('[') {
                    let Some(body) = array_body_after(text, &[*n]) else {
                        return 0;
                    };
                    let mut count = 0;
                    let mut obj_depth = 0;
                    let mut body_in_string = false;
                    let mut body_escaped = false;
                    let mut has_elements = false;
                    for c in body.chars() {
                        if body_escaped {
                            body_escaped = false;
                            continue;
                        }
                        if c == '\\' {
                            body_escaped = true;
                            continue;
                        }
                        if c == '"' {
                            body_in_string = !body_in_string;
                            continue;
                        }
                        if !body_in_string {
                            if c == '{' {
                                if obj_depth == 0 {
                                    count += 1;
                                    has_elements = true;
                                }
                                obj_depth += 1;
                            } else if c == '}' {
                                obj_depth -= 1;
                            } else if c == '[' {
                                obj_depth += 1;
                            } else if c == ']' {
                                obj_depth -= 1;
                            }
                        }
                    }
                    if has_elements {
                        return count;
                    }
                    let trimmed = body.trim();
                    if trimmed.is_empty() {
                        return 0;
                    }
                    let mut comma_count = 0;
                    let mut comma_depth = 0;
                    let mut comma_in_string = false;
                    let mut comma_escaped = false;
                    for c in trimmed.chars() {
                        if comma_escaped {
                            comma_escaped = false;
                            continue;
                        }
                        if c == '\\' {
                            comma_escaped = true;
                            continue;
                        }
                        if c == '"' {
                            comma_in_string = !comma_in_string;
                            continue;
                        }
                        if !comma_in_string {
                            if c == '[' || c == '{' {
                                comma_depth += 1;
                            } else if c == ']' || c == '}' {
                                comma_depth -= 1;
                            } else if c == ',' && comma_depth == 0 {
                                comma_count += 1;
                            }
                        }
                    }
                    return comma_count + 1;
                }
                let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(v) = digits.parse::<usize>() {
                    return v;
                }
            }
        }
        0
    }
    fn type_counts(text: &str) -> Vec<DrcDefectCount> {
        let mut counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let needle = "\"type\"";
        let mut rest = text;
        while let Some(idx) = rest.find(needle) {
            rest = &rest[idx + needle.len()..];
            let Some(colon) = rest.find(':') else {
                break;
            };
            rest = rest[colon + 1..].trim_start();
            let Some(after_quote) = rest.strip_prefix('"') else {
                continue;
            };
            let mut escaped = false;
            let mut end = None;
            for (i, c) in after_quote.char_indices() {
                if escaped {
                    escaped = false;
                    continue;
                }
                if c == '\\' {
                    escaped = true;
                    continue;
                }
                if c == '"' {
                    end = Some(i);
                    break;
                }
            }
            let Some(end) = end else {
                break;
            };
            let kind = after_quote[..end].to_string();
            *counts.entry(kind).or_default() += 1;
            rest = &after_quote[end + 1..];
        }
        counts
            .into_iter()
            .map(|(kind, count)| DrcDefectCount { kind, count })
            .collect()
    }

    let violation_needles = ["\"violations\":", "\"violation_count\":", "violations\":"];
    let raw_violations = count_after(text, &violation_needles);
    let severity_errors = array_body_after(text, &violation_needles)
        .map(|body| count_top_level_objects_matching(body, "\"severity\":\"error\""))
        .unwrap_or(0)
        + array_body_after(text, &violation_needles)
            .map(|body| count_top_level_objects_matching(body, "\"severity\": \"error\""))
            .unwrap_or(0);
    let severity_warnings = array_body_after(text, &violation_needles)
        .map(|body| count_top_level_objects_matching(body, "\"severity\":\"warning\""))
        .unwrap_or(0)
        + array_body_after(text, &violation_needles)
            .map(|body| count_top_level_objects_matching(body, "\"severity\": \"warning\""))
            .unwrap_or(0);
    let severity_classified = severity_errors + severity_warnings > 0;

    DrcReport {
        violations: if severity_classified {
            severity_errors
        } else {
            raw_violations
        },
        unconnected_items: count_after(
            text,
            &[
                "\"unconnected_items\":",
                "\"unconnected\":",
                "unconnected_items\":",
            ],
        ),
        warnings: count_after(
            text,
            &["\"warnings\":", "\"warning_count\":", "warnings\":"],
        ) + severity_warnings,
        defect_counts: type_counts(text),
        raw_json: text.to_string(),
    }
}
