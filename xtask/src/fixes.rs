use anyhow::Result;
use regex::Regex;
use std::collections::HashSet;
use std::fs;
use walkdir::WalkDir;

pub fn apply_fixes() -> Result<()> {
    println!("🔧 Applying automated fixes...");

    // Fix 1: Add missing Debug derives
    add_missing_debug_derives()?;

    println!("✅ Automated fixes applied.");
    Ok(())
}

fn add_missing_debug_derives() -> Result<()> {
    println!("  - Adding missing Debug derives...");
    let src_root = crate::src_root();

    // Regex to match struct or enum definitions
    let struct_re = Regex::new(r"^(\s*)(pub(?:\([^)]+\))?\s+)?(struct|enum)\s+([A-Za-z0-9_]+)").unwrap();

    // Regex to detect manual Debug implementations
    let manual_impl_re = Regex::new(r"impl\s+(?:<[^>]*>\s*)?(?:std::fmt::)?Debug\s+for\s+([A-Za-z0-9_]+)").unwrap();

    let mut modified_count = 0;

    for entry in WalkDir::new(src_root).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().map_or(false, |ext| ext == "rs") {
            let path = entry.path();
            let content = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to read {}: {}", path.display(), e);
                    continue;
                }
            };

            // pre-scan for manual impls
            let mut manual_impls = HashSet::new();
            for cap in manual_impl_re.captures_iter(&content) {
                if let Some(name) = cap.get(1) {
                    manual_impls.insert(name.as_str().to_string());
                }
            }

            let new_content = process_file_content(&content, &struct_re, &manual_impls);

            if content != new_content {
                if let Err(e) = fs::write(path, new_content) {
                     eprintln!("Failed to write {}: {}", path.display(), e);
                } else {
                    modified_count += 1;
                }
            }
        }
    }

    println!("  -> Added Debug derives to {} files.", modified_count);
    Ok(())
}

fn process_file_content(content: &str, struct_re: &Regex, manual_impls: &HashSet<String>) -> String {
    let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let mut i = 0;

    while i < lines.len() {
        if let Some(caps) = struct_re.captures(&lines[i]) {
            let indent = caps.get(1).map_or("", |m| m.as_str());
            let name = caps.get(4).map_or("", |m| m.as_str());

            // Skip if manual implementation exists
            if manual_impls.contains(name) {
                i += 1;
                continue;
            }

            // Heuristic: Check for fields that might not implement Debug
            // We need to scan forward to check the struct body
            // This is simple line-scanning, prone to errors if braces are on same line etc.
            if has_problematic_fields(&lines, i) {
                i += 1;
                continue;
            }

            // Check preceding lines for attributes
            let mut j = i;
            let mut derive_line_idx = None;

            // Scan backwards
            while j > 0 {
                j -= 1;
                let line = lines[j].trim();

                if line.is_empty() || line.starts_with("//") {
                    continue;
                }

                if line.starts_with("#[derive(") {
                    derive_line_idx = Some(j);
                    break;
                }

                if line.starts_with("#[") {
                    continue;
                }

                if line.starts_with("///") || line.starts_with("/*") || line.starts_with("*") || line.starts_with("*/") {
                     continue;
                }

                break;
            }

            if let Some(idx) = derive_line_idx {
                if !lines[idx].contains("Debug") {
                    if let Some(close_pos) = lines[idx].rfind(')') {
                         lines[idx].insert_str(close_pos, ", Debug");
                    }
                }
            } else {
                let new_line = format!("{}#[derive(Debug)]", indent);
                lines.insert(i, new_line);
                i += 1;
            }
        }
        i += 1;
    }

    let mut result = lines.join("\n");
    if content.ends_with('\n') {
        result.push('\n');
    }
    result
}

fn has_problematic_fields(lines: &[String], start_idx: usize) -> bool {
    // Determine end of struct/enum
    let mut depth = 0;
    let mut started = false;

    // Scan lines starting from start_idx
    for line in lines.iter().skip(start_idx) {
        // Simple heuristic for problematic types
        // Note: This matches anywhere in the line, including comments if we aren't careful.
        // But for a heuristic it's okay.

        if line.contains("Box<dyn")
            || line.contains("Arc<dyn")
            || line.contains("Rc<dyn")
            || line.contains("&dyn")
            || line.contains("Fn(")
            || line.contains("FnMut(")
            || line.contains("FnOnce(")
            || line.contains("FftPlanner") // Specific to this codebase
        {
            return true;
        }

        for c in line.chars() {
            match c {
                '{' => {
                    depth += 1;
                    started = true;
                }
                '}' => {
                    depth -= 1;
                }
                _ => {}
            }
        }

        if started && depth == 0 {
            break;
        }

        // Safety break for long structs or parse failures
        if depth > 100 {
            break;
        }
    }

    // Also check if it's a tuple struct on one line: `struct Foo(Box<dyn Bar>);`
    let first_line = &lines[start_idx];
    if !first_line.contains('{') && first_line.contains(';') {
         if first_line.contains("Box<dyn")
            || first_line.contains("Arc<dyn")
            || first_line.contains("Rc<dyn")
            || first_line.contains("&dyn")
            || first_line.contains("Fn(")
            || first_line.contains("FnMut(")
            || first_line.contains("FnOnce(")
            || first_line.contains("FftPlanner")
        {
            return true;
        }
    }

    false
}
