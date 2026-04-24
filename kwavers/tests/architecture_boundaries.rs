use std::fs;
use std::path::{Path, PathBuf};

fn rust_files_under(root: &Path) -> Vec<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();

    while let Some(path) = stack.pop() {
        for entry in fs::read_dir(&path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }

    files.sort();
    files
}

#[test]
fn domain_layer_has_no_solver_or_simulation_imports() {
    let domain_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/domain");
    let mut violations = Vec::new();

    for file in rust_files_under(&domain_root) {
        let source = fs::read_to_string(&file).unwrap();
        for forbidden in ["crate::solver::", "crate::simulation::"] {
            if source.contains(forbidden) {
                violations.push(format!(
                    "{} contains {}",
                    file.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                        .unwrap()
                        .display(),
                    forbidden
                ));
            }
        }
    }

    assert_eq!(violations, Vec::<String>::new());
}

#[test]
fn solver_factory_policy_has_no_domain_or_simulation_imports() {
    let source =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/solver/factory.rs"))
            .unwrap();
    let violations = ["crate::domain::", "crate::simulation::"]
        .into_iter()
        .filter(|forbidden| source.contains(forbidden))
        .collect::<Vec<_>>();

    assert_eq!(violations, Vec::<&str>::new());
}
