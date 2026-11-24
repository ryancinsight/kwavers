"""Run the independent Rust consumer on the registry baseline or a local release archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parent
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
BASELINE = "bde4385ea3a7d18ddecbcbd0e00f7075d3958c3ab437a489768b62de6b5f9023"
ROWS = {
    "neither": (["--no-default-features"], []),
    "default": ([], ["default", "io"]),
    "transducer": (["--no-default-features", "--features", "kwavers"], ["kwavers"]),
    "both": (["--all-features"], ["default", "io", "kwavers"]),
}


def run(args, cwd, *, check=True):
    cmd = ["cargo", "+1.97.0", *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600, check=check)


def replay(options, stage):
    work = stage / "consumer"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns("target", "__pycache__", ".git"))
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest = BASELINE
    if options.archive:
        archive = options.archive.resolve()
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        with tarfile.open(archive, "r:gz") as contents:
            contents.extractall(stage / "archive", filter="data")
        crate, = (stage / "archive").glob("*/Cargo.toml")
        version = tomllib.loads(crate.read_text())["package"]["version"]
        manifest = work / "Cargo.toml"
        manifest.write_text(manifest.read_text().replace('version = "=0.5.0"',
            f'version = "={version}", path = {json.dumps(str(crate.parent))}'))
        # Keep other versions from the registry consumer lock while changing Driver.
        run(["metadata", "--format-version", "1"], work)
    else:
        version = "0.5.0"

    lock = tomllib.loads((work / "Cargo.lock").read_text())
    providers = {p["name"]: p for p in lock["package"] if p["name"] in
                 ("kwavers-driver", "kwavers-transducer", "aequitas", "eunomia")}
    assert providers["kwavers-driver"]["version"] == version
    if not options.archive:
        assert providers["kwavers-driver"]["checksum"] == BASELINE
    record = {"driver_version": version, "archive_sha256": digest,
              "origin": "unpacked local archive" if options.archive else "crates.io",
              "locked_providers": providers, "rows": {}}
    all_pass = True
    for name, (flags, expected) in ROWS.items():
        metadata = json.loads(run(["metadata", "--format-version", "1", "--locked", *flags], work).stdout)
        packages = {p["id"]: p for p in metadata["packages"]}
        nodes = metadata["resolve"]["nodes"]
        driver, = (n for n in nodes if packages[n["id"]]["name"] == "kwavers-driver")
        assert sorted(driver["features"]) == expected
        local = sorted(p["name"] for p in packages.values() if p["source"] != CRATES_IO)
        assert local == (["driver-qualification","kwavers-driver"] if options.archive else ["driver-qualification"])
        deps = sorted(packages[edge["pkg"]]["name"] for edge in driver["deps"])
        assert ("kwavers-transducer" in deps) == ("kwavers" in expected)
        assert any(packages[n["id"]]["name"] == "kwavers-transducer" for n in nodes) == ("kwavers" in expected)
        cmd = ["nextest", "run", "--locked", "--no-fail-fast", "--success-output", "immediate",
               "--hide-progress-bar", *flags]
        result = run(cmd, work, check=False)
        log = result.stdout + result.stderr
        (output / f"{name}.log").write_text(log)
        summary = next((line.strip() for line in reversed(log.splitlines()) if "tests run:" in line), "")
        record["rows"][name] = {"features": expected, "direct_dependencies": deps,
            "command": ["cargo", "+1.97.0", *cmd], "exit_code": result.returncode, "summary": summary}
        print(f"{name}: {summary or log[-600:]}", flush=True)
        all_pass &= result.returncode == 0
    if options.bench:
        result = run(["bench", "--locked", "--bench", "representative"], work)
        record["benchmark"] = result.stdout.strip()
        print(record["benchmark"])
    (output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Optional unpublished .crate to rehearse")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="driver-qualification-") as directory:
        stage = Path(directory)
        os.environ.setdefault("CARGO_TARGET_DIR", str(stage / "target"))
        raise SystemExit(0 if replay(args, stage) else 1)
