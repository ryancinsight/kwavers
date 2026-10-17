#!/usr/bin/env python3
"""Transport and execute the complete Cargo-selected benchmark program set."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


def identity() -> dict:
    return {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'lock': hashlib.sha256(Path('Cargo.lock').read_bytes()).hexdigest(),
            'compiler': subprocess.check_output(['rustc', '-vV'], text=True),
            'profile': 'dev', 'features': ['default', 'plotting']}


def pack(artifacts: Path, output: Path) -> None:
    programs = {}
    for line in artifacts.read_text().splitlines():
        item = json.loads(line)
        if item.get('reason') != 'compiler-artifact' or not item.get('executable'):
            continue
        kinds = item['target']['kind']
        if 'bench' not in kinds and not item.get('profile', {}).get('test'):
            continue
        path = Path(item['executable']).resolve()
        programs[path.name] = path
    if not programs:
        raise SystemExit('Cargo produced no benchmark programs')
    # Record every program emitted by the complete --benches selection, not a
    # handwritten subset. Required-feature filtering remains Cargo's contract.
    info = dict(identity(), programs=list(programs))
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / 'programs.json'
        manifest.write_text(json.dumps(info))
        with tarfile.open(output, 'w:gz', compresslevel=1) as archive:
            archive.add(manifest, arcname='programs.json')
            for name, path in programs.items():
                archive.add(path, arcname=f'programs/{name}')
    print('Complete selected programs:', ', '.join(programs), flush=True)


def run(archive: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with tarfile.open(archive) as bundle:
            bundle.extractall(root, filter='data')
        info = json.loads((root / 'programs.json').read_text())
        programs = info.pop('programs')
        if not programs or len(programs) != len(set(programs)) or info != identity():
            raise SystemExit('benchmark artifact identity/inventory does not match')
        failures = []
        for name in programs:
            executable = root / 'programs' / name
            print('Executing exact compiled target:', name, flush=True)
            result = subprocess.run([str(executable), '--test'])
            if result.returncode:
                failures.append((name, result.returncode))
        if failures:
            raise SystemExit(f'benchmark programs failed: {failures}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    pack_args = sub.add_parser('pack')
    pack_args.add_argument('--artifacts', type=Path, required=True)
    pack_args.add_argument('--output', type=Path, required=True)
    run_args = sub.add_parser('run')
    run_args.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'pack':
        pack(args.artifacts, args.output)
    else:
        run(args.archive)


if __name__ == '__main__':
    main()
