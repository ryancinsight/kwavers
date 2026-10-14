#!/usr/bin/env python3
"""Prepare foundation libraries with the actual facade build's feature sets."""
import argparse
import json
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--profile', choices=['dev', 'test'], required=True)
    parser.add_argument('--features', required=True)
    args = parser.parse_args()
    query = ['cargo', 'build', '--locked', '--manifest-path', args.manifest,
             '-p', 'kwavers', '--profile', args.profile, '--no-default-features',
             '--features', args.features, '-Z', 'unstable-options', '--unit-graph']
    # Bootstrap applies only to Cargo's read-only graph query. Actual build
    # commands use the pinned compiler with its ordinary stable configuration.
    graph = json.loads(subprocess.check_output(query, env=dict(os.environ, RUSTC_BOOTSTRAP='1')))
    selected = ['kwavers-core', 'kwavers-math', 'kwavers-grid', 'kwavers-medium']
    feature_map = {}
    for unit in graph['units']:
        if unit['target']['kind'] == ['lib'] and unit['target']['name'].replace('_', '-') in selected:
            feature_map[unit['target']['name'].replace('_', '-')] = unit['features']
    if set(feature_map) != set(selected):
        raise SystemExit('foundation graph does not contain every required package')
    command = ['cargo', 'build', '--locked', '--manifest-path', args.manifest,
               '--profile', args.profile, '--no-default-features']
    for package in selected:
        command += ['-p', package]
    features = [f'{package}/{feature}' for package in selected for feature in feature_map[package]]
    if features:
        command += ['--features', ','.join(features)]
    print('Matching foundation build:', command, flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
