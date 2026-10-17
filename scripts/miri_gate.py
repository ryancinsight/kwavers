#!/usr/bin/env python3
"""Run a real Miri selection; reject failures and successful empty harnesses."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SUMMARY = re.compile(r"test result: ok\. (\d+) passed; (\d+) failed;")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--filter")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--prepare-matrix", type=Path)
    args = parser.parse_args()
    command = ["cargo", "+nightly", "miri", "test", "--locked", "-p", args.package, "--lib"]
    if args.filter:
        command.append(args.filter)
    if args.prepare_matrix:
        command += ["--", "--list"]
        listed = subprocess.check_output(command, text=True)
        print(listed, flush=True)
        tests = [line[:-6] for line in ANSI.sub("", listed).splitlines() if line.endswith(": test")]
        declared = re.search(r"^(\d+) tests?, \d+ benchmarks?", listed, re.MULTILINE)
        if not tests or len(tests) != len(set(tests)) or not declared or len(tests) != int(declared[1]):
            raise SystemExit("Miri test inventory must be nonempty and unique")
        args.prepare_matrix.write_text(json.dumps({"case": tests}))
        return 0
    if args.exact:
        command += ["--", "--exact"]
    print("Miri selection:", " ".join(command), flush=True)
    summaries = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            if match := SUMMARY.search(ANSI.sub("", line)):
                summaries.append((int(match[1]), int(match[2])))
        result = process.wait()
    if result:
        return result
    if not summaries or any(passed == 0 or failed != 0 for passed, failed in summaries):
        print("error: Miri must execute a nonempty passing test selection", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
