#!/usr/bin/env python3
"""Fail when a `cargo test` log shows a test binary that ran no tests.

Usage: assert_tests_ran.py LOG TESTS_DIR

A suite gated as a whole file by `#![cfg(feature = ...)]` compiles to an empty
binary: cargo prints `running 0 tests`, `test result: ok. 0 passed`, and exits 0.
That is a configuration that runs nothing and reads as a pass (tenferro-sci-rs#30).
This check reads the log and requires, for every `TESTS_DIR/*.rs` integration
suite, a `Running tests/<name>.rs` section whose result reports at least one
passed test, and a non-zero total. It exits non-zero otherwise.
"""
import pathlib
import re
import sys


def main() -> int:
    log, tests_dir = sys.argv[1], pathlib.Path(sys.argv[2])
    expected = sorted(p.name for p in tests_dir.glob("*.rs"))
    if not expected:
        print(f"error: no integration suites under {tests_dir}")
        return 1

    passed = {}
    current = None
    for line in pathlib.Path(log).read_text().splitlines():
        m = re.search(r"Running (\S+) \(", line)
        if m:
            current = m.group(1)
            continue
        m = re.search(r"Doc-tests", line)
        if m:
            current = None
            continue
        m = re.match(r"test result: \w+\. (\d+) passed", line)
        if m and current:
            passed[current] = passed.get(current, 0) + int(m.group(1))

    failures = []
    for name in expected:
        key = f"tests/{name}"
        n = passed.get(key)
        if n is None:
            failures.append(f"{key}: no test run found in the log")
        elif n == 0:
            failures.append(f"{key}: ran 0 tests")
        print(f"{key}: {n if n is not None else 'missing'} passed")

    total = sum(passed.values())
    print(f"total passed across test binaries: {total}")
    if total == 0:
        failures.append("the whole run executed 0 tests")
    if failures:
        print("\nFAILED: a configuration that runs nothing is not a pass:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
