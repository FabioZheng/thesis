#!/usr/bin/env python3
"""Command-line helper for trimming JSON exports.

This script wraps :func:`save_json.trim_json_file` so datasets cropped to a
fixed number of documents can be generated without invoking the full export
pipeline again.
"""

from __future__ import annotations

import argparse

from save_json import trim_json_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim a JSON document to a fixed number of entries."
    )
    parser.add_argument(
        "input",
        help="Path to the source JSON file to trim.",
    )
    parser.add_argument(
        "output",
        help="Destination path for the trimmed JSON file.",
    )
    parser.add_argument(
        "limit",
        type=int,
        help="Maximum number of records to keep from the input JSON document.",
    )
    parser.add_argument(
        "--no-ensure-directory",
        action="store_true",
        help=(
            "Disable automatic creation of the parent directory for the output "
            "file."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_directory = not args.no_ensure_directory

    print(
        "✂️  Trimming {input} to {limit} entries -> {output}".format(
            input=args.input, limit=args.limit, output=args.output
        )
    )
    path, stats = trim_json_file(
        args.input,
        args.output,
        args.limit,
        ensure_directory=ensure_directory,
    )
    print(
        "✅ Saved trimmed JSON to {path} "
        "(approx memory: {mem:.2f} MB, pickle: {pickle:.2f} MB, json: {json:.2f} MB)".format(
            path=path,
            mem=stats.get("approx_memory_mb", 0.0),
            pickle=stats.get("pickle_disk_mb", 0.0),
            json=stats.get("json_disk_mb", 0.0),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
