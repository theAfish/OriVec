from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from ase.io import write

from .core import get_order_parameters

__all__ = ["main"]


def _parse_vector(values: Sequence[str] | str, expected_size: int = 3) -> np.ndarray:
    """Parse a vector from comma and/or space separated floats."""

    if isinstance(values, str):
        token_string = values.replace(",", " ")
        tokens = token_string.split()
    else:
        tokens = list(values)
    if len(tokens) != expected_size:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_size} components, received {len(tokens)}: {tokens}"
        )
    try:
        vector = np.array([float(token) for token in tokens], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Vector components must be floats: {tokens}") from exc
    return vector


def _parse_type_symbol_map(items: Iterable[str] | None) -> Mapping[int | str, str] | None:
    """Parse mappings of the form ``1=Li`` into a dictionary."""

    if not items:
        return None
    mapping: dict[int | str, str] = {}
    for raw_item in items:
        # Support comma and semicolon delimited entries to ease bulk specifications.
        expanded_items = [part.strip() for part in raw_item.replace(";", ",").split(",") if part.strip()]
        for item in expanded_items:
            if "=" not in item:
                raise argparse.ArgumentTypeError(f"Mapping must contain '=': {item}")
            key_str, value = item.split("=", 1)
            key_str = key_str.strip()
            value = value.strip()
            if not value:
                raise argparse.ArgumentTypeError(f"Element symbol missing in mapping '{item}'")
            try:
                key = int(key_str)
            except ValueError:
                key = key_str
            mapping[key] = value
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orivec",
        description="Compute orientation vectors for local motifs in atomistic structures.",
    )
    parser.add_argument("input", type=Path, help="Structure file readable by ASE (e.g. LAMMPS data)")
    parser.add_argument(
        "motif",
        type=Path,
        help="Reference motif file in minimal XYZ format used to define neighbourhoods.",
    )
    parser.add_argument(
        "--ref-orientation",
        metavar="X,Y,Z",
        type=_parse_vector,
        default=np.array([1.0, 0.0, 0.0]),
        help="Reference orientation vector associated with the motif (default: 1 0 0).",
    )
    parser.add_argument(
        "--element-map",
        metavar="TYPE=SYMBOL",
        action="append",
        help=(
            "LAMMPS numeric type to chemical symbol mapping. Repeat or provide comma-separated entries "
            "for multiple values (e.g. 1=Li,2=Mo)."
        ),
    )
    parser.add_argument(
        "--selected-element",
        dest="selected_elements",
        action="append",
        help="Restrict orientation calculation to specific chemical symbols (repeatable).",
    )
    parser.add_argument(
        "--regularize",
        action="store_true",
        help="Force orientations into a consistent hemisphere using the reference orientation.",
    )
    parser.add_argument(
        "--regularize-anchor",
        metavar="X,Y,Z",
        type=_parse_vector,
        help="Optional vector that defines the preferred hemisphere for regularisation.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process atoms in parallel using a thread pool.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of worker threads when --parallel is supplied.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to write the structure with orientation metadata (e.g. output.xyz).",
    )
    parser.add_argument(
        "--output-format",
        default="extxyz",
        help="ASE format string used when writing --output (default: extxyz).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    element_map = _parse_type_symbol_map(args.element_map)
    regularize_anchor = args.regularize_anchor if args.regularize_anchor is not None else args.ref_orientation

    structure = get_order_parameters(
        file_path=str(args.input),
        ref_motif_path=str(args.motif),
        ref_orientation=args.ref_orientation,
        elements=element_map,
        selected_elements=args.selected_elements,
        regularize_orientations=args.regularize,
        regularize_anchor=regularize_anchor,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )

    if args.output:
        write(str(args.output), structure, format=args.output_format)
        print(f"Written structure with orientations to {args.output}")
    else:
        sys.stdout.write(
            "Orientation vectors stored in structure.arrays['orientation']; provide --output to save.\n"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
