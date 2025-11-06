from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from ase.io import write

from .core import get_order_parameters
from .ref_gen import generate_ref_motifs

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


def build_ref_generator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orivec gen-ref",
        description="Generate reference motifs from symmetry-unique atoms.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Structure file readable by ASE (e.g. CIF, POSCAR).",
    )
    parser.add_argument(
        "--rcut",
        required=True,
        type=float,
        help="Radial cutoff in Angstrom for including neighbors (required).",
    )
    parser.add_argument(
        "--output-prefix",
        dest="out_base_name",
        default="ref_motif",
        help="Prefix applied to each generated motif file (default: ref_motif).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where motif files are written (default: current directory).",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=1e-5,
        help="Symmetry tolerance passed to spglib (default: 1e-5).",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=5.0,
        help="Angular tolerance in degrees passed to spglib (default: 5.0).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if argv_list and argv_list[0] in {"gen-ref", "ref-gen"}:
        ref_parser = build_ref_generator_parser()
        ref_args = ref_parser.parse_args(argv_list[1:])
        motifs = generate_ref_motifs(
            base_file=str(ref_args.input),
            r_cut=ref_args.rcut,
            out_base_name=ref_args.out_base_name,
            symprec=ref_args.symprec,
            angle_tolerance=ref_args.angle_tolerance,
            output_dir=ref_args.output_dir,
        )
        output_dir = ref_args.output_dir.resolve() if ref_args.output_dir else Path.cwd()
        print(
            "Generated "
            f"{len(motifs)} motifs from {ref_args.input} using r_cut={ref_args.rcut} Angstrom. "
            f"Files saved to {output_dir} with prefix {ref_args.out_base_name}_*.xyz"
        )
        return 0

    parser = build_parser()
    args = parser.parse_args(argv_list)

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
