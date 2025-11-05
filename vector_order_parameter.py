"""Backward-compatible wrapper that forwards to the OriVec package."""

from __future__ import annotations

from orivec import (
    assign_type_symbols,
    debug_draw_directions,
    dodecahedron_vertex_directions,
    generate_ref_variants,
    get_icp_transformation,
    get_n_nearest_neighbors,
    get_order_parameters,
    get_orientated_frames,
    get_orientated_frames_parallel,
    get_positions_of_neighbors,
    read_reference,
    to_open3d_cloud,
)

__all__ = [
    "assign_type_symbols",
    "debug_draw_directions",
    "dodecahedron_vertex_directions",
    "generate_ref_variants",
    "get_icp_transformation",
    "get_n_nearest_neighbors",
    "get_order_parameters",
    "get_orientated_frames",
    "get_orientated_frames_parallel",
    "get_positions_of_neighbors",
    "read_reference",
    "to_open3d_cloud",
]


def main() -> int:
    """Invoke the OriVec CLI for compatibility with the original script."""

    from orivec.cli import main as _main

    return _main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
