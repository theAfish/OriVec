from __future__ import annotations

"""Core library for OriVec orientation order parameter analysis."""

import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Mapping, Sequence

import numpy as np
import open3d as o3d
from ase import Atoms
from ase.geometry import get_distances
from ase.io import read

__all__ = [
    "read_reference",
    "assign_type_symbols",
    "to_open3d_cloud",
    "dodecahedron_vertex_directions",
    "generate_ref_variants",
    "get_icp_transformation",
    "get_n_nearest_neighbors",
    "get_positions_of_neighbors",
    "get_orientated_frames",
    "get_orientated_frames_parallel",
    "get_order_parameters",
    "debug_draw_directions",
]


def read_reference(filename: str) -> np.ndarray:
    """Read a minimal XYZ-like motif where the first integer is atom count."""

    with open(filename, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    natoms = int(lines[0].strip())
    positions = np.array(
        [list(map(float, line.split())) for line in lines[2 : 2 + natoms]],
        dtype=float,
    )
    return positions


def assign_type_symbols(
    atoms: Atoms,
    type_symbol_map: Sequence[str] | Mapping[int | str, str],
) -> Atoms:
    """Replace chemical symbols using a LAMMPS-style type-to-element mapping."""

    if "type" not in atoms.arrays:
        raise ValueError("Atoms object does not provide per-atom 'type' data; cannot remap symbols")

    type_ids = np.asarray(atoms.arrays["type"], dtype=int)
    if isinstance(type_symbol_map, Mapping):
        mapping = {int(key): value for key, value in type_symbol_map.items()}
    elif isinstance(type_symbol_map, Sequence) and not isinstance(type_symbol_map, (str, bytes)):
        mapping = {index + 1: symbol for index, symbol in enumerate(type_symbol_map)}
    else:
        raise TypeError("type_symbol_map must be a mapping or a non-string sequence of element symbols")

    missing_types = sorted(set(type_ids.tolist()) - set(mapping.keys()))
    if missing_types:
        raise KeyError(f"No element mapping provided for type identifiers: {missing_types}")

    new_symbols = [mapping[int(type_id)] for type_id in type_ids]
    atoms.set_chemical_symbols(new_symbols)
    atoms.info["type_symbol_map"] = mapping
    return atoms


def to_open3d_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Build an Open3D point cloud from a numpy array."""

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def dodecahedron_vertex_directions() -> list[np.ndarray]:
    """Return unit vectors pointing from the origin to the vertices of a dodecahedron."""

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    vertices_set: set[tuple[float, float, float]] = set()
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        vertices_set.add(signs)

    for sx, sy in itertools.product((-1.0, 1.0), repeat=2):
        vertices_set.add((0.0, sx * inv_phi, sy * phi))
        vertices_set.add((sx * inv_phi, sy * phi, 0.0))
        vertices_set.add((sx * phi, 0.0, sy * inv_phi))

    vertices = np.array(sorted(vertices_set), dtype=float)
    if vertices.shape != (20, 3):
        raise RuntimeError("Failed to construct dodecahedron vertices")

    directions: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()
    for vertex in vertices:
        length = np.linalg.norm(vertex)
        if length < 1e-6:
            continue
        direction = vertex / length
        key = tuple(np.round(direction, 6))
        if key in seen:
            continue
        seen.add(key)
        directions.append(direction)

    if len(directions) != 20:
        raise RuntimeError(f"Expected 20 vertex directions, got {len(directions)}")

    return directions


def _normalise_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("Zero-length vector cannot be normalised")
    return vector / norm


def _regularise_orientation(
    vector: np.ndarray,
    anchor: np.ndarray | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """Return a canonical representative for direction vectors differing by sign."""

    direction = np.asarray(vector, dtype=float)
    if not np.any(direction):
        return direction

    if anchor is not None and np.any(anchor):
        anchor_vec = _normalise_vector(np.asarray(anchor, dtype=float))
        dot = float(np.dot(direction, anchor_vec))
        if abs(dot) > tol:
            return direction if dot >= 0.0 else -direction

    major = int(np.argmax(np.abs(direction)))
    if abs(direction[major]) <= tol:
        return direction
    return direction if direction[major] >= 0.0 else -direction


def _cluster_directions(
    vectors: Iterable[np.ndarray] | np.ndarray,
    angle_tol: float = np.deg2rad(45.0),
    treat_opposites_as_equivalent: bool = True,
) -> list[np.ndarray]:
    """Reduce a set of direction vectors to unique orientations up to a tolerance."""

    unique: list[np.ndarray] = []
    for vector in vectors:
        vector = np.asarray(vector, dtype=float)
        try:
            direction = _normalise_vector(vector)
        except ValueError:
            continue
        is_duplicate = False
        for existing in unique:
            cos_angle = float(np.clip(np.dot(direction, existing), -1.0, 1.0))
            angle = abs(np.arccos(cos_angle))
            if angle < angle_tol or (
                treat_opposites_as_equivalent and abs(angle - np.pi) < angle_tol
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(direction)
    return unique


def _ensure_atom_array(atoms: Atoms, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Create a per-atom array when missing and validate its shape."""

    if name not in atoms.arrays:
        atoms.new_array(name, np.zeros(shape, dtype=float))
    array = atoms.arrays[name]
    if array.shape != shape:
        raise ValueError(f"{name} array must have shape {shape}")
    return array


def debug_draw_directions(
    directions: list[np.ndarray],
    scale: float = 5.0,
    angle_tol: float = np.deg2rad(45.0),
    treat_opposites_as_equivalent: bool = True,
) -> None:
    """Visualize the clustered directions as arrows in 3D space."""

    dirs = _cluster_directions(
        directions,
        angle_tol=angle_tol,
        treat_opposites_as_equivalent=treat_opposites_as_equivalent,
    )
    if not dirs:
        print("No directions available for visualisation")
        return

    unit_vectors = np.vstack(dirs)
    endpoints = unit_vectors * float(scale)

    cloud = to_open3d_cloud(endpoints)
    cloud.paint_uniform_color([0.85, 0.2, 0.2])

    origin = np.zeros((1, 3), dtype=float)
    line_points = np.vstack((origin, endpoints))
    lines = [[0, idx] for idx in range(1, len(line_points))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0.2, 0.6, 0.9]] * len(lines))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(scale), origin=[0.0, 0.0, 0.0])

    o3d.visualization.draw_geometries(
        [cloud, line_set, axes],
        window_name="Direction Debug",
    )


def generate_ref_variants(
    ref_motif: np.ndarray,
    ref_orientation: np.ndarray,
    angle_tol: float = np.deg2rad(45.0),
    directions: Sequence[np.ndarray] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build symmetry-aware motif variants and reference orientations."""

    positions = ref_motif
    if len(positions) == 0:
        raise ValueError("Reference motif must contain at least one atom")

    origin = positions[0]
    rel_vectors = positions - origin
    sampled_directions = directions if directions is not None else dodecahedron_vertex_directions()
    sampled_directions = _cluster_directions(
        sampled_directions,
        angle_tol=angle_tol,
        treat_opposites_as_equivalent=False,
    )

    variants: list[tuple[np.ndarray, np.ndarray]] = []
    seen_keys: set[tuple[float, ...]] = set()

    for primary, secondary in itertools.permutations(sampled_directions, 2):
        x_axis = _normalise_vector(primary)
        secondary_proj = secondary - np.dot(secondary, x_axis) * x_axis
        if np.linalg.norm(secondary_proj) < 1e-8:
            continue
        y_axis = _normalise_vector(secondary_proj)
        z_axis = np.cross(x_axis, y_axis)
        if np.linalg.norm(z_axis) < 1e-8:
            continue
        z_axis = _normalise_vector(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        if np.linalg.det(rotation) < 0:
            y_axis = -y_axis
            rotation = np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis)))
        key = tuple(np.round(rotation.flatten(), 6))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rotated_points = rel_vectors @ rotation.T
        rotated_orientation = rotation @ ref_orientation
        variants.append((rotated_points, rotated_orientation))

    identity_variant = (rel_vectors.copy(), ref_orientation.copy())
    variants.insert(0, identity_variant)

    return variants


def get_icp_transformation(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    threshold: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Compute the ICP transformation aligning source points to target points."""

    source_cloud = to_open3d_cloud(source_points)
    target_cloud = to_open3d_cloud(target_points)

    if threshold is None:
        bbox_extent = np.max(target_points, axis=0) - np.min(target_points, axis=0)
        bbox_diag = np.linalg.norm(bbox_extent)
        threshold = max(bbox_diag, 1e-3)

    transformation = np.eye(4)
    transformation[:3, 3] = np.mean(target_points, axis=0) - np.mean(source_points, axis=0)
    reg = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        threshold,
        transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=tolerance,
            relative_rmse=tolerance,
            max_iteration=max_iterations,
        ),
    )
    if len(reg.correspondence_set) == 0:
        raise RuntimeError(
            "ICP failed to find correspondences. Increase the threshold or provide a better initial guess."
        )

    return reg.transformation, reg.fitness, reg.inlier_rmse


def get_n_nearest_neighbors(
    atoms: Atoms,
    selected_types: list[str] | None,
    N: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return N nearest neighbours for each atom of the selected types."""

    symbols = atoms.get_chemical_symbols()
    if selected_types is None:
        selected_types = list(set(symbols))
    selected_indices = np.array([i for i in range(len(atoms)) if symbols[i] in selected_types])

    if len(selected_indices) == 0:
        return {}

    selected_pos = atoms.positions[selected_indices]
    displacements, dists = get_distances(
        selected_pos,
        selected_pos,
        cell=atoms.get_cell(complete=True),
        pbc=atoms.pbc,
    )

    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, center_idx in enumerate(selected_indices):
        neighbor_dists = dists[i]
        sorted_idx = np.argsort(neighbor_dists)[0:]
        n_neighbors = min(len(sorted_idx), N + 1)
        selected_neighbor_idx = sorted_idx[:n_neighbors]
        selected_dists = neighbor_dists[selected_neighbor_idx]
        selected_displacements = displacements[i, selected_neighbor_idx]
        original_neighbor_idx = selected_indices[selected_neighbor_idx]
        neighbors = (original_neighbor_idx, selected_dists, selected_displacements)
        result[center_idx] = neighbors

    return result


def get_positions_of_neighbors(
    atoms: Atoms,
    neighbors_dict: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Convert neighbour displacement vectors into absolute positions."""

    positions = atoms.get_positions()
    neighbor_positions_dict: dict[int, np.ndarray] = {}
    for center_idx, (_, _, relative_vectors) in neighbors_dict.items():
        center_position = positions[center_idx]
        neighbor_positions = center_position + relative_vectors
        neighbor_positions_dict[center_idx] = neighbor_positions
    return neighbor_positions_dict


def _align_center_orientation(
    center_idx: int,
    center_position: np.ndarray,
    neighbor_positions: np.ndarray,
    ref_variants: Sequence[tuple[np.ndarray, np.ndarray]],
    regularize_orientations: bool,
    regularize_anchor: np.ndarray | None,
    regularize_tol: float,
    max_iterations: int,
    tolerance: float,
    threshold: float,
) -> tuple[int, np.ndarray | None, float | None, float | None, str | None]:
    """Determine the best orientation match for a single central atom."""

    neighbor_positions_centered = neighbor_positions - center_position
    best_orientation = None
    best_fitness = None
    best_rmse = None

    for ref_points_variant, ref_orientation_variant in ref_variants:
        try:
            transformation, fitness, inlier_rmse = get_icp_transformation(
                neighbor_positions_centered,
                ref_points_variant,
                max_iterations=max_iterations,
                tolerance=tolerance,
                threshold=threshold,
            )
        except RuntimeError:
            continue

        if best_rmse is not None and inlier_rmse >= best_rmse:
            continue

        rotation = transformation[:3, :3].T
        candidate_orientation = rotation @ ref_orientation_variant
        best_orientation = candidate_orientation
        best_fitness = fitness
        best_rmse = inlier_rmse

    if best_orientation is None:
        return center_idx, None, None, None, (
            f"Atom index {center_idx}: ICP failed for all reference variants"
        )

    try:
        orientation_vec = _normalise_vector(best_orientation)
    except ValueError:
        orientation_vec = best_orientation

    if regularize_orientations:
        orientation_vec = _regularise_orientation(
            orientation_vec,
            anchor=regularize_anchor,
            tol=regularize_tol,
        )

    return center_idx, orientation_vec, best_fitness, best_rmse, None


def get_orientated_frames(
    atoms: Atoms,
    ref_motif: np.ndarray,
    elements: list[str] | None = None,
    ref_orientation: np.ndarray = np.array([1.0, 0.0, 0.0]),
    regularize_orientations: bool = False,
    regularize_anchor: np.ndarray | None = None,
    regularize_tol: float = 1e-8,
) -> Atoms:
    """Attach orientation vectors to selected atoms via ICP alignment."""

    num_neighbors = len(ref_motif)
    neighbors_dict = get_n_nearest_neighbors(atoms, selected_types=elements, N=num_neighbors - 1)
    neighbor_positions_dict = get_positions_of_neighbors(atoms, neighbors_dict)

    n_atoms = len(atoms)
    orientations = _ensure_atom_array(atoms, "orientation", (n_atoms, 3))
    fitness_array = _ensure_atom_array(atoms, "fitness", (n_atoms,))
    inlier_rmse_array = _ensure_atom_array(atoms, "inlier_rmse", (n_atoms,))

    ref_variants = generate_ref_variants(ref_motif, ref_orientation, angle_tol=np.deg2rad(55.0))
    for center_idx, neighbor_positions in neighbor_positions_dict.items():
        center_position = atoms.positions[center_idx]
        anchor = regularize_anchor if regularize_anchor is not None else ref_orientation
        idx, orientation_vec, best_fitness, best_rmse, error_msg = _align_center_orientation(
            center_idx,
            center_position,
            neighbor_positions,
            ref_variants,
            regularize_orientations,
            anchor,
            regularize_tol,
            max_iterations=50,
            tolerance=1e-6,
            threshold=5,
        )
        if orientation_vec is None:
            if error_msg:
                print(error_msg)
            continue

        orientations[idx] = orientation_vec
        fitness_array[idx] = best_fitness if best_fitness is not None else 0.0
        inlier_rmse_array[idx] = best_rmse if best_rmse is not None else 0.0

    return atoms


def get_orientated_frames_parallel(
    atoms: Atoms,
    ref_motif: np.ndarray,
    elements: list[str] | None = None,
    ref_orientation: np.ndarray = np.array([1.0, 0.0, 0.0]),
    regularize_orientations: bool = False,
    regularize_anchor: np.ndarray | None = None,
    regularize_tol: float = 1e-8,
    max_workers: int | None = None,
) -> Atoms:
    """Parallel variant of :func:`get_orientated_frames` using thread pools."""

    num_neighbors = len(ref_motif)
    neighbors_dict = get_n_nearest_neighbors(atoms, selected_types=elements, N=num_neighbors - 1)
    neighbor_positions_dict = get_positions_of_neighbors(atoms, neighbors_dict)

    n_atoms = len(atoms)
    orientations = _ensure_atom_array(atoms, "orientation", (n_atoms, 3))
    fitness_array = _ensure_atom_array(atoms, "fitness", (n_atoms,))
    inlier_rmse_array = _ensure_atom_array(atoms, "inlier_rmse", (n_atoms,))

    ref_variants = generate_ref_variants(ref_motif, ref_orientation, angle_tol=np.deg2rad(55.0))
    anchor = regularize_anchor if regularize_anchor is not None else ref_orientation

    tasks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for center_idx, neighbor_positions in neighbor_positions_dict.items():
        center_position = atoms.positions[center_idx]
        tasks.append((center_idx, center_position, neighbor_positions))

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _align_center_orientation,
                center_idx,
                center_position,
                neighbor_positions,
                ref_variants,
                regularize_orientations,
                anchor,
                regularize_tol,
                50,
                1e-6,
                5,
            ): center_idx
            for center_idx, center_position, neighbor_positions in tasks
        }

        for future in as_completed(futures):
            idx, orientation_vec, best_fitness, best_rmse, error_msg = future.result()
            if orientation_vec is None:
                if error_msg:
                    failures.append(error_msg)
                continue
            orientations[idx] = orientation_vec
            fitness_array[idx] = best_fitness if best_fitness is not None else 0.0
            inlier_rmse_array[idx] = best_rmse if best_rmse is not None else 0.0

    for msg in failures:
        print(msg)

    return atoms


def get_order_parameters(
    file_path: str,
    ref_motif_path: str,
    ref_orientation: np.ndarray,
    elements: Sequence[str] | Mapping[int | str, str] | None = None,
    selected_elements: list[str] | None = None,
    regularize_orientations: bool = False,
    regularize_anchor: np.ndarray | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
) -> Atoms:
    """Compute orientation order parameters for atoms in a structure."""

    structure = read(file_path, format="lammps-data", atom_style="atomic")
    if elements:
        structure = assign_type_symbols(structure, elements)

    ref_motif = read_reference(ref_motif_path)

    orientate = get_orientated_frames_parallel if parallel else get_orientated_frames
    structure = orientate(
        structure,
        ref_motif,
        ref_orientation=ref_orientation,
        elements=selected_elements,
        regularize_orientations=regularize_orientations,
        regularize_anchor=regularize_anchor,
        **({"max_workers": max_workers} if parallel else {}),
    )

    return structure
