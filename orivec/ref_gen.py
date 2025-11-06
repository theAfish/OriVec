from pathlib import Path

from ase.io import read
from ase import Atoms
from ase.neighborlist import neighbor_list
import numpy as np
import spglib


def get_symmetry_unique_indices(
    atoms: Atoms,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
) -> list[int]:
    """Return sorted indices representing symmetry-distinct atoms.

    The routine relies on *spglib* via ASE to determine symmetry-equivalent
    positions. When symmetry information cannot be resolved (missing ``spglib``,
    non-periodic system, ill-defined cell) all atoms are treated as distinct.
    """

    if len(atoms) == 0:
        return []

    # Non-periodic structures do not have translational symmetry; treat all atoms as unique.
    if not np.any(atoms.pbc):
        return list(range(len(atoms)))

    cell_matrix = np.asarray(atoms.get_cell(complete=True))
    if abs(np.linalg.det(cell_matrix)) < 1e-8:
        return list(range(len(atoms)))

    dataset = spglib.get_symmetry_dataset(
        (
            cell_matrix,
            atoms.get_scaled_positions(wrap=True),
            atoms.get_atomic_numbers(),
        ),
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )

    if dataset is None:
        return list(range(len(atoms)))

    equivalent_atoms = getattr(dataset, "equivalent_atoms", None)
    if equivalent_atoms is None and isinstance(dataset, dict):
        equivalent_atoms = dataset.get("equivalent_atoms")

    if equivalent_atoms is None:
        return list(range(len(atoms)))

    equivalent = np.asarray(equivalent_atoms, dtype=int)
    representatives: dict[int, int] = {}
    for idx, label in enumerate(equivalent):
        if label not in representatives:
            representatives[label] = idx

    return sorted(representatives.values())

def get_nearest_neighbors(atoms: Atoms, center: int, r_cut: float) -> list[tuple[int, np.ndarray]]:
    """Return neighbors within ``r_cut`` keeping periodic images distinct."""

    if not np.any(atoms.pbc):
        distances = atoms.get_all_distances(mic=False)[center]
        sorted_indices = np.argsort(distances)
        mask = (sorted_indices != center) & (distances[sorted_indices] <= r_cut)
        return [
            (int(idx), np.zeros(3, dtype=int))
            for idx in sorted_indices[mask]
        ]

    center_pos = atoms.get_positions()[center]
    cell = atoms.get_cell()
    i, j, shifts = neighbor_list(
        "ijS",
        atoms,
        cutoff=r_cut,
        self_interaction=False,
    )

    mask = i == center
    neighbor_indices = j[mask]
    neighbor_shifts = shifts[mask]
    if neighbor_indices.size == 0:
        return []

    displacements = (
    atoms.get_positions()[neighbor_indices]
    + np.dot(neighbor_shifts, cell)
    - center_pos
    )
    order = np.argsort(np.linalg.norm(displacements, axis=1))
    return [
        (int(neighbor_indices[idx]), neighbor_shifts[idx].astype(int))
        for idx in order
    ]

def generate_basic_ref(atoms: Atoms, center_indices: list[int], r_cut: float) -> list[np.ndarray]:
    """Generate reference motifs from a base structure file.

    Each motif is centered on the atom indices specified in
    ``center_indices`` and includes all neighbors within ``r_cut`` of the
    center atom. Coordinates are expressed relative to the center atom, which
    is located at the origin using minimal-image displacements when periodic
    boundaries are present.

    Args:
        base_file: Path to the structure file readable by ASE.
        center_indices: List of atom indices to center the motifs on.
        r_cut: Radial cutoff in Angstrom for including neighboring atoms.

    Returns:
        A list of NumPy arrays representing the generated reference motifs.
    """
    motifs = []

    for center in center_indices:
        neighbor_entries = get_nearest_neighbors(atoms, center, r_cut)

        # Build motif in a local frame with the center atom at the origin.
        center_pos = atoms.get_positions()[center]
        cell = atoms.get_cell()
        neighbor_vectors = []
        for neighbor, shift in neighbor_entries:
            # Account for periodic image translation before forming relative vector.
            shift_vector = np.dot(shift.astype(float), cell)
            neighbor_pos = atoms.get_positions()[neighbor] + shift_vector
            neighbor_vectors.append(neighbor_pos - center_pos)

        motif = np.vstack([np.zeros(3, dtype=float)] + neighbor_vectors)
        motifs.append(motif)

    return motifs

def generate_ref_motifs(
    base_file: str,
    r_cut: float,
    out_base_name: str = "ref_motif",
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = 5.0,
    output_dir: str | Path | None = None,
) -> list[np.ndarray]:
    """Generate reference motifs from symmetry-unique atoms in a base structure file.

    Each motif is centered on symmetry-distinct atoms and includes all neighbors
    within ``r_cut`` of the center atom. Coordinates are expressed relative to
    the center atom, which is located at the origin using minimal-image
    displacements when periodic boundaries are present.

    Args:
        base_file: Path to the structure file readable by ASE.
        r_cut: Radial cutoff in Angstrom for including neighboring atoms.
        out_base_name: Prefix applied to each output motif filename.
        symprec: Positional tolerance passed to spglib symmetry detection.
        angle_tolerance: Angular tolerance passed to spglib symmetry detection.
        output_dir: Directory where motif files will be written (default: current working directory).

    Returns:
        A list of NumPy arrays representing the generated reference motifs.
    """
    atoms = read(base_file)
    unique_indices = get_symmetry_unique_indices(
        atoms,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    motifs = generate_basic_ref(atoms, unique_indices, r_cut)
    output_path = Path(output_dir) if output_dir is not None else Path.cwd()
    output_path.mkdir(parents=True, exist_ok=True)
    for i, motif in enumerate(motifs):
        target_file = output_path / f"{out_base_name}_{i}.xyz"
        np.savetxt(
            target_file,
            motif,
            header=f"{len(motif)}\nMotif {i}",
            comments="",
        )

    return motifs



# if __name__ == "__main__":
#     # demo usage
#     file = "alphaBi-opt.cif"
#     generate_ref_motifs(file, r_cut=6.2, out_base_name="ref_motif_test")


