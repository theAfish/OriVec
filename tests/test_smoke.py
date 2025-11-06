from __future__ import annotations

from ase import Atoms
import numpy as np

import orivec


def test_version_available():
    assert hasattr(orivec, "__version__")


def test_module_exports():
    assert callable(orivec.get_order_parameters)
    assert callable(orivec.generate_ref_variants)
    assert callable(orivec.generate_ref_motifs)


def test_parse_reference(tmp_path):
    xyz = tmp_path / "motif.xyz"
    xyz.write_text("""3\ncomment\n0 0 0\n1 0 0\n0 1 0\n""", encoding="utf-8")
    points = orivec.read_reference(str(xyz))
    assert points.shape == (3, 3)
    assert np.allclose(points[0], [0.0, 0.0, 0.0])


def test_generate_ref_motifs(tmp_path):
    structure = tmp_path / "structure.xyz"
    atoms = Atoms("H2", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
    atoms.write(structure)

    motifs = orivec.generate_ref_motifs(
        base_file=str(structure),
        r_cut=2.0,
        out_base_name="motif",
        output_dir=tmp_path,
    )

    assert len(motifs) == 2
    assert (tmp_path / "motif_0.xyz").exists()
