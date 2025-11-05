# OriVec

OriVec provides tools to compute orientation order parameters for local motifs in atomistic point clouds.

## Installation

```powershell
pip install .
```

## Command Line Usage

```powershell
orivec .\liquid.data .\ref_unit.xyz --element-map 1=Li,2=Mo,3=S --selected-element S --regularize --parallel --output liquid-orientations.xyz
```

This command reads a LAMMPS data file, aligns local motifs to the reference geometry, stores the resulting orientation vectors in per-atom arrays, and writes the augmented structure to `liquid-orientations.xyz`.

## Python API

```python
from orivec import get_order_parameters
import numpy as np

structure = get_order_parameters(
    "liquid.data",
    "ref_unit.xyz",
    ref_orientation=np.array([0.0, 0.0, 1.0]),
    elements={1: "Li", 2: "Mo", 3: "S"},
    selected_elements=["S"],
    regularize_orientations=True,
    regularize_anchors=np.array([0.0, 0.0, 1.0]),
    parallel=True,
    max_workers=8,
)
```

The resulting `ase.Atoms` object stores orientation vectors in `structure.arrays['orientation']`, `structure.arrays['inlier_rmse']`
