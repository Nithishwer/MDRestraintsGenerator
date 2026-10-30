"""Regression tests for angular dispersion used in force-constant fitting."""

import numpy as np
import pytest

from MDRestraintsGenerator.datatypes import VectorData


@pytest.mark.parametrize(
    ("coordinate_type", "values"),
    [
        ("angle", [59.0, 60.0, 61.0]),
        ("dihedral", [179.0, 180.0, -179.0]),
    ],
)
def test_angular_variance_has_squared_degree_units(coordinate_type, values):
    coordinate = VectorData(len(values))
    coordinate.atype = coordinate_type
    coordinate.periodic = True
    coordinate.values = np.array(values)

    coordinate.analyze()

    # Both samples fluctuate by -1, 0, and +1 degrees around their center.
    assert coordinate.var == pytest.approx(2 / 3, rel=1e-3)
