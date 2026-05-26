import numpy as np
import pandas as pd
import pytest


pytest.importorskip("scilslab")
from slx2imzml.slxFileHelper import slxFileHelper


def test_build_final_features_offsets_duplicate_mz_by_ccs_order():
    features = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "mz_low": [700.4449, 700.4449, 701.0000],
            "mz_high": [700.4451, 700.4451, 701.0020],
            "ccs_low": [150.0, 200.0, np.nan],
            "ccs_high": [152.0, 202.0, np.nan],
        }
    )

    out = slxFileHelper._build_final_features(features, mz_offset_step=1e-4)

    # Columns: [id, mz_low, mz_high, centroid_shifted, ccs_low, ccs_high]
    assert out.shape == (3, 6)

    # Extract duplicate centroid rows
    dup = out[np.isclose(out[:, 1], 700.4449) & np.isclose(out[:, 2], 700.4451)]
    assert dup.shape[0] == 2

    # Higher CCS feature (id=11) must receive higher shifted centroid
    c_shift = {int(r[0]): float(r[3]) for r in dup}
    assert c_shift[11] > c_shift[10]
    assert np.isclose(c_shift[11] - c_shift[10], 1e-4)


def test_build_final_features_offsets_duplicate_mz_without_ccs_too():
    features = pd.DataFrame(
        {
            "id": [1, 2],
            "mz_low": [500.0, 500.0],
            "mz_high": [500.2, 500.2],
        }
    )

    out = slxFileHelper._build_final_features(features, mz_offset_step=1e-4)

    assert out.shape == (2, 6)
    assert np.isclose(out[1, 3] - out[0, 3], 1e-4)
