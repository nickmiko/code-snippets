import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from src.economy_blend import apply_projection_weights

def test_apply_projection_weights_no_pl():
    h = pd.DataFrame({"Name": ["H1"], "Base_Value": [10.0]})
    p = pd.DataFrame({"Name": ["P1"], "Base_Value": [10.0]})
    config = {
        'projections_weights': {
            'pitchers': {'atc': 1.0, 'pitcherlist': 0.0}
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        h2, p2 = apply_projection_weights(h, p, config, tmpdir)
        assert p2.loc[p2['Name'] == 'P1', 'Base_Value'].values[0] == 10.0

def test_apply_projection_weights_with_pl():
    h = pd.DataFrame({"Name": ["H1"], "Base_Value": [10.0]})
    p = pd.DataFrame({"Name": ["P1", "P2"], "Base_Value": [10.0, 10.0]})
    config = {
        'projections_weights': {
            'pitchers': {'atc': 0.5, 'pitcherlist': 0.5}
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        pd.DataFrame({
            "Name": ["p1"], 
            "dollar_value": [30.0]
        }).to_csv(Path(tmpdir) / 'pitcherlist_dollar_values.csv', index=False)
        
        h2, p2 = apply_projection_weights(h, p, config, tmpdir)
        
        assert p2.loc[p2['Name'] == 'P1', 'Base_Value'].values[0] == 20.0
        assert p2.loc[p2['Name'] == 'P2', 'Base_Value'].values[0] == 5.0