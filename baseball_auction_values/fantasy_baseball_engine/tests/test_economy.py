import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from src.economy import compute_base_dollars, compute_marginal_dollars, apply_keeper_inflation

@pytest.fixture
def sample_config():
    return {
        'league_settings': {
            'teams': 10,
            'budget': 260,
            'hitter_split': 0.65,
            'pitcher_split': 0.35
        },
        'roster': {
            'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3, 'UT': 1, 'P': 9
        }
    }

def test_compute_base_dollars():
    df_hitters = pd.DataFrame({"Name": ["H1", "H2", "H3"], "Z_Above_Rep": [10.0, 5.0, -1.0]})
    df_pitchers = pd.DataFrame({"Name": ["P1", "P2", "P3"], "Z_Above_Rep": [8.0, 4.0, -2.0]})
    
    h, p = compute_base_dollars(df_hitters, df_pitchers, h_budget=102, p_budget=102, total_roster_spots=20)
    
    assert np.isclose(h.loc[h['Name'] == 'H1', 'Base_Value'].values[0], 67.67, atol=0.01)
    assert np.isclose(h.loc[h['Name'] == 'H2', 'Base_Value'].values[0], 34.33, atol=0.01)
    assert h.loc[h['Name'] == 'H3', 'Base_Value'].values[0] == 0.0
    assert np.isclose(p.loc[p['Name'] == 'P1', 'Base_Value'].values[0], 67.67, atol=0.01)

def test_compute_marginal_dollars(sample_config):
    df_hitters = pd.DataFrame({"Name": ["H1", "H2"], "Z_Above_Rep": [10.0, 5.0]})
    df_pitchers = pd.DataFrame({"Name": ["P1", "P2"], "Z_Above_Rep": [8.0, 4.0]})
    
    h, p = compute_marginal_dollars(df_hitters, df_pitchers, sample_config)
    assert np.isclose(h.loc[h['Name'] == 'H1', 'Base_Value'].values[0], 1126.33, atol=0.01)

def test_apply_keeper_inflation(sample_config):
    df_hitters = pd.DataFrame({"Name": ["Keeper1", "Free1"], "Base_Value": [50.0, 30.0], "Z_Above_Rep": [10.0, 5.0]})
    df_pitchers = pd.DataFrame({"Name": ["Keeper2", "Free2"], "Base_Value": [40.0, 20.0], "Z_Above_Rep": [8.0, 4.0]})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pd.DataFrame({
            "Name": ["Keeper1", "Keeper2"],
            "Cost": [5, 10]
        }).to_csv(Path(tmpdir) / 'keepers.csv', index=False)
        
        h, p = apply_keeper_inflation(df_hitters, df_pitchers, sample_config, tmpdir)
        
        assert h.loc[h['Name'] == 'Keeper1', 'Value'].values[0] == 50.0
        assert h.loc[h['Name'] == 'Keeper1', 'Is_Keeper'].values[0] == True
        assert h.loc[h['Name'] == 'Free1', 'Value'].values[0] > 30.0
        assert p.loc[p['Name'] == 'Free2', 'Value'].values[0] > 20.0

def test_apply_keeper_inflation_no_file(sample_config):
    df_hitters = pd.DataFrame({"Name": ["H1"], "Base_Value": [50.0]})
    df_pitchers = pd.DataFrame({"Name": ["P1"], "Base_Value": [40.0]})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        h, p = apply_keeper_inflation(df_hitters, df_pitchers, sample_config, tmpdir)
        assert h.loc[0, 'Value'] == 50.0
        assert p.loc[0, 'Value'] == 40.0
