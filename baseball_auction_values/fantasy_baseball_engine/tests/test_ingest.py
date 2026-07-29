import pytest
import pandas as pd
import tempfile
import yaml
from pathlib import Path
from src.ingest import load_config, load_projections

def test_load_config():
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
        config_data = {'league_settings': {'teams': 10}}
        with open(tmp.name, 'w') as f:
            yaml.dump(config_data, f)
            
        loaded = load_config(tmp.name)
        assert loaded['league_settings']['teams'] == 10

def test_load_projections():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        
        pd.DataFrame({'Name': ['H1'], 'PlayerId': ['1'], 'HR': [10]}).to_csv(Path(tmpdir) / 'atc_hitter.csv', index=False)
        pd.DataFrame({'Name': ['P1'], 'PlayerId': ['2'], 'QS': [5]}).to_csv(Path(tmpdir) / 'atc_pitcher.csv', index=False)
        
        h, p = load_projections(tmpdir)
        
        assert len(h) == 1
        assert h['Name'].iloc[0] == 'H1'
        assert h['Pos'].iloc[0] == 'DH'
        
        assert len(p) == 1
        assert p['Name'].iloc[0] == 'P1'
        assert p['Pos'].iloc[0] == 'RP'