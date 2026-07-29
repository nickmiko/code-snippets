import pytest
import pandas as pd
import numpy as np
from src.calculate import get_positions, prepare_custom_categories, calculate_zscores, calculate_replacement_levels

@pytest.fixture
def sample_config():
    return {
        'categories': {
            'hitters': ['HR', 'SB', 'AVG', 'OBP'],
            'pitchers': ['QS+W', 'K', 'ERA', 'WHIP']
        },
        'category_weights': {
            'defaults': 1.0,
            'HR': 1.2
        },
        'league_settings': {
            'teams': 2
        },
        'roster': {
            'UT': 1,
            '1B': 1,
            'P': 1,
            'SP': 1
        }
    }

def test_get_positions():
    assert get_positions("SP/RP") == ["SP", "RP"]
    assert get_positions("1B, 2B") == ["1B", "2B"]
    assert get_positions("DH") == ["DH"]

def test_prepare_custom_categories(sample_config):
    df_hitters = pd.DataFrame({"Name": ["A"], "HR": [10]})
    df_pitchers = pd.DataFrame({"Name": ["B"], "QS": [5], "W": [5], "SV": [1], "HLD": [2]})
    
    h, p = prepare_custom_categories(df_hitters, df_pitchers, sample_config)
    assert 'QS+W' in p.columns
    assert p['QS+W'].iloc[0] == 10
    
    config_sv = sample_config.copy()
    config_sv['categories']['pitchers'] = ['SV+HLD']
    h, p = prepare_custom_categories(df_hitters, df_pitchers, config_sv)
    assert 'SV+HLD' in p.columns
    assert p['SV+HLD'].iloc[0] == 3

def test_calculate_replacement_levels(sample_config):
    df_hitters = pd.DataFrame({
        "Name": ["H1", "H2", "H3", "H4", "H5"],
        "Pos": ["1B", "1B", "OF", "SS", "2B"],
        "HR": [30, 20, 10, 5, 2],
        "SB": [5, 15, 25, 30, 40],
        "AVG": [0.300, 0.280, 0.260, 0.240, 0.220],
        "AB": [500, 500, 500, 500, 500],
        "H": [150, 140, 130, 120, 110],
        "OBP": [0.350, 0.340, 0.330, 0.320, 0.310],
        "PA": [550, 550, 550, 550, 550]
    })
    df_pitchers = pd.DataFrame({
        "Name": ["P1", "P2", "P3", "P4", "P5"],
        "Pos": ["SP", "SP", "RP", "SP", "RP"],
        "QS": [15, 10, 0, 5, 0],
        "W": [10, 8, 5, 3, 2],
        "K": [200, 150, 80, 100, 60],
        "ERA": [3.00, 3.50, 2.50, 4.00, 3.80],
        "WHIP": [1.10, 1.20, 1.05, 1.30, 1.25],
        "IP": [200, 180, 60, 150, 50],
        "ER": [67, 70, 17, 67, 21],
        "H": [180, 170, 45, 160, 40],
        "BB": [40, 46, 18, 35, 22]
    })
    
    levels = calculate_replacement_levels(df_hitters, df_pitchers, sample_config)
    assert 'UT' in levels
    assert '1B' in levels
    assert 'P' in levels
    assert 'SP' in levels

def test_calculate_zscores(sample_config):
    df_hitters = pd.DataFrame({
        "Name": ["H1", "H2"],
        "Pos": ["1B", "SS/2B"], 
        "HR": [30, 10],
        "SB": [5, 30],
        "AVG": [0.300, 0.250],
        "AB": [500, 500],
        "H": [150, 125],
        "OBP": [0.350, 0.300],
        "PA": [550, 550]
    })
    df_pitchers = pd.DataFrame({
        "Name": ["P1", "P2"],
        "Pos": ["SP", "RP"],
        "QS": [15, 0],
        "W": [10, 5],
        "K": [200, 80],
        "ERA": [3.00, 2.50],
        "WHIP": [1.10, 1.05],
        "IP": [200, 60],
        "ER": [67, 17],
        "H": [180, 45],
        "BB": [40, 18]
    })
    
    rep_levels = {'1B': -1.0, 'UT': -2.0, 'SP': -1.0, 'P': -2.0}
    h, p = calculate_zscores(df_hitters, df_pitchers, rep_levels, sample_config)
    
    assert 'HR_Z' in h.columns
    assert 'Total_Z' in h.columns
    assert 'QS+W_Z' in p.columns
    assert 'Total_Z' in p.columns