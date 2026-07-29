import pytest
import pandas as pd
from pathlib import Path
from src.calculate import calculate_replacement_levels
from src.economy import compute_marginal_dollars, apply_keeper_inflation
from src.economy_blend import apply_projection_weights

def get_mock_hitters():
    return pd.DataFrame({
        'Name': [f'Hitter {i}' for i in range(20)],
        'Pos': ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UT', 'UT'] * 2,
        'Rank_Score': sorted([float(i) for i in range(20)], reverse=True),
        'Z_Above_Rep': sorted([float(i) for i in range(20)], reverse=True)
    })

def get_mock_pitchers():
    return pd.DataFrame({
        'Name': [f'Pitcher {i}' for i in range(20)],
        'Pos': ['SP', 'SP', 'SP', 'RP', 'RP'] * 4,
        'Rank_Score': sorted([float(i) for i in range(20)], reverse=True),
        'Z_Above_Rep': sorted([float(i) for i in range(20)], reverse=True)
    })

def test_split_changes_economy():
    h = get_mock_hitters()
    p = get_mock_pitchers()
    
    config_50_50 = {
        'league_settings': {'teams': 10, 'budget': 100, 'hitter_split': 0.5, 'pitcher_split': 0.5},
        'roster': {'C': 1}
    }
    
    config_80_20 = {
        'league_settings': {'teams': 10, 'budget': 100, 'hitter_split': 0.8, 'pitcher_split': 0.2},
        'roster': {'C': 1}
    }
    
    h1, p1 = compute_marginal_dollars(h, p, config_50_50)
    h2, p2 = compute_marginal_dollars(h, p, config_80_20)
    
    # The total Base_Value for hitters should be higher in 80/20 than 50/50
    assert h2['Base_Value'].sum() > h1['Base_Value'].sum()
    # P Base_Value should be lower in 80/20 because of the squeeze
    assert p2['Base_Value'].sum() < p1['Base_Value'].sum()

def test_budget_and_teams_changes_total_money():
    h = get_mock_hitters()
    p = get_mock_pitchers()
    
    config_small = {
        'league_settings': {'teams': 10, 'budget': 100, 'hitter_split': 0.6, 'pitcher_split': 0.4},
        'roster': {'C': 1}
    }
    
    config_large = {
        'league_settings': {'teams': 12, 'budget': 260, 'hitter_split': 0.6, 'pitcher_split': 0.4},
        'roster': {'C': 1}
    }
    
    h_small, p_small = compute_marginal_dollars(h, p, config_small)
    h_large, p_large = compute_marginal_dollars(h, p, config_large)
    
    # 12*260 is substantially more money than 10*100
    assert h_large['Base_Value'].sum() > h_small['Base_Value'].sum() * 2
    assert p_large['Base_Value'].sum() > p_small['Base_Value'].sum() * 2

def test_keeper_inflation_changes_remaining_pool():
    # If a keeper is kept for less money than they are worth, inflation goes up.
    h = get_mock_hitters()
    p = get_mock_pitchers()
    
    config = {
        'league_settings': {'teams': 10, 'budget': 100, 'hitter_split': 0.5, 'pitcher_split': 0.5},
        'roster': {'C': 1}
    }
    
    h_base, p_base = compute_marginal_dollars(h, p, config)
    total_base_pool_h = h_base['Base_Value'].sum()
    
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock keeper file where Hitter 0 is kept for $1 (but is worth lots)
        keeper_path = Path(tmpdir) / 'keepers.csv'
        pd.DataFrame({'Name': ['Hitter 0'], 'Cost': [1]}).to_csv(keeper_path, index=False)
        
        h_inf, p_inf = apply_keeper_inflation(h_base, p_base, config, tmpdir)
        
        # Hitter 1 should be worth more in the inflation pool since Hitter 0 is off the board for incredibly cheap
        val_base_h1 = h_base[h_base['Name'] == 'Hitter 1']['Base_Value'].values[0]
        val_inf_h1 = h_inf[h_inf['Name'] == 'Hitter 1']['Value'].values[0]
        
        assert val_inf_h1 > val_base_h1, "Inflation did not correctly increase available player values."

def test_projection_blending():
    h = get_mock_hitters()
    p = get_mock_pitchers()
    
    h['Base_Value'] = 10.0
    p['Base_Value'] = 10.0
    
    config = {
        'projections_weights': {
            'hitters': {'atc': 1.0},
            'pitchers': {'atc': 0.5, 'pitcherlist': 0.5}
        }
    }
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        pl_path = Path(tmpdir) / 'pitcherlist_dollar_values.csv'
        # Hitter 0 gets no change, Pitcher 0 gets PL value of 20
        pd.DataFrame({'Name': ['Pitcher 0'], 'dollar_value': [20.0]}).to_csv(pl_path, index=False)
        
        h_blend, p_blend = apply_projection_weights(h, p, config, tmpdir)
        
        # Pitcher 0: Base 10.0 * 0.5 + PL 20.0 * 0.5 = 15.0
        assert p_blend[p_blend['Name'] == 'Pitcher 0']['Base_Value'].values[0] == 15.0
        # Pitcher 1: Base 10.0 * 0.5 + PL 0.0 * 0.5 = 5.0
        assert p_blend[p_blend['Name'] == 'Pitcher 1']['Base_Value'].values[0] == 5.0
        
        # Hitters identically unharmed
        assert h_blend[h_blend['Name'] == 'Hitter 0']['Base_Value'].values[0] == 10.0
