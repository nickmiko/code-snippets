import yaml
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

@pytest.fixture
def config():
    assert CONFIG_PATH.exists(), "config.yaml does not exist."
    with open(CONFIG_PATH, 'r') as file:
        return yaml.safe_load(file)

def test_config_has_required_top_level_keys(config):
    required_keys = ['league_settings', 'roster', 'categories', 'category_weights', 'projections_weights']
    for key in required_keys:
        assert key in config, f"Missing required top-level key: {key}"

def test_league_settings(config):
    settings = config['league_settings']
    assert isinstance(settings['teams'], int), "Teams must be an integer."
    assert settings['teams'] > 0, "League must have greater than 0 teams."
    
    assert isinstance(settings['budget'], (int, float)), "Budget must be numeric."
    assert settings['budget'] > 0, "Budget must be greater than 0."
    
    assert isinstance(settings['hitter_split'], (int, float)), "Hitter split must be numeric."
    assert isinstance(settings['pitcher_split'], (int, float)), "Pitcher split must be numeric."
    
    total_split = settings['hitter_split'] + settings['pitcher_split']
    assert 0.0 < total_split <= 1.0, f"Hitter and pitcher splits should be <= 1.0, currently {total_split}"

def test_roster_settings(config):
    roster = config['roster']
    required_positions = ['C', '1B', '2B', '3B', 'SS', 'MI', 'CI', 'OF', 'UT', 'SP', 'RP', 'P', 'Bench']
    for pos in required_positions:
        assert pos in roster, f"Missing expected position: {pos}"
        assert isinstance(roster[pos], int), f"Limit for {pos} must be an integer"
        assert roster[pos] >= 0, f"Limit for {pos} cannot be negative"

def test_category_settings(config):
    cats = config['categories']
    assert 'hitters' in cats, "Hitter categories missing."
    assert 'pitchers' in cats, "Pitcher categories missing."
    
    assert isinstance(cats['hitters'], list), "Hitter categories must be a list."
    assert len(cats['hitters']) > 0, "Must have at least one hitting category."
    
    assert isinstance(cats['pitchers'], list), "Pitcher categories must be a list."
    assert len(cats['pitchers']) > 0, "Must have at least one pitching category."

def test_category_weights(config):
    weights = config['category_weights']
    assert 'defaults' in weights, "Missing defaults category weight."
    assert isinstance(weights['defaults'], (int, float)), "Default weight must be numeric."

def test_projections_weights(config):
    proj_weights = config['projections_weights']
    assert 'hitters' in proj_weights, "Missing hitters projection weights."
    assert 'pitchers' in proj_weights, "Missing pitchers projection weights."
    
    # Assert hitter weights sum to <= 1.0 (some users might want to purposefully undercut projections)
    h_sum = sum(proj_weights['hitters'].values())
    p_sum = sum(proj_weights['pitchers'].values())
    
    assert h_sum <= 1.0, "Total sum of hitter projection weights cannot exceed 1.0"
    assert p_sum <= 1.0, "Total sum of pitcher projection weights cannot exceed 1.0"
