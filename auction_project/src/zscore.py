import pandas as pd
import numpy as np

def calculate_auction_values(hitters_df, pitchers_df, budget=260, roster_size=23, num_teams=12, 
                           hitter_budget_pct=0.65,
                           hitter_weights={'HR': 1.0, 'R': 1.0, 'RBI': 1.0, 'SB': 1.0, 'AVG': 1.0},
                           pitcher_weights={'W': 1.0, 'SV': 1.0, 'K': 1.0, 'ERA': 1.0, 'WHIP': 1.0}):
    """
    Calculate auction values based on weighted z-scores with separate hitter/pitcher inputs.
    
    Parameters:
    hitters_df: DataFrame with hitter statistics (HR, R, RBI, SB, AVG)
    pitchers_df: DataFrame with pitcher statistics (W, SV, K, ERA, WHIP)
    budget: Total budget per team (default $260)
    roster_size: Number of players per team (default 23)
    num_teams: Number of teams in league (default 12)
    hitter_budget_pct: Percentage of budget allocated to hitters (default 0.65)
    hitter_weights: Dictionary of weights for hitting categories (default all 1.0)
    pitcher_weights: Dictionary of weights for pitching categories (default all 1.0)
    
    Returns:
    tuple: (hitters_df, pitchers_df) with calculated values
    """
    # Calculate total league budget and split between hitters/pitchers
    total_league_budget = budget * num_teams
    hitter_budget = total_league_budget * hitter_budget_pct
    pitcher_budget = total_league_budget * (1 - hitter_budget_pct)
    
    # Typical roster construction (can be adjusted)
    hitter_roster_size = int(roster_size * 0.6)  # Usually 14 hitters
    pitcher_roster_size = roster_size - hitter_roster_size  # Usually 9 pitchers
    
    # Calculate total rostered players by type
    total_rostered_hitters = hitter_roster_size * num_teams
    total_rostered_pitchers = pitcher_roster_size * num_teams
    
    # Define category lists
    hitting_stats = ['HR', 'R', 'RBI', 'SB', 'AVG']
    pitching_stats = ['W', 'SV', 'K', 'ERA', 'WHIP']
    
    # Calculate weighted z-scores for hitters
    for stat in hitting_stats:
        if stat in hitters_df.columns and stat in hitter_weights:
            hitters_df[f'{stat}_z'] = (
                ((hitters_df[stat] - hitters_df[stat].mean()) / hitters_df[stat].std()) 
                * hitter_weights[stat]
            )
    
    # Calculate weighted z-scores for pitchers
    for stat in pitching_stats:
        if stat in pitchers_df.columns and stat in pitcher_weights:
            # Reverse z-score for ERA and WHIP since lower is better
            if stat in ['ERA', 'WHIP']:
                pitchers_df[f'{stat}_z'] = (
                    -((pitchers_df[stat] - pitchers_df[stat].mean()) / pitchers_df[stat].std())
                    * pitcher_weights[stat]
                )
            else:
                pitchers_df[f'{stat}_z'] = (
                    ((pitchers_df[stat] - pitchers_df[stat].mean()) / pitchers_df[stat].std())
                    * pitcher_weights[stat]
                )
    
    # Sum weighted z-scores for each group
    hitter_z_cols = [col for col in hitters_df.columns if col.endswith('_z')]
    pitcher_z_cols = [col for col in pitchers_df.columns if col.endswith('_z')]
    
    hitters_df['total_z'] = hitters_df[hitter_z_cols].sum(axis=1)
    pitchers_df['total_z'] = pitchers_df[pitcher_z_cols].sum(axis=1)
    
    # Add individual category contributions
    for stat in hitting_stats:
        if f'{stat}_z' in hitters_df.columns:
            hitters_df[f'{stat}_contribution'] = (
                hitters_df[f'{stat}_z'] / hitters_df['total_z'].abs().mean()
            ).round(3)
    
    for stat in pitching_stats:
        if f'{stat}_z' in pitchers_df.columns:
            pitchers_df[f'{stat}_contribution'] = (
                pitchers_df[f'{stat}_z'] / pitchers_df['total_z'].abs().mean()
            ).round(3)
    
    # Calculate value above replacement for each group
    for df, roster_spots in [(hitters_df, total_rostered_hitters), 
                            (pitchers_df, total_rostered_pitchers)]:
        df.sort_values('total_z', ascending=False, inplace=True)
        replacement_level_z = df.iloc[roster_spots-1]['total_z']
        df['value_above_replacement'] = df['total_z'] - replacement_level_z
        df['value_above_replacement'] = df['value_above_replacement'].clip(lower=0)
    
    # Calculate dollar values for hitters
    total_hitter_var = hitters_df['value_above_replacement'].sum()
    hitter_dollars_per_var = (hitter_budget - total_rostered_hitters) / total_hitter_var
    hitters_df['dollar_value'] = (hitters_df['value_above_replacement'] * hitter_dollars_per_var) + 1
    
    # Calculate dollar values for pitchers
    total_pitcher_var = pitchers_df['value_above_replacement'].sum()
    pitcher_dollars_per_var = (pitcher_budget - total_rostered_pitchers) / total_pitcher_var
    pitchers_df['dollar_value'] = (pitchers_df['value_above_replacement'] * pitcher_dollars_per_var) + 1
    
    # Round dollar values
    hitters_df['dollar_value'] = hitters_df['dollar_value'].round(1)
    pitchers_df['dollar_value'] = pitchers_df['dollar_value'].round(1)
    
    return hitters_df, pitchers_df