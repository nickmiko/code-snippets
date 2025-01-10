import pandas as pd
import numpy as np
from typing import Dict, List
import sys

def preprocess_hitter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for analysis"""
    # Remove rows with null values in key columns
    key_columns = ['Name', 'Team', 'PA', 'HR', 'RBI', 'R', 'SB', 'OBP']
    return df.dropna(subset=key_columns)

def calculate_weighted_zscore(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Calculate weighted z-scores for each statistic"""
    try:
        # Create z-score DataFrame
        z_scores = pd.DataFrame()
        
        # Calculate z-score for each metric
        for metric, weight in weights.items():
            z_scores[metric] = (df[metric] - df[metric].mean()) / df[metric].std()
            z_scores[metric] = z_scores[metric] * weight
        
        # Calculate final weighted score
        df['weighted_zscore'] = z_scores.sum(axis=1)
        return df.sort_values('weighted_zscore', ascending=False)
    
    except Exception as e:
        print(f"Error calculating z-scores: {e}")
        sys.exit(1)
def process_hitter_data():
    try:
        # Configuration
        weights = {
            'OBP': 0.20,
            'HR': 0.20,
            'RBI': 0.15,
            'R': 0.20,
            'SB': 0.15,
            'PA': 0.10
        }
        
        display_columns = ['Name', 'Team', 'OBP', 'HR', 'RBI', 'R', 'SB', 'PA', 'weighted_zscore']
        
        # Read and process data
        df = pd.read_csv('fangraphs-leaderboard-projections_hitter.csv')
        df = preprocess_hitter_data(df)
        
        # Calculate rankings
        ranked_players = calculate_weighted_zscore(df, weights)
        
        # Display results
        print("\nTop 25 Players by Weighted Z-Score:")
        print(ranked_players[display_columns].head(25).to_string())
        
        # Save results
        ranked_players.to_csv('player_rankings.csv', index=False)
        
    except FileNotFoundError:
        print("Error: Input CSV file not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def preprocess_pitcher_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for analysis"""
    # Remove rows with null values in key columns
    key_columns = ['Name', 'Team', 'IP', 'K/9', 'ERA', 'WHIP', 'W', 'SV']
    return df.dropna(subset=key_columns)

def process_pitcher_data():
    try:
        # Configuration
        weights = {
            'K/9': 0.20,
            'ERA': -0.20,
            'WHIP': -0.20,
            'W': 0.15,
            'SV': 0.15,
            'IP': 0.10
        }
        
        display_columns = ['Name', 'Team', 'IP', 'K/9', 'ERA', 'WHIP', 'W', 'SV', 'weighted_zscore']
        
        # Read and process data
        df = pd.read_csv('fangraphs-leaderboard-projections_pitcher.csv')
        df = preprocess_pitcher_data(df)
        
        # Calculate rankings
        ranked_players = calculate_weighted_zscore(df, weights)
        
        # Display results
        print("\nTop 25 Players by Weighted Z-Score:")
        print(ranked_players[display_columns].head(25).to_string())
        
        # Save results
        ranked_players.to_csv('pitcher_rankings.csv', index=False)
        
    except FileNotFoundError:
        print("Error: Input CSV file not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
def main():
    process_hitter_data()
    process_pitcher_data()

if __name__ == "__main__":
    main()