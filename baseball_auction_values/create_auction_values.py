import pandas as pd
import numpy as np
from typing import Dict
import sys

class PlayerRankings:
    def __init__(self, input_file: str, output_file: str, weights: Dict[str, float]):
        self.input_file = input_file
        self.output_file = output_file
        self.weights = weights
        self.df = None

    def read_data(self):
        """Read CSV data into DataFrame"""
        try:
            self.df = pd.read_csv(self.input_file)
        except FileNotFoundError:
            print("Error: Input CSV file not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading data: {e}")
            sys.exit(1)

    def preprocess_data(self, player_type: str):
        """Clean and prepare data for analysis"""
        key_columns = {
            'hitter': ['Name', 'Team', 'PA', 'HR', 'RBI', 'R', 'SB', 'OBP'],
            'pitcher': ['Name', 'Team', 'IP', 'K/9', 'QS', 'SV', 'ERA', 'WHIP']
        }
        if player_type not in key_columns:
            print("Error: Invalid player type")
            sys.exit(1)
        self.df = self.df.dropna(subset=key_columns[player_type])

    def calculate_weighted_zscore(self):
        """Calculate weighted z-scores for each statistic"""
        try:
            z_scores = pd.DataFrame()
            for metric, weight in self.weights.items():
                z_scores[metric] = (self.df[metric] - self.df[metric].mean()) / self.df[metric].std()
                z_scores[metric] = z_scores[metric] * weight
            self.df['weighted_zscore'] = z_scores.sum(axis=1)
            self.df = self.df.sort_values('weighted_zscore', ascending=False)
        except Exception as e:
            print(f"Error calculating z-scores: {e}")
            sys.exit(1)

    def calculate_auction_values(self):
        """Calculate the dollar value to be spent on each player"""
        try:
            total_teams = 12
            budget = 260
            players_per_team = 23  # Total players per team (11 hitters + 12 pitchers)
            league_budget = total_teams * budget
            rostered_players = total_teams * players_per_team

            total_zscore = self.df['weighted_zscore'].sum()
            if total_zscore == 0:
                print("Error: Total z-score is zero, cannot divide by zero")
                sys.exit(1)

            dollar_per_zscore = league_budget / total_zscore
            self.df['auction_value'] = (self.df['weighted_zscore'] * dollar_per_zscore).round(1)
            self.df = self.df.sort_values('auction_value', ascending=False)
        except Exception as e:
            print(f"Error calculating auction values: {e}")
            sys.exit(1)

    def save_results(self, player_type: str):
        """Save the ranked players to a CSV file"""
        try:
            self.df.to_csv(f'{player_type}_{self.output_file}', index=False)
        except Exception as e:
            print(f"Error saving results: {e}")
            sys.exit(1)

    def run(self, player_type: str, input_file: str):
        """Run the full ranking process"""
        self.input_file = input_file
        self.read_data()
        self.preprocess_data(player_type)
        self.calculate_weighted_zscore()
        self.calculate_auction_values()
        self.save_results(player_type)
        return self.df[['Name', 'auction_value']]

if __name__ == "__main__":
    hitter_weights = {
        'OBP': 0.20,
        'HR': 0.20,
        'RBI': 0.15,
        'R': 0.20,
        'SB': 0.15,
        'PA': 0.10
    }
    pitcher_weights = {
        'IP': 0.20,
        'K/9': 0.20,
        'QS': 0.15,
        'SV': 0.15,
        'ERA': 0.15,
        'WHIP': 0.15
    }
    hitter_input_file = 'fangraphs-leaderboard-projections_hitters.csv'
    pitcher_input_file = 'fangraphs-leaderboard-projections_pitchers.csv'
    output_file = 'player_rankings.csv'
    
    rankings = PlayerRankings(hitter_input_file, output_file, hitter_weights)
    hitter_results = rankings.run('hitter', hitter_input_file)
    
    rankings.weights = pitcher_weights
    pitcher_results = rankings.run('pitcher', pitcher_input_file)
    
    # Combine hitter and pitcher results
    results = pd.concat([hitter_results, pitcher_results])
    results = results.sort_values('auction_value', ascending=False)
    results.to_csv(output_file, index=False)