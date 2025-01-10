import pandas as pd
import numpy as np
from typing import Dict, List
import sys

class PlayerRankings:
    def __init__(self, input_file: str, output_file: str, weights: Dict[str, float]):
        self.input_file = input_file
        self.output_file = output_file
        self.weights = weights
        self.df = None
        self.players = {}

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
        if player_type == 'hitter':
            key_columns = ['Name', 'Team', 'PA', 'HR', 'RBI', 'R', 'SB', 'OBP']
        elif player_type == 'pitcher':
            key_columns = ['Name', 'Team', 'IP', 'K/9', 'QS', 'SV', 'ERA', 'WHIP']
        else:
            print("Error: Invalid player type")
            sys.exit(1)
        self.df = self.df.dropna(subset=key_columns)

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

    def display_top_players(self, top_n: int = 25):
        """Display top N players by weighted z-score"""
        display_columns = ['Name', 'Team', 'OBP', 'HR', 'RBI', 'R', 'SB', 'PA', 'weighted_zscore']
        print(f"\nTop {top_n} Players by Weighted Z-Score:")
        print(self.df[display_columns].head(top_n).to_string())

    def save_results(self):
        """Save the ranked players to a CSV file"""
        try:
            self.df.to_csv(self.output_file, index=False)
        except Exception as e:
            print(f"Error saving results: {e}")
            sys.exit(1)
    def store_results(self):
        """ Store the ranked players and their z-scores in a dictionary"""
        try:
            self.players = self.df.set_index('Name')['weighted_zscore'].to_dict()
        except Exception as e:
            print(f"Error storing results: {e}")
            sys.exit(1)

    def run(self, player_type: str):
        """Run the full ranking process"""
        self.read_data()
        self.preprocess_data(player_type)
        self.calculate_weighted_zscore()
        # self.display_top_players()
        self.save_results()
        self.store_results()
        self.calculate_auction_values()
    
    def calculate_auction_values(self):
        """ Create a dollar amount auction value for each player based on their z-score using a fixed budget and amount of players to be drafted to each team"""
        try:
            # Set the budget and number of players to be drafted
            budget = 260
            players_per_team = 23
            teams = 12
            # Calculate the total number of players in the league
            total_players = players_per_team * teams
            # Calculate the total budget for the league
            total_budget = budget * teams
            # Calculate the average z-score for all players
            average_zscore = self.df['weighted_zscore'].mean()
            # Calculate the standard deviation of z-scores for all players
            std_zscore = self.df['weighted_zscore'].std()
            # Calculate the z-score for the replacement player
            replacement_zscore = average_zscore - std_zscore
            # Calculate the total z-score for all players
            total_zscore = self.df['weighted_zscore'].sum()
            # Calculate the z-score for each player as a percentage of the total z-score
            self.df['zscore_percent'] = self.df['weighted_zscore'] / total_zscore
            # Calculate the dollar value for each player based on the total budget and z-score percentage
            self.df['auction_value'] = self.df['zscore_percent'] * total_budget


            output_file = 'auction_values.csv'
            self.df.to_csv(output_file, index=False)
        except Exception as e:
            print(f"Error calculating auction values: {e}")
            sys.exit(1)
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
    hitter_output_file = 'hitter_rankings.csv'
    pitcher_output_file = 'pitcher_rankings.csv'
    
    hitter_rankings = PlayerRankings(hitter_input_file, hitter_output_file, hitter_weights)
    hitter_rankings.run('hitter')
    
    pitcher_rankings = PlayerRankings(pitcher_input_file, pitcher_output_file, pitcher_weights)
    pitcher_rankings.run('pitcher')