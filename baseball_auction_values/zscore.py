import pandas as pd
import sys
import os

class PlayerRankings:
    def __init__(self, hitter_input_file: str, pitcher_input_file: str, keeper_file:str, output_file: str):
        self.hitter_input_file = hitter_input_file
        self.pitcher_input_file = pitcher_input_file
        self.output_file = output_file
        self.keeper_file = keeper_file
        self.df = None
        self.keepers = None

    def read_data(self, input_file):
        """Read CSV data into DataFrame"""
        try:
            return pd.read_csv(input_file)
        except FileNotFoundError:
            print("Error: Input CSV file not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading data: {e}")
            sys.exit(1)
    def load_data(self):
        """Load hitter and pitcher data from CSV files"""
        try:
            hitters_df = self.read_data(self.hitter_input_file)
            pitchers_df = self.read_data(self.pitcher_input_file)
        except FileNotFoundError:
            print("Error: Input CSV file not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading data: {e}")
            sys.exit(1)
        return hitters_df, pitchers_df

    def calculate_auction_values(self, hitters_df, pitchers_df, keeper_df, budget=260, roster_size=20, num_teams=12, 
                            hitter_budget_pct=0.675,
                            hitter_weights={'HR': 1.25, 'R': 1.0, 'RBI': 1.0, 'SB': 0.75, 'BB/K': 1.5, 'OBP': 1.5, 'PA': 1.25},
                            pitcher_weights={'QS': 1.25, 'SV': 0.25, 'HLD': 0.25, 'K': 1.0, 'K-BB%' : 1.5, 'ERA': 1.0, 'WHIP': 1.0, 'IP': 1.25}):
        """
        Calculate auction values based on weighted z-scores with separate hitter/pitcher inputs.
        
        Parameters:
        hitters_df: DataFrame with hitter statistics
        pitchers_df: DataFrame with pitcher statistics
        keeper_df: DataFrame with keeper players
        budget: Total budget per team (default $260)
        roster_size: Number of players per team (default 23)
        num_teams: Number of teams in league (default 12)
        hitter_budget_pct: Percentage of budget allocated to hitters (default 0.80)
        hitter_weights: Dictionary of weights for hitting categories
        pitcher_weights: Dictionary of weights for pitching categories
        
        Returns:
        tuple: (hitters_df, pitchers_df) with calculated values
        """
        # Ensure hitters_df and pitchers_df are defined
        if hitters_df is None or pitchers_df is None:
            raise ValueError("hitters_df and pitchers_df must be provided")

        # Remove keepers from hitters and pitchers dataframes
        if keeper_df is not None:
            hitters_df = hitters_df.loc[~hitters_df['Name'].isin(keeper_df['Name'])].copy()
            pitchers_df = pitchers_df.loc[~pitchers_df['Name'].isin(keeper_df['Name'])].copy()

        # Calculate total league budget and split between hitters/pitchers
        total_league_budget = budget * num_teams
        hitter_budget = total_league_budget * hitter_budget_pct
        pitcher_budget = total_league_budget * (1 - hitter_budget_pct)
        
        # Typical roster construction (can be adjusted)
        hitter_roster_size = int(roster_size * 0.5)  # Usually 14 hitters
        pitcher_roster_size = roster_size - hitter_roster_size  # Usually 9 pitchers
        
        # Calculate total rostered players by type
        total_rostered_hitters = hitter_roster_size * num_teams
        total_rostered_pitchers = pitcher_roster_size * num_teams
        
        # Define category lists
        hitting_stats = list(hitter_weights.keys())
        pitching_stats = list(pitcher_weights.keys())
        
        # Calculate weighted z-scores for hitters
        for stat in hitting_stats:
            if stat in hitters_df.columns:
                hitters_df[f'{stat}_z'] = (
                    ((hitters_df[stat] - hitters_df[stat].mean()) / hitters_df[stat].std())
                    * hitter_weights[stat]
                )
        
        # Calculate weighted z-scores for pitchers
        for stat in pitching_stats:
            if stat in pitchers_df.columns:
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
        hitters_df['total_z'] = hitters_df[[f'{stat}_z' for stat in hitting_stats if f'{stat}_z' in hitters_df.columns]].sum(axis=1)
        pitchers_df['total_z'] = pitchers_df[[f'{stat}_z' for stat in pitching_stats if f'{stat}_z' in pitchers_df.columns]].sum(axis=1)
        
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
    def combine_data(self, hitters_df, pitchers_df):
        """Combine hitter and pitcher data into single DataFrame"""
        hitters_df['Position'] = 'H'
        pitchers_df['Position'] = 'P'
        combined_df = pd.concat([hitters_df, pitchers_df], ignore_index=True)
        return combined_df

    def save_results(self, combined_results):
        """Save the ranked players to a CSV file"""
        try:
            output_dir = 'auction_values'
            combined_results.to_csv(os.path.join(output_dir, self.output_file), index=False)
        except Exception as e:
            print(f"Error saving results: {e}")
            sys.exit(1)

    def run(self):
        """Run the full ranking process"""
        hitters_df, pitchers_df = self.load_data()
        keepers = self.read_data(self.keeper_file)
        hitters_df, pitchers_df = self.calculate_auction_values(hitters_df, pitchers_df, keepers)
        combined_results = self.combine_data(hitters_df, pitchers_df)
        self.df = combined_results
        # self.save_results(combined_results)
        if 'Name' in self.df.columns:
            return self.df[['Name', 'dollar_value']]
        else:
            return self.df[['dollar_value']]
if __name__ == "__main__":
    hitter_input_file = 'fangraphs-leaderboard-projections_hitters.csv'
    pitcher_input_file = 'fangraphs-leaderboard-projections_pitchers.csv'
    output_file = 'player_rankings.csv'
    keeper_file = 'keepers.csv'
    rankings = PlayerRankings(hitter_input_file, pitcher_input_file, keeper_file, output_file)
    results = rankings.run()