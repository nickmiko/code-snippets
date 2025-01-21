import pandas as pd
import numpy as np
import sys

class RankingsZScore:
    def __init__(self, rankings_file, total_budget, roster_size=23, num_teams=12):
        self.rankings_file = rankings_file
        self.total_budget = total_budget
        self.roster_size = roster_size
        self.num_teams = num_teams

    def load_data(self):
        """Load rankings data from CSV file"""
        try:
            df = pd.read_csv(self.rankings_file)
        except FileNotFoundError:
            print("Error: Rankings CSV file not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading data: {e}")
            sys.exit(1)
        return df

    def calculate_dollar_values(self, df):
        """Calculate dollar values based on player rankings"""
        total_players = self.roster_size * self.num_teams
        df = df.head(total_players)  # Limit to the top N players

        # Assign dollar values based on rank
        df = df.copy()
        df.loc[:, 'rank'] = np.arange(1, len(df) + 1)
        df.loc[:, 'dollar_value'] = (self.total_budget / total_players) * (total_players - df['rank'] + 1)
        df.loc[:, 'dollar_value'] = df['dollar_value'].round(2)

        # Include player names in the returned DataFrame
        df = df[['Name', 'dollar_value']]

        return df

    def run(self):
        """Run the full ranking process"""
        df = self.load_data()
        df = self.calculate_dollar_values(df)
        return df

# Example usage
if __name__ == "__main__":
    rankings_file = 'ibw_rankings.csv'
    total_budget = 260

    zscore = RankingsZScore(rankings_file, total_budget)
    df = zscore.run()
    df.to_csv('calculated_auction_values.csv', index=False)