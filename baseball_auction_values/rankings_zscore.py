import pandas as pd
import numpy as np

class RankingsZScore:
    def __init__(self, rankings, total_budget):
        self.rankings = rankings
        self.total_budget = total_budget

    def generate_dollar_values(self):
        # Read the rankings from the CSV file
        df = pd.read_csv(self.rankings, header=None, names=['Name'])
        
        # Initialize dollar values
        dollar_values = []
        
        # Define the budget distribution
        top_30_budget = self.total_budget * 0.7  # 70% of the budget for top 30
        remaining_budget = self.total_budget * 0.3  # 30% of the budget for the rest
        
        # Calculate dollar values for top 30 using linear distribution
        top_30_values = np.linspace(top_30_budget / 30, top_30_budget, 30)[::-1]
        dollar_values.extend(top_30_values)
        
        # Calculate dollar values for the rest using flat distribution
        remaining_value = remaining_budget / (len(df) - 30)
        dollar_values.extend([remaining_value] * (len(df) - 30))
        
        # Add dollar values to the DataFrame
        df['dollar_value'] = dollar_values
        
        return df

# Example usage
if __name__ == "__main__":
    pitcherlist_rankings = '/home/shmick/Documents/repos/code-snippets/baseball_auction_values/projections/pitcherlist_rankings.csv'
    sporer_rankings = '/home/shmick/Documents/repos/code-snippets/baseball_auction_values/projections/sporer_rankings.csv'
    
    total_budget = 80  # Example total budget
    
    pitcherlist_zscore = RankingsZScore(pitcherlist_rankings, total_budget)
    sporer_zscore = RankingsZScore(sporer_rankings, total_budget)
    
    pitcherlist_df = pitcherlist_zscore.generate_dollar_values()
    sporer_df = sporer_zscore.generate_dollar_values()

    # Save the DataFrames to new CSV files
    pitcherlist_df.to_csv('/home/shmick/Documents/repos/code-snippets/baseball_auction_values/projections/pitcherlist_dollar_values.csv', index=False)
    sporer_df.to_csv('/home/shmick/Documents/repos/code-snippets/baseball_auction_values/projections/sporer_dollar_values.csv', index=False)