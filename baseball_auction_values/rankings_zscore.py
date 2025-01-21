import pandas as pd
import numpy as np
import os

class RankingsZScore:
    def __init__(self, rankings, total_budget):
        self.rankings = rankings
        self.total_budget = total_budget

    def generate_dollar_values(self, top_n=40):
        # Read the rankings from the CSV file
        df = pd.read_csv(self.rankings, header=None, names=['Name'])
        
        # Initialize dollar values
        dollar_values = []
        
        # Define the budget distribution
        top_n_budget = self.total_budget * 0.7  # 70% of the budget for top N
        remaining_budget = self.total_budget * 0.3  # 30% of the budget for the rest
        
        # Calculate dollar values for top N using linear distribution
        top_n_values = np.linspace(top_n_budget / top_n, top_n_budget, top_n)[::-1]
        dollar_values.extend(top_n_values)
        
        # Calculate dollar values for the rest using flat distribution
        remaining_value = np.ceil(remaining_budget / (len(df) - top_n))
        dollar_values.extend([remaining_value] * (len(df) - top_n))
        
        # Add dollar values to the DataFrame
        df['dollar_value'] = dollar_values
        
        return df

# Example usage
if __name__ == "__main__":
    projections_folder = 'projections'
    total_budget = 80  # Example total budget

    for filename in os.listdir(projections_folder):
        if filename.endswith('_rankings.csv'):
            if filename.startswith('ibw'):
                total_budget = 260
                top_n = 100
            else:
                total_budget = 80
                top_n = 40
            rankings_path = os.path.join(projections_folder, filename)
            zscore = RankingsZScore(rankings_path, total_budget)
            df = zscore.generate_dollar_values(top_n=top_n)
            output_path = os.path.join('auction_values', filename)
            df.to_csv(output_path, index=False)
