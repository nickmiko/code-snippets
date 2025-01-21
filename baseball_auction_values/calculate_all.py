import zscore
import os
import pandas as pd
import sys
from rankings_zscore import RankingsZScore

class CalculateAllProjections:
    def __init__(self):
        self.hitter_input_file = ""
        self.pitcher_input_file = ""
        self.auction_values = []
        self.result = None

    def calculate_rankings(self):
        hitter_input = self.hitter_input_file
        pitcher_input = self.pitcher_input_file
        projection_system = os.path.basename(hitter_input).split('_')[0]
        output_file = f'{projection_system}_player_rankings.csv'
        calculator = zscore.PlayerRankings(hitter_input, pitcher_input, output_file)
        return calculator.run(), projection_system

    def process_all_files(self, folder_path):

        for file_name in os.listdir(folder_path):
            if file_name.endswith('_hitter.csv'):
                self.hitter_input_file = os.path.join(folder_path, file_name)
                projection_system = file_name.split('_')[0]
                pitcher_file_name = f'{projection_system}_pitcher.csv'
                self.pitcher_input_file = os.path.join(folder_path, pitcher_file_name)
                rankings, projection_system = self.calculate_rankings()
                rankings['dollar_value'] = rankings['dollar_value'].fillna(0)
                rankings['system'] = projection_system
                self.auction_values.append((projection_system, rankings))
        return self.auction_values
    def process_rankings_files(self):
        projections_folder = 'projections'

        for filename in os.listdir(projections_folder):
            if filename.endswith('_rankings.csv'):
                if filename.startswith('ibw'):
                    total_budget = 80
                    top_n = 100
                else:
                    total_budget = 52
                    top_n = 40
                rankings_path = os.path.join(projections_folder, filename)
                rankings_zscore = RankingsZScore(rankings_path, total_budget)
                df = rankings_zscore.generate_dollar_values(top_n=top_n)
                output_path = os.path.join('auction_values', filename)
                df.to_csv(output_path, index=False)
                projection_system = filename.split('_')[0]
                df['system'] = projection_system
                self.auction_values.append((projection_system, df))
        return self.auction_values
    def calculate_weighted_average(self, df):
        weights = {'depthcharts': 0.025, 'oopsy': 0.025, 'steamer': 0.05, 'atc': 0.5, 'pitcherlist' : 0.2, 'sporer' : 0.1, 'ibw' : 0.1}
        for col in weights.keys():
            if col not in df.columns:
                df[col] = 0
        df['weighted_average'] = sum(df[col] * weight for col, weight in weights.items())
        return df

    def save_results(self):
        all_auction_values = pd.concat([df for _, df in self.auction_values], ignore_index=True)
        all_auction_values = all_auction_values.pivot_table(index='Name', columns='system', values='dollar_value', aggfunc='sum', fill_value=000).reset_index()
        non_system_columns = [col for col in all_auction_values.columns if col != 'Name']
        all_auction_values['average'] = all_auction_values[non_system_columns].apply(lambda row: row[row > 0].mean(), axis=1)
        all_auction_values = self.calculate_weighted_average(all_auction_values)
        all_auction_values = all_auction_values.sort_values(by='weighted_average', ascending=False)
        output_dir = 'auction_values'
        os.makedirs(output_dir, exist_ok=True)
        all_auction_values.to_csv(f'{output_dir}/all_auction_values.csv', index=False)

if __name__ == "__main__":
    folder_path = 'projections'
    calculator = CalculateAllProjections()
    calculator.process_all_files(folder_path)
    calculator.process_rankings_files()
    calculator.save_results()

