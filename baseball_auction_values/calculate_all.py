import zscore
import os
import pandas as pd
import sys

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
                rankings['system'] = projection_system
                self.auction_values.append((projection_system, rankings))
        return self.auction_values
    def calculate_weighted_average(self, df):
        weights = {'depthcharts': 0.2, 'oopsy': 0.1, 'steamer': 0.7}
        df['weighted_average'] = (
            df['depthcharts'] * weights['depthcharts'] +
            df['oopsy'] * weights['oopsy'] +
            df['steamer'] * weights['steamer']
        )
        return df

    def save_results(self):
        all_auction_values = pd.concat([df for _, df in self.auction_values])
        all_auction_values = all_auction_values.pivot_table(index='Name', columns='system', values='dollar_value', aggfunc='sum').reset_index()
        non_system_columns = [col for col in all_auction_values.columns if col != 'Name']
        all_auction_values['average'] = all_auction_values[non_system_columns].mean(axis=1)
        all_auction_values = self.calculate_weighted_average(all_auction_values)
        all_auction_values = all_auction_values.sort_values(by='weighted_average', ascending=False)
        all_auction_values.to_csv('all_auction_values.csv', index=False)

if __name__ == "__main__":
    folder_path = 'projections'
    calculator = CalculateAllProjections()
    calculator.process_all_files(folder_path)
    calculator.save_results()

