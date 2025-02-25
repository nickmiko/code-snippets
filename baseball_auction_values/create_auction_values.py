import zscore
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
from rankings_zscore import RankingsZScore

@dataclass
class ProjectionConfig:
    PROJECTION_WEIGHTS: Dict[str, float] = {
        'depthcharts': 0.05, 
        'oopsy': 0.05, 
        'steamer': 0.05, 
        'thebat': 0.2, 
        'atc': 0.65
    }
    DEFAULT_BUDGET: int = 35
    DEFAULT_ROSTER_SIZE: int = 10
    IBW_BUDGET: int = 60
    IBW_ROSTER_SIZE: int = 23

class CalculateAllProjections:
    def __init__(self):
        self.auction_values: List[Tuple[str, pd.DataFrame]] = []
        self.keeper_file = Path('keepers.csv')
        self.config = ProjectionConfig()

    def calculate_rankings(self, hitter_path: Path, pitcher_path: Path) -> Tuple[pd.DataFrame, str]:
        """Calculate rankings for a single projection system"""
        projection_system = hitter_path.stem.split('_')[0]
        output_file = f'{projection_system}_player_rankings.csv'
        
        calculator = zscore.PlayerRankings(
            str(hitter_path), 
            str(pitcher_path), 
            str(self.keeper_file), 
            output_file
        )
        return calculator.run(), projection_system

    def process_all_files(self, folder_path: str) -> None:
        """Process all projection files in the given folder"""
        folder = Path(folder_path)
        hitter_files = list(folder.glob('*_hitter.csv'))
        
        for hitter_file in hitter_files:
            projection_system = hitter_file.stem.split('_')[0]
            pitcher_file = folder / f'{projection_system}_pitcher.csv'
            
            if not pitcher_file.exists():
                continue
                
            rankings, system = self.calculate_rankings(hitter_file, pitcher_file)
            if not rankings.empty:
                rankings['dollar_value'] = pd.to_numeric(rankings['dollar_value'], errors='coerce').fillna(0)
                self.auction_values.append((system, rankings))

    def calculate_weighted_average(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted averages using vectorized operations"""
        weights = pd.Series(self.config.PROJECTION_WEIGHTS)
        available_systems = weights.index.intersection(df.columns)
        
        if not available_systems.empty:
            normalized_weights = weights[available_systems] / weights[available_systems].sum()
            df['projection_weighted_average'] = df[available_systems].mul(normalized_weights).sum(axis=1)
            
        return df

    def save_results(self) -> None:
        """Save processed results to CSV"""
        if not self.auction_values:
            raise ValueError("No auction values to process")

        # Concatenate all DataFrames efficiently
        all_values = pd.concat(
            [df.assign(system=sys) for sys, df in self.auction_values],
            ignore_index=True
        )

        # Create pivot table with optimized settings
        pivot_values = pd.pivot_table(
            all_values,
            index='Name',
            columns='system',
            values='dollar_value',
            aggfunc='first',  # More efficient than 'sum' when values are unique
            fill_value=0
        ).reset_index()

        # Calculate weighted averages
        pivot_values = self.calculate_weighted_average(pivot_values)
        
        # Sort by weighted average
        if 'projection_weighted_average' in pivot_values.columns:
            pivot_values = pivot_values.sort_values(
                'projection_weighted_average', 
                ascending=False
            )

        # Save results efficiently
        output_dir = Path('auction_values')
        output_dir.mkdir(exist_ok=True)
        pivot_values.to_csv(
            output_dir / 'all_auction_values.csv',
            index=False,
            float_format='%.2f'  # Limit decimal places for better readability
        )

def main():
    folder_path = 'projections'
    calculator = CalculateAllProjections()
    calculator.process_all_files(folder_path)
    calculator.save_results()

if __name__ == "__main__":
    main()