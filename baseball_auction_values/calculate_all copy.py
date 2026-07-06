import zscore
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import logging

def get_projection_weights() -> Dict[str, float]:
    """Factory function for projection system weights"""
    return {
        'depthcharts': 0.05, 
        'oopsy': 0.05, 
        'steamer': 0.05, 
        'thebat': 0.2, 
        'atc': 0.65
    }

@dataclass
class ProjectionConfig:
    PROJECTION_WEIGHTS: Dict[str, float] = field(default_factory=get_projection_weights)
    DEFAULT_BUDGET: int = 35
    DEFAULT_ROSTER_SIZE: int = 10
    IBW_BUDGET: int = 60
    IBW_ROSTER_SIZE: int = 23

class CalculateAllProjections:
    def __init__(self):
        self.auction_values: List[Tuple[str, pd.DataFrame]] = []
        self.keeper_file = Path('keepers.csv')
        self.config = ProjectionConfig()
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('auction_calculations.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def calculate_rankings(self, hitter_path: Path, pitcher_path: Path) -> Tuple[pd.DataFrame, str]:
        """Calculate rankings for a single projection system"""
        projection_system = hitter_path.stem.split('_')[0]
        output_file = f'{projection_system}_player_rankings.csv'
        
        self.logger.info(f"Calculating rankings for {projection_system}")
        
        calculator = zscore.PlayerRankings(
            str(hitter_path), 
            str(pitcher_path), 
            str(self.keeper_file), 
            output_file
        )
        
        rankings = calculator.run()
        
        # Log rankings data
        self.logger.info(f"Rankings returned: {rankings.shape[0]} rows")
        
        return rankings, projection_system

    def process_all_files(self, folder_path: str) -> None:
        """Process all projection files in the given folder"""
        folder = Path(folder_path)
        hitter_files = list(folder.glob('*_hitter.csv'))
        
        self.logger.info(f"Processing {len(hitter_files)} projection systems...")
        
        for hitter_file in hitter_files:
            projection_system = hitter_file.stem.split('_')[0]
            pitcher_file = folder / f'{projection_system}_pitcher.csv'
            
            self.logger.info(f"Processing {projection_system}")
            
            # Validate input files
            if not hitter_file.exists():
                self.logger.error(f"Error: Hitter file {hitter_file} not found")
                continue
            
            if not pitcher_file.exists():
                self.logger.error(f"Error: Pitcher file {pitcher_file} not found")
                continue
            
            try:
                # Validate hitter data
                hitter_df = pd.read_csv(hitter_file)
                if 'Name' not in hitter_df.columns:
                    self.logger.error(f"Error: Name column missing in {hitter_file}")
                    continue
                    
                rankings, system = self.calculate_rankings(hitter_file, pitcher_file)
                if rankings is None or rankings.empty:
                    self.logger.warning(f"Empty rankings for {system}")
                    continue
                    
                if 'Name' not in rankings.columns or 'dollar_value' not in rankings.columns:
                    self.logger.error(f"Error: Required columns missing in {system} rankings")
                    continue
                        
                self.logger.info(f"Rankings processed: {len(rankings)} rows")
                rankings['dollar_value'] = pd.to_numeric(rankings['dollar_value'], errors='coerce').fillna(0)

                self.auction_values.append((system, rankings))
                self.logger.info(f"Added {system} to auction_values")
                    
            except Exception as e:
                self.logger.exception(f"Error processing {projection_system}: {str(e)}")
                continue

    def create_pivot_table(self) -> pd.DataFrame:
        """Create a pivot table from the auction values"""
        if not self.auction_values:
            raise ValueError("No auction values to process")
        
        # Step 1: Extract columns efficiently to reduce memory usage
        all_values = pd.concat(
            [df.assign(system=sys)[['Name', 'system', 'dollar_value']] 
            for sys, df in self.auction_values],
            ignore_index=True
        )
        
        # Step 2: Create pivot table with optimized settings
        self.logger.info(f"Creating pivot table from {len(all_values)} rows")
        pivot_values = pd.pivot_table(
            all_values,
            index='Name',
            columns='system',
            values='dollar_value',
            aggfunc='first',
            fill_value=0
        ).reset_index()
        
        self.logger.info(f"Pivot table created with {len(pivot_values)} players")
        return pivot_values

    def calculate_weighted_average(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted averages using vectorized operations"""
        weights = pd.Series(self.config.PROJECTION_WEIGHTS)
        available_systems = weights.index.intersection(df.columns)
        
        if not available_systems.empty:
            normalized_weights = weights[available_systems] / weights[available_systems].sum()
            self.logger.info(f"Using normalized weights: {normalized_weights.to_dict()}")
            
            df['projection_weighted_average'] = df[available_systems].mul(normalized_weights).sum(axis=1)
        else:
            self.logger.warning("No projection systems found in data that match the configured weights")
            
        return df

    def save_results(self) -> None:
        """Save processed results to CSV"""
        # Create pivot table
        pivot_values = self.create_pivot_table()
        
        # Calculate weighted averages
        pivot_values = self.calculate_weighted_average(pivot_values)
        
        # Sort by weighted average
        if 'projection_weighted_average' in pivot_values.columns:
            pivot_values = pivot_values.sort_values('projection_weighted_average', ascending=False)
        
        # Save results
        output_dir = Path('auction_values')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'all_auction_values.csv'
        
        pivot_values.to_csv(output_path, index=False, float_format='%.2f')
        
        self.logger.info(f"Output saved to {output_path}")
        self.logger.info(f"Total players: {len(pivot_values)}")
        self.logger.info(f"Systems included: {[col for col in pivot_values.columns if col not in ['Name', 'projection_weighted_average']]}")

def main():
    try:
        folder_path = 'projections'
        calculator = CalculateAllProjections()
        calculator.process_all_files(folder_path)
        calculator.save_results()
    except Exception as e:
        logging.exception(f"Fatal error in main: {str(e)}")

if __name__ == "__main__":
    main()