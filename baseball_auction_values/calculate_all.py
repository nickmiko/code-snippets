import zscore
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

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

    def calculate_rankings(self, hitter_path: Path, pitcher_path: Path) -> Tuple[pd.DataFrame, str]:
        """Calculate rankings for a single projection system"""
        projection_system = hitter_path.stem.split('_')[0]
        output_file = f'{projection_system}_player_rankings.csv'
        
        print(f"\nCalculating rankings for {projection_system}")
        
        calculator = zscore.PlayerRankings(
            str(hitter_path), 
            str(pitcher_path), 
            str(self.keeper_file), 
            output_file
        )
        
        rankings = calculator.run()
        
        # Debug rankings data
        print(f"Rankings returned: {rankings.shape} rows")
        print(f"Columns in rankings: {rankings.columns.tolist()}")
        print(f"First few names: {rankings['Name'].head().tolist()}")
        
        return rankings, projection_system

    def process_all_files(self, folder_path: str) -> None:
        """Process all projection files in the given folder"""
        folder = Path(folder_path)
        hitter_files = list(folder.glob('*_hitter.csv'))
        
        print(f"\nProcessing {len(hitter_files)} projection systems...")
        
        for hitter_file in hitter_files:
            projection_system = hitter_file.stem.split('_')[0]
            pitcher_file = folder / f'{projection_system}_pitcher.csv'
            
            print(f"\nProcessing {projection_system}:")
            print(f"Hitter file: {hitter_file}")
            print(f"Pitcher file: {pitcher_file}")
            
            # Validate input files
            if not hitter_file.exists():
                print(f"Error: Hitter file {hitter_file} not found")
                continue
            
            if not pitcher_file.exists():
                print(f"Error: Pitcher file {pitcher_file} not found")
                continue
            
            try:
                # Load and check hitter data
                hitter_df = pd.read_csv(hitter_file)
                print(f"Hitter file loaded: {len(hitter_df)} rows")
                if 'Name' not in hitter_df.columns:
                    print(f"Error: Name column missing in {hitter_file}")
                    continue
                    
                rankings, system = self.calculate_rankings(hitter_file, pitcher_file)
                if not rankings.empty:
                    if 'Name' not in rankings.columns or 'dollar_value' not in rankings.columns:
                        print(f"Error: Required columns missing in {system} rankings")
                        continue
                        
                    print(f"Rankings processed: {len(rankings)} rows")
                    rankings['dollar_value'] = pd.to_numeric(rankings['dollar_value'], errors='coerce').fillna(0)

                    self.auction_values.append((system, rankings))
                    print(f"Added {system} to auction_values")
                else:
                    print(f"Empty rankings for {system}")
                    
            except Exception as e:
                print(f"Error processing {projection_system}: {str(e)}")
                continue

    def calculate_weighted_average(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted averages using vectorized operations"""
        weights = pd.Series(self.config.PROJECTION_WEIGHTS)
        available_systems = weights.index.intersection(df.columns)
        
        if not available_systems.empty:
            normalized_weights = weights[available_systems] / weights[available_systems].sum()
            df['projection_weighted_average'] = df[available_systems].mul(normalized_weights).sum(axis=1)
            
        return df

    def save_results(self) -> None:
        """Save processed results to CSV with enhanced debugging"""
        if not self.auction_values:
            raise ValueError("No auction values to process")

        print("\n=== Data Flow Debugging ===")
        # Step 2: Concatenate with system column
        all_values = pd.concat(
            [df.assign(system=sys)[['Name', 'system', 'dollar_value']] 
            for sys, df in self.auction_values],
            ignore_index=True
        )
        
        # Step 3: Create pivot table with debugging
        print("\n3. Pivot Table Creation:")
        print(f"Input shape: {all_values.shape}")
        print(f"Unique names: {len(all_values['Name'].unique())}")
        print(f"Unique systems: {all_values['system'].unique().tolist()}")
        
        pivot_values = pd.pivot_table(
            all_values,
            index='Name',
            columns='system',
            values='dollar_value',
            aggfunc='first',
            fill_value=0
        ).reset_index()
        
        # Step 4: Calculate weighted averages
        weights = pd.Series(self.config.PROJECTION_WEIGHTS)
        available_systems = weights.index.intersection(pivot_values.columns)
        
        if not available_systems.empty:
            normalized_weights = weights[available_systems] / weights[available_systems].sum()
            print("\n4. Weight Calculation:")
            print(f"Available systems: {available_systems.tolist()}")
            print(f"Normalized weights: {normalized_weights.to_dict()}")
            
            pivot_values['projection_weighted_average'] = (
                pivot_values[available_systems].mul(normalized_weights).sum(axis=1)
            )
        # Save results
        output_dir = Path('auction_values')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'all_auction_values.csv'
        
        # Sort by weighted average before saving
        if 'projection_weighted_average' in pivot_values.columns:
            pivot_values = pivot_values.sort_values('projection_weighted_average', ascending=False)
        
        pivot_values.to_csv(output_path, index=False, float_format='%.2f')
        
        print(f"\nOutput Summary:")
        print(f"Total rows saved: {len(pivot_values)}")
        print(f"Total columns: {len(pivot_values.columns)}")
        print(f"Systems included: {[col for col in pivot_values.columns if col not in ['Name', 'projection_weighted_average']]}")

def main():
    folder_path = 'projections'
    calculator = CalculateAllProjections()
    calculator.process_all_files(folder_path)
    calculator.save_results()

if __name__ == "__main__":
    main()