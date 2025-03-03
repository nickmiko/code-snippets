import zscore
from rankings_zscore import RankingsZScore
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import logging
import os

def get_projection_weights() -> Dict[str, float]:
    """Factory function for projection system weights"""
    return {
        'depthcharts': 0.025, 
        'oopsy': 0.05, 
        'steamer': 0.025, 
        'thebat': 0.4, 
        'atc': 0.4,
        'pitcherlisting_rank': 0.05,
        'sarris_rank': 0.025,
        'sporer_rank': 0.025
    }

@dataclass
class ProjectionConfig:
    PROJECTION_WEIGHTS: Dict[str, float] = field(default_factory=get_projection_weights)
    DEFAULT_BUDGET: int = 35
    DEFAULT_ROSTER_SIZE: int = 10
    IBW_BUDGET: int = 60
    IBW_ROSTER_SIZE: int = 23
    HITTER_CATEGORIES: List[str] = field(default_factory=lambda: ['HR', 'R', 'RBI', 'SB', 'AVG'])
    PITCHER_CATEGORIES: List[str] = field(default_factory=lambda: ['W', 'SV', 'K', 'ERA', 'WHIP'])
    INVERSE_STATS: List[str] = field(default_factory=lambda: ['ERA', 'WHIP'])
    STRENGTH_THRESHOLD: float = 1.0  # Z-score threshold for strengths
    WEAKNESS_THRESHOLD: float = -0.5  # Z-score threshold for weaknesses

class CalculateAllProjections:
    def __init__(self):
        self.auction_values: List[Tuple[str, pd.DataFrame]] = []
        self.keeper_file = Path('keepers.csv')
        self.notes_file = Path('notes.csv')
        self.config = ProjectionConfig()
        self._setup_logging()
        self.hitter_zscores = {}
        self.pitcher_zscores = {}
        
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
        """Calculate rankings for a single projection system and store z-scores"""
        projection_system = hitter_path.stem.split('_')[0]
        output_file = f'{projection_system}_player_rankings.csv'
        
        self.logger.info(f"Calculating rankings for {projection_system}")
        
        # Create calculator instance
        calculator = zscore.PlayerRankings(
            str(hitter_path), 
            str(pitcher_path), 
            str(self.keeper_file), 
            output_file
        )
        
        # Process the data and get rankings
        rankings = calculator.run()
        
        # Store z-scores for later analysis
        self._calculate_and_store_zscores(hitter_path, pitcher_path, projection_system)
        
        return rankings, projection_system
    
    def _calculate_and_store_zscores(self, hitter_path: Path, pitcher_path: Path, system: str):
        """Calculate and store z-scores for each category for later analysis"""
        try:
            # Load hitter data and calculate z-scores
            hitters_df = pd.read_csv(hitter_path)
            if 'Name' in hitters_df.columns and not hitters_df.empty:
                hitter_zscores = {}
                for category in self.config.HITTER_CATEGORIES:
                    if category in hitters_df.columns:
                        # Convert to numeric and handle missing values
                        series = pd.to_numeric(hitters_df[category], errors='coerce')
                        
                        # Remove outliers (outside 3 standard deviations)
                        mean, std = series.mean(), series.std()
                        clean_series = series[np.abs(series - mean) <= 3 * std]
                        
                        # Calculate z-scores
                        if not clean_series.empty:
                            mean, std = clean_series.mean(), clean_series.std()
                            if std > 0:
                                zscores = (series - mean) / std
                                # Invert ERA and WHIP so negative is good
                                if category in self.config.INVERSE_STATS:
                                    zscores = -zscores
                                
                                # Store z-scores by player name
                                for idx, player in enumerate(hitters_df['Name']):
                                    if pd.notna(zscores.iloc[idx]):
                                        if player not in hitter_zscores:
                                            hitter_zscores[player] = {}
                                        hitter_zscores[player][category] = zscores.iloc[idx]
                
                # Store in the instance dictionary
                self.hitter_zscores[system] = hitter_zscores
            
            # Load pitcher data and calculate z-scores
            pitchers_df = pd.read_csv(pitcher_path)
            if 'Name' in pitchers_df.columns and not pitchers_df.empty:
                pitcher_zscores = {}
                for category in self.config.PITCHER_CATEGORIES:
                    if category in pitchers_df.columns:
                        # Convert to numeric and handle missing values
                        series = pd.to_numeric(pitchers_df[category], errors='coerce')
                        
                        # Remove outliers (outside 3 standard deviations)
                        mean, std = series.mean(), series.std()
                        clean_series = series[np.abs(series - mean) <= 3 * std]
                        
                        # Calculate z-scores
                        if not clean_series.empty:
                            mean, std = clean_series.mean(), clean_series.std()
                            if std > 0:
                                zscores = (series - mean) / std
                                # Invert ERA and WHIP so negative is good
                                if category in self.config.INVERSE_STATS:
                                    zscores = -zscores
                                
                                # Store z-scores by player name
                                for idx, player in enumerate(pitchers_df['Name']):
                                    if pd.notna(zscores.iloc[idx]):
                                        if player not in pitcher_zscores:
                                            pitcher_zscores[player] = {}
                                        pitcher_zscores[player][category] = zscores.iloc[idx]
                
                # Store in the instance dictionary
                self.pitcher_zscores[system] = pitcher_zscores
                
        except Exception as e:
            self.logger.error(f"Error calculating z-scores for {system}: {str(e)}")
    
    def determine_strengths_and_weaknesses(self, pivot_values: pd.DataFrame) -> pd.DataFrame:
        """Determine strengths and weaknesses for each player based on z-scores"""
        self.logger.info("Determining player strengths and weaknesses...")
        
        # Choose primary projection system for z-score analysis (atc or thebat)
        primary_system = 'atc' if 'atc' in self.hitter_zscores else 'thebat'
        
        if primary_system not in self.hitter_zscores and primary_system not in self.pitcher_zscores:
            self.logger.warning(f"No z-scores found for {primary_system}, skipping strength/weakness analysis")
            pivot_values['strengths'] = ''
            pivot_values['weaknesses'] = ''
            return pivot_values
            
        result_df = pivot_values.copy()
        result_df['strengths'] = ''
        result_df['weaknesses'] = ''
        
        # Process each player
        for idx, row in result_df.iterrows():
            player_name = row['Name']
            strengths = set()
            weaknesses = set()
            
            # Check hitter categories
            if player_name in self.hitter_zscores.get(primary_system, {}):
                player_zscores = self.hitter_zscores[primary_system][player_name]
                
                for category, zscore in player_zscores.items():
                    if zscore >= self.config.STRENGTH_THRESHOLD:
                        strengths.add(category)
                    elif zscore <= self.config.WEAKNESS_THRESHOLD:
                        weaknesses.add(category)
            
            # Check pitcher categories
            if player_name in self.pitcher_zscores.get(primary_system, {}):
                player_zscores = self.pitcher_zscores[primary_system][player_name]
                
                for category, zscore in player_zscores.items():
                    if zscore >= self.config.STRENGTH_THRESHOLD:
                        strengths.add(category)
                    elif zscore <= self.config.WEAKNESS_THRESHOLD:
                        weaknesses.add(category)
            
            # Update the dataframe with strengths and weaknesses
            result_df.at[idx, 'strengths'] = ', '.join(sorted(strengths)) if strengths else ''
            result_df.at[idx, 'weaknesses'] = ', '.join(sorted(weaknesses)) if weaknesses else ''
        
        return result_df

    def load_notes(self) -> pd.DataFrame:
        """Load notes from notes.csv if it exists"""
        try:
            if self.notes_file.exists():
                notes_df = pd.read_csv(self.notes_file)
                self.logger.info(f"Loaded {len(notes_df)} notes entries")
                return notes_df
            else:
                self.logger.warning(f"Notes file {self.notes_file} not found")
                return pd.DataFrame(columns=['Name', 'Target', 'Notes'])
        except Exception as e:
            self.logger.error(f"Error loading notes: {str(e)}")
            return pd.DataFrame(columns=['Name', 'Target', 'Notes'])

    def process_all_files(self, folder_path: str) -> None:
        """Process all projection files and rankings files"""
        # Process traditional projection files
        self._process_projection_files(folder_path)
        
        # Process rankings files - NEW!
        self._process_rankings_files(folder_path)

    def _process_projection_files(self, folder_path: str) -> None:
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

    def _process_rankings_files(self, folder_path: str) -> None:
        """Process all ranking files in the given folder"""
        folder = Path(folder_path)
        ranking_files = list(folder.glob('*_rankings.csv'))
        
        self.logger.info(f"Processing {len(ranking_files)} ranking files...")
        
        for ranking_file in ranking_files:
            # Extract system name from filename (e.g., "fangraphs_rankings.csv" -> "fangraphs")
            system_name = ranking_file.stem.split('_')[0]
            system_id = f"{system_name}_rank"  # Add suffix to distinguish from projections
            
            self.logger.info(f"Processing {system_name} rankings file")
            
            try:
                # Use the RankingsZScore class to convert rankings to dollar values
                calculator = RankingsZScore(
                    rankings_file=str(ranking_file),
                    total_budget=self.config.IBW_BUDGET,  # Using your specified budget
                    roster_size=self.config.IBW_ROSTER_SIZE,
                    num_teams=12,  # Standard league size
                    hitter_pct=0.67,  # 67% to hitters, 33% to pitchers
                    exponential_factor=0.9  # Controls value curve steepness
                )
                
                # Generate dollar values
                values_df = calculator.run()
                
                # Validate results
                if values_df is None or values_df.empty:
                    self.logger.warning(f"Empty dollar values for {system_name} rankings")
                    continue
                
                if 'Name' not in values_df.columns or 'dollar_value' not in values_df.columns:
                    self.logger.error(f"Required columns missing in {system_name} values")
                    continue
                
                # Ensure consistent format
                values_df['dollar_value'] = pd.to_numeric(values_df['dollar_value'], errors='coerce').fillna(0)
                
                # Add to auction values with the ranking-specific ID
                self.auction_values.append((system_id, values_df[['Name', 'dollar_value']]))
                self.logger.info(f"Added {system_id} to auction_values with {len(values_df)} players")
                
            except Exception as e:
                self.logger.exception(f"Error processing ranking file {ranking_file}: {str(e)}")
                continue
            
    def create_pivot_table(self) -> pd.DataFrame:
        """Create pivot table with better performance"""
        if not self.auction_values:
            raise ValueError("No auction values to process")
        
        # Extract columns with minimal memory usage
        data = []
        systems = []
        
        self.logger.info(f"Input shape: ({sum(len(df) for _, df in self.auction_values)}, 3)")
        self.logger.info(f"Unique names: {len(set(name for _, df in self.auction_values for name in df['Name']))}") 
        self.logger.info(f"Unique systems: {sorted(set(sys for sys, _ in self.auction_values))}")
        
        # Build data more efficiently
        for system, df in self.auction_values:
            systems.append(system)
            data.append(df[['Name', 'dollar_value']].assign(system=system))
        
        # Concatenate once
        all_values = pd.concat(data, ignore_index=True)
        
        # Create pivot with optimizations
        pivot = pd.pivot_table(
            all_values,
            index='Name',
            columns='system',
            values='dollar_value',
            aggfunc='first',
            fill_value=0
        ).reset_index()
        
        return pivot

    def calculate_weighted_average(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted averages excluding zero values for each player"""
        weights = pd.Series(self.config.PROJECTION_WEIGHTS)
        available_systems = weights.index.intersection(df.columns)
        
        if not available_systems.empty:
            self.logger.info(f"Calculating weighted averages with systems: {list(available_systems)}")
            
            # Create a copy to avoid SettingWithCopyWarning
            result_df = df.copy()
            result_df['projection_weighted_average'] = 0.0
            
            # Process each player individually to handle their specific non-zero projections
            for idx, row in result_df.iterrows():
                # Get systems with non-zero values for this player
                non_zero_systems = [sys for sys in available_systems if row[sys] > 0]
                
                if non_zero_systems:
                    # Get weights for non-zero systems and normalize them
                    player_weights = weights[non_zero_systems]
                    normalized_weights = player_weights / player_weights.sum()
                    
                    # Calculate weighted average using only non-zero systems
                    weighted_sum = sum(row[sys] * normalized_weights[sys] for sys in non_zero_systems)
                    result_df.at[idx, 'projection_weighted_average'] = weighted_sum
                else:
                    # If all projections are zero, keep the weighted average as zero
                    result_df.at[idx, 'projection_weighted_average'] = 0
            
            # Log some stats about the calculation
            non_zero_counts = (df[available_systems] > 0).sum()
            self.logger.info(f"System coverage (non-zero values): {non_zero_counts.to_dict()}")
            
            return result_df
        else:
            self.logger.warning("No projection systems found in data that match the configured weights")
            return df

    def save_results(self) -> None:
        """Save processed results to CSV with strengths, weaknesses, and notes"""
        # Create pivot table
        pivot_values = self.create_pivot_table()
        
        # Calculate weighted averages
        pivot_values = self.calculate_weighted_average(pivot_values)
        
        # Determine strengths and weaknesses
        pivot_values = self.determine_strengths_and_weaknesses(pivot_values)
        
        # Load notes and merge with pivot table
        notes_df = self.load_notes()
        if not notes_df.empty and 'Name' in notes_df.columns:
            # Process target column to make it consistent
            if 'Target' in notes_df.columns:
                notes_df['Target'] = notes_df['Target'].fillna('').apply(
                    lambda x: 'Target' if x and str(x).lower() == 'target' else '')
            else:
                notes_df['Target'] = ''
                
            # Process notes column
            if 'Notes' not in notes_df.columns:
                notes_df['Notes'] = ''
            else:
                notes_df['Notes'] = notes_df['Notes'].fillna('')
            
            # Merge with main data
            pivot_values = pd.merge(
                pivot_values,
                notes_df[['Name', 'Target', 'Notes']],
                on='Name',
                how='left'
            )
            
            # Fill missing values
            pivot_values['Target'] = pivot_values['Target'].fillna('')
            pivot_values['Notes'] = pivot_values['Notes'].fillna('')
        else:
            # Add empty columns if no notes
            pivot_values['Target'] = ''
            pivot_values['Notes'] = ''
        
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
        self.logger.info(f"Systems included: {[col for col in pivot_values.columns if col not in ['Name', 'projection_weighted_average', 'strengths', 'weaknesses', 'Target', 'Notes']]}")

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