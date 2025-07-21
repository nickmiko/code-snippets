import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass, field

def get_hitter_weights() -> Dict[str, float]:
    return {'HR': 1.0, 'R': 1.0, 'RBI': 1.0, 'SB': 0.75, 'OBP': 1.25}

def get_pitcher_weights() -> Dict[str, float]:
    return {'QS': 1.0, 'SV': 0.75, 'HLD': 1.0, 'K': 1.25, 'ERA': 1.0, 'WHIP': 1.0}

@dataclass
class LeagueSettings:
    BUDGET: int = 260
    ROSTER_SIZE: int = 20
    NUM_TEAMS: int = 12
    HITTER_BUDGET_PCT: float = 0.675
    HITTER_ROSTER_PCT: float = 0.5
    HITTER_WEIGHTS: Dict[str, float] = field(default_factory=get_hitter_weights)
    PITCHER_WEIGHTS: Dict[str, float] = field(default_factory=get_pitcher_weights)
    INVERSE_STATS: tuple = ('ERA', 'WHIP')

class PlayerRankings:
    def __init__(self, hitter_input_file: str, pitcher_input_file: str, keeper_file: str, output_file: str):
        self.hitter_input_file = Path(hitter_input_file)
        self.pitcher_input_file = Path(pitcher_input_file)
        self.keeper_file = Path(keeper_file)
        self.output_file = Path(output_file)
        self.settings = LeagueSettings()
        self._setup_logging()

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def read_data(self, input_file: Path) -> pd.DataFrame:
        """Read CSV data with optimized settings and column name normalization"""
        try:
            df = pd.read_csv(
                input_file,
                low_memory=False,
                dtype={'Name': str}
            )
            
            # Normalize column names for keepers file
            if input_file.name == 'keepers.csv':
                # Map possible variations of dollar value column names
                dollar_cols = ['dollar_value', 'Dollar_Value', 'DollarValue', 'Value', 'Cost']
                for col in df.columns:
                    if col.lower().replace('_', '') in [c.lower().replace('_', '') for c in dollar_cols]:
                        df = df.rename(columns={col: 'dollar_value'})
                        break
                
                if 'dollar_value' not in df.columns:
                    logging.error(f"No dollar value column found in {input_file}")
                    # Create dollar_value column with default values if missing
                    df['dollar_value'] = 0
                    
            return df
            
        except Exception as e:
            logging.error(f"Error reading {input_file}: {e}")
            raise

    # Improved zscore.py calculation method
    def calculate_zscores(self, df: pd.DataFrame, weights: Dict[str, float], inverse_stats: tuple) -> pd.Series:
        """Calculate z-scores with vectorized operations for better performance"""
        # Validate input early
        if df.empty:
            return pd.Series(0, index=df.index)
            
        # Determine valid stats once (avoid repeated lookups)
        valid_stats = [stat for stat in weights if stat in df.columns]
        if not valid_stats:
            logging.warning(f"No valid stats found in columns: {df.columns.tolist()}")
            return pd.Series(0, index=df.index)
        
        # Pre-compute normalized weights (avoid repeated division)
        weight_sum = sum(weights[stat] for stat in valid_stats)
        adjusted_weights = {stat: (weights[stat] / weight_sum) for stat in valid_stats} if weight_sum > 0 else {stat: 0 for stat in valid_stats}
        
        # Use numpy arrays for faster computation
        z_scores = np.zeros((len(df), len(valid_stats)))
        
        for i, stat in enumerate(valid_stats):
            series = pd.to_numeric(df[stat], errors='coerce')
            
            # Handle missing values efficiently
            if series.isna().all():
                continue
                
            # Vectorized outlier removal
            q1, q3 = np.nanpercentile(series, [25, 75])
            iqr = q3 - q1
            bounds = (q1 - 3 * iqr, q3 + 3 * iqr)
            filtered_values = series[(series >= bounds[0]) & (series <= bounds[1])]
            
            if len(filtered_values) < 2:
                continue
                
            # Robust z-score calculation
            median = np.nanmedian(filtered_values)
            mad = np.nanmedian(np.abs(filtered_values - median)) * 1.4826
            
            if mad == 0:
                mad = np.nanstd(filtered_values)
                if mad == 0:
                    continue
            
            # Apply z-score with single vectorized operation
            multiplier = -1 if stat in inverse_stats else 1
            z = multiplier * (series - median) / mad * adjusted_weights[stat]
            z_scores[:, i] = np.clip(z, -3, 3)
        
        # Sum along rows for total z-score
        return pd.Series(np.nansum(z_scores, axis=1), index=df.index)

    # Improved dollar value calculation in process_player_group
    def process_player_group(self, df: pd.DataFrame, roster_spots: int, budget: float) -> pd.DataFrame:
        """Process players with improved value calculation algorithm"""
        if df.empty or roster_spots <= 0:
            return pd.DataFrame(columns=['Name', 'dollar_value'])
        
        # One-time name normalization
        df = df.copy()
        df['Name_norm'] = df['Name'].str.lower()
        
        # Determine statistics type
        is_hitter = 'HR' in df.columns
        weights = self.settings.HITTER_WEIGHTS if is_hitter else self.settings.PITCHER_WEIGHTS
        inverse_stats = () if is_hitter else self.settings.INVERSE_STATS
        
        # Calculate z-scores efficiently
        df['total_z'] = self.calculate_zscores(df, weights, inverse_stats)
        
        # Sort once and create ranks (avoid repeated sorting)
        df.sort_values('total_z', ascending=False, inplace=True)
        df['rank'] = np.arange(1, len(df) + 1)
        
        # Use vectorized calculation for replacement level
        num_rostered = min(roster_spots, len(df))
        replacement_level = df.iloc[num_rostered-1]['total_z'] if num_rostered > 0 else 0
        
        # Vectorized calculation of value above replacement
        df['value_above_replacement'] = np.maximum(df['total_z'] - replacement_level, 0)
        
        # More accurate dollar value calculation that handles edge cases
        total_var = df['value_above_replacement'].sum()
        min_value = 1.0  # Minimum player value
        
        if total_var > 0:
            available_budget = budget - (num_rostered * min_value)
            if available_budget <= 0:
                df['dollar_value'] = min_value
            else:
                # Distributes budget proportionally with minimum value guarantee
                df['dollar_value'] = (df['value_above_replacement'] * (available_budget / total_var) + min_value).round(1)
        else:
            df['dollar_value'] = min_value
        
        # Return only required columns to save memory
        return df[['Name', 'Name_norm', 'dollar_value', 'rank']].copy()

# Improved keeper handling in calculate_auction_values
    def calculate_auction_values(self, hitters_df: pd.DataFrame, pitchers_df: pd.DataFrame, 
                            keeper_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate auction values with better data integrity"""
        # Normalize names consistently with vectorized operations
        hitters_df = hitters_df.copy()
        pitchers_df = pitchers_df.copy()
        
        # One-time normalization
        hitters_df['Name_norm'] = hitters_df['Name'].str.lower()
        pitchers_df['Name_norm'] = pitchers_df['Name'].str.lower()
        
        # Process keepers with better validation
        keeper_budget = 0
        keeper_hitters = pd.DataFrame(columns=['Name', 'Name_norm', 'dollar_value', 'Position', 'is_keeper'])
        keeper_pitchers = pd.DataFrame(columns=['Name', 'Name_norm', 'dollar_value', 'Position', 'is_keeper'])
        
        if keeper_df is not None and not keeper_df.empty:
            keeper_df = keeper_df.copy()
            
            # Normalize keeper names the same way
            keeper_df['Name_norm'] = keeper_df['Name'].str.lower()
            
            # Better column standardization
            dollar_cols = ['dollar_value', 'value', 'cost']
            for col in keeper_df.columns:
                if col.lower() in dollar_cols:
                    keeper_df = keeper_df.rename(columns={col: 'dollar_value'})
                    break
            
            if 'dollar_value' not in keeper_df.columns:
                logging.warning("No dollar_value column in keepers, creating with default values")
                keeper_df['dollar_value'] = 0
                
            # Safely convert to numeric
            keeper_df['dollar_value'] = pd.to_numeric(keeper_df['dollar_value'], errors='coerce').fillna(0)
            keeper_budget = keeper_df['dollar_value'].sum()
            
            # Use efficient lookup with sets for better performance
            hitter_names = set(hitters_df['Name_norm'])
            pitcher_names = set(pitchers_df['Name_norm'])
            
            # Process keepers with proper position assignment
            for _, keeper in keeper_df.iterrows():
                name_norm = keeper['Name_norm']
                
                # Create keeper row with all required fields
                keeper_row = pd.DataFrame({
                    'Name': [keeper['Name']],
                    'Name_norm': [name_norm],
                    'dollar_value': [keeper['dollar_value']],
                    'Position': ['Unknown'],
                    'is_keeper': [True]
                })
                
                # Add to appropriate position group
                if name_norm in hitter_names:
                    keeper_row['Position'] = 'H'
                    keeper_hitters = pd.concat([keeper_hitters, keeper_row])
                elif name_norm in pitcher_names:
                    keeper_row['Position'] = 'P'
                    keeper_pitchers = pd.concat([keeper_pitchers, keeper_row])
                else:
                    logging.warning(f"Keeper {keeper['Name']} not found in player data")
            
            # Remove keepers efficiently (use Series.isin with the set)
            keeper_names = set(keeper_df['Name_norm'])
            hitters_df = hitters_df[~hitters_df['Name_norm'].isin(keeper_names)]
            pitchers_df = pitchers_df[~pitchers_df['Name_norm'].isin(keeper_names)]
        
        # Calculate budgets
        total_budget = self.settings.BUDGET * self.settings.NUM_TEAMS - keeper_budget
        hitter_budget = total_budget * self.settings.HITTER_BUDGET_PCT
        pitcher_budget = total_budget * (1 - self.settings.HITTER_BUDGET_PCT)
        
        # Calculate roster spots
        hitter_roster_size = int(self.settings.ROSTER_SIZE * self.settings.HITTER_ROSTER_PCT)
        pitcher_roster_size = self.settings.ROSTER_SIZE - hitter_roster_size
        total_hitters = (hitter_roster_size * self.settings.NUM_TEAMS) - len(keeper_hitters)
        total_pitchers = (pitcher_roster_size * self.settings.NUM_TEAMS) - len(keeper_pitchers)
        
        # Process players
        hitter_values = self.process_player_group(hitters_df, total_hitters, hitter_budget)
        pitcher_values = self.process_player_group(pitchers_df, total_pitchers, pitcher_budget)
        
        # Ensure consistent columns
        required_cols = ['Name', 'Name_norm', 'dollar_value']
        for df in [hitter_values, pitcher_values]:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logging.error(f"Missing columns after processing: {missing}")
                return pd.DataFrame(columns=required_cols), pd.DataFrame(columns=required_cols)
        
        # Add position and keeper flags
        hitter_values['Position'] = 'H'
        pitcher_values['Position'] = 'P'
        hitter_values['is_keeper'] = False
        pitcher_values['is_keeper'] = False
        
        # Combine with keepers
        final_hitters = pd.concat([keeper_hitters, hitter_values], ignore_index=True)
        final_pitchers = pd.concat([keeper_pitchers, pitcher_values], ignore_index=True)
        
        # Validate results
        for name, df in [('hitters', final_hitters), ('pitchers', final_pitchers)]:
            if df.empty:
                logging.error(f"No {name} data after processing")
            else:
                logging.info(f"Processed {len(df)} {name}, max value: ${df['dollar_value'].max():.1f}")
        
        return final_hitters, final_pitchers

    # Improved combine_data method to reduce memory usage
    def combine_data(self, hitters_df: pd.DataFrame, pitchers_df: pd.DataFrame) -> pd.DataFrame:
        """Combine hitter and pitcher data with reduced memory usage"""
        logging.info(f"Combining data - Hitters: {len(hitters_df)} rows, Pitchers: {len(pitchers_df)} rows")
        
        # Only keep necessary columns
        required_cols = ['Name', 'Name_norm', 'dollar_value', 'Position', 'is_keeper']
        
        # Use DataFrame.loc for efficient column selection
        hitters = hitters_df.loc[:, [col for col in required_cols if col in hitters_df.columns]]
        pitchers = pitchers_df.loc[:, [col for col in required_cols if col in pitchers_df.columns]]
        
        # Add missing columns efficiently
        for df in [hitters, pitchers]:
            if 'Name_norm' not in df.columns and 'Name' in df.columns:
                df['Name_norm'] = df['Name'].str.lower()
        
        # Use pd.concat once with optimized parameters
        combined = pd.concat([hitters, pitchers], ignore_index=True, sort=False)
        
        # Use inplace sorting when possible
        combined.sort_values('dollar_value', ascending=False, inplace=True)
        combined.reset_index(drop=True, inplace=True)
        
        return combined
    
    def save_results(self, df: pd.DataFrame) -> None:
        """Save results efficiently"""
        output_dir = Path('auction_values')
        output_dir.mkdir(exist_ok=True)
        df.to_csv(
            output_dir / self.output_file,
            index=False,
            float_format='%.1f'
        )
    
    # Improved validation in run method
    def run(self) -> pd.DataFrame:
        """Execute full process with better validation and error handling"""
        try:
            # Validate input files
            for file_path, file_type in [
                (self.hitter_input_file, 'hitter'), 
                (self.pitcher_input_file, 'pitcher')
            ]:
                if not file_path.exists():
                    logging.error(f"{file_type.capitalize()} file not found: {file_path}")
                    return pd.DataFrame(columns=['Name', 'dollar_value'])
            
            # Read data with validation
            hitters_df = self.read_data(self.hitter_input_file)
            if hitters_df.empty:
                logging.error("Empty hitter data")
                return pd.DataFrame(columns=['Name', 'dollar_value'])
                
            pitchers_df = self.read_data(self.pitcher_input_file)
            if pitchers_df.empty:
                logging.error("Empty pitcher data")
                return pd.DataFrame(columns=['Name', 'dollar_value'])
                
            # Process keepers if available
            keepers_df = None
            if self.keeper_file.exists():
                keepers_df = self.read_data(self.keeper_file)
                if keepers_df.empty:
                    logging.warning("Empty keeper file, proceeding without keepers")
                    keepers_df = None

            # Calculate values with validation
            final_hitters, final_pitchers = self.calculate_auction_values(hitters_df, pitchers_df, keepers_df)
            
            # Validate outputs
            for df_name, df in [('hitters', final_hitters), ('pitchers', final_pitchers)]:
                if 'dollar_value' not in df.columns:
                    logging.error(f"Missing dollar_value in {df_name}")
                    return pd.DataFrame(columns=['Name', 'dollar_value'])
                    
            # Combine with validation
            combined_results = self.combine_data(final_hitters, final_pitchers)
            if combined_results.empty:
                logging.error("Combined results are empty")
                return pd.DataFrame(columns=['Name', 'dollar_value'])
                
            # Return well-formed output
            return combined_results[['Name', 'dollar_value']]
            
        except pd.errors.EmptyDataError:
            logging.error("Empty data file encountered")
            return pd.DataFrame(columns=['Name', 'dollar_value'])
        except pd.errors.ParserError as e:
            logging.error(f"CSV parsing error: {e}")
            return pd.DataFrame(columns=['Name', 'dollar_value'])
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            logging.exception("Full traceback:")
            return pd.DataFrame(columns=['Name', 'dollar_value'])

if __name__ == "__main__":
    rankings = PlayerRankings(
        hitter_input_file='fangraphs-leaderboard-projections_hitters.csv',
        pitcher_input_file='fangraphs-leaderboard-projections_pitchers.csv',
        keeper_file='keepers.csv',
        output_file='player_rankings.csv'
    )
    results = rankings.run()