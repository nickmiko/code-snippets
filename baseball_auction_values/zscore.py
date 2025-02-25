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

    def calculate_zscores(self, df: pd.DataFrame, weights: Dict[str, float], inverse_stats: tuple) -> pd.Series:
        """Calculate z-scores with robust outlier handling and validation"""
        df = df.copy()
        stats = list(weights.keys())
        z_scores = pd.DataFrame(index=df.index)
        
        # Determine valid stats for this dataset
        valid_stats = [stat for stat in stats if stat in df.columns]
        if not valid_stats:
            logging.warning(f"No valid stats found in columns: {df.columns.tolist()}")
            return pd.Series(0, index=df.index)
        
        # Adjust weights for available stats
        weight_sum = sum(weights[stat] for stat in valid_stats)
        adjusted_weights = {
            stat: (weights[stat] / weight_sum) if weight_sum > 0 else 0 
            for stat in valid_stats
        }
        
        def calculate_robust_zscore(series: pd.Series, weight: float, inverse: bool = False) -> pd.Series:
            """Calculate robust z-score using median and MAD with improved outlier handling"""
            series = pd.to_numeric(series, errors='coerce')
            series_clean = series.dropna()
            
            if series_clean.empty:
                return pd.Series(0, index=series.index)
                
            # Use percentile-based outlier removal
            q1, q3 = series_clean.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - (3 * iqr)
            upper_bound = q3 + (3 * iqr)
            series_filtered = series_clean[(series_clean >= lower_bound) & (series_clean <= upper_bound)]
            
            if series_filtered.empty or series_filtered.std() == 0:
                return pd.Series(0, index=series.index)
            
            median = series_filtered.median()
            mad = series_filtered.sub(median).abs().median() * 1.4826
            
            if mad == 0:
                mad = series_filtered.std()
                if mad == 0:
                    return pd.Series(0, index=series.index)
            
            z = (series - median) / mad
            multiplier = -1 if inverse else 1
            return (multiplier * z * weight).clip(-3, 3)  # More conservative clipping
        
        # Calculate z-scores only for available stats
        for stat in valid_stats:
            z_scores[f'{stat}_z'] = calculate_robust_zscore(
                df[stat], 
                adjusted_weights[stat],
                stat in inverse_stats
            )
        
        total_z = z_scores.sum(axis=1)
        
        return total_z

    def process_player_group(self, df: pd.DataFrame, roster_spots: int, budget: float) -> pd.DataFrame:
        """Process a group of players (hitters or pitchers) with improved value retention"""
        if df.empty:
            logging.warning("Empty DataFrame passed to process_player_group")
            return pd.DataFrame(columns=['Name', 'dollar_value'])
        
        # Make a safe copy and normalize names early
        df = df.copy()
        df['Name_norm'] = df['Name'].str.strip().str.lower()
        
        # Determine if processing hitters or pitchers
        is_hitter = 'HR' in df.columns
        weights = self.settings.HITTER_WEIGHTS if is_hitter else self.settings.PITCHER_WEIGHTS
        inverse_stats = () if is_hitter else self.settings.INVERSE_STATS
        
        # Calculate z-scores
        df['total_z'] = self.calculate_zscores(df, weights, inverse_stats)
        df = df.sort_values('total_z', ascending=False)
        
        # Calculate replacement level using actual roster spots
        num_rostered = min(roster_spots, len(df))
        replacement_level = df.iloc[num_rostered-1]['total_z'] if num_rostered > 0 else 0
        
        # Calculate value above replacement
        df['value_above_replacement'] = (df['total_z'] - replacement_level).clip(lower=0)
        
        # Calculate dollar values with minimum value of 1
        total_var = df['value_above_replacement'].sum()
        if total_var > 0:
            dollars_per_var = (budget - roster_spots) / total_var
            df['dollar_value'] = (df['value_above_replacement'] * dollars_per_var + 1).round(1)
        else:
            df['dollar_value'] = 1.0
        
        # Return all necessary columns
        result = df[['Name', 'Name_norm', 'dollar_value', 'total_z']].copy()
        logging.info(f"Processed {len(result)} players, max value: ${result['dollar_value'].max():.1f}")
        
        return result

    def calculate_auction_values(self, hitters_df: pd.DataFrame, pitchers_df: pd.DataFrame, 
                            keeper_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate auction values with improved validation and data preservation"""
        # Create copies and normalize names
        hitters_df = hitters_df.copy()
        pitchers_df = pitchers_df.copy()
        hitters_df['Name_norm'] = hitters_df['Name'].str.strip().str.lower()
        pitchers_df['Name_norm'] = pitchers_df['Name'].str.strip().str.lower()
        
        # Initialize keeper structures
        keeper_budget = 0
        keeper_hitters = pd.DataFrame(columns=['Name', 'Name_norm', 'dollar_value', 'Position', 'is_keeper'])
        keeper_pitchers = pd.DataFrame(columns=['Name', 'Name_norm', 'dollar_value', 'Position', 'is_keeper'])
        
        # Process keepers if present
        if keeper_df is not None and not keeper_df.empty:
            keeper_df = keeper_df.copy()
            keeper_df['Name_norm'] = keeper_df['Name'].str.strip().str.lower()
            
            # Ensure dollar_value exists and is numeric
            if 'dollar_value' not in keeper_df.columns:
                logging.warning("No dollar_value column in keepers, using default values")
                keeper_df['dollar_value'] = 0
            
            keeper_df['dollar_value'] = pd.to_numeric(keeper_df['dollar_value'], errors='coerce').fillna(0)
            keeper_budget = keeper_df['dollar_value'].sum()
            logging.info(f"Total keeper cost: ${keeper_budget}")
            
            for _, keeper in keeper_df.iterrows():
                keeper_row = pd.DataFrame({
                    'Name': [keeper['Name']],
                    'Name_norm': [keeper['Name_norm']],
                    'dollar_value': [keeper['dollar_value']],
                    'Position': ['Unknown'],
                    'is_keeper': [True]
                })
                
                if keeper['Name_norm'] in hitters_df['Name_norm'].values:
                    keeper_row['Position'] = 'H'
                    keeper_hitters = pd.concat([keeper_hitters, keeper_row])
                elif keeper['Name_norm'] in pitchers_df['Name_norm'].values:
                    keeper_row['Position'] = 'P'
                    keeper_pitchers = pd.concat([keeper_pitchers, keeper_row])
                
            
            # Remove keepers from available players
            hitters_df = hitters_df[~hitters_df['Name_norm'].isin(keeper_df['Name_norm'])]
            pitchers_df = pitchers_df[~pitchers_df['Name_norm'].isin(keeper_df['Name_norm'])]
        
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

    def combine_data(self, hitters_df: pd.DataFrame, pitchers_df: pd.DataFrame) -> pd.DataFrame:
        """Combine hitter and pitcher data with improved data preservation"""
        logging.info(f"Combining data - Hitters: {len(hitters_df)} rows, Pitchers: {len(pitchers_df)} rows")
        
        required_cols = ['Name', 'Name_norm', 'dollar_value', 'Position', 'is_keeper']
        
        # Verify required columns
        for df, df_type in [(hitters_df, 'hitters'), (pitchers_df, 'pitchers')]:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logging.error(f"Missing columns in {df_type}: {missing}")
                if 'Name_norm' in missing and 'Name' in df.columns:
                    df['Name_norm'] = df['Name'].str.strip().str.lower()
                    missing.remove('Name_norm')
                if missing:  # If there are still missing columns
                    return pd.DataFrame(columns=['Name', 'dollar_value'])
        
        # Make copies to avoid modifying originals
        hitters = hitters_df.copy()
        pitchers = pitchers_df.copy()
         
        # Combine and sort
        combined = pd.concat([hitters, pitchers], ignore_index=True)
        combined = combined.sort_values('dollar_value', ascending=False).reset_index(drop=True)
        
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

    def run(self) -> pd.DataFrame:
        """Execute the full process with proper error handling and data preservation"""
        try:
            # Read input data
            hitters_df = self.read_data(self.hitter_input_file)
            pitchers_df = self.read_data(self.pitcher_input_file)
            keepers_df = self.read_data(self.keeper_file) if self.keeper_file.exists() else None

            # Calculate auction values
            final_hitters, final_pitchers = self.calculate_auction_values(hitters_df, pitchers_df, keepers_df)

            # Ensure required columns exist and preserve Name_norm
            columns_to_keep = ['Name', 'Name_norm', 'dollar_value', 'Position', 'is_keeper']
            
            # Add Name_norm if missing
            for df in [final_hitters, final_pitchers]:
                if 'Name_norm' not in df.columns:
                    df['Name_norm'] = df['Name'].str.strip().str.lower()

            # Combine data with all necessary columns
            combined_results = self.combine_data(
                final_hitters[columns_to_keep],
                final_pitchers[columns_to_keep]
            )
            
            if combined_results.empty:
                logging.error("Combined results are empty")
                return pd.DataFrame(columns=['Name', 'dollar_value'])

            # Return final results
            return combined_results[['Name', 'dollar_value']]
            
        except Exception as e:
            logging.error(f"Error in processing: {str(e)}")
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