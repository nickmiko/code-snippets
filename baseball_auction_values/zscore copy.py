import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
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
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('player_rankings.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_df_stats(self, df: pd.DataFrame, description: str) -> None:
        """Log key statistics about a dataframe for debugging"""
        if df is None or df.empty:
            self.logger.warning(f"{description}: Empty DataFrame")
            return
            
        self.logger.info(f"{description}: {len(df)} rows, columns: {df.columns.tolist()}")
        
        # Log numeric column stats if available
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols and 'dollar_value' in numeric_cols:
            self.logger.info(f"{description} dollar_value stats: min={df['dollar_value'].min():.2f}, "
                            f"max={df['dollar_value'].max():.2f}, "
                            f"mean={df['dollar_value'].mean():.2f}")

    def read_data(self, input_file: Path) -> pd.DataFrame:
        """Read CSV data with optimized settings and column name normalization"""
        if not input_file.exists():
            self.logger.warning(f"File not found: {input_file}")
            return pd.DataFrame()
            
        try:
            # Use optimized CSV reading settings
            df = pd.read_csv(
                input_file,
                low_memory=False,
                dtype={'Name': str},
                usecols=lambda col: col.lower() in ['name', 'hr', 'r', 'rbi', 'sb', 'obp', 
                                                   'qs', 'sv', 'hld', 'k', 'era', 'whip',
                                                   'dollar_value', 'dollarvalue', 'value', 'cost'] 
                                                   or col == 'Name' or col == 'dollar_value'
            )
            
            # Early validation
            if 'Name' not in df.columns:
                self.logger.error(f"'Name' column missing in {input_file}")
                return pd.DataFrame()
                
            # Normalize column names for keepers file
            if input_file.name.lower() == 'keepers.csv':
                # Map possible variations of dollar value column names
                dollar_cols = ['dollar_value', 'dollarvalue', 'value', 'cost']
                found = False
                
                for col in df.columns:
                    if col.lower().replace('_', '') in [c.lower().replace('_', '') for c in dollar_cols]:
                        df = df.rename(columns={col: 'dollar_value'})
                        found = True
                        break
                
                if not found:
                    self.logger.warning(f"No dollar value column found in {input_file}, creating default")
                    df['dollar_value'] = 0
                    
            # Log stats about the loaded data
            self.log_df_stats(df, f"Loaded {input_file.name}")
            return df
            
        except Exception as e:
            self.logger.exception(f"Error reading {input_file}: {e}")
            return pd.DataFrame()

    def calculate_zscores(self, df: pd.DataFrame, weights: Dict[str, float], inverse_stats: tuple) -> pd.Series:
        """Calculate z-scores with robust outlier handling and validation"""
        if df.empty:
            self.logger.warning("Empty DataFrame passed to calculate_zscores")
            return pd.Series(0, index=pd.Index([]))
            
        # Make a copy to avoid warnings
        df = df.copy()
        
        # Get stats that exist in both weights and df columns
        valid_stats = [stat for stat in weights.keys() if stat in df.columns]
        
        if not valid_stats:
            self.logger.warning(f"No valid stats found in columns: {df.columns.tolist()}")
            return pd.Series(0, index=df.index)
        
        # Normalize weights
        weight_sum = sum(weights[stat] for stat in valid_stats)
        adjusted_weights = {
            stat: (weights[stat] / weight_sum) if weight_sum > 0 else 0 
            for stat in valid_stats
        }
        
        # Convert to numeric (more efficient to do once per column)
        for stat in valid_stats:
            df[stat] = pd.to_numeric(df[stat], errors='coerce')
        
        # Preallocate z-score array for better performance
        z_scores = np.zeros((len(df), len(valid_stats)))
        
        # Calculate z-scores for each stat
        for i, stat in enumerate(valid_stats):
            series = df[stat].dropna()
            
            if series.empty or series.nunique() <= 1:
                continue
                
            # Use vectorized operations for speed
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            
            # Handle zero IQR case
            if iqr == 0:
                # If IQR is zero, use standard deviation
                std = series.std()
                if std == 0:
                    continue
                    
                z = (df[stat] - series.mean()) / std
            else:
                # Normal case - use IQR for outlier detection
                lower_bound = q1 - (3 * iqr)
                upper_bound = q3 + (3 * iqr)
                
                # Use robust statistics
                filtered = series[(series >= lower_bound) & (series <= upper_bound)]
                if filtered.empty:
                    continue
                    
                median = filtered.median()
                mad = filtered.sub(median).abs().median() * 1.4826
                
                if mad == 0:
                    # Fall back to standard deviation if MAD is zero
                    std = filtered.std()
                    if std == 0:
                        continue
                    z = (df[stat] - median) / std
                else:
                    z = (df[stat] - median) / mad
            
            # Apply weight and direction adjustment
            weight = adjusted_weights[stat]
            multiplier = -1 if stat in inverse_stats else 1
            z_scores[:, i] = (z * weight * multiplier).clip(-3, 3).fillna(0)
        
        # Sum across stats
        total_z = pd.Series(z_scores.sum(axis=1), index=df.index)
        
        return total_z

    def process_player_group(self, df: pd.DataFrame, roster_spots: int, budget: float, player_type: str) -> pd.DataFrame:
        """Process a group of players (hitters or pitchers) with improved value retention"""
        if df.empty:
            self.logger.warning(f"Empty DataFrame passed to process_player_group for {player_type}")
            return pd.DataFrame(columns=['Name', 'Name_norm', 'dollar_value', 'total_z', 'Position', 'is_keeper'])
        
        # Make a safe copy
        df = df.copy()
        
        # Normalize names
        df['Name_norm'] = df['Name'].str.strip().str.lower()
        
        # Determine weights based on player type
        is_hitter = player_type.lower() == 'hitter'
        weights = self.settings.HITTER_WEIGHTS if is_hitter else self.settings.PITCHER_WEIGHTS
        inverse_stats = self.settings.INVERSE_STATS if not is_hitter else ()
        
        # Calculate z-scores
        df['total_z'] = self.calculate_zscores(df, weights, inverse_stats)
        
        # Sort players by z-score
        df = df.sort_values('total_z', ascending=False)
        
        # Calculate replacement level
        num_rostered = min(roster_spots, len(df))
        if num_rostered > 0:
            replacement_level = df.iloc[num_rostered-1]['total_z']
        else:
            replacement_level = 0
            
        # Calculate value above replacement
        df['value_above_replacement'] = (df['total_z'] - replacement_level).clip(lower=0)
        
        # Calculate dollar values ensuring minimum value of 1
        total_var = df['value_above_replacement'].sum()
        if total_var > 0:
            dollars_per_var = (budget - roster_spots) / total_var
            df['dollar_value'] = (df['value_above_replacement'] * dollars_per_var + 1).round(1)
        else:
            df['dollar_value'] = 1.0
        
        # Set position and keeper status
        df['Position'] = 'H' if is_hitter else 'P'
        df['is_keeper'] = False
        
        # Log summary stats
        self.logger.info(f"Processed {len(df)} {player_type}s, max value: ${df['dollar_value'].max():.1f}")
        
        # Return necessary columns
        result = df[['Name', 'Name_norm', 'dollar_value', 'total_z', 'Position', 'is_keeper']].copy()
        
        return result

    def calculate_auction_values(self, hitters_df: pd.DataFrame, pitchers_df: pd.DataFrame, 
                            keeper_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate auction values with improved validation and data preservation"""
        # Early validation
        if hitters_df.empty and pitchers_df.empty:
            self.logger.error("Both hitters and pitchers DataFrames are empty")
            return (pd.DataFrame(columns=['Name', 'Name_norm', 'dollar_value', 'Position', 'is_keeper']),
                    pd.DataFrame(columns=['Name', 'Name_norm', 'dollar_value', 'Position', 'is_keeper']))
        
        # Create copies and normalize names
        hitters_df = hitters_df.copy() if not hitters_df.empty else pd.DataFrame(columns=['Name'])
        pitchers_df = pitchers_df.copy() if not pitchers_df.empty else pd.DataFrame(columns=['Name'])
        
        if not hitters_df.empty:
            hitters_df['Name_norm'] = hitters_df['Name'].str.strip().str.lower()
        
        if not pitchers_df.empty:
            pitchers_df['Name_norm'] = pitchers_df['Name'].str.strip().str.lower()
        
        # Create empty keeper dataframes with required columns
        keeper_columns = ['Name', 'Name_norm', 'dollar_value', 'Position', 'is_keeper']
        keeper_hitters = pd.DataFrame(columns=keeper_columns)
        keeper_pitchers = pd.DataFrame(columns=keeper_columns)
        keeper_budget = 0
        
        # Process keepers if present
        if keeper_df is not None and not keeper_df.empty:
            self.logger.info(f"Processing {len(keeper_df)} keepers")
            
            keeper_df = keeper_df.copy()
            keeper_df['Name_norm'] = keeper_df['Name'].str.strip().str.lower()
            
            # Ensure dollar_value is numeric
            if 'dollar_value' not in keeper_df.columns:
                self.logger.warning("No dollar_value column in keepers, using default values")
                keeper_df['dollar_value'] = 0
            else:
                keeper_df['dollar_value'] = pd.to_numeric(keeper_df['dollar_value'], errors='coerce').fillna(0)
            
            keeper_budget = keeper_df['dollar_value'].sum()
            self.logger.info(f"Total keeper cost: ${keeper_budget:.2f}")
            
            # Classify keepers by position efficiently
            keeper_names_norm = keeper_df['Name_norm'].tolist()
            
            # Create a mask for hitter keepers
            if not hitters_df.empty:
                hitter_mask = hitters_df['Name_norm'].isin(keeper_names_norm)
                hitter_keepers = hitters_df.loc[hitter_mask].copy()
                
                if not hitter_keepers.empty:
                    # Add keeper information
                    hitter_keepers['is_keeper'] = True
                    hitter_keepers['Position'] = 'H'
                    
                    # Merge with keeper costs
                    hitter_keepers = pd.merge(
                        hitter_keepers,
                        keeper_df[['Name_norm', 'dollar_value']],
                        on='Name_norm',
                        how='left',
                        suffixes=('', '_keeper')
                    )
                    
                    # Use keeper dollar value
                    hitter_keepers['dollar_value'] = hitter_keepers['dollar_value_keeper']
                    hitter_keepers = hitter_keepers.drop('dollar_value_keeper', axis=1)
                    
                    # Add to keeper hitters
                    keeper_hitters = hitter_keepers[keeper_columns].copy()
                    
                    # Remove from available hitters
                    hitters_df = hitters_df.loc[~hitter_mask].copy()
            
            # Create a mask for pitcher keepers
            if not pitchers_df.empty:
                pitcher_mask = pitchers_df['Name_norm'].isin(keeper_names_norm)
                pitcher_keepers = pitchers_df.loc[pitcher_mask].copy()
                
                if not pitcher_keepers.empty:
                    # Add keeper information
                    pitcher_keepers['is_keeper'] = True
                    pitcher_keepers['Position'] = 'P'
                    
                    # Merge with keeper costs
                    pitcher_keepers = pd.merge(
                        pitcher_keepers,
                        keeper_df[['Name_norm', 'dollar_value']],
                        on='Name_norm',
                        how='left',
                        suffixes=('', '_keeper')
                    )
                    
                    # Use keeper dollar value
                    pitcher_keepers['dollar_value'] = pitcher_keepers['dollar_value_keeper']
                    pitcher_keepers = pitcher_keepers.drop('dollar_value_keeper', axis=1)
                    
                    # Add to keeper pitchers
                    keeper_pitchers = pitcher_keepers[keeper_columns].copy()
                    
                    # Remove from available pitchers
                    pitchers_df = pitchers_df.loc[~pitcher_mask].copy()
        
        # Calculate budgets
        total_budget = self.settings.BUDGET * self.settings.NUM_TEAMS - keeper_budget
        hitter_budget = total_budget * self.settings.HITTER_BUDGET_PCT
        pitcher_budget = total_budget * (1 - self.settings.HITTER_BUDGET_PCT)
        
        # Calculate roster spots
        hitter_roster_size = int(self.settings.ROSTER_SIZE * self.settings.HITTER_ROSTER_PCT)
        pitcher_roster_size = self.settings.ROSTER_SIZE - hitter_roster_size
        total_hitters = (hitter_roster_size * self.settings.NUM_TEAMS) - len(keeper_hitters)
        total_pitchers = (pitcher_roster_size * self.settings.NUM_TEAMS) - len(keeper_pitchers)
        
        # Log budget and roster information
        self.logger.info(f"Hitters: Budget=${hitter_budget:.2f}, Spots={total_hitters}")
        self.logger.info(f"Pitchers: Budget=${pitcher_budget:.2f}, Spots={total_pitchers}")
        
        # Process players
        hitter_values = (
            self.process_player_group(hitters_df, total_hitters, hitter_budget, 'hitter') 
            if not hitters_df.empty else 
            pd.DataFrame(columns=keeper_columns)
        )
        
        pitcher_values = (
            self.process_player_group(pitchers_df, total_pitchers, pitcher_budget, 'pitcher')
            if not pitchers_df.empty else
            pd.DataFrame(columns=keeper_columns)
        )
        
        # Combine with keepers efficiently
        final_hitters = pd.concat([keeper_hitters, hitter_values], ignore_index=True)
        final_pitchers = pd.concat([keeper_pitchers, pitcher_values], ignore_index=True)
        
        return final_hitters, final_pitchers

    def combine_data(self, hitters_df: pd.DataFrame, pitchers_df: pd.DataFrame) -> pd.DataFrame:
        """Combine hitter and pitcher data efficiently"""
        self.logger.info(f"Combining data - Hitters: {len(hitters_df)} rows, Pitchers: {len(pitchers_df)} rows")
        
        # Required columns for final output
        required_cols = ['Name', 'dollar_value', 'Position', 'is_keeper']
        optional_cols = ['Name_norm', 'total_z']
        
        # Create combined dataframe with only necessary columns
        combined_dfs = []
        
        # Add hitters if present
        if not hitters_df.empty:
            # Ensure all required columns exist
            for col in required_cols + optional_cols:
                if col not in hitters_df.columns and col != 'Position':
                    if col == 'Name_norm':
                        hitters_df['Name_norm'] = hitters_df['Name'].str.strip().str.lower()
                    else:
                        hitters_df[col] = 0.0 if col == 'dollar_value' or col == 'total_z' else False
            
            if 'Position' not in hitters_df.columns:
                hitters_df['Position'] = 'H'
                
            combined_dfs.append(hitters_df)
        
        # Add pitchers if present
        if not pitchers_df.empty:
            # Ensure all required columns exist
            for col in required_cols + optional_cols:
                if col not in pitchers_df.columns and col != 'Position':
                    if col == 'Name_norm':
                        pitchers_df['Name_norm'] = pitchers_df['Name'].str.strip().str.lower()
                    else:
                        pitchers_df[col] = 0.0 if col == 'dollar_value' or col == 'total_z' else False
            
            if 'Position' not in pitchers_df.columns:
                pitchers_df['Position'] = 'P'
                
            combined_dfs.append(pitchers_df)
        
        if not combined_dfs:
            self.logger.warning("No data to combine")
            return pd.DataFrame(columns=required_cols)
        
        # Combine and sort
        combined = pd.concat(combined_dfs, ignore_index=True)
        combined = combined.sort_values('dollar_value', ascending=False).reset_index(drop=True)
        
        # Log combined results
        self.log_df_stats(combined, "Combined player data")
        
        return combined
    
    def save_results(self, df: pd.DataFrame) -> None:
        """Save results to file"""
        if df.empty:
            self.logger.warning("No data to save")
            return
            
        output_dir = Path('auction_values')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / self.output_file
        
        # Save with minimal columns for final output
        final_df = df[['Name', 'dollar_value']].copy()
        final_df.to_csv(output_path, index=False, float_format='%.1f')
        
        self.logger.info(f"Saved {len(final_df)} players to {output_path}")

    def run(self) -> pd.DataFrame:
        """Execute the full process with proper error handling"""
        try:
            # Read input data
            hitters_df = self.read_data(self.hitter_input_file)
            pitchers_df = self.read_data(self.pitcher_input_file)
            
            if hitters_df.empty and pitchers_df.empty:
                self.logger.error("No valid input data found")
                return pd.DataFrame(columns=['Name', 'dollar_value'])
                
            # Read keepers if file exists
            keepers_df = None
            if self.keeper_file.exists():
                keepers_df = self.read_data(self.keeper_file)
                
            # Calculate auction values
            final_hitters, final_pitchers = self.calculate_auction_values(hitters_df, pitchers_df, keepers_df)

            # Combine data
            combined_results = self.combine_data(final_hitters, final_pitchers)
            
            if combined_results.empty:
                self.logger.error("Combined results are empty")
                return pd.DataFrame(columns=['Name', 'dollar_value'])

            # Save results
            self.save_results(combined_results)
            
            # Return results for further processing
            return combined_results[['Name', 'dollar_value']]
            
        except Exception as e:
            self.logger.exception(f"Error in processing: {str(e)}")
            return pd.DataFrame(columns=['Name', 'dollar_value'])

if __name__ == "__main__":
    rankings = PlayerRankings(
        hitter_input_file='fangraphs-leaderboard-projections_hitters.csv',
        pitcher_input_file='fangraphs-leaderboard-projections_pitchers.csv',
        keeper_file='keepers.csv',
        output_file='player_rankings.csv'
    )
    results = rankings.run()