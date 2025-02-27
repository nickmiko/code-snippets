import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

class RankingsZScore:
    """Convert player rankings to fantasy baseball dollar values using advanced methods"""
    
    def __init__(
        self, 
        rankings_file: str, 
        total_budget: int, 
        roster_size: int = 23, 
        num_teams: int = 12,
        hitter_pct: float = 0.67,  # Typical 67/33 split between hitters and pitchers
        position_adjustments: Optional[Dict[str, float]] = None,
        exponential_factor: float = 0.9  # Controls how quickly values drop (0.8-0.95 typical)
    ):
        """
        Initialize the rankings converter
        
        Args:
            rankings_file: Path to CSV with player rankings
            total_budget: Total league budget (e.g., $260 per team)
            roster_size: Roster spots per team
            num_teams: Number of teams in league
            hitter_pct: Percentage of budget allocated to hitters
            position_adjustments: Optional dict mapping positions to scarcity multipliers
            exponential_factor: Controls value distribution curve steepness
        """
        self.rankings_file = rankings_file
        self.total_budget = total_budget
        self.roster_size = roster_size
        self.num_teams = num_teams
        self.hitter_pct = hitter_pct
        self.league_budget = total_budget * num_teams
        self.exponential_factor = exponential_factor
        
        # Default position adjustments (scarcity-based)
        self.position_adjustments = position_adjustments or {
            'C': 1.15,  # Catchers are scarce
            '1B': 0.95, # First basemen are abundant
            '2B': 1.05,
            '3B': 1.05,
            'SS': 1.10,
            'OF': 0.95,
            'MI': 1.05, # Middle infield
            'CI': 1.00, # Corner infield
            'UT': 0.90, # Utility
            'SP': 1.05, # Starting pitchers
            'RP': 0.95, # Relief pitchers
            'P': 1.00   # Generic pitcher
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """Load rankings data with validation"""
        try:
            file_path = Path(self.rankings_file)
            if not file_path.exists():
                raise FileNotFoundError(f"Rankings file not found: {self.rankings_file}")
                
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['Name']
            position_col = False
            
            # Check if we have position data
            for col in df.columns:
                if col.lower() in ('position', 'pos'):
                    df.rename(columns={col: 'Position'}, inplace=True)
                    position_col = True
            
            # Check for required columns
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
                
            # Handle missing position data
            if not position_col:
                self.logger.warning("No position column found - player type differentiation disabled")
                df['Position'] = 'Unknown'
                
            # Add player type column
            df['player_type'] = df['Position'].apply(
                lambda x: 'Hitter' if x in ('C', '1B', '2B', '3B', 'SS', 'OF', 'MI', 'CI', 'UT') else 'Pitcher'
            )
            
            # Normalize names
            df['Name'] = df['Name'].str.strip()
            
            return df
            
        except pd.errors.EmptyDataError:
            self.logger.error("Empty rankings file")
            sys.exit(1)
        except pd.errors.ParserError:
            self.logger.error("Error parsing CSV file")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error reading data: {str(e)}")
            sys.exit(1)

    def calculate_dollar_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dollar values with exponential distribution and position adjustments"""
        if df.empty:
            return pd.DataFrame(columns=['Name', 'dollar_value'])
            
        # Split budget between hitters and pitchers
        hitter_budget = self.league_budget * self.hitter_pct
        pitcher_budget = self.league_budget * (1 - self.hitter_pct)
        
        # Calculate how many of each type to value
        roster_hitters = int(self.roster_size * self.hitter_pct)
        roster_pitchers = self.roster_size - roster_hitters
        total_hitters = roster_hitters * self.num_teams
        total_pitchers = roster_pitchers * self.num_teams
        
        # Separate into hitters and pitchers
        hitters = df[df['player_type'] == 'Hitter'].copy()
        pitchers = df[df['player_type'] == 'Pitcher'].copy()
        
        # Apply ranking-based valuation to each group
        hitters = self._apply_exponential_values(hitters, hitter_budget, total_hitters)
        pitchers = self._apply_exponential_values(pitchers, pitcher_budget, total_pitchers)
        
        # Apply position adjustments
        hitters = self._apply_position_adjustments(hitters)
        pitchers = self._apply_position_adjustments(pitchers)
        
        # Combine and sort by dollar value
        valued_players = pd.concat([hitters, pitchers], ignore_index=True)
        valued_players = valued_players.sort_values('dollar_value', ascending=False)
        
        # Ensure minimum value of $1
        valued_players['dollar_value'] = valued_players['dollar_value'].clip(lower=1)
        
        # Round to a reasonable precision
        valued_players['dollar_value'] = valued_players['dollar_value'].round(1)
        
        # Return most important columns
        return valued_players[['Name', 'Position', 'dollar_value']]

    def _apply_exponential_values(self, df: pd.DataFrame, total_budget: float, 
                                total_players: int) -> pd.DataFrame:
        """Apply exponential value distribution based on rank"""
        # If no players, return empty DataFrame
        if df.empty:
            return df
            
        # Limit to the required number of players
        df = df.head(total_players).copy()
        
        # Calculate rank
        df['rank'] = np.arange(1, len(df) + 1)
        
        # Calculate exponential weights (higher values for better ranks)
        # Using exponential decay: a * (b^rank) where b is the decay factor
        exp_factor = self.exponential_factor
        df['exp_weight'] = np.power(exp_factor, df['rank'])
        
        # Calculate dollar value distribution
        total_weight = df['exp_weight'].sum()
        if total_weight > 0:
            df['dollar_value'] = df['exp_weight'] * (total_budget / total_weight)
        else:
            df['dollar_value'] = 0
            
        return df

    def _apply_position_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply position scarcity adjustments"""
        if df.empty or 'Position' not in df.columns:
            return df
            
        df = df.copy()
        
        # Apply position adjustments
        for pos, adjustment in self.position_adjustments.items():
            mask = df['Position'] == pos
            if mask.any():
                df.loc[mask, 'dollar_value'] = df.loc[mask, 'dollar_value'] * adjustment
                
        return df

    def run(self) -> pd.DataFrame:
        """Execute the full process with timing and validation"""
        self.logger.info(f"Processing rankings from {self.rankings_file}")
        
        try:
            df = self.load_data()
            self.logger.info(f"Loaded {len(df)} players from rankings")
            
            result = self.calculate_dollar_values(df)
            self.logger.info(f"Generated values for {len(result)} players")
            
            # Log some stats
            if not result.empty:
                self.logger.info(f"Value range: ${result['dollar_value'].min():.1f} to ${result['dollar_value'].max():.1f}")
                self.logger.info(f"Top valued player: {result.iloc[0]['Name']} (${result.iloc[0]['dollar_value']:.1f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in valuation process: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame(columns=['Name', 'dollar_value'])