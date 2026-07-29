"""
This module provides post-draft analysis functionality.
"""

import pandas as pd
from .fantrax_api import FantraxAPI

class PostDraftAnalyzer:
    """
    Provides post-draft analysis, including team analysis, trade suggestions,
    and trade evaluation.
    """

    def __init__(self, league_id, projections_df, config):
        """
        Initializes the PostDraftAnalyzer.

        Args:
            league_id (str): The Fantrax league ID.
            projections_df (pd.DataFrame): DataFrame with player projections.
            config (dict): The league configuration.
        """
        self.fantrax_api = FantraxAPI(league_id=league_id)
        self.projections_df = projections_df
        self.config = config
        self.player_id_map = self._get_player_id_map()
        self.rosters = self._get_rosters()
        self.team_projections = self._calculate_all_team_projections()

    def _get_player_id_map(self):
        """
        Creates a mapping from Fantrax player ID to player name.
        """
        print("Fetching player ID map...")
        player_data = self.fantrax_api.get_player_ids()
        if not player_data:
            print("Could not retrieve player ID map. API returned empty.")
            return {}
        
        id_map = {}
        for pid, info in player_data.items():
            name = info.get('name', '')
            # Convert "Last, First" to "First Last"
            if ',' in name:
                parts = name.split(',', 1)
                name = f"{parts[1].strip()} {parts[0].strip()}"
            id_map[pid] = name
            
        return id_map

    def _get_rosters(self):
        """
        Retrieves and processes team rosters from the Fantrax API.

        Returns:
            dict: A dictionary with team names as keys and lists of player names as values.
        """
        roster_data = self.fantrax_api.get_team_rosters()

        if not roster_data or 'rosters' not in roster_data:
            print("Could not retrieve roster data or 'rosters' key is missing.")
            return {}

        rosters = {}
        for team_data in roster_data['rosters'].values():
            team_name = team_data['teamName']
            player_ids = [item['id'] for item in team_data.get('rosterItems', [])]
            
            # Map IDs to names, handling cases where an ID might not be in the map
            players = [self.player_id_map.get(pid, f"Unknown Player (ID: {pid})") for pid in player_ids]
            rosters[team_name] = players
        return rosters

    def _calculate_team_projections(self, player_list):
        """
        Calculates the total projected stats for a list of players.

        Args:
            player_list (list): A list of player names.

        Returns:
            pd.Series: A series with the total projected stats.
        """
        team_df = self.projections_df[self.projections_df['Name'].isin(player_list)]
        return team_df[self.config['categories']['hitters'] + self.config['categories']['pitchers']].sum()

    def _calculate_all_team_projections(self):
        """
        Calculates projected stats for all teams in the league.

        Returns:
            pd.DataFrame: A DataFrame with teams as rows and categories as columns.
        """
        all_team_projections = {}
        for team_name, players in self.rosters.items():
            all_team_projections[team_name] = self._calculate_team_projections(players)
        return pd.DataFrame(all_team_projections).T

    def get_team_standings_z_scores(self):
        """
        Calculates the z-scores for each team in each category.

        Returns:
            pd.DataFrame: A DataFrame of z-scores for each team.
        """
        stats = self.team_projections
        z_scores = (stats - stats.mean()) / stats.std()
        
        # Calculate overall z-score sum
        z_scores['Overall'] = z_scores.sum(axis=1)
        # Sort by overall score
        z_scores = z_scores.sort_values(by='Overall', ascending=False)
        
        show_ranks = self.config.get('league_settings', {}).get('show_ranks', False)
        
        if show_ranks:
            # Rank descending (highest z-score gets rank 1)
            ranks = z_scores.rank(ascending=False, method='min')
            
            def to_ordinal(n):
                if pd.isna(n): return n
                n = int(n)
                if 11 <= (n % 100) <= 13:
                    suffix = 'th'
                else:
                    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
                return f"{n}{suffix}"
                
            return ranks.apply(lambda col: col.map(to_ordinal))
            
        return z_scores

    def evaluate_trade(self, team1_players_out, team2_players_out, team1_name, team2_name):
        """
        Evaluates a trade between two teams.

        Args:
            team1_players_out (list): Players the user's team is trading away.
            team2_players_out (list): Players the other team is trading away.
            team1_name (str): The name of the user's team.
            team2_name (str): The name of the other team.

        Returns:
            dict: A dictionary showing the impact of the trade on both teams.
        """
        # Get original projections
        original_team1_proj = self.team_projections.loc[team1_name].copy()
        original_team2_proj = self.team_projections.loc[team2_name].copy()

        # Get projections for players being traded
        team1_out_proj = self._calculate_team_projections(team1_players_out)
        team2_out_proj = self._calculate_team_projections(team2_players_out)

        # Calculate new team projections
        new_team1_proj = original_team1_proj - team1_out_proj + team2_out_proj
        new_team2_proj = original_team2_proj - team2_out_proj + team1_out_proj

        # Create a summary DataFrame
        impact = pd.DataFrame({
            f'{team1_name} (Original)': original_team1_proj,
            f'{team1_name} (New)': new_team1_proj,
            f'{team2_name} (Original)': original_team2_proj,
            f'{team2_name} (New)': new_team2_proj,
        })

        impact[f'{team1_name} Change'] = new_team1_proj - original_team1_proj
        impact[f'{team2_name} Change'] = new_team2_proj - original_team2_proj

        return impact

    def suggest_trades(self, my_team_name, top_n=100, partner_team=None):
        """
        Analyzes up to 2-for-3 trade permutations to find mutually beneficial trades.
        Uses a needs-based weighting algorithm where categories a team is weak in 
        are valued exponentially higher than categories they are already strong in.

        Args:
            my_team_name (str): The name of the user's team.
            top_n (int): Number of top trades to return.

        Returns:
            pd.DataFrame: A DataFrame of recommended trades.
        """
        import numpy as np
        from itertools import combinations

        if my_team_name not in self.rosters:
            print(f"Error: Could not find team '{my_team_name}' in rosters.")
            return pd.DataFrame()

        print("Preparing optimized trade permutations (evaluating millions of combos)...")

        stats = self.team_projections
        mean = stats.mean()
        std = stats.std()

        # Inverse categories adjustment for ERA/WHIP so higher Z is always better
        inverse_cats = ['ERA', 'WHIP']
        direction = np.ones(len(stats.columns))
        for i, col in enumerate(stats.columns):
            if col in inverse_cats:
                direction[i] = -1

        # Current z-scores relative to league
        curr_z = (stats - mean) / std * direction

        # Calculate needs weights for each team 
        # (exponential so weak categories [Z < 0] have High Weight, strong categories have Low Weight)
        team_weights = {}
        for team in self.rosters.keys():
            # cap values between -3 and 3 to prevent astronomical exponential inflation
            clipped_z = np.clip(curr_z.loc[team].values, -3, 3)
            team_weights[team] = np.exp(-clipped_z)

        my_weights = team_weights[my_team_name]
        my_roster = self.rosters[my_team_name]

        # Precompute player Z deltas relative to the league standard deviation
        player_z_array = {}
        for team_name, team_players in self.rosters.items():
            for player in team_players:
                if player not in player_z_array:
                    p_proj = self._calculate_team_projections([player])
                    # Divide raw stat by league std dev to scale into Z-units properly
                    player_z_array[player] = np.nan_to_num((p_proj.values / std.values) * direction)

        def get_combo_arrays(roster, max_size):
            combos = []
            for size in range(1, max_size + 1):
                combos.extend(list(combinations(roster, size)))
            
            z_matrix = np.zeros((len(combos), len(stats.columns)))
            for i, c in enumerate(combos):
                for player in c:
                    z_matrix[i] += player_z_array[player]
            return list(combos), z_matrix

        # Get my side of potential trades (Sizes 1, 2)
        my_combos, my_z_matrix = get_combo_arrays(my_roster, 2)
        my_c_my_val = my_z_matrix.dot(my_weights)

        suggested_trades = []

        for partner_name, partner_roster in self.rosters.items():
            if partner_name == my_team_name:
                continue

            if partner_team and partner_name != partner_team:
                continue

            partner_weights = team_weights[partner_name]
            # Get partner side of potential trades (Sizes 1, 2, 3)
            partner_combos, p_z_matrix = get_combo_arrays(partner_roster, 3)

            # Dot products to evaluate the absolute "Needs-Adjusted Value" of each side 
            # of the trade for both teams
            my_c_p_val = my_z_matrix.dot(partner_weights)
            p_c_my_val = p_z_matrix.dot(my_weights)
            p_c_p_val = p_z_matrix.dot(partner_weights)

            # my_val: partner combo value to ME - my combo value to ME
            # p_val: my combo value to PARTNER - partner combo value to PARTNER
            my_val_matrix = p_c_my_val[np.newaxis, :] - my_c_my_val[:, np.newaxis]
            p_val_matrix = my_c_p_val[:, np.newaxis] - p_c_p_val[np.newaxis, :]

            # Find mutually beneficial trades. Threshold filters out insignificant lateral moves.
            valid_i, valid_j = np.where((my_val_matrix > 0.5) & (p_val_matrix > 0.5))

            for i, j in zip(valid_i, valid_j):
                my_val = my_val_matrix[i, j]
                p_val = p_val_matrix[i, j]
                
                suggested_trades.append({
                    'Partner': partner_name,
                    'You Give': ", ".join(my_combos[i]),
                    'You Get': ", ".join(partner_combos[j]),
                    'Your Need +': round(my_val, 2),
                    'Partner Need +': round(p_val, 2),
                    'Win-Win Score': round(my_val + p_val, 2)
                })

        if not suggested_trades:
            return pd.DataFrame()

        # Build DataFrame and sort according to round-robin rules and fairness
        df = pd.DataFrame(suggested_trades)
        
        # Calculate how equal the trade is (absolute difference between needs)
        df['Need Diff'] = abs(df['Your Need +'] - df['Partner Need +'])
        
        if partner_team:
            # If a specific partner is requested, sort by equality then total score
            df = df.sort_values(by=['Need Diff', 'Win-Win Score'], ascending=[True, False])
        else:
            # Else, show one option for each other team round-robin style
            # Create a rank column within each partner group based on equality then Win-Win Score
            df = df.sort_values(by=['Partner', 'Need Diff', 'Win-Win Score'], ascending=[True, True, False])
            df['TeamRank'] = df.groupby('Partner').cumcount()
            
            # Sort by TeamRank first (0th, 1st, 2nd trade for each team), 
            # then Need Diff, then Win-Win Score to break ties among the same rank
            df = df.sort_values(by=['TeamRank', 'Need Diff', 'Win-Win Score'], ascending=[True, True, False])
            df = df.drop(columns=['TeamRank'])
        
        # Drop the temporary column to keep output clean but sorted based on it
        df = df.drop(columns=['Need Diff'])

        # Keep just the top n distinctly impactful trades
        df = df.head(top_n)
        df.index = range(1, len(df) + 1)
        return df
