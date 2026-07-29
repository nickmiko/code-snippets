"""
This module contains the FantraxAPI class for interacting with the Fantrax API.
"""

import requests

class FantraxAPI:
    """
    A class to interact with the Fantrax API.
    """
    BASE_URL = "https://www.fantrax.com/fxea/general"

    def __init__(self, league_id=None, user_secret_id=None):
        """
        Initializes the FantraxAPI client.
        Args:
            league_id (str, optional): The league ID. Defaults to None.
            user_secret_id (str, optional): The user's secret ID. Defaults to None.
        """
        self.league_id = league_id
        self.user_secret_id = user_secret_id

    def _make_request(self, endpoint, params=None, data=None):
        """
        Makes a request to the Fantrax API.
        Args:
            endpoint (str): The API endpoint to call.
            params (dict, optional): Query string parameters. Defaults to None.
            data (dict, optional): JSON data for POST requests. Defaults to None.
        Returns:
            dict: The JSON response from the API.
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        # Add user_secret_id to params if it exists
        if self.user_secret_id:
            if params is None:
                params = {}
            params['userSecretId'] = self.user_secret_id

        try:
            if data:
                response = requests.post(url, json=data, timeout=10)
            else:
                response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error making request to {url}: {e}")
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            return None

    def get_player_ids(self, sport="MLB"):
        """
        Retrieves player IDs for a given sport.
        Args:
            sport (str): The sport (e.g., 'MLB', 'NFL').
        Returns:
            dict: API response.
        """
        return self._make_request("getPlayerIds", params={"sport": sport})

    def get_adp(self, sport="MLB", position=None, show_all_positions=False, start=None, limit=None, order=None):
        """
        Retrieves ADP info for all players.
        Args:
            sport (str): The sport.
            position (str, optional): Player position. Defaults to None.
            show_all_positions (bool, optional): Show all positions. Defaults to False.
            start (int, optional): Start index. Defaults to None.
            limit (int, optional): Limit results. Defaults to None.
            order (str, optional): Order of results. Defaults to None.
        Returns:
            dict: API response.
        """
        params = {"sport": sport}
        if position:
            params["position"] = position
        if show_all_positions:
            params["showAllPositions"] = "true"
        if start is not None:
            params["start"] = start
        if limit is not None:
            params["limit"] = limit
        if order:
            params["order"] = order
        return self._make_request("getAdp", params=params)

    def get_leagues(self):
        """
        Retrieves the list of leagues for the user.
        Returns:
            dict: API response.
        """
        if not self.user_secret_id:
            raise ValueError("user_secret_id is required to get leagues.")
        return self._make_request("getLeagues", params={"userSecretId": self.user_secret_id})

    def get_league_info(self):
        """
        Retrieves information about a specific league.
        Returns:
            dict: API response.
        """
        if not self.league_id:
            raise ValueError("league_id is required to get league info.")
        return self._make_request("getLeagueInfo", params={"leagueId": self.league_id})

    def get_draft_picks(self):
        """
        Retrieves future and current draft picks in a specific league.
        Returns:
            dict: API response.
        """
        if not self.league_id:
            raise ValueError("league_id is required to get draft picks.")
        return self._make_request("getDraftPicks", params={"leagueId": self.league_id})

    def get_draft_results(self):
        """
        Retrieves the draft results of a specific league.
        Returns:
            dict: API response.
        """
        if not self.league_id:
            raise ValueError("league_id is required to get draft results.")
        return self._make_request("getDraftResults", params={"leagueId": self.league_id})

    def get_team_rosters(self, period=None):
        """
        Retrieves data on all rosters for a given period.
        Args:
            period (int, optional): The lineup period. Defaults to None.
        Returns:
            dict: API response.
        """
        if not self.league_id:
            raise ValueError("league_id is required to get team rosters.")
        params = {"leagueId": self.league_id}
        if period:
            params["period"] = period
        return self._make_request("getTeamRosters", params=params)

    def get_standings(self):
        """
        Retrieves the current standings of the league.
        Returns:
            dict: API response.
        """
        if not self.league_id:
            raise ValueError("league_id is required to get standings.")
        return self._make_request("getStandings", params={"leagueId": self.league_id})
