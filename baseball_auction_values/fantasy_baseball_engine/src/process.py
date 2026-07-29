"""
This module handles pre-processing of projection dataframes,
such as creating combined stat columns.
"""
import pandas as pd

def create_combined_pitcher_categories(p_proj: pd.DataFrame) -> pd.DataFrame:
    """
    Creates combined pitcher category columns if they don't already exist.

    Args:
        p_proj (pd.DataFrame): The pitcher projections DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with combined columns.
    """
    if 'QS' in p_proj.columns and 'W' in p_proj.columns and 'QS+W' not in p_proj.columns:
        print("Creating 'QS+W' column...")
        p_proj['QS+W'] = p_proj['QS'] + p_proj['W']
    
    if 'SV' in p_proj.columns and 'HLD' in p_proj.columns and 'SV+HLD' not in p_proj.columns:
        print("Creating 'SV+HLD' column...")
        p_proj['SV+HLD'] = p_proj['SV'] + p_proj['HLD']
        
    return p_proj
