import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CHEAT_SHEET_PATH = OUTPUT_DIR / "cheat_sheet_values.csv"
CLEAN_VALUES_PATH = OUTPUT_DIR / "clean_auction_values.csv"
KEEPERS_PATH = BASE_DIR / "data" / "keepers" / "keepers.csv"

def test_cheat_sheet_exists():
    assert CHEAT_SHEET_PATH.exists(), "Cheat sheet file was not generated."

def test_cheat_sheet_columns():
    df = pd.read_csv(CHEAT_SHEET_PATH)
    expected_columns = ['Tier', 'Pos_Tier', 'Primary_Pos', 'Name', 'Profile', 'Draft_Day_Inflation_Value']
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"
    # Target column is optional based on whether targets.csv exists
    valid_lengths = [len(expected_columns), len(expected_columns) + 1]
    assert len(df.columns) in valid_lengths, "Total columns count mismatch."

def test_cheat_sheet_has_no_keepers():
    df = pd.read_csv(CHEAT_SHEET_PATH)
    if KEEPERS_PATH.exists():
        keepers_df = pd.read_csv(KEEPERS_PATH)
        keepers = keepers_df.get('Name', pd.Series()).tolist()
        # Ensure no keeper is present in the final output
        for keeper in keepers:
            assert keeper not in df['Name'].values, f"Keeper {keeper} unexpectedly found in cheat sheet."

def test_cheat_sheet_values_are_numeric_and_finite():
    df = pd.read_csv(CHEAT_SHEET_PATH)
    assert pd.api.types.is_numeric_dtype(df['Draft_Day_Inflation_Value']), "Draft_Day_Inflation_Value is not numeric."
    assert df['Draft_Day_Inflation_Value'].notna().all(), "There are null values in Draft_Day_Inflation_Value."

def test_cheat_sheet_matches_clean_auction_values():
    if not CLEAN_VALUES_PATH.exists():
        return
    df_cheat = pd.read_csv(CHEAT_SHEET_PATH)
    df_clean = pd.read_csv(CLEAN_VALUES_PATH)
    
    assert len(df_cheat) == len(df_clean), "Row count mismatch between cheat sheet and clean values (they should have the identical player pool)."
    
    # Check if the exact player pool and ordering matches
    assert list(df_cheat['Name']) == list(df_clean['Name']), "Player names or ordering do not identically match between sheets."
    
def test_cheat_sheet_is_sorted():
    df = pd.read_csv(CHEAT_SHEET_PATH)
    values = df['Draft_Day_Inflation_Value'].tolist()
    # It should be sorted completely largest to smallest
    sorted_values = sorted(values, reverse=True)
    assert values == sorted_values, "Cheat sheet is not strictly sorted by Draft_Day_Inflation_Value in descending order. Found values out of order."
