import pandas as pd
from pathlib import Path

def apply_projection_weights(df_hitters: pd.DataFrame, df_pitchers: pd.DataFrame, config: dict, data_dir: str) -> tuple:
    hitters = df_hitters.copy()
    pitchers = df_pitchers.copy()
    
    weights = config.get('projections_weights', {})
    p_weights = weights.get('pitchers', {}) if 'pitchers' in weights else weights
    
    atc_weight = p_weights.get('atc', 1.0)
    pl_weight = p_weights.get('pitcherlist', 0.0)
    
    if pl_weight > 0.0:
        pl_file = Path(data_dir) / 'pitcherlist_dollar_values.csv'
        if pl_file.exists():
            print(f"Blending PitcherList projections: {atc_weight*100}% ATC / {pl_weight*100}% PL...")
            pl_df = pd.read_csv(pl_file)
            
            # Use output.py's normalize_player_name if possible, otherwise simple lowercase matching
            def normalize_name(name):
                import re
                if pd.isna(name): return ""
                return re.sub(r'[^a-z0-9]', '', str(name).lower())
            
            pl_df['Match_Name'] = pl_df['Name'].apply(normalize_name)
            pitchers['Match_Name'] = pitchers['Name'].apply(normalize_name)
            
            pl_dict = dict(zip(pl_df['Match_Name'], pl_df['dollar_value']))
            
            def blend(row):
                base = row['Base_Value']
                pl_val = pl_dict.get(row['Match_Name'], 0.0)
                # Weighted average
                return round((base * atc_weight) + (pl_val * pl_weight), 2)
            
            pitchers['Base_Value'] = pitchers.apply(blend, axis=1)
            pitchers.drop(columns=['Match_Name'], inplace=True, errors='ignore')
        else:
            print("WARNING: pitcherlist_dollar_values.csv not found in raw_projections folder.")
            
    return hitters, pitchers
