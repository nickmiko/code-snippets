import pandas as pd
from pathlib import Path
import unicodedata
import re

def normalize_player_name(name):
    if pd.isna(name): return ""
    # Remove common fantasy labels
    name = str(name).replace(' (Batter)', '').replace(' (Pitcher)', '')
    # Remove accents and diacritics
    name = ''.join(c for c in unicodedata.normalize('NFKD', name) if unicodedata.category(c) != 'Mn')
    name = name.lower()
    # Remove periods and apostrophes
    name = re.sub(r"[.']", "", name)
    # Replace hyphens with spaces
    name = re.sub(r"[-]", " ", name)
    # Remove common suffixes like Jr, Sr, II
    name = re.sub(r"\b(jr|sr|ii|iii)\b", "", name)
    # Trim multiple spaces
    return " ".join(name.split())

def generate_tiers(df):
    draftable = df[df['Value'] > 0]
    if len(draftable) < 2:
        df['Tier'] = 7
        return df
        
    v_mean = draftable['Value'].mean()
    v_std = draftable['Value'].std()
    
    def get_tier(v):
        if v <= 0: return 7
        z = (v - v_mean) / v_std
        if z >= 2.0: return 1
        elif z >= 1.0: return 2
        elif z >= 0.5: return 3
        elif z >= 0.0: return 4
        elif z >= -0.5: return 5
        else: return 6
        
    df['Tier'] = df['Value'].apply(get_tier)
    return df

def generate_positional_tiers(df):
    import re
    def get_primary_pos(p_str):
        p_list = [p.strip() for p in re.split(r'[/,]', str(p_str)) if p.strip()]
        non_dh = [p for p in p_list if p.upper() != 'DH']
        return non_dh[0] if non_dh else (p_list[0] if p_list else 'DH')

    df['Primary_Pos'] = df['Pos'].apply(get_primary_pos)
    df['Pos_Tier'] = df['Primary_Pos'] + ' 7'
    
    for pos, group in df.groupby('Primary_Pos'):
        draftable = group[group['Value'] > 0]
        if len(draftable) < 2:
            continue
            
        v_mean = draftable['Value'].mean()
        v_std = draftable['Value'].std()
        
        if v_std == 0:
            continue
            
        def get_pos_tier(v):
            if v <= 0: return f"{pos} 7"
            z = (v - v_mean) / v_std
            if z >= 2.0: return f"{pos} 1"
            elif z >= 1.0: return f"{pos} 2"
            elif z >= 0.5: return f"{pos} 3"
            elif z >= 0.0: return f"{pos} 4"
            elif z >= -0.5: return f"{pos} 5"
            else: return f"{pos} 6"
            
        df.loc[group.index, 'Pos_Tier'] = group['Value'].apply(get_pos_tier)
        
    return df

def export_values(df_hitters: pd.DataFrame, df_pitchers: pd.DataFrame, output_dir: str = 'output', config: dict = None):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if 'Is_Keeper' in df_hitters.columns:
        df_hitters = df_hitters[~df_hitters['Is_Keeper']].copy()
        df_hitters.drop(columns=['Is_Keeper'], inplace=True)
    if 'Is_Keeper' in df_pitchers.columns:
        df_pitchers = df_pitchers[~df_pitchers['Is_Keeper']].copy()
        df_pitchers.drop(columns=['Is_Keeper'], inplace=True)
        
    # Add Target logic before tiers
    targets_file = Path("data/targets/targets.csv")
    target_map = None
    has_targets = False
    if targets_file.exists():
        try:
            targets_df = pd.read_csv(targets_file)
            targets_df.columns = targets_df.columns.str.strip()
            if "Name" in targets_df.columns and "Target" in targets_df.columns:
                targets_df["norm_name_target"] = targets_df["Name"].apply(normalize_player_name)
                target_map = dict(zip(targets_df["norm_name_target"], targets_df["Target"].astype(bool)))
                has_targets = True
        except Exception:
            pass

    if has_targets:
        df_hitters["norm"] = df_hitters["Name"].apply(normalize_player_name)
        df_hitters["Target"] = df_hitters["norm"].map(target_map).fillna(False)
        df_hitters.loc[df_hitters["Target"] == True, "Value"] = df_hitters["Value"].clip(lower=1.0)
        df_hitters.drop(columns=["norm"], inplace=True)
        
        df_pitchers["norm"] = df_pitchers["Name"].apply(normalize_player_name)
        df_pitchers["Target"] = df_pitchers["norm"].map(target_map).fillna(False)
        df_pitchers.loc[df_pitchers["Target"] == True, "Value"] = df_pitchers["Value"].clip(lower=1.0)
        df_pitchers.drop(columns=["norm"], inplace=True)

    df_hitters = generate_tiers(df_hitters)
    df_hitters = generate_positional_tiers(df_hitters)
    df_pitchers = generate_tiers(df_pitchers)
    df_pitchers = generate_positional_tiers(df_pitchers)
    
    # Save hitting and pitching (Full Debug Versions)
    df_hitters.to_csv(out_path / 'hitter_values.csv', index=False)
    df_pitchers.to_csv(out_path / 'pitcher_values.csv', index=False)
    
    # Combined Overall (Full Debug Version)
    overall = pd.concat([df_hitters, df_pitchers], ignore_index=True)
    overall.sort_values(by='Value', ascending=False, inplace=True)
    
    # --- ADP Bargain Value Logic ---
    adp_file = Path(__file__).parent.parent / 'data' / 'adp' / 'FantasyPros_2026_Overall_MLB_ADP_Rankings.csv'
    if adp_file.exists():
        adp_df = pd.read_csv(adp_file)
        if 'Player' in adp_df.columns and 'Rank' in adp_df.columns:
            # Rigorously normalize both sets of names using advanced matching
            adp_df['Match_Name'] = adp_df['Player'].apply(normalize_player_name)
            overall['Match_Name'] = overall['Name'].apply(normalize_player_name)
            
            adp_map = dict(zip(adp_df['Match_Name'], adp_df['Rank']))
            
            # Map ADP
            overall['ADP_Rank'] = overall['Match_Name'].map(adp_map).fillna(9999)
            
            # Sort by ADP to map values to the market's preferred players
            overall = overall.sort_values('ADP_Rank')
            
            # Extract highest values available
            market_vals = sorted(overall.loc[overall['Value'] > 0, 'Value'].tolist(), reverse=True)
            
            # Map to market values
            market_array = [0.0] * len(overall)
            for i, val in enumerate(market_vals):
                if i < len(market_array):
                    market_array[i] = round(val, 2)
                    
            overall['Market_Value'] = market_array
            overall['Bargain_Value'] = round(overall['Value'] - overall['Market_Value'], 2)
            
            # Clean up the temporary column and sort back by 'Value'
            overall.drop(columns=['Match_Name'], inplace=True)
            overall = overall.sort_values(by='Value', ascending=False)
            
    # --- ADDED INSIGHTS ---
    overall['Overall_Rank'] = range(1, len(overall) + 1)
    overall['Pos_Rank'] = overall.groupby('Primary_Pos')['Value'].rank(ascending=False, method='first').astype(int)
    
    if 'Bargain_Value' in overall.columns and 'Market_Value' in overall.columns:
        overall['ROI_Percent'] = ((overall['Bargain_Value'] / overall['Market_Value'].apply(lambda x: x if x > 0 else 1.0)) * 100).round(1)
        
    z_cats = [c for c in overall.columns if c.endswith('_Z') and c != 'Total_Z']
    if z_cats:
        def get_profile(row):
            zdf = pd.to_numeric(row[z_cats], errors='coerce').fillna(-999)
            if zdf.max() > 2.0:
                best_cat = zdf.idxmax().replace('Impact_', '').replace('_Z', '')
                return f"{best_cat} Specialist"
            return "Balanced"
        overall['Profile'] = overall.apply(get_profile, axis=1)
    else:
        overall['Profile'] = "Balanced"
            
    overall.to_csv(out_path / 'overall_values_debug.csv', index=False)
    
    # Create the "Clean" Final Export (Stripping out the Z-score math)
    # Define columns we want to keep for the final clean sheet
    base_cols = ['Overall_Rank', 'Tier', 'Pos_Rank', 'Pos_Tier', 'Primary_Pos', 'Name', 'Team', 'Pos', 'PlayerId']
    
    if config and 'categories' in config:
        h_cats = config['categories'].get('hitters', [])
        p_cats = config['categories'].get('pitchers', [])
        stat_cols = list(dict.fromkeys(h_cats + p_cats))
    else:
        stat_cols = ['PA', 'AB', 'H', 'HR', 'RBI', 'R', 'SB', 'AVG', 'IP', 'W', 'SV', 'K', 'ER', 'BB', 'HA']
        
    value_cols = ['Base_Value', 'Market_Value', 'Bargain_Value', 'ROI_Percent', 'Profile', 'Is_Keeper', 'Value', 'Total_Z']
    
    # Check for targets.csv
    targets_file = Path('data/targets/targets.csv')
    if targets_file.exists():
        try:
            targets_df = pd.read_csv(targets_file)
            targets_df.columns = targets_df.columns.str.strip() # Handle ' Target' vs 'Target'
            if 'Name' in targets_df.columns and 'Target' in targets_df.columns:
                
                targets_df['norm_name_target'] = targets_df['Name'].apply(normalize_player_name)
                overall['norm_name_overall'] = overall['Name'].apply(normalize_player_name)
                
                target_map = dict(zip(targets_df['norm_name_target'], targets_df['Target'].astype(bool)))
                overall['Target'] = overall['norm_name_overall'].map(target_map).fillna(False)
                value_cols.append('Target')
                
                # Check for unmatched targets
                unmatched = targets_df[~targets_df['norm_name_target'].isin(overall['norm_name_overall'])]
                if not unmatched.empty:
                    print("\nWARNING: The following Target players could not be found in the projections:")
                    for _, row in unmatched.iterrows():
                        print(f" - {row['Name']}")
                    print("")
        except pd.errors.EmptyDataError:
            pass

    # Ensure individual Z-scores are included
    z_score_cols = [c for c in overall.columns if c.endswith('_Z') and c != 'Total_Z']
    
    # Only keep the columns that actually exist in the dataframe
    keep_cols = [c for c in (base_cols + stat_cols + value_cols + z_score_cols) if c in overall.columns]
    
    clean_overall = overall[keep_cols].copy()
    
    # Rename Value columns to be extremely explicit
    clean_overall.rename(columns={
        'Base_Value': 'True_Value_Uninflated',
        'Value': 'Draft_Day_Inflation_Value'
    }, inplace=True)
    
    # Round statistical columns for reading clarity
    for col in stat_cols:
        if col in clean_overall.columns:
            clean_overall[col] = clean_overall[col].round(3)
            
    clean_overall.to_csv(out_path / 'clean_auction_values.csv', index=False)
    
    # Create the secondary compact "Cheat Sheet" Export
    cheat_sheet_cols = ['Tier', 'Pos_Tier', 'Primary_Pos', 'Name', 'Profile', 'Draft_Day_Inflation_Value']
    if 'Target' in clean_overall.columns:
        cheat_sheet_cols.append('Target')
    cheat_sheet = clean_overall[cheat_sheet_cols].copy()
    cheat_sheet.to_csv(out_path / 'cheat_sheet_values.csv', index=False)
    
    # Create the third "rankings" export
    if 'Team' in clean_overall.columns:
        rankings = clean_overall[['Name', 'Team']].copy()
        rankings.rename(columns={'Team': 'rankings'}, inplace=True)
    else:
        rankings = clean_overall[['Name']].copy()
        rankings['rankings'] = ''
    rankings.to_csv(out_path / 'rankings.csv', index=False)
    
    print(f"Values exported successfully to {out_path}/")
    print(f"-> Generated 'clean_auction_values.csv' for Draft Day!")
    print(f"-> Generated 'cheat_sheet_values.csv' (compact view) for Draft Day!")
    print(f"-> Generated 'rankings.csv'!")
