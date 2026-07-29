import pandas as pd
from pathlib import Path

def compute_base_dollars(df_hitters, df_pitchers, h_budget, p_budget, total_roster_spots):
    hitters = df_hitters.copy()
    pitchers = df_pitchers.copy()
    
    # Calculate Marginal Dollars
    h_min_bids = sum(hitters['Z_Above_Rep'] > 0)
    p_min_bids = sum(pitchers['Z_Above_Rep'] > 0)
    
    h_marginal = h_budget - h_min_bids
    p_marginal = p_budget - p_min_bids
    
    total_h_z = hitters.loc[hitters['Z_Above_Rep'] > 0, 'Z_Above_Rep'].sum()
    total_p_z = pitchers.loc[pitchers['Z_Above_Rep'] > 0, 'Z_Above_Rep'].sum()
    
    hitter_d_per_z = h_marginal / total_h_z if total_h_z > 0 else 0
    pitcher_d_per_z = p_marginal / total_p_z if total_p_z > 0 else 0
    
    hitters['Base_Value'] = hitters['Z_Above_Rep'].apply(lambda z: round(1 + (z * hitter_d_per_z), 2) if z > 0 else 0)
    pitchers['Base_Value'] = pitchers['Z_Above_Rep'].apply(lambda z: round(1 + (z * pitcher_d_per_z), 2) if z > 0 else 0)
    
    return hitters, pitchers

def compute_marginal_dollars(df_hitters: pd.DataFrame, df_pitchers: pd.DataFrame, config: dict) -> tuple:
    teams = config['league_settings']['teams']
    budget = config['league_settings']['budget']
    h_split = config['league_settings']['hitter_split']
    p_split = config['league_settings']['pitcher_split']
    
    total_money = teams * budget
    total_roster_spots = sum(config['roster'].values())
    
    h_budget = total_money * h_split
    p_budget = total_money * p_split
    
    return compute_base_dollars(df_hitters, df_pitchers, h_budget, p_budget, total_roster_spots)

def apply_keeper_inflation(df_hitters: pd.DataFrame, df_pitchers: pd.DataFrame, config: dict, data_dir: str) -> tuple:
    keeper_file = Path(data_dir) / 'keepers.csv'
    
    if not keeper_file.exists():
        print("No keepers.csv found. Skipping inflation adjustment...")
        df_hitters['Value'] = df_hitters['Base_Value']
        df_pitchers['Value'] = df_pitchers['Base_Value']
        return df_hitters, df_pitchers
    
    print("Reading keepers.csv and applying inflation...")
    keepers = pd.read_csv(keeper_file)
    
    teams = config['league_settings']['teams']
    total_budget = teams * config['league_settings']['budget']
    h_split = config['league_settings']['hitter_split']
    p_split = config['league_settings']['pitcher_split']
    
    # Track Keeper Money Kept and Base Value Kept
    h_kept_money = 0
    p_kept_money = 0
    h_spent_money = 0
    p_spent_money = 0
    
    df_hitters['Is_Keeper'] = False
    df_pitchers['Is_Keeper'] = False
    
    for _, row in keepers.iterrows():
        # Match keeper by name
        name = row.get('Name')
        cost = float(row.get('Cost', row.get('Keeper_Cost', 1)))
        
        if name in df_hitters['Name'].values:
            idx = df_hitters[df_hitters['Name'] == name].index[0]
            df_hitters.at[idx, 'Is_Keeper'] = True
            h_kept_money += df_hitters.at[idx, 'Base_Value']
            h_spent_money += cost
        if name in df_pitchers['Name'].values:
            idx = df_pitchers[df_pitchers['Name'] == name].index[0]
            df_pitchers.at[idx, 'Is_Keeper'] = True
            p_kept_money += df_pitchers.at[idx, 'Base_Value']
            p_spent_money += cost

    # Calculate Inflation. If players are kept for cheaper than base value, money goes UP
    h_budget = (total_budget * h_split) - h_spent_money
    p_budget = (total_budget * p_split) - p_spent_money
    
    # Filter only available players to calculate new base D per Z ratio
    av_hitters = df_hitters[~df_hitters['Is_Keeper']]
    av_pitchers = df_pitchers[~df_pitchers['Is_Keeper']]
    
    # Note total_roster_spots needs adjustment for keepers, omitting complex calculation for demo
    av_h, av_p = compute_base_dollars(av_hitters, av_pitchers, h_budget, p_budget, sum(config['roster'].values()))
    
    # We want to map the newly calculated base value (from available pool) to the 'Value' column
    # while preserving original 'Base_Value' for everyone to show draft inflation.
    
    # Rename for merge to avoid overriding Base_Value
    av_h.rename(columns={'Base_Value': 'Value'}, inplace=True)
    av_p.rename(columns={'Base_Value': 'Value'}, inplace=True)
    
    df_hitters['Value'] = df_hitters['Base_Value']
    df_pitchers['Value'] = df_pitchers['Base_Value']
    
    # Map back available inflation-adjusted values into the 'Value' column
    df_hitters.update(av_h[['Value']])
    df_pitchers.update(av_p[['Value']])
    
    return df_hitters, df_pitchers
