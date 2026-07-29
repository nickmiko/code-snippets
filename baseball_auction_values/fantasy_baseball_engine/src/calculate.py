import pandas as pd

import re

def get_positions(pos_string):
    return [p.strip() for p in re.split(r'[/,]', str(pos_string)) if p.strip()]

def prepare_custom_categories(df_hitters, df_pitchers, config):
    hitters = df_hitters.copy()
    pitchers = df_pitchers.copy()
    
    cats_p = config['categories'].get('pitchers', [])
    
    if 'QS+W' in cats_p:
        if 'QS' in pitchers.columns and 'W' in pitchers.columns:
            pitchers['QS+W'] = pitchers['QS'] + pitchers['W']
        elif 'W' in pitchers.columns:
            pitchers['QS+W'] = pitchers['W']
        elif 'QS' in pitchers.columns:
            pitchers['QS+W'] = pitchers['QS']
            
    if 'SV+HLD' in cats_p:
        if 'SV' in pitchers.columns and 'HLD' in pitchers.columns:
            pitchers['SV+HLD'] = pitchers['SV'] + pitchers['HLD']
        elif 'SV' in pitchers.columns:
            pitchers['SV+HLD'] = pitchers['SV']
        elif 'HLD' in pitchers.columns:
            pitchers['SV+HLD'] = pitchers['HLD']
            
    return hitters, pitchers

def calculate_replacement_levels(df_hitters: pd.DataFrame, df_pitchers: pd.DataFrame, config: dict) -> dict:
    teams = config['league_settings']['teams']
    roster_limits = config['roster']
    
    # Calculate initial Z scores with an empty replacement proxy to find pure statistical values
    ht, pt = calculate_zscores(df_hitters, df_pitchers, {}, config)
    
    ht['Rank_Score'] = ht['Total_Z']
    pt['Rank_Score'] = pt['Total_Z']

    rep_levels = {}
    
    ht = ht.sort_values(by='Rank_Score', ascending=False)
    pt = pt.sort_values(by='Rank_Score', ascending=False)

    for pos, limit in roster_limits.items():
        if limit == 0: continue
        total_drafted = limit * teams
        
        if pos == 'UT':
            eligible = ht
        elif pos == 'MI':
            eligible = ht[ht['Pos'].str.contains('2B|SS', regex=True, na=False)]
        elif pos == 'CI':
            eligible = ht[ht['Pos'].str.contains('1B|3B', regex=True, na=False)]
        elif pos in ['C', '1B', '2B', '3B', 'SS', 'OF']:
            eligible = ht[ht['Pos'].str.contains(pos, na=False)]
        elif pos == 'P':
            eligible = pt
        elif pos in ['SP', 'RP']:
            eligible = pt[pt['Pos'].str.contains(pos, na=False)]
        elif pos == 'Bench':
            continue
        else:
            continue
            
        if len(eligible) > total_drafted:
            rep_player = eligible.iloc[total_drafted - 1]
            rep_levels[pos] = rep_player['Rank_Score']
        else:
            rep_levels[pos] = 0

    return rep_levels

def calculate_zscores(df_hitters: pd.DataFrame, df_pitchers: pd.DataFrame, replacement_levels: dict, config: dict) -> tuple:
    hitters, pitchers = prepare_custom_categories(df_hitters, df_pitchers, config)
    
    cats_h = config['categories']['hitters']
    cats_p = config['categories']['pitchers']
    
    if 'AB' in hitters.columns and 'H' in hitters.columns:
        pool_avg = hitters['H'].sum() / hitters['AB'].sum()
        hitters['Impact_AVG'] = hitters['AB'] * (hitters['AVG'] - pool_avg)
        
    if 'PA' in hitters.columns and 'OBP' in hitters.columns:
        pool_obp = (hitters['OBP'] * hitters['PA']).sum() / hitters['PA'].sum()
        hitters['Impact_OBP'] = hitters['PA'] * (hitters['OBP'] - pool_obp)
        
    h_z_cats = []
    for cat in cats_h:
        if cat == 'AVG': h_z_cats.append(('Impact_AVG', cat))
        elif cat == 'OBP': h_z_cats.append(('Impact_OBP', cat))
        else: h_z_cats.append((cat, cat))
        
    weights = config.get('category_weights', {})

    for calc_col, orig_col in h_z_cats:
        if calc_col in hitters.columns:
            weight = weights.get(orig_col, weights.get('defaults', 1.0)) * (-1 if orig_col == 'K' else 1)
            hitters[f'{calc_col}_Z'] = ((hitters[calc_col] - hitters[calc_col].mean()) / hitters[calc_col].std()) * weight
            
            # Elite Category Anchor Premium (SB)
            if orig_col == 'SB':
                hitters[f'{calc_col}_Z'] = hitters[f'{calc_col}_Z'].apply(lambda x: x * 1.15 if pd.notna(x) and x > 3.0 else x)

    z_cols = [f"{c[0]}_Z" for c in h_z_cats if c[0] in hitters.columns]
    hitters['Total_Z'] = hitters[z_cols].sum(axis=1) if len(z_cols) > 0 else 1.5
    
    if 'AB' in hitters.columns:
        hitters['Total_Z'] += ((hitters['AB'] - hitters['AB'].mean()) / hitters['AB'].std()).fillna(0) * 0.25
        
    if 'InterSD' in hitters.columns:
        hitters['Total_Z'] -= ((hitters['InterSD'] - hitters['InterSD'].mean()) / hitters['InterSD'].std()).fillna(0) * 0.1
        
    if 'wRC+' in hitters.columns:
        hitters['Total_Z'] += ((hitters['wRC+'] - 100).clip(lower=0) / 200).fillna(0)

    def calc_pos_bonus(p_str):
        p_list = [p for p in get_positions(p_str) if p.upper() != 'DH']
        if not p_list:
            return -1.0 # Penalty for strictly DH-only players due to lack of flexibility
        return max(0, len(p_list) - 1) * 0.25
        
    hitters['Total_Z'] = hitters['Total_Z'] + hitters['Pos'].apply(calc_pos_bonus)
    
    def get_rep_adj_hitter(row):
        pos_list = get_positions(row['Pos'])
        reps = [replacement_levels.get(p, 0) for p in pos_list if p in replacement_levels]
        return min(reps) if reps else 0

    hitters['Rep_Level'] = hitters.apply(get_rep_adj_hitter, axis=1)
    hitters['Z_Above_Rep'] = hitters['Total_Z'] - hitters['Rep_Level']
        
    if 'IP' in pitchers.columns and 'ER' in pitchers.columns and 'H' in pitchers.columns and 'BB' in pitchers.columns:
        pool_era = (pitchers['ER'].sum() / pitchers['IP'].sum()) * 9
        pitchers['Impact_ERA'] = pitchers['IP'] * (pool_era - (pitchers['ER'] / pitchers['IP']) * 9) / 9
        pool_whip = (pitchers['H'].sum() + pitchers['BB'].sum()) / pitchers['IP'].sum()
        pitchers['Impact_WHIP'] = pitchers['IP'] * (pool_whip - ((pitchers['H'] + pitchers['BB']) / pitchers['IP']))
        
    p_z_cats = []
    for cat in cats_p:
        if cat == 'ERA': p_z_cats.append(('Impact_ERA', cat))
        elif cat == 'WHIP': p_z_cats.append(('Impact_WHIP', cat))
        else: p_z_cats.append((cat, cat))
        
    for calc_col, orig_col in p_z_cats:
        if calc_col in pitchers.columns:
            weight = weights.get(orig_col, weights.get('defaults', 1.0))
            pitchers[f'{calc_col}_Z'] = ((pitchers[calc_col] - pitchers[calc_col].mean()) / pitchers[calc_col].std()) * weight
            
            # Elite Category Anchor Premium (SV or SV+HLD)
            if orig_col in ['SV', 'SV+HLD']:
                pitchers[f'{calc_col}_Z'] = pitchers[f'{calc_col}_Z'].apply(lambda x: x * 1.15 if pd.notna(x) and x > 3.0 else x)

    pz_cols = [f"{c[0]}_Z" for c in p_z_cats if c[0] in pitchers.columns]
    pitchers['Total_Z'] = pitchers[pz_cols].sum(axis=1) if len(pz_cols) > 0 else 1.0

    if 'IP' in pitchers.columns:
        pitchers['Total_Z'] += ((pitchers['IP'] - pitchers['IP'].mean()) / pitchers['IP'].std()).fillna(0) * 0.25
        
    if 'InterSD' in pitchers.columns:
        pitchers['Total_Z'] -= ((pitchers['InterSD'] - pitchers['InterSD'].mean()) / pitchers['InterSD'].std()).fillna(0) * 0.1
        
    if 'WAR' in pitchers.columns:
        pitchers['Total_Z'] += (pitchers['WAR'].clip(lower=0) * 0.05).fillna(0)

    pitchers['Total_Z'] = pitchers['Total_Z'] + pitchers['Pos'].apply(calc_pos_bonus)

    def get_rep_adj_pitcher(row):
        pos_list = get_positions(row['Pos']) + ['P']
        reps = [replacement_levels.get(p, 0) for p in pos_list if p in replacement_levels]
        return min(reps) if reps else 0

    pitchers['Rep_Level'] = pitchers.apply(get_rep_adj_pitcher, axis=1)
    pitchers['Z_Above_Rep'] = pitchers['Total_Z'] - pitchers['Rep_Level']

    return hitters, pitchers

