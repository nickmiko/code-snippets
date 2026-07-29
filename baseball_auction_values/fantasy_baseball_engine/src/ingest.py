import pandas as pd
import yaml
from pathlib import Path

def load_config(config_path: str = 'config.yaml') -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_projections(data_dir: str = 'data/raw_projections', config: dict = None) -> tuple:
    print(f"Loading projections from {data_dir}...")
    
    # Load raw projections
    h_file = Path(data_dir) / 'atc_hitter.csv'
    p_file = Path(data_dir) / 'atc_pitcher.csv'
    
    hitters = pd.read_csv(h_file) if h_file.exists() else pd.DataFrame()
    pitchers = pd.read_csv(p_file) if p_file.exists() else pd.DataFrame()

    # Load positions from a known mapping file because ATC does not have Pos logic built in directly
    base_dir = Path(__file__).resolve().parent.parent.parent
    mapping_file = base_dir / 'old_outputs' / 'fangraphs-auction-calculator.csv'
    
    if mapping_file.exists():
        mappings = pd.read_csv(mapping_file)
        if 'PlayerId' in mappings.columns and 'POS' in mappings.columns:
            mappings['PlayerId'] = mappings['PlayerId'].astype(str)
            pos_dict = dict(zip(mappings['PlayerId'], mappings['POS']))
            
            if not hitters.empty and 'PlayerId' in hitters.columns:
                hitters['PlayerId'] = hitters['PlayerId'].astype(str)
                # Initialize Pos
                hitters['Pos'] = hitters['PlayerId'].map(pos_dict)

                # Fallback to FantasyPros by Name
                fp_file = base_dir / 'FantasyPros_2026_Overall_MLB_ADP_Rankings.csv'
                if fp_file.exists():
                    import unicodedata
                    def remove_accents(input_str):
                        if not isinstance(input_str, str): return input_str
                        nfkd_form = unicodedata.normalize('NFKD', input_str)
                        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

                    fp_df = pd.read_csv(fp_file)
                    def parse_pos(p_str):
                        if not isinstance(p_str, str): return 'DH'
                        spots = [s.strip() for s in p_str.split(',')]
                        valid = [s for s in spots if s not in ('DH', 'UT')]
                        return valid[0] if valid else spots[0]
                    
                    fp_df['norm_player'] = fp_df['Player'].apply(remove_accents)
                    fp_dict = dict(zip(fp_df['norm_player'], fp_df['Positions'].apply(parse_pos)))
                    
                    if 'Name' in hitters.columns:
                        hitters['norm_name'] = hitters['Name'].apply(remove_accents)
                        fp_mapped = hitters['norm_name'].map(fp_dict)
                        # Override if missing, or if it is a dump-bucket like DH/UT 
                        mask = hitters['Pos'].isna() | hitters['Pos'].isin(['DH', 'UT'])
                        hitters.loc[mask & fp_mapped.notna(), 'Pos'] = fp_mapped
                
                # Final manual overrides for deep prospects or missing players before falling back to DH
                if 'Name' in hitters.columns:
                    manual_dict = {
                        'Charlie Condon': '3B', 'Braden Montgomery': 'OF', 'Jett Williams': 'SS', 
                        'James Triantos': '2B', 'Edwin Arroyo': 'SS', 'Dominic Smith': '1B', 
                        'Eloy Jiménez': 'OF', 'Jorge Alfaro': 'C', 'Myles Straw': 'OF', 
                        'Andrés Chaparro': '1B', 'Alan Roden': 'OF', 'Tirso Ornelas': 'OF', 
                        'Yohel Pozo': 'C', 'Yohandy Morales': '3B', 'Victor Mesa Jr.': 'OF', 
                        'Troy Johnston': '1B', "Tre' Morgan": '1B', 'Robert Hassell III': 'OF', 
                        'Sebastián Rivero': 'C', 'John Rave': 'OF', 'Gabriel Rincones Jr.': 'OF', 
                        'Jahmai Jones': '2B', 'Dom Keegan': 'C', 'Akil Baddoo': 'OF', 
                        'Niko Kavadas': '1B', 'Joey Meneses': '1B', 'Mickey Gasper': 'C'
                    }
                    mask = hitters['Pos'].isna() | hitters['Pos'].isin(['DH', 'UT'])
                    manual_mapped = hitters['Name'].map(manual_dict)
                    hitters.loc[mask & manual_mapped.notna(), 'Pos'] = manual_mapped
                
                hitters['Pos'] = hitters['Pos'].fillna('DH')
                hitters['Pos'] = hitters['Pos'].replace({'LF': 'OF', 'CF': 'OF', 'RF': 'OF'})
                
            if not pitchers.empty and 'PlayerId' in pitchers.columns:
                pitchers['PlayerId'] = pitchers['PlayerId'].astype(str)
                pitchers['Pos'] = pitchers['PlayerId'].map(pos_dict).fillna('P')
    
    if 'Pos' not in hitters.columns and not hitters.empty:
        hitters['Pos'] = 'DH'
        
    if not pitchers.empty:
        # If Pos is just "P" or completely missing, let's infer SP/RP from standard columns
        if 'Pos' not in pitchers.columns:
            pitchers['Pos'] = 'P'
            
        def infer_pitcher_pos(row):
            pos = str(row.get('Pos', 'P'))
            if pos == 'P':
                # Distinguish based on Games Started vs Games
                g = row.get('G', 0)
                gs = row.get('GS', 0)
                # If they start more than 30% of their games, call them an SP, otherwise RP
                if g > 0 and (gs / g) >= 0.3:
                    return 'SP'
                else:
                    return 'RP'
            return pos
            
        pitchers['Pos'] = pitchers.apply(infer_pitcher_pos, axis=1)
        
    if not hitters.empty and 'SO' in hitters.columns:
        hitters.rename(columns={'SO': 'K'}, inplace=True)
    if not pitchers.empty and 'SO' in pitchers.columns:
        pitchers.rename(columns={'SO': 'K'}, inplace=True)
        
    return hitters, pitchers