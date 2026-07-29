import pandas as pd
mappings = pd.read_csv('../old_outputs/fangraphs-auction-calculator.csv')
print("Mapping PlayerId:", mappings['PlayerId'].head().tolist())

hitters = pd.read_csv('data/raw_projections/atc_hitter.csv')
print("Hitter PlayerId:", hitters['PlayerId'].head().tolist())
mappings['PlayerId'] = mappings['PlayerId'].astype(str)
pos_dict = dict(zip(mappings['PlayerId'], mappings['POS']))
hitters['PlayerId'] = hitters['PlayerId'].astype(str)
hitters['Pos'] = hitters['PlayerId'].map(pos_dict)
print(hitters[['Name', 'PlayerId', 'Pos']].head())
