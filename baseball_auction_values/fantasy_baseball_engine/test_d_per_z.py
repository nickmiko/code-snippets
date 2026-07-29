import pandas as pd
h = pd.read_csv('output/hitter_values.csv')
p = pd.read_csv('output/pitcher_values.csv')

total_hz = h.loc[h['Z_Above_Rep'] > 0, 'Z_Above_Rep'].sum()
total_pz = p.loc[p['Z_Above_Rep'] > 0, 'Z_Above_Rep'].sum()

print("Hitter Total Z:", total_hz)
print("Pitcher Total Z:", total_pz)

h_marg = (260 * 12 * 0.5) - sum(h['Z_Above_Rep'] > 0)
p_marg = (260 * 12 * 0.5) - sum(p['Z_Above_Rep'] > 0)

print("Hitter D/Z:", h_marg / total_hz)
print("Pitcher D/Z:", p_marg / total_pz)

print("\nTop 5 H Z_Above_Rep:\n", h[['Name', 'Z_Above_Rep']].head(5))
print("\nTop 5 P Z_Above_Rep:\n", p[['Name', 'Z_Above_Rep']].head(5))
