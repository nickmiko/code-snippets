from src.ingest import load_config, load_projections
c = load_config()
h, p = load_projections(config=c)
print(h['Pos'].value_counts())
