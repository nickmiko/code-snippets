from src.calculate import calculate_replacement_levels, calculate_zscores
from src.ingest import load_config, load_projections
c = load_config()
h, p = load_projections(config=c)
rep = calculate_replacement_levels(h, p, c)
print("Rep Levels:", rep)
