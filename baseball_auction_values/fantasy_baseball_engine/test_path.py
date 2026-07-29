from pathlib import Path
print("Path resolves to:", Path('src/ingest.py').resolve().parent.parent.parent / 'old_outputs' / 'fangraphs-auction-calculator.csv')
