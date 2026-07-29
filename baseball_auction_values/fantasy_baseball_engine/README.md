# Fantasy Baseball Auction Engine (V2)

A robust, modular data pipeline designed to calculate custom fantasy baseball auction/salary cap values based on raw player projections and customized league settings.

Unlike simple averaging calculators, this engine blends raw *baseball statistics* first, accurately calculates volume-weighted category Z-Scores considering positional scarcity, dynamically shifts available dollars based on keeper inflation, and provides a live draft-day dashboard.

## Features
* **Custom League Settings:** Easily configure budget, team count, roster composition, and hitting/pitching splits.
* **Projection Blending:** Drop in multiple projection systems (ATC, Steamer, The BAT, etc.) and weight their stats based on your preferences.
* **Smart Valuation Math:** Accurately accounts for positional scarcity, replacement levels, and volume-adjusted rate stats (AVG, ERA, WHIP).
* **Keeper Inflation Engine:** Input your league's saved keepers to automatically adjust the marginal dollar value for all remaining free agents in the pool.
* **Clean Exports:** Generates a clean, draft-ready `clean_auction_values.csv` sheet containing both pure value (uninflated) and draft day value (inflated) target bids.
* **Live Draft App:** Included Streamlit application to mark drafted players and watch the available economy values update on the board.

---

## 1. Installation

Requires Python 3.9+.

```bash
cd fantasy_baseball_engine
pip install -r requirements.txt
```

---

## 2. Configuration & Data Setup

Before running the engine, ensure your data is populated in the correct directories and your `config.yaml` matches your league parameters.

### `config.yaml`
Edit this file to match your league:
* **teams:** Number of teams in the league.
* **budget:** Total salary cap per team (e.g., $260).
* **splits:** Desired percentage of budget allocated to hitting vs pitching.
* **roster limits:** The number of starters at each position.
* **projections_weights:** Weighted variables for each projection source you use (e.g. `atc: 0.6`, `steamer: 0.4`).

### Folders structure
Drop your CSV data into the following `data/` subdirectories:

1. **`data/raw_projections/`**: Place raw CSV files from Fangraphs here (e.g., `atc_hitters.csv`, `steamer_pitchers.csv`). The engine will read all of them and aggregate them based on the weights.
2. **`data/mappings/`**: Optional. Add `chadwick_register.csv` here to map and standardize player IDs universally.
3. **`data/keepers/`**: Add a `keepers.csv` with at least a `Name` and `Cost` column. If this exists, the engine will extract these players from the free agent pool and redistribute their surplus/deficit value into the draft-day inflation logic.

---

## 3. Running the Pipeline

Once your data is loaded and your config is set, generate the custom auction values with the Typer CLI:

```bash
python main.py run-pipeline
```

*(You can also use a custom config file: `python main.py run-pipeline --config-name myleague_config.yaml`)*

**Outputs:**
The generated CSVs will be deposited in the `output/` directory:
* `clean_auction_values.csv` - The final, human-readable draft cheat sheet.
* `hitter_values.csv` & `pitcher_values.csv` - Split raw data for deeper review.
* `overall_values_debug.csv` - Complete table including intermediate mathematical z-scores and true replacement level math for diagnostics.

---

## 5. Post-Draft Analysis

The engine also includes tools for post-draft analysis, allowing you to evaluate your team and potential trades.

### `config.yaml`
For post-draft analysis, make sure you have the following in your `config.yaml`:
* **league_id:** Your Fantrax league ID.
* **my_team_name:** Your team name as it appears on Fantrax.

### Commands

#### Standings
View the projected standings for your league based on Z-Scores.

```bash
python main.py post-draft standings
```

If you haven't set your `league_id` in the config, you can provide it as an option:
```bash
python main.py post-draft standings --league-id YOUR_LEAGUE_ID
```

#### Evaluate Trade
Evaluate a potential trade between your team and another team in the league.

```bash
python main.py post-draft evaluate-trade --team2-name "Other Team Name" --team1-players "Player A, Player B" --team2-players "Player C, Player D"
```

The command will use your `league_id` and `my_team_name` from the config file. You can also provide them as options if needed.

---

## Architecture Overview

**`src/ingest.py`**: Normalizes ID mappings and blends weighted stat lines across all raw projection CSVs.
**`src/calculate.py`**: Generates replacement levels based on configuration thresholds, weights volume stats, and calculates positional Z-Scores for every category.
**`src/economy.py`**: Subtracts $1 minimum bids, applies Keeper-specific savings/deficits, and calculates the true `Dollars per Unit of Z-Score` ratio to assign monetary value.
**`src/output.py`**: Formats the technical mathematical DataFrames into clean, readable CSV files.
