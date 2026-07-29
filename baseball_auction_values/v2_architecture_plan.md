# Fantasy Baseball Auction Value Calculator - V2 Architecture Plan

If I were to build a fantasy baseball auction value calculator from scratch today, I would architect it as a robust, modular data pipeline. The biggest challenges in these projects are **data normalization (player name mapping)**, **handling variable league settings**, and **accurate replacement-level calculations**.

Here is the blueprint for how I would accomplish this from scratch using modernizing the approach:

## 1. Tech Stack & Architecture
*   **Language & Core Libraries:** Python, `pandas` (for heavy data manipulation), and `numpy` (for math).
*   **Validation:** `pydantic` to enforce strict data schemas (ensuring every projection source has standard fields like `PA`, `HR`, `SB` before processing).
*   **Configuration:** A `config.yaml` file to store league settings (number of teams, budget, roster spots by position, hitting/pitching split percentage) so the code doesn't need to be touched if settings change.
*   **CLI / GUI:** `typer` or `argparse` for a clean command-line interface, and potentially a `Streamlit` web interface for the draft instead of a local Tkinter app (which handles search and filtering beautifully).

## 2. Phase 1: Data Ingestion & ID Normalization
The most common point of failure is matching "Shohei Ohtani" in one file to "Shohei Ohtani (DH)" in another, or "Ronald Acuña Jr." vs "Ronald Acuna".
*   **Universal ID Mapping:** I would use the **Chadwick Bureau Register** (a publicly available mapping of MLBAM, Fangraphs, Baseball-Reference, and RetroSheet IDs).
*   **Ingestion Pipeline:** Read all CSVs in the `projections/` folder. Map their native IDs to the universal `Fangraphs ID` or `MLBAM ID`. Drop the names for the merge, relying entirely on the ID.
*   **Projection Blending:** Write an aggregator that allows me to apply weights in the config (e.g., `ATC: 60%, The BAT X: 40%`) to create a single, master projection dataframe.

## 3. Phase 2: Positional Scarcity & Replacement Levels
You can't just compare a catcher's stats to a first baseman's stats directly.
*   **Position Eligibility:** Create a definitive dictionary of player positions. For multi-eligible players, assign them to their most "scarce" or valuable position for replacement calculations.
*   **Find Replacement Level:** If a 12-team league starts 2 Catchers, 24 Catchers will be drafted. The replacement level Catcher is the 25th best Catcher. The pipeline will sort projected points/stats per position and identify the baseline replacement player for *every* position.

## 4. Phase 3: The Valuation Math (Z-Scores or SGP)
Instead of arbitrary rankings, build a mathematical foundation base:
*   **Calculate Base Z-Scores:** For 5x5 category leagues, calculate the Z-Score (standard deviations above the replacement level mean) for each category (HR, RBI, R, SB, OBP).
*   *Note on Rates:* For rate stats (AVG, ERA, WHIP), weight them by volume (AB or IP). A .300 AVG over 600 ABs is much more valuable than .300 over 200 ABs.
*   **Sum Z-Scores:** Sum the category Z-Scores into an arbitrary `Total Value` number for every player above replacement. 

## 5. Phase 4: Dollar Conversion & Keeper Inflation
This is where draft strategy kicks in.
*   **League Economy:** Total Budget = (12 teams * $260) = $3,120.
*   **Marginal Dollars:** Subtract the mandated $1 minimum bids for all drafted roster spots. The remainder is the "Marginal Money".
*   **Divide and Assign:** Divide the total Marginal Money by the sum of all positive Z-Scores in the player pool. This gives you a `Dollars per Unit of Z-Score` ratio. Multiply each player's Z-Score by this ratio, add $1, and you have their Auction Value.
*   **Keeper Adjustment (`keepers.csv`):** Read the keepers file. Calculate what those players *should* have cost vs. what they *are being kept for*. Take the surplus/deficit money and redistribute it back into the remaining pool space to calculate exact draft-day inflation values.

## 6. Phase 5: Output & Draft Day Tool
*   **Outputs:** Generate clean `hitter_values.csv`, `pitcher_values.csv`, and `overall_values.csv` artifacts. 
*   **The Draft App:** Instead of static CSVs, I would feed the final dataframe into a local `Streamlit` app. This allows you to have a live draft dashboard where you can check off players as they are drafted, instantly recalculating inflation and adjusting available dollar values on the fly.

## Directory Structure Blueprint
```text
fantasy_baseball_engine/
├── config.yaml          # League rules ($260, 5x5, Roster limits, Source weights)
├── data/
│   ├── raw_projections/ # Steamer, ATC, The Bat CSVs
│   ├── mappings/        # Chadwick Bureau ID map
│   └── keepers/         # keepers.csv
├── src/
│   ├── ingest.py        # Normalizes inputs and maps IDs
│   ├── calculate.py     # Z-Score and SGP math based on replacement levels
│   ├── economy.py       # Dollar conversions and keeper inflation
│   └── output.py        # Formatting exports
├── app.py               # Streamlit draft-day dashboard
└── main.py              # The CLI orchestrator
```
