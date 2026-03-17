# Baseball Auction Value Calculator

This script calculates fantasy baseball auction values for players based on multiple projection systems. It processes raw player projection data, calculates z-score based rankings and dollar values for each projection system, and then consolidates these values into a single CSV file. The script also computes a weighted average of the dollar values from the different projection systems to provide a unified valuation.

## Description

The script is designed to help fantasy baseball managers prepare for their auction drafts. By combining multiple sources of player projections, it aims to provide a more robust and reliable set of auction values than any single projection system could offer. It calculates player values for both hitters and pitchers and then merges them into a comprehensive list.

The core logic involves:
1.  Reading player projection data from various sources (e.g., ATC, Steamer, Depth Charts).
2.  For each projection system, calculating z-scores for relevant statistical categories.
3.  Converting these z-scores into a dollar value for each player.
4.  Aggregating the dollar values from all projection systems into a single master file.
5.  Calculating a weighted average dollar value for each player based on configurable weights for each projection system.

## Usage

To run the script, first change into the `baseball_auction_values/` directory and then execute:

```bash
cd baseball_auction_values
python create_auction_values.py
```

Once you are inside the `baseball_auction_values/` directory, you can also run the script directly:

```bash
python create_auction_values.py
```

If you are using `uv`, you can run the pre-configured script from inside `baseball_auction_values/`:

```bash
uv run create-values
```

## Dependencies

The project dependencies are declared in `baseball_auction_values/pyproject.toml`:

*   `pandas>=2.0`
*   `numpy>=1.24`
*   `streamlit>=1.30`
*   `pyarrow>=14.0`

The recommended way to install dependencies is via [`uv`](https://github.com/astral-sh/uv). From inside the `baseball_auction_values/` directory:

```bash
uv sync
```

Alternatively, you can install with pip:

```bash
pip install pandas numpy streamlit pyarrow
```

The script also uses the following local modules, which must be present in the same directory:
*   `zscore.py`
*   `rankings_zscore.py`

## File Structure

### Input Files

The script expects the following files and directories to be in place:

*   `projections/`: This directory should contain the raw projection data as CSV files. For each projection system, there should be two files: one for hitters and one for pitchers. The files should be named in the format `<system>_hitter.csv` and `<system>_pitcher.csv`.
    *   Example: `projections/atc_hitter.csv`, `projections/atc_pitcher.csv`

*   `keepers.csv`: This file should contain a list of players who are designated as "keepers" and will be excluded from the auction value calculations. The file should have a `Name` column with player names and a `dollar_value` column with the keeper cost.

### Output Files

The script will generate the following output:

*   `auction_values/`: This directory will be created if it doesn't exist.
*   `auction_values/all_auction_values.csv`: This is the main output file. It contains a table of players with the following columns:
    *   `Name`: The player's name.
    *   One column for each projection system (e.g., `atc`, `steamer`), containing the calculated dollar value for that player from that system.
    *   `projection_weighted_average`: The weighted average of the dollar values across all projection systems.
    *   `projection_std`: The standard deviation of the dollar values, which can be used as a measure of the consensus (or lack thereof) on a player's value.

## Configuration

The script can be configured by modifying the `ProjectionConfig` dataclass within `create_auction_values.py`:

*   `PROJECTION_WEIGHTS`: A dictionary where keys are the names of the projection systems (matching the prefixes of the filenames in the `projections` directory) and values are their corresponding weights for the weighted average calculation.
*   `DEFAULT_BUDGET`: The default auction budget for the league.
*   `DEFAULT_ROSTER_SIZE`: The default number of players on a roster.
