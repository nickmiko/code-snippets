# Auction Project

This project is designed to calculate auction values for players in a fantasy sports league based on their performance statistics. It provides tools for both hitters and pitchers, allowing users to evaluate player value effectively.

## Project Structure

The project consists of the following files:

- `src/create_auction_values.py`: Contains a class that handles the creation of auction values for players based on their statistics. It includes methods for reading data, preprocessing it, calculating weighted z-scores, calculating auction values, saving results, and running the full ranking process.

- `src/zscore.py`: Defines a function `calculate_auction_values` that calculates auction values based on weighted z-scores for hitters and pitchers. It takes DataFrames for hitters and pitchers, along with various parameters for budget and weights, and returns updated DataFrames with calculated dollar values.

- `src/__init__.py`: Marks the directory as a Python package. This file can be empty or contain initialization code for the package.

- `requirements.txt`: Lists the dependencies required for the project, typically including libraries like pandas and numpy.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data for hitters and pitchers in the form of pandas DataFrames.
2. Use the `calculate_auction_values` function from `zscore.py` to compute the auction values.
3. For more advanced usage, you can utilize the class in `create_auction_values.py` to manage the entire process from data reading to auction value calculation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.