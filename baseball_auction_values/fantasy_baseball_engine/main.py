import pandas as pd
import typer
from pathlib import Path
from typing import Optional

from src.calculate import (
    calculate_replacement_levels,
    calculate_zscores,
)
from src.economy import compute_marginal_dollars, apply_keeper_inflation
from src.ingest import load_projections, load_config
from src.output import export_values
from src.process import create_combined_pitcher_categories
from src.post_draft_analyzer import PostDraftAnalyzer

app = typer.Typer()
post_draft_app = typer.Typer()
app.add_typer(post_draft_app, name="post-draft", help="Commands for post-draft analysis.")


@app.command()
def run_pipeline(config_name: str = "config.yaml"):
    """
    Runs the full data pipeline to calculate auction values.
    """
    config = load_config(config_name)
    raw_data_dir = Path("data/raw_projections")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Ingest
    h_proj, p_proj = load_projections(str(raw_data_dir), config)

    # Process
    p_proj = create_combined_pitcher_categories(p_proj)

    # Calculate
    replacement_levels = calculate_replacement_levels(h_proj, p_proj, config)
    hitter_ranks, pitcher_ranks = calculate_zscores(h_proj, p_proj, replacement_levels, config)
    
    # Economy
    hitter_ranks, pitcher_ranks = compute_marginal_dollars(hitter_ranks, pitcher_ranks, config)
    
    # Keepers
    keeper_path = Path("data/keepers/keepers.csv")
    if keeper_path.exists():
        hitter_ranks, pitcher_ranks = apply_keeper_inflation(hitter_ranks, pitcher_ranks, config, str(keeper_path.parent))
    
    # Combine and Rank
    overall_ranks = pd.concat([hitter_ranks, pitcher_ranks]).sort_values(by="Z_Above_Rep", ascending=False)
    overall_ranks['Rank'] = overall_ranks['Z_Above_Rep'].rank(ascending=False, method='first')

    # Output
    export_values(overall_ranks, hitter_ranks, pitcher_ranks, str(output_dir))

    print("✅ Pipeline complete. Auction values saved to `output/` directory.")


@post_draft_app.command("standings")
def get_standings(
    league_id: Optional[str] = typer.Option(None, "--league-id", "-l"),
    config_name: str = "config.yaml",
):
    """
    Display projected standings for the league.
    """
    cfg = load_config(config_name)
    final_league_id = league_id or cfg.get("league_settings", {}).get("league_id")
    if not final_league_id:
        print("Error: League ID must be provided via --league-id or in the config file.")
        raise typer.Exit(code=1)

    config = load_config(config_name)
    raw_data_dir = Path("data/raw_projections")
    h_proj, p_proj = load_projections(str(raw_data_dir), config)
    p_proj = create_combined_pitcher_categories(p_proj)
    all_projections = pd.concat([h_proj, p_proj], ignore_index=True)

    analyzer = PostDraftAnalyzer(final_league_id, all_projections, config)

    print("Projected Standings (Z-Scores):")
    standings = analyzer.get_team_standings_z_scores()
    print(standings)

@post_draft_app.command("evaluate-trade")
def evaluate_trade(
    team2_name: str = typer.Option(..., "--team2-name", help="Name of the other team in the trade."),
    team1_players: str = typer.Option(..., "--team1-players", help="Comma-separated list of players your team gives."),
    team2_players: str = typer.Option(..., "--team2-players", help="Comma-separated list of players your team receives."),
    league_id: Optional[str] = typer.Option(None, help="Fantrax league ID (will use from config if not provided)"),
    my_team_name: Optional[str] = typer.Option(None, help="Your team name (will use from config if not provided)"),
    config_name: str = "config.yaml",
):
    """
    Evaluate a trade between two teams.
    """
    cfg = load_config(config_name)
    final_league_id = league_id or cfg.get("league_settings", {}).get("league_id")
    my_team_final_name = my_team_name or cfg.get("league_settings", {}).get("my_team_name")

    if not final_league_id or not my_team_final_name:
        print("Error: League ID and My Team Name must be provided or set in the config file.")
        raise typer.Exit(code=1)

    config = load_config(config_name)
    raw_data_dir = Path("data/raw_projections")
    h_proj, p_proj = load_projections(str(raw_data_dir), config)
    p_proj = create_combined_pitcher_categories(p_proj)
    all_projections = pd.concat([h_proj, p_proj], ignore_index=True)

    analyzer = PostDraftAnalyzer(final_league_id, all_projections, config)

    team1_player_list = [p.strip() for p in team1_players.split(',')]
    team2_player_list = [p.strip() for p in team2_players.split(',')]

    trade_impact = analyzer.evaluate_trade(team1_player_list, team2_player_list, my_team_final_name, team2_name)
    
    print("\nTrade Evaluation:")
    print(trade_impact)

@post_draft_app.command("suggest-trades")
def suggest_trades(
    league_id: Optional[str] = typer.Option(None, help="Fantrax league ID (will use from config if not provided)"),
    my_team_name: Optional[str] = typer.Option(None, "--my-team", help="Your team name"),
    partner_team: Optional[str] = typer.Option(None, "--partner-team", help="Specific team you want to trade with"),
    top_n: int = typer.Option(100, "--top-n", help="Number of top trades to display"),
    config_name: str = "config.yaml",
):
    """
    Suggest mutually beneficial trades to improve your team (1-for-1, 2-for-1, 2-for-2, 2-for-3).
    """
    cfg = load_config(config_name)
    final_league_id = league_id or cfg.get("league_settings", {}).get("league_id")
    my_team_final_name = my_team_name or cfg.get("league_settings", {}).get("my_team_name")

    if not final_league_id or not my_team_final_name:
        print("Error: League ID and My Team Name must be provided or set in the config file.")
        raise typer.Exit(code=1)

    config = load_config(config_name)
    raw_data_dir = Path("data/raw_projections")
    h_proj, p_proj = load_projections(str(raw_data_dir), config)
    p_proj = create_combined_pitcher_categories(p_proj)
    all_projections = pd.concat([h_proj, p_proj], ignore_index=True)

    analyzer = PostDraftAnalyzer(final_league_id, all_projections, config)

    print(f"Analyzing multi-player trade permutations for '{my_team_final_name}'...")
    if partner_team:
        print(f"Focusing specifically on trades with '{partner_team}'...")
    else:
        print("Looking for mutually beneficial trades across the whole league (round-robin by team)...")
    
    trades_df = analyzer.suggest_trades(my_team_final_name, top_n=top_n, partner_team=partner_team)
    
    if trades_df.empty:
        print("\nNo mutually beneficial trades found.")
    else:
        print(f"\nTop {top_n} Suggested Trades:")
        print(trades_df.to_string())

if __name__ == "__main__":
    app()
