import csv
import os
from pathlib import Path

def load_csv_data(file_path):
    """Load CSV data and return a list of dictionaries"""
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def save_draft_value(data, player_name, draft_value, csv_file_path):
    """Save the entered draft value for the player"""
    # Find the player in the data and update the draft value
    player_found = False
    for row in data:
        if row['Name'].lower() == player_name.lower():
            row['draft_value'] = draft_value
            player_found = True
            break
    
    if not player_found:
        print(f"Player not found: {player_name}")
        return False
    
    # Write back to CSV
    try:
        # Get all column names including draft_value if it doesn't exist
        fieldnames = list(data[0].keys())
        if 'draft_value' not in fieldnames:
            fieldnames.append('draft_value')
        
        # Create backup of original file
        backup_path = Path(csv_file_path).with_suffix('.bak')
        if Path(csv_file_path).exists():
            Path(csv_file_path).rename(backup_path)
        
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Draft value for {player_name} saved successfully!")
        return True
        
    except Exception as e:
        # Restore from backup if we failed
        if backup_path.exists():
            backup_path.rename(Path(csv_file_path))
        print(f"Failed to save draft value: {str(e)}")
        return False

def display_full_row(player):
    """Display full row data in an organized format"""
    print(f"\n{'=' * 100}")
    print(f"Player: {player['Name']}")
    print(f"{'=' * 100}")
    
    # Show column headers and values in a table-like format
    headers = []
    values = []
    
    for col, value in player.items():
        if col != 'Name':  # Skip name as we already displayed it
            headers.append(col)
            values.append(value)
    
    # Format as a table
    header_row = " | ".join(f"{h:<18}" for h in headers)
    value_row = " | ".join(f"{v:<18}" for v in values)
    
    print(header_row)
    print("-" * len(header_row))
    print(value_row)
    print(f"{'=' * 100}")

def main():
    # Get the current directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not script_dir:  # If script is run from current directory
        script_dir = os.getcwd()
        
    csv_file_path = os.path.join(script_dir, "auction_values", "all_auction_values.csv")
    
    # Load the data
    data = load_csv_data(csv_file_path)
    
    if not data:
        print(f"Failed to load CSV data from {csv_file_path}.")
        return
    
    while True:
        # Display menu
        print("\n===== Fantasy Baseball Auction Values =====")
        print("1. Search for a player")
        print("2. Update draft value")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            search_term = input("Enter player name to search: ").lower()
            matched_players = [row for row in data if search_term in row['Name'].lower()]
            
            if matched_players:
                print(f"\nFound {len(matched_players)} matching players:")
                for i, player in enumerate(matched_players, 1):
                    draft_val = player.get('draft_value', 'Not set')
                    print(f"{i}. {player['Name']} - Proj: ${player['projection_weighted_average']} - Draft: ${draft_val}")
                
                player_idx = input("\nEnter number to see details (or Enter to continue): ")
                if player_idx.isdigit() and 1 <= int(player_idx) <= len(matched_players):
                    player = matched_players[int(player_idx)-1]
                    display_full_row(player)
            else:
                print("No matching players found.")
                
        elif choice == '2':
            player_name = input("Enter exact player name: ")
            draft_value = input("Enter draft value: $")
            
            try:
                float(draft_value)  # Validate it's a number
                save_draft_value(data, player_name, draft_value, csv_file_path)
            except ValueError:
                print("Draft value must be a number.")
                
        elif choice == '3':
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()