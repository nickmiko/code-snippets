import csv
import tkinter as tk
from tkinter import ttk, messagebox
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

def update_autocomplete(event=None):
    """Update the autocomplete listbox based on current input"""
    search_term = entry.get().lower()
    listbox.delete(0, tk.END)
    
    if search_term:
        matches = [name for name in name_list if search_term in name.lower()]
        for name in matches[:10]:  # Limit to 10 suggestions
            listbox.insert(tk.END, name)
    
    # Also update search results
    search_and_display()

def use_autocomplete(event=None):
    """Fill the entry with the selected item from listbox"""
    if listbox.curselection():
        selected = listbox.get(listbox.curselection())
        entry.delete(0, tk.END)
        entry.insert(0, selected)
        listbox.delete(0, tk.END)  # Clear the listbox
        search_and_display()

def search_and_display(event=None):
    """Search for the entered name and display results"""
    search_term = entry.get().lower()
    
    # Clear previous results
    result_text.delete(1.0, tk.END)
    
    # Clear draft value entry
    draft_value_entry.delete(0, tk.END)
    
    global current_player
    current_player = None
    
    if search_term:
        # Find rows where the name column matches the search term
        matched_rows = [row for row in data if search_term in row[name_column].lower()]
        
        if matched_rows:
            # If we find an exact match, use that player
            exact_matches = [row for row in matched_rows if row[name_column].lower() == search_term.lower()]
            if exact_matches:
                current_player = exact_matches[0]
            else:
                current_player = matched_rows[0]  # Use first match
            
            # Set the draft value if it exists
            if 'draft_value' in current_player and current_player['draft_value']:
                draft_value_entry.insert(0, current_player['draft_value'])
            
            # Display all matching players
            for row in matched_rows:
                # Format the output nicely
                result_text.insert(tk.END, f"Player: {row[name_column]}\n")
                result_text.insert(tk.END, "=" * 50 + "\n")
                
                # Display all values except the name (already shown)
                for k, v in row.items():
                    if k != name_column:
                        result_text.insert(tk.END, f"{k}: {v}\n")
                
                result_text.insert(tk.END, "\n\n")
        else:
            result_text.insert(tk.END, "No matches found.")

def save_draft_value():
    """Save the entered draft value for the current player"""
    draft_value = draft_value_entry.get().strip()
    
    if not current_player:
        messagebox.showwarning("Warning", "No player selected. Search for a player first.")
        return
    
    player_name = current_player[name_column]
    
    # Validate draft value is a number
    try:
        if draft_value:
            float(draft_value)  # Just to validate it's a number
    except ValueError:
        messagebox.showerror("Error", "Draft value must be a number")
        return
    
    # Find the player in the data and update the draft value
    for row in data:
        if row[name_column] == player_name:
            row['draft_value'] = draft_value
            break
    
    # Write back to CSV
    try:
        # Get all column names including the draft_value if it doesn't exist
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
        
        messagebox.showinfo("Success", f"Draft value for {player_name} saved successfully!")
        
        # Update the display
        search_and_display()
        
    except Exception as e:
        # Restore from backup if we failed
        if backup_path.exists():
            backup_path.rename(Path(csv_file_path))
        messagebox.showerror("Error", f"Failed to save draft value: {str(e)}")

# Get the current directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# File path to your CSV (in the same directory as the script)
csv_file_path = os.path.join(script_dir, "auction_values", "all_auction_values.csv")

# Load the data
data = load_csv_data(csv_file_path)
current_player = None

if data:
    # Determine which column contains names (assuming first column is for names)
    name_column = list(data[0].keys())[0]  # First column name
    
    # Create a list of all names
    name_list = [row[name_column] for row in data if row[name_column]]
    
    # Create the GUI
    root = tk.Tk()
    root.title("Fantasy Baseball Auction Values")
    
    # Frame for input
    input_frame = ttk.Frame(root, padding="10")
    input_frame.pack(fill=tk.X)
    
    # Label
    ttk.Label(input_frame, text=f"Player Name:").pack(side=tk.LEFT, padx=5)
    
    # Entry widget
    entry = ttk.Entry(input_frame, width=30)
    entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    entry.bind("<KeyRelease>", update_autocomplete)
    
    # Autocomplete listbox
    listbox = tk.Listbox(root, height=5)
    listbox.pack(fill=tk.X, padx=15)
    listbox.bind("<<ListboxSelect>>", use_autocomplete)
    
    # Draft value frame
    draft_frame = ttk.Frame(root, padding="10")
    draft_frame.pack(fill=tk.X)
    
    ttk.Label(draft_frame, text="Draft Value ($):").pack(side=tk.LEFT, padx=5)
    draft_value_entry = ttk.Entry(draft_frame, width=10)
    draft_value_entry.pack(side=tk.LEFT, padx=5)
    
    save_button = ttk.Button(draft_frame, text="Save Draft Value", command=save_draft_value)
    save_button.pack(side=tk.LEFT, padx=20)
    
    # Results text widget
    result_frame = ttk.Frame(root, padding="10")
    result_frame.pack(fill=tk.BOTH, expand=True)
    
    result_text = tk.Text(result_frame, wrap=tk.WORD, height=20, width=80)
    result_text.pack(fill=tk.BOTH, expand=True)
    
    # Add a scrollbar
    scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=result_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    result_text.configure(yscrollcommand=scrollbar.set)
    
    # Status bar
    status_var = tk.StringVar()
    status_var.set("Ready")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    root.mainloop()
else:
    print(f"Failed to load CSV data from {csv_file_path}. Exiting.")