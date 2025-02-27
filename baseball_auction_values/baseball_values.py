import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Get the CSV file path
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_dir, "auction_values", "all_auction_values.csv")

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv(csv_file_path)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df = load_data()

# App title
st.title('Fantasy Baseball Auction Values')

# Search functionality
search_term = st.text_input('Search for a player:')

if search_term:
    filtered_df = df[df['Name'].str.contains(search_term, case=False)]
    
    if not filtered_df.empty:
        st.write(f"Found {len(filtered_df)} matching players")
        
        # Display players as a dropdown
        player_names = filtered_df['Name'].tolist()
        selected_player = st.selectbox('Select player:', player_names)
        
        # Get player data
        player_data = filtered_df[filtered_df['Name'] == selected_player].iloc[0]
        
        # Display player data
        st.subheader(selected_player)
        
        # Format as columns
        col1, col2 = st.columns(2)
        
        with col1:
            for idx, (key, value) in enumerate(player_data.items()):
                if idx % 2 == 0 and key != 'Name':
                    st.write(f"**{key}**: {value}")
        
        with col2:
            for idx, (key, value) in enumerate(player_data.items()):
                if idx % 2 == 1 and key != 'Name':
                    st.write(f"**{key}**: {value}")
        
        # Draft value input
        draft_value = st.text_input('Enter draft value ($):', 
                                   value=player_data.get('draft_value', ''))
        
        if st.button('Save Draft Value'):
            try:
                # Validate input
                if draft_value.strip():
                    float(draft_value)  # Check if it's a valid number
                
                # Update dataframe
                idx = df[df['Name'] == selected_player].index[0]
                df.loc[idx, 'draft_value'] = draft_value
                
                # Save back to CSV
                df.to_csv(csv_file_path, index=False)
                st.success(f"Draft value for {selected_player} saved successfully!")
            except ValueError:
                st.error("Draft value must be a number")
            except Exception as e:
                st.error(f"Error saving draft value: {str(e)}")
    else:
        st.write("No players found matching your search.")