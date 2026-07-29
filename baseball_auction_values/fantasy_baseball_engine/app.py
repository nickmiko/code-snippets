import streamlit as st
import pandas as pd
from pathlib import Path
import os

st.set_page_config(layout="wide", page_title="Fantasy Baseball Draft Hub")
st.title("Fantasy Baseball Draft Hub")

# Setup Paths
base_dir = Path(__file__).parent.resolve()
output_file = base_dir / 'output' / 'overall_values_debug.csv'

@st.cache_data
def load_data():
    if not output_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(output_file)
    if 'Drafted' not in df.columns:
        df['Drafted'] = False
    return df

df = load_data()

if df.empty:
    st.warning("No projection data found. Please run `uv run main.py` first to generate the outputs.")
    st.stop()
else:
    # Sidebar Filters
    st.sidebar.header("Filters")
    pos_filter = st.sidebar.text_input("Search Position (e.g. OF, SP) :")
    search_filter = st.sidebar.text_input("Search Player Name :")
    hide_drafted = st.sidebar.checkbox("Hide Drafted Players", value=True)

    # Filter Data
    view_df = df.copy()
    if pos_filter:
        view_df = view_df[view_df['Pos'].str.contains(pos_filter.upper(), na=False)]
    if search_filter:
        view_df = view_df[view_df['Name'].str.contains(search_filter, case=False, na=False)]
    if hide_drafted:
        view_df = view_df[~view_df['Drafted']]

    st.subheader("Available Player Pool")
    # Using experimental data editor so users can click checkboxes
    edited_df = st.data_editor(
        view_df[['Tier', 'Pos_Tier', 'Name', 'Pos', 'Value', 'Market_Value', 'Bargain_Value', 'Total_Z', 'Drafted']], 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "Drafted": st.column_config.CheckboxColumn("Drafted?", help="Check if drafted")
        }
    )

    # Note: Streamlit restarts top to bottom. If we wanted to save the drafted state,
    # we would write `edited_df` back to CSV or st.session_state here.
    
    # Simple Economy Tracker
    st.subheader("Draft Economy Tracker")
    col1, col2, col3 = st.columns(3)
    col1.metric("Players Remaining", len(view_df))
    # Approximation of total value
    col2.metric("Total Value On Board", f"${view_df['Value'].sum():.2f}")
    col3.metric("Highest Available Z-Score", round(view_df['Total_Z'].max(), 3) if not view_df.empty else 0)
