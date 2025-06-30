import streamlit as st
from appwrite.client import Client
from appwrite.services.storage import Storage
import pandas as pd
import json
import os
from dotenv import load_dotenv

st.cache_data.clear()  # This will clear all cached data

# @st.cache_data(ttl=3600)  # Cache the data for 1 hour
def get_football_data(file_id="nfdStatsAllModels"):
    """
    Retrieve football data from Appwrite storage
    """
    # Load environment variables
    load_dotenv()
    
    # Appwrite credentials
    API_ENDPOINT = 'https://cloud.appwrite.io/v1'
    PROJECT_ID = os.getenv('APPWRITE_PROJECT_ID')
    API_KEY = os.getenv('APPWRITE_API_KEY')
    STORAGE_ID = os.getenv('NFD_MODEL_STORAGE')
    
    try:
        # Initialize Appwrite client
        client = Client()
        client.set_endpoint(API_ENDPOINT)
        client.set_project(PROJECT_ID)
        client.set_key(API_KEY)
        
        # Initialize storage service
        storage = Storage(client)
        
        # Use direct method to download file
        file_content = storage.get_file_view(bucket_id=STORAGE_ID, file_id=file_id)
        
        # Parse JSON
        json_data = json.loads(file_content.decode('utf-8'))
        
        # Extract data and metadata
        if 'data' in json_data and 'metadata' in json_data:
            # Create DataFrame
            df = pd.DataFrame(json_data['data'])
            
            # Get update timestamp from metadata
            update_time = json_data['metadata'].get('updated_at', 'unknown')
            
            return df, update_time
        else:
            return None, None
        
    except Exception as e:
        st.error(f"Error retrieving data: {str(e)}")
        return None, None

# Streamlit app
st.title("Football Statistics Dashboard")

# Load data with a spinner
with st.spinner("Loading football data..."):
    # dfOne, update_timeOne = get_football_data(file_id="echStatsOne")
    # dfTwo, update_timeTwo = get_football_data(file_id="echStatsTwo")
    # dfThree, update_time = get_football_data(file_id="echStatsThree")
    # df = pd.concat([dfOne, dfTwo, dfThree], ignore_index=True)

    df, update_time = get_football_data(file_id="nfdStatsAllModels")

# Display data if available
if df is not None:
    st.success(f"✅ Data loaded successfully! Last updated: {update_time}")
    
    # Show data summary
    st.subheader("Data Summary")
    # st.write(f"dfOne: {dfOne.shape}")
    # st.write(f"dfTwo: {dfTwo.shape}")
    # st.write(f"dfThree: {dfThree.shape}")
    st.write(f"Total matches: {len(df)}")
    st.write(f"Seasons included: {', '.join(df['season'].unique())}")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Team Statistics", "Match Analysis"])
    
    with tab1:
        st.dataframe(df.head(10))
        
    with tab2:
        # Team selection
        teams = sorted(pd.unique(df[['home_team', 'away_team']].values.ravel('K')))
        selected_team = st.selectbox("Select a team", teams)
        
        # Filter data for selected team
        team_matches = df[(df['home_team'] == selected_team) | (df['away_team'] == selected_team)]
        
        # Display team statistics
        st.write(f"Total matches: {len(team_matches)}")
        home_matches = team_matches[team_matches['home_team'] == selected_team]
        away_matches = team_matches[team_matches['away_team'] == selected_team]
        
        st.write(f"Home matches: {len(home_matches)}, Away matches: {len(away_matches)}")
        
        # Display recent matches
        st.write("Recent matches:")
        st.dataframe(team_matches.sort_values('date', ascending=False).head(5)[
            ['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'ft_result']
        ])
        
    with tab3:
        # Add your match analysis features here
        st.write("This tab will contain match analysis features")
else:
    st.error("Failed to load football data. Please check the connection to Appwrite.")