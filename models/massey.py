import os
import math
import time

import json
import numpy as np
import pandas as pd
from io import BytesIO
# from appwrite.id import ID
# from appwrite.services.storage import Storage
import statsmodels.api as sm
from datetime import datetime
from dotenv import load_dotenv
from scipy.stats import poisson
from appwrite.query import Query
from appwrite.client import Client
from scipy.optimize import minimize
from appwrite.services.databases import Databases


import warnings
warnings.filterwarnings('ignore')


load_dotenv()

# def main(context):

# Replace these with your actual Appwrite credentials
API_ENDPOINT = 'https://cloud.appwrite.io/v1'
PROJECT_ID = os.getenv('APPWRITE_PROJECT_ID')
API_KEY = os.getenv('APPWRITE_API_KEY')
DATABASE_ID = os.getenv('APPWRITE_DB_ID')
STORAGE_ID = os.getenv('NFD_MODEL_STORAGE')

# List of collection IDs to retrieve data from
COLLECTION_IDS = [
    # os.getenv('SEASON_MATCHES_NFD20_21'),
    # os.getenv('SEASON_MATCHES_NFD21_22'),  
    # os.getenv('SEASON_MATCHES_NFD22_23'),
    # os.getenv('SEASON_MATCHES_NFD23_24'),
    # os.getenv('SEASON_MATCHES_NFD24_25'),  
    os.getenv('SEASON_MATCHES_ECH24_25'), 
]

# OUTPUT_FILENAME = "ech_data.xlsx"

# User-configurable parameter: number of past games to consider for rolling statistics
# Default is 6, but can be changed as needed
NUM_PREVIOUS_GAMES = 6
HALF_LIFE=10
PRIOR_STRENGTH=0.6,
UPDATE_INTERVAL=10
# User-configurable parameter: home advantage factor for expected goals calculation
# Default is 1.2, but can be changed as needed

MIN_HOME_ADVANTAGE=0.3
HOME_ADVANTAGE = 1 + MIN_HOME_ADVANTAGE
PYTHAGOREAN_EXPONENT=1.83

# Initialize Appwrite client
client = Client()
client.set_endpoint(API_ENDPOINT)
client.set_project(PROJECT_ID)
client.set_key(API_KEY)

# Initialize Databases service
databases = Databases(client)

########################################################

def save_to_appwrite_storage(dataframe, bucket_id, file_id="echStats", client=None, appwrite_endpoint=None, project_id=None, api_key=None):
    """
    Save DataFrame to a single file in Appwrite Storage with consistent name.
    Will replace existing file if it exists.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The DataFrame containing football statistics
    bucket_id : str
        The Appwrite bucket ID where the file will be stored
    file_id : str
        The file ID to use (default: "echStats")
    client : appwrite.Client, optional
        An initialized Appwrite client (if not provided, one will be created)
    appwrite_endpoint : str, optional
        The Appwrite endpoint URL (required if client is None)
    project_id : str, optional
        The Appwrite project ID (required if client is None)
    api_key : str, optional
        The Appwrite API key (required if client is None)
        
    Returns:
    --------
    str or None
        The file ID if successful, None otherwise
    """
    import time
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import json
    import os
    import tempfile
    
    # Initialize Appwrite client and services if not provided
    if client is None:
        from appwrite.client import Client
        from appwrite.services.storage import Storage
        from appwrite.exception import AppwriteException
        
        if not appwrite_endpoint or not project_id or not api_key:
            raise ValueError("If client is not provided, you must provide appwrite_endpoint, project_id, and api_key")
        
        # Initialize client with provided credentials
        client = Client()
        client.set_endpoint(appwrite_endpoint)
        client.set_project(project_id)
        client.set_key(api_key)
    
    # Initialize storage service
    from appwrite.services.storage import Storage
    from appwrite.exception import AppwriteException
    from appwrite.input_file import InputFile
    from appwrite.permission import Permission
    storage = Storage(client)
    
    start_time = time.time()
    
    # Make a shallow copy to avoid modifying the original
    df = dataframe.copy()
    # df = dataframe.iloc[380:].copy()
    print(f"Preparing to save {len(df)} records to storage as '{file_id}'")
    
    # Check for and process specific columns
    if 'date' in df.columns:
        print("Column 'date' found in the dataframe")
        print("Formatting date column...")
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Handle column renames if needed
    if 'full_time_result' in df.columns and 'ft_result' not in df.columns:
        print("Column 'full_time_result' found in the dataframe")
        print("Renamed 'full_time_result' to 'ft_result'")
        df = df.rename(columns={'full_time_result': 'ft_result'})
    
    # Reorder columns if needed
    if all(col in df.columns for col in ['home_goals', 'away_goals', 'goal_diff']):
        # Get the current column order
        cols = df.columns.tolist()
        # Find the position of 'away_goals'
        away_goals_pos = cols.index('away_goals')
        # Remove 'goal_diff' from its current position
        if 'goal_diff' in cols:
            cols.remove('goal_diff')
            # Insert 'goal_diff' after 'away_goals'
            cols.insert(away_goals_pos + 1, 'goal_diff')
            # Reorder the dataframe
            df = df[cols]
            print("Moved 'goal_diff' column to come after 'away_goals'")
    
    # Handle boolean columns
    if 'rolling_stats_valid' in df.columns:
        df['rolling_stats_valid'] = df['rolling_stats_valid'].fillna(False).astype(bool)
    
    # Process numeric columns to ensure proper JSON serialization
    integer_columns = [
        'homeID', 'awayID', 'winningTeam', 'result', 
        'home_goals', 'away_goals', 'goal_diff',
        'home_team_goals_scored_total', 
        'home_team_goals_conceded_total',
        'away_team_goals_scored_total', 
        'away_team_goals_conceded_total'
    ]
    
    float_columns = [
        'odds_ft_1', 'odds_ft_x', 'odds_ft_2',
        'home_xg_odds', 'away_xg_odds',
        'home_team_goals_scored_average', 'home_team_goals_conceded_average',
        'away_team_goals_scored_average', 'away_team_goals_conceded_average',
        'home_xg', 'away_xg', 'home_xg_elo', 'away_xg_elo',
        'home_xg_dc', 'away_xg_dc', 'home_xg_bt', 'away_xg_bt',
        'home_xg_pyth', 'away_xg_pyth', 'home_xg_bayesian', 'away_xg_bayesian',
        'home_xg_twr', 'away_xg_twr'
    ]
    
    # Clean up data for serialization
    for col in integer_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)
    
    # Add metadata as part of the JSON
    metadata = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "record_count": len(df)
    }
    
    # Create a JSON object with metadata and data
    json_data = {
        "metadata": metadata,
        "data": df.to_dict(orient='records')
    }
    
    # Save JSON to a temporary file (Appwrite expects a file path, not a BytesIO object)
    # Create a temporary file to store the JSON data
    temp_fd, temp_path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(temp_fd, 'w') as tmp:
            json.dump(json_data, tmp, default=str)
        
        print(f"Saved data to temporary file: {temp_path}")
        
        # First, check if file exists and delete it
        try:
            # Try to get file info to check existence
            storage.get_file(bucket_id=bucket_id, file_id=file_id)
            
            # If we reach here, file exists, so delete it
            print(f"Existing file '{file_id}' found. Deleting...")
            storage.delete_file(bucket_id=bucket_id, file_id=file_id)
            print(f"Existing file deleted successfully.")
        except AppwriteException as e:
            # File doesn't exist, which is fine
            if "not found" in str(e).lower():
                print(f"No existing file with ID '{file_id}' found. Creating new file.")
            else:
                print(f"Warning when checking existing file: {str(e)}")
        
        # Try two approaches for permissions
        try:
            # First approach - without specifying permissions (use bucket defaults)
            result = storage.create_file(
                bucket_id=bucket_id,
                file_id=file_id,
                file=InputFile.from_path(temp_path)
                # No permissions parameter
            )
        except AppwriteException as e1:
            # If that fails, try with explicit permissions
            if "permissions" in str(e1).lower():
                try:
                    # Second approach - specify explicit 'read' permission
                    result = storage.create_file(
                        bucket_id=bucket_id,
                        file_id=file_id,
                        file=InputFile.from_path(temp_path),
                        permissions=['read']  # Allow read permission only
                    )
                except AppwriteException as e2:
                    # If that also fails, print both errors and give up
                    print(f"Failed with default permissions: {str(e1)}")
                    print(f"Failed with explicit 'read' permission: {str(e2)}")
                    raise e2
            else:
                # Not a permissions error, rethrow
                raise e1
        
        total_time = time.time() - start_time
        print(f"✅ Successfully saved {len(df)} records to storage in {total_time:.2f}s")
        print(f"File ID: {result['$id']}")
        
        return result['$id']
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Error uploading to storage after {total_time:.2f}s: {str(e)}")
        return None
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Deleted temporary file: {temp_path}")
    
########################################################

# Function to convert odds to raw probabilities
def odds_to_raw_probs(home_odds, draw_odds, away_odds):
    home_prob = 1 / home_odds
    draw_prob = 1 / draw_odds
    away_prob = 1 / away_odds
    return home_prob, draw_prob, away_prob

# Function to normalize probabilities (account for overround)
def normalize_probs(home_prob, draw_prob, away_prob):
    total = home_prob + draw_prob + away_prob
    return home_prob/total, draw_prob/total, away_prob/total

# Function to calculate match outcome probabilities from Poisson parameters
def poisson_match_probs(params):
    home_xg, away_xg = params
    
    # Calculate probabilities for different scorelines (0-0, 1-0, 0-1, etc.)
    max_goals = 10  # Consider up to 10 goals for each team
    home_probs = np.exp(-home_xg) * np.power(home_xg, np.arange(max_goals)) / np.array([math.factorial(i) for i in range(max_goals)])
    away_probs = np.exp(-away_xg) * np.power(away_xg, np.arange(max_goals)) / np.array([math.factorial(i) for i in range(max_goals)])
    
    # Calculate match outcome probabilities
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0
    
    for i in range(max_goals):
        for j in range(max_goals):
            p = home_probs[i] * away_probs[j]
            if i > j:
                home_win_prob += p
            elif i == j:
                draw_prob += p
            else:
                away_win_prob += p
    
    return home_win_prob, draw_prob, away_win_prob

# Function to minimize (difference between bookmaker probabilities and Poisson probabilities)
def objective_function(params, target_probs):
    home_win_prob, draw_prob, away_win_prob = poisson_match_probs(params)
    target_home_prob, target_draw_prob, target_away_prob = target_probs
    
    # Sum of squared differences
    return (
        (home_win_prob - target_home_prob)**2 + 
        (draw_prob - target_draw_prob)**2 + 
        (away_win_prob - target_away_prob)**2
    )

# Function to solve for xG values
def solve_for_xg(home_odds, draw_odds, away_odds):
    # Convert odds to normalized probabilities
    raw_probs = odds_to_raw_probs(home_odds, draw_odds, away_odds)
    target_probs = normalize_probs(*raw_probs)
    
    # Initial guess for xG values (reasonable starting point)
    initial_guess = [1.5, 1.0]
    
    # Bounds to ensure positive xG values
    bounds = [(0.01, 5), (0.01, 5)]
    
    # Solve the optimization problem
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(target_probs,), 
        method='L-BFGS-B', 
        bounds=bounds
    )
    
    # Return the optimized parameters (home_xg, away_xg)
    return result.x

# Apply to the DataFrame
def add_xg_columns(df):
    # Initialize empty lists for home and away xG
    home_xg_list = []
    away_xg_list = []
    
    # Process each row
    for _, row in df.iterrows():
        try:
            home_odds = row['odds_ft_1']
            draw_odds = row['odds_ft_x']
            away_odds = row['odds_ft_2']
            
            # Check if odds are valid
            if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds) or home_odds <= 0 or draw_odds <= 0 or away_odds <= 0:
                # Set to 0 instead of np.nan for missing odds
                home_xg_list.append(0.0)
                away_xg_list.append(0.0)
            else:
                home_xg, away_xg = solve_for_xg(home_odds, draw_odds, away_odds)
                home_xg_list.append(round(home_xg,2))
                away_xg_list.append(round(away_xg,2))
        except Exception as e:
            print(f"Error processing row: {e}")
            # Set to 0 instead of np.nan for errors too
            home_xg_list.append(0.0)
            away_xg_list.append(0.0)
    
    # Add columns to DataFrame
    df['home_xg_odds'] = home_xg_list
    df['away_xg_odds'] = away_xg_list
    
    return df

########################################################

# Function to calculate expected goals using Bayesian parameters
def calculate_expected_goals(home_team, away_team, attack_params, defense_params, home_advantage=1.2):
    """
    Calculate expected goals for a match using Bayesian parameters.
    
    Parameters:
    -----------
    home_team, away_team : str
        Team names
    attack_params, defense_params : dict
        Dictionaries containing attack and defense parameters for all teams
    home_advantage : float
        Home advantage factor (default: 1.2)
    
    Returns:
    --------
    tuple
        (expected_home_goals, expected_away_goals)
    """
    # Get parameters - default to middle value (1.5) if not available
    home_attack_alpha, _ = attack_params.get(home_team, (1.5, 1.0))
    home_defense_alpha, _ = defense_params.get(home_team, (1.5, 1.0))
    away_attack_alpha, _ = attack_params.get(away_team, (1.5, 1.0))
    away_defense_alpha, _ = defense_params.get(away_team, (1.5, 1.0))
    
    # For defense, higher values mean better defense (fewer goals conceded)
    # We need to convert defense rating to a factor that reduces expected goals
    home_defense_factor = (3.0 - home_defense_alpha) / 3.0  # Transform to a 0-1 scale
    away_defense_factor = (3.0 - away_defense_alpha) / 3.0  # Transform to a 0-1 scale
    
    # Calculate expected goals based on team's attack strength against opponent's defense
    # Home team expected goals = home team attack * away team defense factor * home advantage
    # Away team expected goals = away team attack * home team defense factor
    expected_home_goals = home_attack_alpha * away_defense_factor * home_advantage
    expected_away_goals = away_attack_alpha * home_defense_factor
    
    # Ensure expected goals are within the 0-3 range
    expected_home_goals = min(max(expected_home_goals, 0.0), 3.0)
    expected_away_goals = min(max(expected_away_goals, 0.0), 3.0)
    
    return expected_home_goals, expected_away_goals

########################################################

# Function to calculate rolling statistics for teams
def add_team_rolling_stats(df, num_previous_games=6, home_advantage=1.2):
    """
    Add rolling statistics for each team based on their past n games.
    Statistics are calculated independently for each season.
    
    Args:
        df: DataFrame containing match data
        num_previous_games: Number of previous games to consider for rolling stats
        home_advantage: Home advantage factor for expected goals calculation
        
    Returns:
        DataFrame with additional columns for team statistics
    """
    print(f"Calculating rolling statistics based on the last {num_previous_games} games...")
    
    # Ensure the DataFrame is sorted by date
    df = df.sort_values(by=['season', 'match_date'])
    
    # Initialize new columns for rolling statistics
    stats_columns = [
        'home_team_goals_scored_total', 'home_team_goals_conceded_total',
        'away_team_goals_scored_total', 'away_team_goals_conceded_total',
        'home_team_goals_scored_average', 'home_team_goals_conceded_average',
        'away_team_goals_scored_average', 'away_team_goals_conceded_average',
        'home_xg', 'away_xg'
    ]
    
    for col in stats_columns:
        df[col] = 0.0
    
    # Process each season separately
    seasons = df['season'].unique()
    
    for season in seasons:
        print(f"Processing season: {season}")
        
        # Filter data for current season
        season_df = df[df['season'] == season].copy()
        
        # Get all unique teams in this season
        home_teams = season_df['homeID'].unique()
        away_teams = season_df['awayID'].unique()
        all_teams = np.union1d(home_teams, away_teams)
        
        # Initialize dictionary to track team performance
        team_performance = {team_id: [] for team_id in all_teams}
        
        # Initialize dictionaries for attack and defense parameters
        attack_params = {}
        defense_params = {}
        
        # Build team performance history
        for idx, row in season_df.iterrows():
            # For home team, add match to history
            home_team_id = row['homeID']
            away_team_id = row['awayID']
            
            # Skip if match is not complete
            if row['status'] != 'complete':
                continue
                
            # Add match to home team history
            team_performance[home_team_id].append({
                'match_date': row['match_date'],
                'is_home': True,
                'goals_scored': row['homeGoalCount'],
                'goals_conceded': row['awayGoalCount']
            })
            
            # Add match to away team history
            team_performance[away_team_id].append({
                'match_date': row['match_date'],
                'is_home': False,
                'goals_scored': row['awayGoalCount'],
                'goals_conceded': row['homeGoalCount']
            })
        
        # Update attack and defense parameters based on team performance
        for team_id, matches in team_performance.items():
            if not matches:
                continue
                
            # Use up to the most recent num_previous_games matches
            recent_matches = matches[-num_previous_games:] if len(matches) > num_previous_games else matches
            
            # Calculate average goals scored and conceded
            goals_scored = [match['goals_scored'] for match in recent_matches]
            goals_conceded = [match['goals_conceded'] for match in recent_matches]
            
            avg_goals_scored = sum(goals_scored) / len(recent_matches) if recent_matches else 1.5
            avg_goals_conceded = sum(goals_conceded) / len(recent_matches) if recent_matches else 1.5
            
            # Map averages to parameters in 0-3 range
            # For attack, higher is better - this represents the team's goal-scoring ability
            attack_alpha = min(max(avg_goals_scored, 0.0), 3.0)
            
            # For defense, lower conceded is better, so we use an inverse scale
            # Higher defense_alpha means better defense (fewer goals conceded)
            defense_alpha = min(max(3.0 - avg_goals_conceded, 0.0), 3.0)
            
            # Store parameters
            attack_params[team_id] = (attack_alpha, 1.0)  # using fixed beta for simplicity
            defense_params[team_id] = (defense_alpha, 1.0)  # using fixed beta for simplicity
        
        # Now calculate rolling statistics for each match
        for idx, row in season_df.iterrows():
            match_date = row['match_date']
            home_team_id = row['homeID']
            away_team_id = row['awayID']
            
            # Calculate home team stats
            home_team_history = team_performance[home_team_id]
            # Filter history to only include matches before current match
            previous_home_matches = [
                match for match in home_team_history
                if match['match_date'] < match_date
            ]

            # Get the number of previous matches for home team
            num_home_previous_all = len(previous_home_matches)
            
            # Use the specified number of previous games or all available if less
            previous_home_matches = previous_home_matches[-num_previous_games:] if previous_home_matches else []
            num_home_previous = len(previous_home_matches)
            
            # Calculate home team totals
            home_goals_scored = sum(match['goals_scored'] for match in previous_home_matches)
            home_goals_conceded = sum(match['goals_conceded'] for match in previous_home_matches)
            
            # Calculate home team averages
            home_goals_scored_avg = home_goals_scored / num_home_previous if num_home_previous > 0 else 0
            home_goals_conceded_avg = home_goals_conceded / num_home_previous if num_home_previous > 0 else 0
            
            # Calculate away team stats
            away_team_history = team_performance[away_team_id]
            # Filter history to only include matches before current match
            previous_away_matches = [
                match for match in away_team_history
                if match['match_date'] < match_date
            ]

            # Get the number of previous matches for away team
            num_away_previous_all = len(previous_away_matches)
            
            # Use the specified number of previous games or all available if less
            previous_away_matches = previous_away_matches[-num_previous_games:] if previous_away_matches else []
            num_away_previous = len(previous_away_matches)
            
            # Calculate away team totals
            away_goals_scored = sum(match['goals_scored'] for match in previous_away_matches)
            away_goals_conceded = sum(match['goals_conceded'] for match in previous_away_matches)
            
            # Calculate away team averages
            away_goals_scored_avg = away_goals_scored / num_away_previous if num_away_previous > 0 else 0
            away_goals_conceded_avg = away_goals_conceded / num_away_previous if num_away_previous > 0 else 0
            
            # Calculate expected goals using Bayesian parameters
            expected_home_goals, expected_away_goals = calculate_expected_goals(
                home_team_id, 
                away_team_id, 
                attack_params, 
                defense_params, 
                home_advantage
            )
            
            # Update the DataFrame with calculated values
            df.loc[idx, 'home_team_goals_scored_total'] = int(home_goals_scored)
            df.loc[idx, 'home_team_goals_conceded_total'] = int(home_goals_conceded)
            df.loc[idx, 'away_team_goals_scored_total'] = int(away_goals_scored)
            df.loc[idx, 'away_team_goals_conceded_total'] = int(away_goals_conceded)
            df.loc[idx, 'home_team_goals_scored_average'] = round(home_goals_scored_avg, 2)
            df.loc[idx, 'home_team_goals_conceded_average'] = round(home_goals_conceded_avg, 2)
            df.loc[idx, 'away_team_goals_scored_average'] = round(away_goals_scored_avg, 2)
            df.loc[idx, 'away_team_goals_conceded_average'] = round(away_goals_conceded_avg, 2)
            df.loc[idx, 'home_xg'] = round(expected_home_goals, 2)
            df.loc[idx, 'away_xg'] = round(expected_away_goals, 2)

            # Set the rolling_stats_valid flag
            # Flag is 1 only if both teams have played at least num_previous_games games in this season
            if num_home_previous_all >= num_previous_games and num_away_previous_all >= num_previous_games:
                df.loc[idx, 'rolling_stats_valid'] = 1
            else:
                df.loc[idx, 'rolling_stats_valid'] = 0
    
    return df

########################################################

# Function to retrieve all documents from a collection
def get_collection_documents(collection_id):
    all_documents = []
    limit = 400
    offset = 0
    
    while True:
        try:
            # Set up query options for offset-based pagination
            query_options = [
                Query.limit(limit),
                Query.offset(offset)
            ]
            
            # Get a batch of documents
            response = databases.list_documents(
                database_id=DATABASE_ID,
                collection_id=collection_id,
                queries=query_options
            )
            
            documents = response['documents']
            
            # If no documents were returned, we've reached the end
            if not documents:
                break
                
            # Add documents to our list
            all_documents.extend(documents)
            
            print(f"Retrieved {len(all_documents)} documents from collection {collection_id} so far...")
            
            # If we got fewer documents than the limit, we've reached the end
            if len(documents) < limit:
                break
                
            # Update offset for the next page
            offset += limit
            
        except Exception as e:
            print(f"Error retrieving documents from collection {collection_id}: {str(e)}")
            break
    
    return all_documents

# Create an empty list to store dataframes from each collection
all_dataframes = []

# Process each collection and create a dataframe
for collection_id in COLLECTION_IDS:
    if collection_id:  # Skip if collection ID is None or empty
        print(f"\nProcessing collection: {collection_id}")
        documents = get_collection_documents(collection_id)
        
        if documents:
            # Create a DataFrame for this collection
            df = pd.DataFrame(documents)
            
            # Add a column to identify which collection this data came from
            df['source_collection'] = collection_id
            
            # Add this dataframe to our list
            all_dataframes.append(df)
            print(f"Added {len(df)} rows from collection {collection_id}")
        else:
            print(f"No documents found in collection {collection_id}")

# Combine all dataframes into one
if all_dataframes:
    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\nCombined DataFrame has {len(combined_df)} rows")
    
    # Replace -1 with 0 only in numeric columns
    numeric_columns = combined_df.select_dtypes(include=['number']).columns
    combined_df[numeric_columns] = combined_df[numeric_columns].replace(-1, 0)
    
    # Create a copy of the DataFrame to defragment it
    combined_df = combined_df.copy()
    
    # Create a date column if date_unix exists
    if 'date_unix' in combined_df.columns:
        # Process all dates at once using vectorized operations
        combined_df['date_unix'] = pd.to_numeric(combined_df['date_unix'], errors='coerce')
        
        # Create a datetime column for sorting
        combined_df['datetime_temp'] = pd.to_datetime(combined_df['date_unix'], unit='s')
        
        # Create a formatted date string column
        combined_df['match_date'] = combined_df['datetime_temp'].dt.strftime('%Y-%m-%d')
        
        # Sort by the datetime column from earliest to latest
        combined_df = combined_df.sort_values(by='datetime_temp')
        
        # Drop the temporary datetime column (optional)
        combined_df = combined_df.drop(columns=['datetime_temp'])
        
    # Create the 'winningTeamName' column using a lambda function
    combined_df['winningTeamName'] = combined_df.apply(
        lambda row: row['home_name'] if row['winningTeam'] == row['homeID'] 
                    else row['away_name'] if row['winningTeam'] == row['awayID'] 
                    else 'Draw' if row['winningTeam'] == 0 
                    else None,  # This handles any unexpected values
        axis=1  
    )   

    # Filter for essential columns first
    combined_df = combined_df[['match_date', 'season', 'status', 'homeID', 'home_name', 'awayID', 'away_name', 'winningTeam', 'winningTeamName',
                            'homeGoalCount','awayGoalCount', 'odds_ft_1', 'odds_ft_x', 'odds_ft_2']]
    
    # Apply xG calculations
    combined_df = add_xg_columns(combined_df)
    
    # Add the new rolling statistics including expected goals with the specified home advantage
    combined_df = add_team_rolling_stats(combined_df, num_previous_games=NUM_PREVIOUS_GAMES, home_advantage=HOME_ADVANTAGE)

    # Save to Excel
    # combined_df.to_excel(OUTPUT_FILENAME, index=False)
    # print(f"\nData saved to {OUTPUT_FILENAME}")
    
    # print first few rows
    # print("\nFirst 5 rows of the combined data:")
    # print(combined_df.head())
else:
    print("No data found in any of the collections")

def formatDataframe(df):



        
    # Check if 'date' column exists
    if 'date' in df.columns:
        print("Column 'date' found in the dataframe")
    else:
        print("Column 'date' NOT found. Available columns:", df.columns.tolist())
        
        # Check for case-insensitive matches
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            print(f"Found possible date columns: {date_cols}")
            # Use the first match
            df = df.rename(columns={date_cols[0]: 'date'})
        else:
            # If no date column is found, create a column to avoid errors
            print("No date column found. Using a placeholder.")
            df['date'] = pd.NaT
            
    # Now we can safely format the date column
    print("Formatting date column...")
    df['date'] = pd.to_datetime(df['date']).dt.date

    # 2. Add new column 'result' that maps full_time_result values
    # First check if full_time_result column exists
    if 'full_time_result' in df.columns:
        print("Column 'full_time_result' found in the dataframe")
        result_mapping = {'H': 1, 'D': 0, 'A': 2}
        df['result'] = df['full_time_result'].map(result_mapping)
    else:
        print("Column 'full_time_result' NOT found. Available columns:", df.columns.tolist())
        # Check for alternative column names
        result_cols = [col for col in df.columns if 'result' in col.lower() and 'full' in col.lower()]
        if result_cols:
            print(f"Found possible result columns: {result_cols}")
            # Use the first match
            result_col = result_cols[0]
            df = df.rename(columns={result_col: 'full_time_result'})
            result_mapping = {'H': 1, 'D': 0, 'A': 2}
            df['result'] = df['full_time_result'].map(result_mapping)
        else:
            print("No suitable result column found. Creating a placeholder.")
            df['full_time_result'] = None
            df['result'] = None

    # 3. Rename full_time_result to ft_result
    if 'full_time_result' in df.columns:
        df = df.rename(columns={'full_time_result': 'ft_result'})
        print("Renamed 'full_time_result' to 'ft_result'")
    else:
        print("Cannot rename 'full_time_result' as it does not exist in the dataframe")

    # 4. Move result and ft_result columns to come after winningTeamName column
    # First, get the index of 'winningTeamName' column
    winning_team_idx = df.columns.get_loc('winningTeamName')

    # Extract the columns we want to move
    result_col = df['result']
    ft_result_col = df['ft_result']

    # Drop those columns from the DataFrame
    df = df.drop(['result', 'ft_result'], axis=1)

    # Insert them after 'winningTeamName' column
    df.insert(winning_team_idx + 1, 'result', result_col)
    df.insert(winning_team_idx + 2, 'ft_result', ft_result_col)

    # 5. Move goal_diff column to come after away_goals
    if 'goal_diff' in df.columns and 'away_goals' in df.columns:
        # Get the index of away_goals column
        away_goals_idx = df.columns.get_loc('away_goals')
        
        # Extract the goal_diff column
        goal_diff_col = df['goal_diff']
        
        # Drop the column
        df = df.drop('goal_diff', axis=1)
        
        # Insert it after away_goals
        df.insert(away_goals_idx + 1, 'goal_diff', goal_diff_col)
        print("Moved 'goal_diff' column to come after 'away_goals'")
    else:
        print("'goal_diff' or 'away_goals' column not found - cannot move goal_diff")

    return df

def load_and_process_data(df):
    """
    Load match data from Excel and prepare for analysis.
    With specific handling for various column naming conventions.
    
    Parameters:
    file_path - Path to the Excel file
    
    Returns:
    Processed DataFrame
    """
    try:
        # Load the Excel file
        # df = pd.read_excel(file_path)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns.")
        print(f"Columns found: {df.columns.tolist()}")
        
        # Specifically map the known column names in your file
        column_mapping = {}
        
        # Add the specific mappings you identified
        if 'homeGoalCount' in df.columns:
            column_mapping['homeGoalCount'] = 'home_goals'
        
        if 'awayGoalCount' in df.columns:
            column_mapping['awayGoalCount'] = 'away_goals'
        
        # Additional mappings for other columns we need
        column_mappings = {
            'home_team': ['hometeam', 'home_team', 'homename', 'home_name', 'home team', 'home', 'home_id', 'homeid', 'hometeam'],
            'away_team': ['awayteam', 'away_team', 'awayname', 'away_name', 'away team', 'away', 'away_id', 'awayid', 'awayteam'],
            'date': ['date', 'match_date', 'gamedate', 'game_date', 'datetime', 'date_time'],
            'season': ['season', 'seasonid', 'season_id', 'year', 'competition_year'],
            'full_time_result': ['ftr', 'full_time_result', 'result', 'outcome', 'match_result']
        }
        
        # Map other necessary columns
        for target_col, possible_names in column_mappings.items():
            # Skip if we already have this column
            if target_col in df.columns or target_col in column_mapping.values():
                continue
                
            # Try various matching approaches
            for possible_name in possible_names:
                if possible_name in df.columns:
                    column_mapping[possible_name] = target_col
                    break
            
            # If still not found, try case-insensitive matching
            if target_col not in column_mapping.values():
                for col in df.columns:
                    if col.lower() in possible_names:
                        column_mapping[col] = target_col
                        break
        
        # Apply the mappings found
        if column_mapping:
            print(f"\nRenaming columns: {column_mapping}")
            df = df.rename(columns=column_mapping)
        
        # Check for still missing required columns
        required_cols = ['home_team', 'away_team', 'home_goals', 'away_goals']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Special handling for team columns
            if 'home_team' in missing_cols:
                home_team_candidates = [col for col in df.columns if 'home' in col.lower() and ('team' in col.lower() or 'name' in col.lower())]
                if home_team_candidates:
                    df['home_team'] = df[home_team_candidates[0]]
                    print(f"Using '{home_team_candidates[0]}' as home_team")
                    missing_cols.remove('home_team')
                elif 'homeID' in df.columns:
                    df['home_team'] = df['homeID'].astype(str)
                    print("Using homeID as home_team")
                    missing_cols.remove('home_team')
            
            if 'away_team' in missing_cols:
                away_team_candidates = [col for col in df.columns if 'away' in col.lower() and ('team' in col.lower() or 'name' in col.lower())]
                if away_team_candidates:
                    df['away_team'] = df[away_team_candidates[0]]
                    print(f"Using '{away_team_candidates[0]}' as away_team")
                    missing_cols.remove('away_team')
                elif 'awayID' in df.columns:
                    df['away_team'] = df['awayID'].astype(str)
                    print("Using awayID as away_team")
                    missing_cols.remove('away_team')
            
            # If still missing required columns, raise error
            if missing_cols:
                print("\nStill missing required columns. Here are all available columns:")
                for i, col in enumerate(df.columns):
                    print(f"{i}: {col}")
                
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Try to create full_time_result if it doesn't exist
        if 'full_time_result' not in df.columns and 'home_goals' in df.columns and 'away_goals' in df.columns:
            df['full_time_result'] = np.where(df['home_goals'] > df['away_goals'], 'H',
                                np.where(df['home_goals'] < df['away_goals'], 'A', 'D'))
            print("Created 'full_time_result' from goals comparison")
        
        # Ensure date column is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Calculate goal difference
        df['goal_diff'] = df['home_goals'] - df['away_goals']
        
        # Try to create season if it doesn't exist
        if 'season' not in df.columns and 'date' in df.columns:
            try:
                if pd.api.types.is_datetime64_dtype(df['date']):
                    # Extract season (assuming seasons span calendar years, e.g., 2020-2021)
                    df['season'] = df['date'].dt.year.astype(str) + '-' + (df['date'].dt.year + 1).astype(str)
                    print("Created 'season' column from 'date'")
            except Exception as e:
                print(f"Could not extract season from date: {e}")
        
        print("\nProcessed data summary:")
        print(f"Rows: {len(df)}")
        if 'home_team' in df.columns:
            unique_teams = set(df['home_team'].unique()).union(set(df['away_team'].unique()))
            print(f"Unique teams: {len(unique_teams)}")
        if 'season' in df.columns:
            print(f"Seasons: {df['season'].unique()}")
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nTrying to display file contents for debugging:")
        try:
            # Try to just read the raw Excel without processing
            # raw_df = pd.read_excel(file_path)
            raw_df=df
            print(f"Raw columns: {raw_df.columns.tolist()}")
            print(raw_df.head(3))
        except Exception as sub_e:
            print(f"Could not read raw file: {sub_e}")
        return None

clean_dataframe = load_and_process_data(combined_df)




###################################################################################################
#################################     ELO MODEL    ################################################



# Function to calculate match outcome probabilities from Poisson parameters
def poisson_match_probs(home_xg, away_xg):
    """
    Calculate match outcome probabilities using Poisson distribution.
    
    Parameters:
    -----------
    home_xg : float
        Expected goals for home team
    away_xg : float
        Expected goals for away team
        
    Returns:
    --------
    tuple (home_win_prob, draw_prob, away_win_prob)
        Probabilities for each match outcome
    """
    # Calculate probabilities for different scorelines (0-0, 1-0, 0-1, etc.)
    max_goals = 10  # Consider up to 10 goals for each team
    home_probs = np.exp(-home_xg) * np.power(home_xg, np.arange(max_goals)) / np.array([math.factorial(i) for i in range(max_goals)])
    away_probs = np.exp(-away_xg) * np.power(away_xg, np.arange(max_goals)) / np.array([math.factorial(i) for i in range(max_goals)])
    
    # Calculate match outcome probabilities
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0
    
    for i in range(max_goals):
        for j in range(max_goals):
            p = home_probs[i] * away_probs[j]
            if i > j:
                home_win_prob += p
            elif i == j:
                draw_prob += p
            else:
                away_win_prob += p
    
    return home_win_prob, draw_prob, away_win_prob

# Function to solve for xG values that match target probabilities
def solve_for_xg(target_probs, league_avg_home_xg=1.35, league_avg_away_xg=1.1):
    """
    Solve for xG values that would produce the target match probabilities.
    
    Parameters:
    -----------
    target_probs : tuple
        Target probabilities (home_win, draw, away_win)
    league_avg_home_xg : float
        League average for home team expected goals
    league_avg_away_xg : float
        League average for away team expected goals
        
    Returns:
    --------
    tuple (home_xg, away_xg)
        Expected goals values that produce probabilities closest to target
    """
    target_home_prob, target_draw_prob, target_away_prob = target_probs
    
    # Function to minimize (difference between Poisson probabilities and target probabilities)
    def objective_function(params):
        home_xg, away_xg = params
        home_win_prob, draw_prob, away_win_prob = poisson_match_probs(home_xg, away_xg)
        
        # Sum of squared differences
        return (
            (home_win_prob - target_home_prob)**2 + 
            (draw_prob - target_draw_prob)**2 + 
            (away_win_prob - target_away_prob)**2
        )
    
    # Initial guess for xG values (reasonable starting points based on league averages)
    initial_guess = [league_avg_home_xg, league_avg_away_xg]
    
    # Bounds to ensure positive xG values
    bounds = [(0.01, 5), (0.01, 5)]
    
    # Solve the optimization problem
    result = minimize(
        objective_function, 
        initial_guess, 
        method='L-BFGS-B', 
        bounds=bounds
    )
    
    # Return the optimized parameters (home_xg, away_xg)
    return result.x

# def add_elo_xg_only(file_path, output_file=None, num_recent_games=6, home_advantage=35, league_home_xg=1.35, league_away_xg=1.1, def_elo=1500, def_f_factor=32):
def add_elo_xg_only(dataframe, num_recent_games=6, home_advantage=35, league_home_xg=1.35, league_away_xg=1.1, def_elo=1500, def_f_factor=32):
    """
    Calculate expected goals based on Elo ratings and add only the xG values to the football dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_xg_only' to original filename)
    num_recent_games : int
        Number of recent games to use for calculating league averages
    home_advantage : int
        Home advantage in Elo points
    league_home_xg : float
        Default league average for home team expected goals
    league_away_xg : float
        Default league average for away team expected goals
    def_elo : int
        Default Elo rating for teams
    def_f_factor : int
        K-factor for Elo rating updates
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with only the xG columns added
    """
    # Set default output file name if not provided
    # if output_file is None:
    #     output_file = file_path.replace('.xlsx', '_xg_only.xlsx')
    
    # Load and process the data
    # df = load_and_process_data(file_path)
    df = dataframe
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize only the xG columns in the final dataframe
    output_columns = [
        'home_xg_elo',
        'away_xg_elo'
    ]
    
    # We'll still calculate the Elo ratings and probabilities temporarily
    temp_columns = [
        'home_team_elo',
        'away_team_elo',
        'home_win_probability',
        'draw_probability',
        'away_win_probability'
    ]
    
    # Initialize all temporary and output columns with default values
    all_columns = output_columns + temp_columns
    for col in all_columns:
        df[col] = 0.0
    
    # Analyze historical data to get league averages for xG
    # These will be used as initial values for the optimization
    league_avg_home_xg = league_home_xg  # Default if we can't calculate from data
    league_avg_away_xg = league_away_xg  # Default if we can't calculate from data
    
    # Try to calculate from data if we have enough completed matches
    completed_matches = df[df['status'].isin(['complete', 'finished'])]
    if len(completed_matches) >= num_recent_games:
        league_avg_home_xg = completed_matches['home_goals'].mean()
        league_avg_away_xg = completed_matches['away_goals'].mean()
        print(f"Calculated league averages - Home xG: {league_avg_home_xg:.2f}, Away xG: {league_avg_away_xg:.2f}")
    else:
        print(f"Using default league averages - Home xG: {league_avg_home_xg:.2f}, Away xG: {league_avg_away_xg:.2f}")
    
    # Process each season separately
    for season in df['season'].unique():
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Dictionary to store team Elo ratings for this season
        elo_ratings = {}  # Dictionary to store team Elo ratings
        default_elo = def_elo  # Starting Elo rating for teams without history
        k_factor = def_f_factor  # K-factor determines how quickly ratings change
        
        # Home field advantage in Elo points
        home_advantage_elo = home_advantage
        
        # Initialize Elo ratings for teams in this season
        teams_in_season = set(season_df['home_team'].unique()) | set(season_df['away_team'].unique())
        for team in teams_in_season:
            elo_ratings[team] = default_elo
        
        # Process each match chronologically within the season
        for match_idx in season_indices:
            home_team = df.at[match_idx, 'home_team']
            away_team = df.at[match_idx, 'away_team']
            
            # Set current Elo ratings for this match (before updating)
            current_home_elo = elo_ratings.get(home_team, default_elo)
            current_away_elo = elo_ratings.get(away_team, default_elo)
            
            # Store Elo ratings temporarily (won't be included in final output)
            df.at[match_idx, 'home_team_elo'] = current_home_elo
            df.at[match_idx, 'away_team_elo'] = current_away_elo
            
            # Calculate win probabilities based on Elo with home advantage
            home_elo_adjusted = current_home_elo + home_advantage_elo
            elo_diff = (home_elo_adjusted - current_away_elo) / 400.0
            home_win_prob_raw = 1.0 / (1.0 + 10.0 ** (-elo_diff))
            
            # Calculate away win probability directly from Elo
            away_elo_adjusted = current_away_elo + home_advantage_elo  # If they were home
            away_elo_diff = (away_elo_adjusted - current_home_elo) / 400.0
            away_win_prob_raw = 1.0 / (1.0 + 10.0 ** (-away_elo_diff))
            
            # Calibrate draw probability 
            base_draw_prob = 0.28
            
            # Adjust draw probability based on how close the teams are
            elo_diff_abs = abs(current_home_elo - current_away_elo)
            draw_adjustment = max(0, 0.06 - (elo_diff_abs / 2000))
            draw_prob = base_draw_prob + draw_adjustment
            
            # Recalibrate win probabilities to account for draw and ensure all sum to 1
            remaining_prob = 1.0 - draw_prob
            
            # Calculate relative strengths of home and away teams
            total_win_prob_raw = home_win_prob_raw + away_win_prob_raw
            relative_home_strength = home_win_prob_raw / total_win_prob_raw
            relative_away_strength = away_win_prob_raw / total_win_prob_raw
            
            # Distribute the remaining probability according to relative strengths
            home_win_prob = relative_home_strength * remaining_prob
            away_win_prob = relative_away_strength * remaining_prob
            
            # Final check to ensure probabilities are in valid range and sum to 1
            total_prob = home_win_prob + draw_prob + away_win_prob
            if abs(total_prob - 1.0) > 0.0001:  # If not very close to 1
                home_win_prob /= total_prob
                draw_prob /= total_prob
                away_win_prob /= total_prob
            
            # Store probabilities temporarily (won't be included in final output)
            df.at[match_idx, 'home_win_probability'] = home_win_prob
            df.at[match_idx, 'draw_probability'] = draw_prob
            df.at[match_idx, 'away_win_probability'] = away_win_prob
            
            # Calculate expected goals based on these probabilities
            try:
                target_probs = (home_win_prob, draw_prob, away_win_prob)
                home_xg, away_xg = solve_for_xg(target_probs, league_avg_home_xg, league_avg_away_xg)
                
                df.at[match_idx, 'home_xg_elo'] = round(home_xg, 2)
                df.at[match_idx, 'away_xg_elo'] = round(away_xg, 2)
            except Exception as e:
                print(f"Error calculating xG for match {match_idx}: {e}")
                # Fallback calculation for xG based on relative team strength
                home_strength_ratio = 10 ** (elo_diff)
                
                # More balanced xG values even when optimization fails
                if home_strength_ratio > 1:  # Home team is stronger
                    df.at[match_idx, 'home_xg_elo'] = round(league_avg_home_xg * (home_strength_ratio ** 0.15), 2)
                    df.at[match_idx, 'away_xg_elo'] = round(league_avg_away_xg * (1 / home_strength_ratio ** 0.1), 2)
                else:  # Away team is stronger
                    df.at[match_idx, 'home_xg_elo'] = round(league_avg_home_xg * (home_strength_ratio ** 0.1), 2)
                    df.at[match_idx, 'away_xg_elo'] = round(league_avg_away_xg * (1 / home_strength_ratio ** 0.15), 2)
            
            # Update Elo ratings after the match if it has been played
            if match_idx in df.index and df.at[match_idx, 'status'] in ['complete', 'finished']:
                # Get match result
                match_result = df.at[match_idx, 'full_time_result']
                
                # Only update if we have a result
                if pd.notna(match_result):
                    # Convert result to score for Elo calculation
                    if match_result == 'H':
                        actual_score = 1.0  # Home win
                    elif match_result == 'A':
                        actual_score = 0.0  # Away win
                    else:  # Draw
                        actual_score = 0.5
                    
                    # Calculate expected score based on Elo difference (including draw probability)
                    expected_score = home_win_prob + (0.5 * draw_prob)
                    
                    # Calculate Elo updates
                    elo_change = k_factor * (actual_score - expected_score)
                    
                    # Update Elo ratings for next matches
                    elo_ratings[home_team] = current_home_elo + elo_change
                    elo_ratings[away_team] = current_away_elo - elo_change
    
    # Create a new DataFrame with only the original columns plus xG columns
    # This ensures we don't include the temporary Elo and probability columns
    original_columns = [col for col in df.columns if col not in all_columns]
    final_columns = original_columns + output_columns
    output_df = df[final_columns].copy()

    # output_df = output_df.drop(['full_time_result'], axis=1)
    
    # Save the enhanced dataframe with only the xG columns
    # output_df.to_excel(output_file, index=False)
    # print(f"Data with only xG values saved to {output_file}")
    
    return output_df

# Example usage
df_elo = add_elo_xg_only(clean_dataframe, num_recent_games=NUM_PREVIOUS_GAMES, 
                                home_advantage=35, league_home_xg=1.35, league_away_xg=1.1,
                                def_elo=1500, def_f_factor=32)

# Save the enhanced dataframe with only the xG columns
# df_elo.to_excel(OUTPUT_FILENAME, index=False)
# print(f"Data with only xG values saved to {OUTPUT_FILENAME}")

# ###################################################################################################
# #########################     DIXON-COLES MODEL    ################################################

def _get_team_matches_dixonCole(df, team):
    """Get all matches for a team (both home and away)"""
    return df[(df['home_team'] == team) | (df['away_team'] == team)]

def _dixon_coles_correction(home_goals, away_goals, home_rate, away_rate, rho):
    """Dixon-Coles correction function for low scoring matches."""
    if home_goals == 0 and away_goals == 0:
        return 1 - (home_rate * away_rate * rho)
    elif home_goals == 1 and away_goals == 0:
        return 1 + (away_rate * rho)
    elif home_goals == 0 and away_goals == 1:
        return 1 + (home_rate * rho)
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0

def _dixon_coles_loglikelihood(params, match_data, teams_attack, teams_defense, num_teams):
    """Calculate the negative log-likelihood for the Dixon-Coles model."""
    # Extract parameters
    home_advantage = params[0]
    rho = params[1]
    
    # Calculate negative log-likelihood for matches
    nll = 0.0
    
    for home_idx, away_idx, home_goals, away_goals in match_data:
        # Calculate expected goals
        home_rate = np.exp(teams_attack[home_idx] + teams_defense[away_idx] + home_advantage)
        away_rate = np.exp(teams_attack[away_idx] + teams_defense[home_idx])
        
        # Apply Dixon-Coles correction for low-scoring matches
        correction = _dixon_coles_correction(home_goals, away_goals, home_rate, away_rate, rho)
        
        # Poisson probability with Dixon-Coles correction
        home_prob = np.exp(-home_rate) * (home_rate ** home_goals) / np.math.factorial(home_goals)
        away_prob = np.exp(-away_rate) * (away_rate ** away_goals) / np.math.factorial(away_goals)
        
        # Add to negative log-likelihood
        if correction > 0:
            nll -= np.log(home_prob * away_prob * correction)
        else:
            # Avoid numerical issues
            nll += 10  # Penalize impossible scenarios
    
    # Add regularization to prevent extreme values
    regularization_strength = 0.1
    for team_idx in range(num_teams):
        nll += regularization_strength * (teams_attack[team_idx]**2 + teams_defense[team_idx]**2)
    
    return nll

def _fit_dixon_coles_model(match_data, teams, initial_values=None, min_home_advantage=0.3):
    """
    Fit the Dixon-Coles model to match data.
    
    Parameters:
    -----------
    match_data : list of tuples
        List of (home_idx, away_idx, home_goals, away_goals) tuples
    teams : list
        List of team names
    initial_values : dict or None
        Dictionary of initial attack and defense values for teams
    min_home_advantage : float
        Minimum home advantage value to enforce (default: 0.3)
        
    Returns:
    --------
    tuple (attack_strengths, defense_strengths, home_advantage, rho)
        Fitted model parameters
    """
    num_teams = len(teams)
    teams_idx = {team: i for i, team in enumerate(teams)}
    
    # Initial values for team parameters
    if initial_values is None:
        # Default initial values
        teams_attack = {i: 0.0 for i in range(num_teams)}
        teams_defense = {i: 0.0 for i in range(num_teams)}
    else:
        # Use provided initial values
        teams_attack = {teams_idx[team]: values['attack'] for team, values in initial_values.items() if team in teams_idx}
        teams_defense = {teams_idx[team]: values['defense'] for team, values in initial_values.items() if team in teams_idx}
        
        # Set default values for any missing teams
        for i in range(num_teams):
            if i not in teams_attack:
                teams_attack[i] = 0.0
            if i not in teams_defense:
                teams_defense[i] = 0.0
    
    # Initial values for global parameters
    # Start with a higher initial home advantage to encourage the model to find values above our minimum
    initial_params = np.array([max(0.5, min_home_advantage), -0.1])  # home advantage and rho
    
    # Define objective function for the global parameters optimization
    def objective(params):
        return _dixon_coles_loglikelihood(params, match_data, teams_attack, teams_defense, num_teams)
    
    # Optimize global parameters with lower bound for home advantage
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=[(min_home_advantage, 1.0), (-1.0, 0.0)],  # home advantage [min,1], rho [-1,0]
        options={'maxiter': 100}
    )
    
    # Extract optimized parameters
    home_advantage = max(result.x[0], min_home_advantage)  # Enforce minimum even if optimizer went below
    rho = result.x[1]
    
    # Create a Poisson regression dataset for estimating team parameters
    X_data = []
    y_data = []
    
    for home_idx, away_idx, home_goals, away_goals in match_data:
        # For home goals
        x_row = np.zeros(2 * num_teams + 1)
        x_row[home_idx] = 1  # Home attack
        x_row[num_teams + away_idx] = 1  # Away defense
        x_row[-1] = 1  # Home advantage
        X_data.append(x_row)
        y_data.append(home_goals)
        
        # For away goals
        x_row = np.zeros(2 * num_teams + 1)
        x_row[away_idx] = 1  # Away attack
        x_row[num_teams + home_idx] = 1  # Home defense
        X_data.append(x_row)
        y_data.append(away_goals)
    
    # Fit Poisson regression model
    X = np.array(X_data)
    y = np.array(y_data)
    
    try:
        model = sm.GLM(y, X, family=sm.families.Poisson())
        result = model.fit(disp=0)
        
        # Extract parameters
        coeffs = result.params
        
        # Update team parameters
        for i in range(num_teams):
            teams_attack[i] = coeffs[i]
            teams_defense[i] = coeffs[num_teams + i]
        
        # Home advantage from Poisson model, but ensure it's above minimum
        home_advantage = max(coeffs[-1], min_home_advantage)
    except:
        print("  Error fitting Poisson model, using partial estimates")
    
    # Ensure identifiability by centering the attack and defense parameters
    avg_attack = sum(teams_attack.values()) / num_teams
    avg_defense = sum(teams_defense.values()) / num_teams
    
    for i in range(num_teams):
        teams_attack[i] -= avg_attack
        teams_defense[i] -= avg_defense
    
    # Convert parameters to team ratings
    attack_strengths = {teams[i]: teams_attack[i] for i in range(num_teams)}
    defense_strengths = {teams[i]: teams_defense[i] for i in range(num_teams)}
    
    return attack_strengths, defense_strengths, home_advantage, rho

# Function to calculate expected goals using team ratings
def calculate_expected_goals_DixonCole(home_team, away_team, attack_params, defense_params, home_advantage=1.3):
    """
    Calculate expected goals for a match using team attack and defense parameters.
    
    Parameters:
    -----------
    home_team, away_team : str
        Team names
    attack_params, defense_params : dict
        Dictionaries containing attack and defense parameters for all teams
    home_advantage : float
        Home advantage factor (default: 1.2)
    
    Returns:
    --------
    tuple
        (expected_home_goals, expected_away_goals)
    """
    # Get parameters - default to middle value (1.0) if not available
    home_attack = attack_params.get(home_team, 1.0)
    home_defense = defense_params.get(home_team, 1.0)
    away_attack = attack_params.get(away_team, 1.0)
    away_defense = defense_params.get(away_team, 1.0)
    
    # For defense_factor, higher values mean better defense (fewer goals conceded)
    # We need to convert defense rating to a factor that reduces expected goals
    home_defense_factor = (3.0 - home_defense) / 3.0
    away_defense_factor = (3.0 - away_defense) / 3.0
    
    # Calculate expected goals
    # Home team expected goals = home team attack * away team defense factor * home advantage
    # Away team expected goals = away team attack * home team defense factor
    expected_home_goals = home_attack * away_defense_factor * home_advantage
    expected_away_goals = away_attack * home_defense_factor
    
    # Ensure expected goals are within a reasonable range (0-3)
    expected_home_goals = min(max(expected_home_goals, 0.0), 3.0)
    expected_away_goals = min(max(expected_away_goals, 0.0), 3.0)
    
    return expected_home_goals, expected_away_goals

def add_dixon_coles_ratings(dataframe, num_recent_games=6, min_home_advantage=0.3, home_advantage_factor=1.3):
    """
    Add Dixon-Coles model rating columns to the football dataset.
    Ratings are explicitly scaled to be positive and within the range 0-5.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    num_recent_games : int, optional
        Number of recent games to consider for calculations (default: 6)
    min_home_advantage : float, optional
        Minimum home advantage value to enforce (default: 0.3)
    home_advantage_factor : float, optional
        Home advantage factor for expected goals calculation (default: 1.2)
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_dc' to original filename)
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with Dixon-Coles rating columns
    """
    # Set default output file name if not provided
    # if output_file is None:
    #     output_file = file_path.replace('.xlsx', '_dc_fixed.xlsx')
    
    # Load and process the data
    # df = load_and_process_data(file_path)
    df = dataframe

    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize Dixon-Coles rating columns
    dc_columns = [
        'home_team_attack',
        'home_team_defense',
        'away_team_attack',
        'away_team_defense',
        'dc_home_advantage',
        'home_xg_dc',  # New column for home team expected goals
        'away_xg_dc'   # New column for away team expected goals
    ]
    
    # Initialize all columns with default values
    for col in dc_columns:
        if col.endswith('_attack') or col.endswith('_defense'):
            df[col] = 0.5  # Default rating of 0.5
        elif col == 'dc_home_advantage':
            df[col] = min_home_advantage  # Set to minimum home advantage
        elif col.endswith('_xg_dc'):
            df[col] = 0.0  # Default expected goals of 0.0
        else:
            df[col] = 0.0
    
    # Process each season separately
    seasons = sorted(df['season'].unique())
    for i, season in enumerate(seasons):
        print(f"Processing season: {season}")
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Dixon-Coles model parameters for this season
        team_attack = {}  # Dictionary to store attack parameters
        team_defense = {}  # Dictionary to store defense parameters
        home_advantage = min_home_advantage  # Initialize with minimum home advantage
        rho = 0.0  # Dixon-Coles correlation parameter (still used internally but not exported)
        
        # Initialize ratings for teams in this season
        teams_in_season = list(set(season_df['home_team'].unique()) | set(season_df['away_team'].unique()))
        for team in teams_in_season:
            team_attack[team] = 0.0
            team_defense[team] = 0.0
        
        # Track the model fitting frequency
        last_fit_index = -1
        
        # Adjust fit interval based on season length
        season_length = len(season_indices)
        fit_interval = max(num_recent_games, int(season_length / max(1, season_length/num_recent_games)))
        
        # Adjust minimum matches needed based on season position
        if i == 0:
            # First season - start fitting earlier
            min_matches_needed = num_recent_games
        else:
            # Subsequent seasons - standard threshold
            min_matches_needed = num_recent_games
        
        # Process each match chronologically within the season
        for match_idx_position, match_idx in enumerate(season_indices):
            home_team = df.at[match_idx, 'home_team']
            away_team = df.at[match_idx, 'away_team']
            
            # Get all previous matches in this season up to this match
            prev_season_matches = season_df[season_df.index < match_idx]
            
            # Check if we should recalculate ratings
            enough_matches = len(prev_season_matches) >= min_matches_needed
            enough_new_matches = match_idx_position - last_fit_index >= fit_interval
            
            if enough_matches and (enough_new_matches or match_idx_position == len(season_indices) - 1):
                print(f"  Fitting model at match {match_idx_position+1}/{len(season_indices)}")
                last_fit_index = match_idx_position
                
                # Create a list of match data for fitting
                match_data = []
                
                # Teams that appear in recent matches
                teams_with_recent_matches = set()
                
                # For each team, get their most recent matches
                for team in teams_in_season:
                    team_matches = _get_team_matches_dixonCole(prev_season_matches, team)
                    
                    # Only use the most recent num_recent_games for each team
                    if len(team_matches) > 0:
                        recent_team_matches = team_matches.tail(min(len(team_matches), num_recent_games))
                        
                        # Add to the set of teams with recent matches
                        teams_with_recent_matches.add(team)
                        
                        # Add each match to the data if not already added
                        for _, match in recent_team_matches.iterrows():
                            home_team_match = match['home_team']
                            away_team_match = match['away_team']
                            
                            # Add both teams to the set
                            teams_with_recent_matches.add(home_team_match)
                            teams_with_recent_matches.add(away_team_match)
                
                # Filter to only teams with recent matches
                teams_to_fit = list(teams_with_recent_matches)
                teams_idx = {team: i for i, team in enumerate(teams_to_fit)}
                
                # Prepare match data for model fitting
                for _, match in prev_season_matches.iterrows():
                    home_team_match = match['home_team']
                    away_team_match = match['away_team']
                    
                    # Only include matches between teams in our index
                    if home_team_match in teams_idx and away_team_match in teams_idx:
                        home_idx = teams_idx[home_team_match]
                        away_idx = teams_idx[away_team_match]
                        home_goals = match['home_goals']
                        away_goals = match['away_goals']
                        
                        match_data.append((home_idx, away_idx, home_goals, away_goals))
                
                if len(match_data) >= (num_recent_games-1) and len(teams_to_fit) >= (num_recent_games-1):
                    # Prepare initial values
                    initial_values = {}
                    for team in teams_to_fit:
                        if team in team_attack and team in team_defense:
                            initial_values[team] = {
                                'attack': team_attack[team],
                                'defense': team_defense[team]
                            }
                    
                    # Fit the Dixon-Coles model with minimum home advantage
                    try:
                        attack, defense, home_advantage, rho = _fit_dixon_coles_model(
                            match_data, teams_to_fit, initial_values, min_home_advantage
                        )
                        
                        # Update team parameters
                        for team, att in attack.items():
                            team_attack[team] = att
                        
                        for team, defs in defense.items():
                            team_defense[team] = defs
                        
                        print(f"  Model fit successful. Home advantage: {home_advantage:.2f}")
                    except Exception as e:
                        print(f"  Error fitting Dixon-Coles model: {e}")
                        # If optimization fails, keep existing ratings but ensure home advantage is at least minimum
                        home_advantage = max(home_advantage, min_home_advantage)
            
            # Set current Dixon-Coles ratings for this match
            current_home_attack = team_attack.get(home_team, 0.0)
            current_home_defense = team_defense.get(home_team, 0.0)
            current_away_attack = team_attack.get(away_team, 0.0)
            current_away_defense = team_defense.get(away_team, 0.0)
            
            # Scale ratings to 0-3 range
            all_attack_values = list(team_attack.values())
            all_defense_values = list(team_defense.values())
            
            if len(all_attack_values) >= 2:
                # Find min and max values
                min_attack = min(all_attack_values)
                max_attack = max(all_attack_values)
                min_defense = min(all_defense_values)
                max_defense = max(all_defense_values)
                
                # Range for scaling
                attack_range = max_attack - min_attack if max_attack > min_attack else 1.0
                defense_range = max_defense - min_defense if max_defense > min_defense else 1.0
                
                # Scale to 0-3 range and shift to ensure minimum value is 0.5
                # For attack, higher raw value = better attack
                home_attack_scaled = 0.5 + 2.5 * (current_home_attack - min_attack) / attack_range
                away_attack_scaled = 0.5 + 2.5 * (current_away_attack - min_attack) / attack_range
                
                # For defense, lower raw value = better defense, so invert
                home_defense_scaled = 0.5 + 2.5 * (max_defense - current_home_defense) / defense_range
                away_defense_scaled = 0.5 + 2.5 * (max_defense - current_away_defense) / defense_range
                
                # Clip to ensure 0.5-3 range even if distribution is skewed
                home_attack_scaled = min(max(home_attack_scaled, 0.5), 3.0)
                home_defense_scaled = min(max(home_defense_scaled, 0.5), 3.0)
                away_attack_scaled = min(max(away_attack_scaled, 0.5), 3.0)
                away_defense_scaled = min(max(away_defense_scaled, 0.5), 3.0)
                
                # Create a dictionary of scaled attack and defense values for expected goals calculation
                attack_params = {
                    home_team: home_attack_scaled,
                    away_team: away_attack_scaled
                }
                
                defense_params = {
                    home_team: home_defense_scaled,
                    away_team: away_defense_scaled
                }
                
                # Calculate expected goals using the formula from the first script
                home_expected_goals, away_expected_goals = calculate_expected_goals_DixonCole(
                    home_team, 
                    away_team, 
                    attack_params, 
                    defense_params, 
                    home_advantage_factor
                )
            else:
                # Not enough teams for scaling, use default value
                home_attack_scaled = 0.5
                home_defense_scaled = 0.5
                away_attack_scaled = 0.5
                away_defense_scaled = 0.5
                home_expected_goals = 0.0
                away_expected_goals = 0.0
            
            # Store ratings in the dataframe
            df.at[match_idx, 'home_team_attack'] = home_attack_scaled
            df.at[match_idx, 'home_team_defense'] = home_defense_scaled
            df.at[match_idx, 'away_team_attack'] = away_attack_scaled
            df.at[match_idx, 'away_team_defense'] = away_defense_scaled
            df.at[match_idx, 'dc_home_advantage'] = home_advantage
            df.at[match_idx, 'home_xg_dc'] = round(home_expected_goals, 2)
            df.at[match_idx, 'away_xg_dc'] = round(away_expected_goals, 2)
    
    # Round all rating columns to 2 decimal places
    for col in dc_columns:
        df[col] = df[col].round(2)
    
    # Final check to ensure all home advantage values are at least the minimum
    df['dc_home_advantage'] = np.maximum(df['dc_home_advantage'], min_home_advantage)

    # drop unwanted columns
    df = df.drop(['home_team_attack', 'home_team_defense', 'away_team_attack', 'away_team_defense', 'dc_home_advantage'], axis=1)
    
    # Save the enhanced dataframe
    # df.to_excel(output_file, index=False)
    # print(f"Data with Dixon-Coles ratings saved to {output_file}")
    
    return df

# Example usage
df_dc = add_dixon_coles_ratings(
    clean_dataframe,
    num_recent_games=NUM_PREVIOUS_GAMES,
    min_home_advantage=MIN_HOME_ADVANTAGE,  # Setting minimum home advantage
    home_advantage_factor=HOME_ADVANTAGE,  # Home advantage factor for expected goals calculation
)


save_to_appwrite_storage(df_dc, STORAGE_ID, file_id="echStatsDixon", client=client, 
                        #  appwrite_endpoint=API_ENDPOINT, 
                        #  project_id=PROJECT_ID, 
                        #  api_key=API_KEY
                         )

# return context.res.empty()
