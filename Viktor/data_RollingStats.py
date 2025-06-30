import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from appwrite.query import Query
from appwrite.client import Client
from scipy.optimize import minimize
from appwrite.services.databases import Databases

load_dotenv()

# Replace these with your actual Appwrite credentials
API_ENDPOINT = 'https://cloud.appwrite.io/v1'
PROJECT_ID = os.getenv('APPWRITE_PROJECT_ID')
API_KEY = os.getenv('APPWRITE_API_KEY')
DATABASE_ID = os.getenv('APPWRITE_DB_ID')

# List of collection IDs to retrieve data from
COLLECTION_IDS = [
    os.getenv('SEASON_MATCHES_NFD20_21'),
    os.getenv('SEASON_MATCHES_NFD21_22'),  
    os.getenv('SEASON_MATCHES_NFD22_23'),
    os.getenv('SEASON_MATCHES_NFD23_24'),
    os.getenv('SEASON_MATCHES_NFD24_25'),  
]

OUTPUT_FILENAME = "nfd_data_rolling.xlsx"

# User-configurable parameter: number of past games to consider for rolling statistics
# Default is 6, but can be changed as needed
NUM_PREVIOUS_GAMES = 6

# User-configurable parameter: home advantage factor for expected goals calculation
# Default is 1.2, but can be changed as needed
HOME_ADVANTAGE = 1.3

# Initialize Appwrite client
client = Client()
client.set_endpoint(API_ENDPOINT)
client.set_project(PROJECT_ID)
client.set_key(API_KEY)

# Initialize Databases service
databases = Databases(client)

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
    # combined_df = combined_df[['match_date', 'season', 'status', 'homeID', 'home_name', 'awayID', 'away_name', 'winningTeam', 'winningTeamName',
    #                            'homeGoalCount','awayGoalCount', 'odds_ft_1', 'odds_ft_x', 'odds_ft_2']]
    
    # Apply xG calculations
    combined_df = add_xg_columns(combined_df)
    
    # Add the new rolling statistics including expected goals with the specified home advantage
    combined_df = add_team_rolling_stats(combined_df, num_previous_games=NUM_PREVIOUS_GAMES, home_advantage=HOME_ADVANTAGE)

    # Save to Excel
    combined_df.to_excel(OUTPUT_FILENAME, index=False)
    print(f"\nData saved to {OUTPUT_FILENAME}")
    
    # Print first few rows
    # print("\nFirst 5 rows of the combined data:")
    # print(combined_df.head())
else:
    print("No data found in any of the collections")