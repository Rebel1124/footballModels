import pandas as pd
import numpy as np
from scipy.optimize import minimize

def load_and_process_data(file_path):
    """
    Load match data from Excel and prepare for analysis.
    With specific handling for various column naming conventions.
    """
    # [Keep the existing implementation]
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
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
            raw_df = pd.read_excel(file_path)
            print(f"Raw columns: {raw_df.columns.tolist()}")
            print(raw_df.head(3))
        except Exception as sub_e:
            print(f"Could not read raw file: {sub_e}")
        return None

def _get_team_matches(df, team):
    """Get all matches for a team (both home and away)"""
    return df[(df['home_team'] == team) | (df['away_team'] == team)]

def _bradley_terry_loglikelihood(params, match_data, teams_idx, teams_count):
    """
    Calculate the negative log-likelihood for the Bradley-Terry model with regularization.
    
    Parameters:
    -----------
    params : array
        Array of parameters: team strengths followed by home advantage
    match_data : list of tuples
        List of (home_idx, away_idx, home_win, draw) tuples
    teams_idx : dict
        Mapping of team names to index in strengths array
    teams_count : int
        Number of teams
        
    Returns:
    --------
    float
        Negative log-likelihood with regularization
    """
    # Separate team strengths and home advantage
    strengths = params[:teams_count]
    home_advantage = params[-1]
    
    # Add constraint to ensure sum of strengths is 0 (identifiability constraint)
    strengths = strengths - np.mean(strengths)
    
    # Calculate negative log-likelihood for matches
    nll = 0.0
    for home_idx, away_idx, home_win, draw in match_data:
        # Get team strengths
        s_i = strengths[home_idx] + home_advantage
        s_j = strengths[away_idx]
        
        # Difference in strengths determines win probability
        diff = s_i - s_j
        
        # Modified logistic function for probability calculation
        p_i = 1.0 / (1.0 + np.exp(-diff))
        
        # For draws, use an ordered logit model
        # Higher probability of draw when teams are more evenly matched
        p_draw = max(0.0, 1.0 - abs(p_i - 0.5) * 2.0)  # Simple draw model
        p_draw = min(p_draw, 0.5)  # Cap the draw probability
        
        # Adjust win/loss probabilities
        p_i_win = p_i * (1.0 - p_draw)
        p_j_win = (1.0 - p_i) * (1.0 - p_draw)
        
        # Add to negative log-likelihood based on actual result
        if draw:
            nll -= np.log(max(p_draw, 1e-10))
        elif home_win:
            nll -= np.log(max(p_i_win, 1e-10))
        else:
            nll -= np.log(max(p_j_win, 1e-10))
    
    # Add regularization to prevent extreme values (L2 regularization)
    regularization_strength = 0.1
    nll += regularization_strength * np.sum(strengths**2)
    
    return nll

def add_bradley_terry_ratings(file_path, num_recent_games=6, output_file=None):
    """
    Add Bradley-Terry model rating columns to the football dataset.
    Ratings are scaled to be positive and within the range 0-5 with average at 1.0.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    num_recent_games : int, optional
        Number of recent games to consider for calculations (default: 6)
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_bt' to original filename)
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with Bradley-Terry rating columns
    """
    # Set default output file name if not provided
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_bt_fixed.xlsx')
    
    # Load and process the data
    df = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize Bradley-Terry rating columns
    bt_columns = [
        'home_team_bt',
        'away_team_bt',
        'bt_home_advantage'
    ]
    
    # Initialize all columns with default values
    for col in bt_columns:
        df[col] = 0.0
    
    # Process each season separately
    for season in df['season'].unique():
        print(f"Processing season: {season}")
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Bradley-Terry model parameters
        bt_ratings = {}  # Dictionary to store Bradley-Terry ratings
        bt_home_advantage = 0.0  # Home advantage parameter for Bradley-Terry model
        
        # Initialize ratings for teams in this season
        teams_in_season = list(set(season_df['home_team'].unique()) | set(season_df['away_team'].unique()))
        for team in teams_in_season:
            bt_ratings[team] = 0.0  # Start with neutral rating
        
        # Create a mapping from team names to indices
        teams_idx = {team: i for i, team in enumerate(teams_in_season)}
        
        # Track the model fitting frequency - don't need to fit every match
        last_fit_index = -1
        # fit_interval = max(5, len(season_df) // 20)  # Fit approximately 20 times per season
        fit_interval = max(num_recent_games, len(season_df) // (len(season_df)/num_recent_games))  # Fit approximately (len(season_df)/num_recent_games) times per season
        
        # Process each match chronologically within the season
        for match_idx_position, match_idx in enumerate(season_indices):
            home_team = df.at[match_idx, 'home_team']
            away_team = df.at[match_idx, 'away_team']
            
            # Get all previous matches in this season up to this match
            prev_season_matches = season_df[season_df.index < match_idx]
            
            # Recalculate Bradley-Terry ratings periodically
            # enough_matches = len(prev_season_matches) >= 15  # Need matches for reliable Bradley-Terry
            enough_matches = len(prev_season_matches) >= num_recent_games  # Need matches for reliable Bradley-Terry: num_recent_games
            enough_new_matches = match_idx_position - last_fit_index >= fit_interval
            
            if enough_matches and (enough_new_matches or match_idx_position == len(season_indices) - 1):
                print(f"  Fitting model at match {match_idx_position+1}/{len(season_indices)}")
                last_fit_index = match_idx_position
                
                # Create a list of match data for optimization
                match_data = []
                
                # Create a set to track teams with recent matches
                teams_with_recent_matches = set()
                
                # For each team, get their most recent matches
                for team in teams_in_season:
                    team_matches = _get_team_matches(prev_season_matches, team)
                    
                    # Only use the most recent num_recent_games for each team
                    recent_team_matches = team_matches.tail(min(len(team_matches), num_recent_games))
                    
                    # Add to the set of teams with recent matches
                    if len(recent_team_matches) > 0:
                        teams_with_recent_matches.add(team)
                        
                        # Add each match to match_data if not already added
                        for _, match in recent_team_matches.iterrows():
                            home_team_match = match['home_team']
                            away_team_match = match['away_team']
                            
                            # Both teams need to be in our index
                            if home_team_match in teams_idx and away_team_match in teams_idx:
                                home_idx = teams_idx[home_team_match]
                                away_idx = teams_idx[away_team_match]
                                home_win = match['full_time_result'] == 'H'
                                draw = match['full_time_result'] == 'D'
                                
                                # Check if this match is already in match_data
                                match_tuple = (home_idx, away_idx, home_win, draw)
                                if match_tuple not in match_data:
                                    match_data.append(match_tuple)
                
                # if len(match_data) >= 10:  # Ensure enough matches for optimization
                if len(match_data) >= num_recent_games:  # Ensure enough matches for optimization: num_recent_games
                    # Initial values - start with current ratings if available
                    initial_strengths = np.zeros(len(teams_in_season))
                    for team, idx in teams_idx.items():
                        initial_strengths[idx] = bt_ratings.get(team, 0.0)
                    
                    # Add home advantage parameter
                    initial_params = np.append(initial_strengths, bt_home_advantage if bt_home_advantage != 0 else 0.1)
                    
                    # Optimize with constraints and regularization
                    try:
                        result = minimize(
                            lambda params: _bradley_terry_loglikelihood(
                                params, match_data, teams_idx, len(teams_in_season)
                            ),
                            initial_params,
                            method='L-BFGS-B',  # More stable method
                            bounds=[(-3, 3)] * len(teams_in_season) + [(0, 1)],  # Bounds to prevent extreme values
                            options={'maxiter': 1000}
                        )
                        
                        # Extract results
                        optimized_strengths = result.x[:len(teams_in_season)]
                        bt_home_advantage = result.x[-1]
                        
                        # Center strengths to ensure identifiability
                        optimized_strengths = optimized_strengths - np.mean(optimized_strengths)
                        
                        # Transform ratings to 0-5 scale with average at 1.0
                        # First, normalize to make variance approximately 0.5
                        if len(optimized_strengths) > 1 and np.std(optimized_strengths) > 0:
                            optimized_strengths = optimized_strengths / (2 * np.std(optimized_strengths))
                        
                        # Then shift and scale to 0-5 range with average at 1.0
                        optimized_strengths = 1.0 + optimized_strengths  # Center at 1.0
                        
                        # Clip to ensure range bounds (0-5)
                        optimized_strengths = np.clip(optimized_strengths, 0, 5)
                        
                        # Update team ratings
                        for team, idx in teams_idx.items():
                            bt_ratings[team] = optimized_strengths[idx]
                        
                        print(f"  Model fit successful. Home advantage: {bt_home_advantage:.2f}")
                    except Exception as e:
                        print(f"  Error fitting Bradley-Terry model: {e}")
                        # If optimization fails, keep existing ratings
            
            # Set current Bradley-Terry ratings for this match
            current_home_bt = bt_ratings.get(home_team, 1.0)  # Default to average (1.0) if not available
            current_away_bt = bt_ratings.get(away_team, 1.0)
            
            # Store ratings in the dataframe
            df.at[match_idx, 'home_team_bt'] = current_home_bt
            df.at[match_idx, 'away_team_bt'] = current_away_bt
            df.at[match_idx, 'bt_home_advantage'] = bt_home_advantage
    
    # Round all rating columns to 2 decimal places
    for col in bt_columns:
        df[col] = df[col].round(2)
    
    # Save the enhanced dataframe
    df.to_excel(output_file, index=False)
    print(f"Data with Bradley-Terry ratings saved to {output_file}")
    
    return df

# Example usage
df_with_bet = add_bradley_terry_ratings(file_path='nfd_data.xlsx', num_recent_games=6, output_file='bradleyterry.xlsx')
