import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize

def load_and_process_data(file_path):
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

def add_elo_xg_only(file_path, output_file=None, num_recent_games=6, home_advantage=35, league_home_xg=1.35, league_away_xg=1.1, def_elo=1500, def_f_factor=32):
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
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_xg_only.xlsx')
    
    # Load and process the data
    df = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize only the xG columns in the final dataframe
    output_columns = [
        'home_xg_from_elo',
        'away_xg_from_elo'
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
                
                df.at[match_idx, 'home_xg_from_elo'] = round(home_xg, 2)
                df.at[match_idx, 'away_xg_from_elo'] = round(away_xg, 2)
            except Exception as e:
                print(f"Error calculating xG for match {match_idx}: {e}")
                # Fallback calculation for xG based on relative team strength
                home_strength_ratio = 10 ** (elo_diff)
                
                # More balanced xG values even when optimization fails
                if home_strength_ratio > 1:  # Home team is stronger
                    df.at[match_idx, 'home_xg_from_elo'] = round(league_avg_home_xg * (home_strength_ratio ** 0.15), 2)
                    df.at[match_idx, 'away_xg_from_elo'] = round(league_avg_away_xg * (1 / home_strength_ratio ** 0.1), 2)
                else:  # Away team is stronger
                    df.at[match_idx, 'home_xg_from_elo'] = round(league_avg_home_xg * (home_strength_ratio ** 0.1), 2)
                    df.at[match_idx, 'away_xg_from_elo'] = round(league_avg_away_xg * (1 / home_strength_ratio ** 0.15), 2)
            
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
    
    # Save the enhanced dataframe with only the xG columns
    output_df.to_excel(output_file, index=False)
    print(f"Data with only xG values saved to {output_file}")
    
    return output_df

# Example usage
df_with_xg_only = add_elo_xg_only('nfd_data.xlsx', output_file='elo.xlsx', num_recent_games=6, 
                                   home_advantage=35, league_home_xg=1.35, league_away_xg=1.1,
                                   def_elo=1500, def_f_factor=32)