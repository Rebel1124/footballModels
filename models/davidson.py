import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(file_path):
    """
    Load match data from Excel and prepare for analysis.
    With specific handling for various column naming conventions.
    """
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns.")
        
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
            unique_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
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

def _scale_to_range(value, old_min, old_max, new_min, new_max):
    """
    Scale a value from one range to another.
    
    Parameters:
    -----------
    value : float
        Value to scale
    old_min, old_max : float
        Original range
    new_min, new_max : float
        Target range
    
    Returns:
    --------
    float
        Scaled value
    """
    # Check if old range is valid
    if old_max == old_min:
        return (new_max + new_min) / 2  # Return midpoint of new range
    
    # Check if value is outside old range
    if value < old_min:
        return new_min
    if value > old_max:
        return new_max
    
    # Perform scaling
    scaled_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    
    # Ensure value is within new range
    return min(max(scaled_value, new_min), new_max)

def _construct_team_goal_matrix(matches_df, team_names):
    """
    Construct matrices for the Davidson model based on the matches data.
    
    Parameters:
    -----------
    matches_df : pandas.DataFrame
        DataFrame containing match data
    team_names : list
        List of team names
    
    Returns:
    --------
    tuple
        (team_indices, team_goals_for, team_goals_against, team_games)
    """
    n_teams = len(team_names)
    team_indices = {team: i for i, team in enumerate(team_names)}
    
    # Initialize arrays
    team_goals_for = np.zeros(n_teams)
    team_goals_against = np.zeros(n_teams)
    team_games = np.zeros(n_teams)
    
    # Pre-compute indices for faster lookup
    home_teams = matches_df['home_team'].values
    away_teams = matches_df['away_team'].values
    home_goals = matches_df['home_goals'].values
    away_goals = matches_df['away_goals'].values
    
    # Fill in the arrays
    for i in range(len(matches_df)):
        home_team = home_teams[i]
        away_team = away_teams[i]
        
        if home_team in team_indices and away_team in team_indices:
            home_idx = team_indices[home_team]
            away_idx = team_indices[away_team]
            
            # Update goals for and against
            team_goals_for[home_idx] += home_goals[i]
            team_goals_against[home_idx] += away_goals[i]
            
            team_goals_for[away_idx] += away_goals[i]
            team_goals_against[away_idx] += home_goals[i]
            
            # Update games count
            team_games[home_idx] += 1
            team_games[away_idx] += 1
    
    return team_indices, team_goals_for, team_goals_against, team_games

def _davidson_likelihood(params, teams_data, matches_data, min_home_advantage=1.1):
    """
    Calculate the negative log-likelihood of the Davidson model parameters.
    
    Parameters:
    -----------
    params : numpy.array
        Parameters of the Davidson model (offensive strengths + [home_advantage])
    teams_data : dict
        Dictionary containing team indices
    matches_data : tuple
        Tuple with (home_teams, away_teams, home_goals, away_goals) arrays
    min_home_advantage : float
        Minimum home advantage value (default: 1.1)
    
    Returns:
    --------
    float
        Negative log-likelihood
    """
    team_indices = teams_data
    home_teams, away_teams, home_goals, away_goals = matches_data
    n_teams = len(team_indices)
    
    # Extract parameters
    offensive_strengths = params[:n_teams]
    defensive_strengths = 1.0 / offensive_strengths  # Defensive strength is reciprocal of offensive
    home_advantage = max(params[n_teams], min_home_advantage)  # Apply minimum home advantage
    
    # Initialize log-likelihood
    log_likelihood = 0.0
    factorial_cache = {}  # Cache factorial calculations
    
    # Process each match
    for i in range(len(home_teams)):
        home_team = home_teams[i]
        away_team = away_teams[i]
        
        if home_team in team_indices and away_team in team_indices:
            home_idx = team_indices[home_team]
            away_idx = team_indices[away_team]
            h_goals = home_goals[i]
            a_goals = away_goals[i]
            
            # Expected goals
            home_expected = offensive_strengths[home_idx] * defensive_strengths[away_idx] * home_advantage
            away_expected = offensive_strengths[away_idx] * defensive_strengths[home_idx]
            
            # Log-likelihood (Poisson distribution)
            if home_expected > 0:
                # Calculate log factorial using cache for efficiency
                h_goals_int = int(h_goals)
                if h_goals_int not in factorial_cache:
                    factorial_cache[h_goals_int] = np.log(np.math.factorial(h_goals_int))
                
                log_likelihood += h_goals * np.log(home_expected) - home_expected - factorial_cache[h_goals_int]
            
            if away_expected > 0:
                # Calculate log factorial using cache for efficiency
                a_goals_int = int(a_goals)
                if a_goals_int not in factorial_cache:
                    factorial_cache[a_goals_int] = np.log(np.math.factorial(a_goals_int))
                
                log_likelihood += a_goals * np.log(away_expected) - away_expected - factorial_cache[a_goals_int]
    
    # Return negative log-likelihood for minimization
    return -log_likelihood

def _fit_davidson_model(matches_df, min_home_advantage=1.1):
    """
    Fit the Davidson model to the match data.
    
    Parameters:
    -----------
    matches_df : pandas.DataFrame
        DataFrame containing match data
    min_home_advantage : float
        Minimum home advantage value (default: 1.1)
    
    Returns:
    --------
    tuple
        (team_indices, offensive_strengths, defensive_strengths, home_advantage)
    """
    # Get unique teams
    teams = sorted(set(matches_df['home_team'].unique()) | set(matches_df['away_team'].unique()))
    n_teams = len(teams)
    
    # Create team indices
    team_indices, team_goals_for, team_goals_against, team_games = _construct_team_goal_matrix(matches_df, teams)
    
    # Prepare data for likelihood calculation
    home_teams = matches_df['home_team'].values
    away_teams = matches_df['away_team'].values
    home_goals = matches_df['home_goals'].values
    away_goals = matches_df['away_goals'].values
    matches_data = (home_teams, away_teams, home_goals, away_goals)
    
    # Initial parameter estimates
    # Use goals ratio as initial estimate for offensive strength
    initial_offensive = np.ones(n_teams)
    for team, idx in team_indices.items():
        if team_games[idx] > 0:
            gf = team_goals_for[idx]
            ga = team_goals_against[idx]
            
            # Avoid division by zero
            if ga > 0:
                initial_offensive[idx] = np.sqrt(gf / ga)
            elif gf > 0:
                initial_offensive[idx] = 2.0  # Arbitrary positive value if no goals against
    
    # Ensure initial values are positive
    initial_offensive = np.clip(initial_offensive, 0.2, 5.0)
    
    # Initial parameters with home advantage
    initial_params = np.append(initial_offensive, max(1.3, min_home_advantage))  # Start with standard home advantage
    
    # Bounds for parameters (all must be positive)
    bounds = [(0.1, 10.0)] * n_teams + [(min_home_advantage, 2.5)]  # Enforce minimum home advantage
    
    # Constraint: product of offensive strengths = 1
    def constraint(params):
        return np.prod(params[:n_teams]) - 1.0
    
    constraints = {'type': 'eq', 'fun': constraint}
    
    # Minimize negative log-likelihood
    try:
        result = minimize(
            _davidson_likelihood,
            initial_params,
            args=(team_indices, matches_data, min_home_advantage),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'disp': False}
        )
        
        # Extract parameters
        offensive_strengths = result.x[:n_teams]
        home_advantage = max(result.x[n_teams], min_home_advantage)  # Apply minimum home advantage
        
        # Defensive strengths are reciprocal of offensive
        defensive_strengths = 1.0 / offensive_strengths
        
        return team_indices, offensive_strengths, defensive_strengths, home_advantage
    
    except Exception as e:
        print(f"Error fitting Davidson model: {e}")
        # Return default values
        offensive_strengths = np.ones(n_teams)
        defensive_strengths = np.ones(n_teams)
        home_advantage = min_home_advantage  # Set default to minimum
        return team_indices, offensive_strengths, defensive_strengths, home_advantage

def add_davidson_ratings(file_path, num_recent_games=6, min_home_advantage=1.1, output_file=None):
    """
    Add Davidson model ratings to the football dataset.
    All ratings are scaled to the 0.5-3.0 range.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    num_recent_games : int, optional
        Number of recent games to consider for calculations (default: 6)
    min_home_advantage : float, optional
        Minimum home advantage value (default: 1.1)
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_davidson' to original filename)
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with Davidson model rating columns
    """
    # Set default output file name if not provided
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_davidson_0.5_3.xlsx')
    
    # Load and process the data
    df = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize rating columns
    davidson_columns = [
        'home_team_attack_davidson',
        'home_team_defense_davidson',
        'away_team_attack_davidson',
        'away_team_defense_davidson',
        'home_advantage_davidson',
    ]
    
    # Initialize all columns with default values in the 0.5-3 range
    for col in davidson_columns:
        if col == 'home_advantage_davidson':
            df[col] = min_home_advantage  # Default home advantage
        else:
            df[col] = 1.75  # Default strength is middle of 0.5-3 range
    
    # Process each season separately
    seasons = sorted(df['season'].unique())
    for season in seasons:
        print(f"\nProcessing season: {season}")
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Minimum matches needed before we start calculating
        min_matches_needed = num_recent_games - 1
        
        # Track teams and their recent matches
        team_match_history = {}
        match_cache = {}  # Cache for model fitting results
        
        # Process each match chronologically within the season
        for match_idx_position, match_idx in enumerate(season_indices):
            if match_idx_position % 100 == 0:
                print(f"  Processing match {match_idx_position+1}/{len(season_indices)}")
            
            home_team = df.at[match_idx, 'home_team']
            away_team = df.at[match_idx, 'away_team']
            
            # Get all previous matches in this season up to this match
            prev_season_matches = season_df[season_df.index < match_idx]
            
            # Check if we have enough matches to fit the model
            if len(prev_season_matches) >= min_matches_needed:
                # Update team match history
                for team in [home_team, away_team]:
                    team_matches = prev_season_matches[(prev_season_matches['home_team'] == team) | 
                                                     (prev_season_matches['away_team'] == team)]
                    
                    if len(team_matches) > 0:
                        team_match_history[team] = team_matches.sort_index(ascending=False).head(num_recent_games)
                
                # Create a set of unique match indices to include in the model fitting
                match_indices_to_include = set()
                for team in [home_team, away_team]:
                    if team in team_match_history:
                        match_indices_to_include.update(team_match_history[team].index)
                
                # Compute a hash for the match set to check if we've already fitted this model
                match_set_hash = frozenset(match_indices_to_include)
                
                # Use cached model parameters if available
                if match_set_hash in match_cache:
                    model_params = match_cache[match_set_hash]
                    team_indices, offensive_dict, defensive_dict, home_advantage = model_params
                else:
                    # Filter previous matches to only include the most recent ones for each team
                    recent_matches = prev_season_matches[prev_season_matches.index.isin(match_indices_to_include)]
                    
                    # Skip if recent matches don't contain both teams in the current match
                    if not (home_team in set(recent_matches['home_team'].unique()) | set(recent_matches['away_team'].unique()) and
                           away_team in set(recent_matches['home_team'].unique()) | set(recent_matches['away_team'].unique())):
                        continue
                    
                    # Fit Davidson model
                    team_indices, offensive_strengths, defensive_strengths, home_advantage = _fit_davidson_model(
                        recent_matches, min_home_advantage=min_home_advantage
                    )
                    
                    # Convert team strengths to dictionaries for easier lookup
                    offensive_dict = {team: offensive_strengths[idx] for team, idx in team_indices.items()}
                    defensive_dict = {team: defensive_strengths[idx] for team, idx in team_indices.items()}
                    
                    # Cache the model parameters
                    match_cache[match_set_hash] = (team_indices, offensive_dict, defensive_dict, home_advantage)
                
                # Get the raw ratings for this match
                home_attack_raw = offensive_dict.get(home_team, 1.0)
                home_defense_raw = defensive_dict.get(home_team, 1.0)
                away_attack_raw = offensive_dict.get(away_team, 1.0)
                away_defense_raw = defensive_dict.get(away_team, 1.0)
                
                # Scale ratings to 0.5-3 range
                # Offensive strengths typically range from 0.5 to 2.0
                # Defensive strengths (reciprocal of offensive) also typically range from 0.5 to 2.0
                home_attack_scaled = _scale_to_range(home_attack_raw, 0.5, 2.0, 0.5, 3.0)
                home_defense_scaled = _scale_to_range(home_defense_raw, 0.5, 2.0, 0.5, 3.0)
                away_attack_scaled = _scale_to_range(away_attack_raw, 0.5, 2.0, 0.5, 3.0)
                away_defense_scaled = _scale_to_range(away_defense_raw, 0.5, 2.0, 0.5, 3.0)
                
                # Scale home advantage to 0.5-3 range (typical range is 1.0-1.5)
                home_advantage_scaled = _scale_to_range(home_advantage, 1.0, 1.5, 0.5, 3.0)
                
                # Update ratings for current match - all rounded to 2 decimal places
                df.at[match_idx, 'home_team_attack_davidson'] = round(home_attack_scaled, 2)
                df.at[match_idx, 'home_team_defense_davidson'] = round(home_defense_scaled, 2)
                df.at[match_idx, 'away_team_attack_davidson'] = round(away_attack_scaled, 2)
                df.at[match_idx, 'away_team_defense_davidson'] = round(away_defense_scaled, 2)
                df.at[match_idx, 'home_advantage_davidson'] = round(home_advantage_scaled, 2)
                
                # Print some debug info occasionally
                if match_idx_position % 200 == 0:
                    print(f"  Match {match_idx_position+1}: {home_team} vs {away_team}")
                    print(f"  Raw ratings - Home attack: {home_attack_raw:.2f}, Home defense: {home_defense_raw:.2f}")
                    print(f"  Raw ratings - Away attack: {away_attack_raw:.2f}, Away defense: {away_defense_raw:.2f}")
                    print(f"  Scaled ratings - Home attack: {home_attack_scaled:.2f}, Home defense: {home_defense_scaled:.2f}")
                    print(f"  Scaled ratings - Away attack: {away_attack_scaled:.2f}, Away defense: {away_defense_scaled:.2f}")
                    print(f"  Home advantage: Raw {home_advantage:.2f}, Scaled: {home_advantage_scaled:.2f}")
    
    # Fill any remaining NaN values with default values
    for col in davidson_columns:
        if col == 'home_advantage_davidson':
            df[col] = df[col].fillna(min_home_advantage)
        else:
            df[col] = df[col].fillna(1.75)
        
        # Ensure all values are rounded to 2 decimal places
        if col in df.columns:
            df[col] = df[col].round(2)
    
    # Final safety check to ensure all values are in the 0.5-3 range
    for col in davidson_columns:
        df[col] = df[col].apply(lambda x: min(max(x, 0.5), 3.0))
    
    # Save the enhanced dataframe
    df.to_excel(output_file, index=False)
    print(f"\nData with Davidson model ratings saved to {output_file}")
    print(f"All values are scaled to be between 0.5-3")
    
    return df

# Example usage
df_with_davidson = add_davidson_ratings(
    file_path='nfd_data.xlsx', 
    num_recent_games=6, 
    min_home_advantage=1.1,  # Set minimum home advantage to 1.1
    output_file='davidson_0.5_3.xlsx'
)