import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, LogisticRegression
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

def _calculate_time_weights(match_indices, half_life=10):
    """
    Calculate time weights for matches based on their recency.
    
    Parameters:
    -----------
    match_indices : array-like
        Indices of matches in chronological order
    half_life : int, optional
        Number of matches after which the weight is halved (default: 10)
    
    Returns:
    --------
    numpy.array
        Array of weights, with more recent matches having higher weights
    """
    # Convert to numpy array for efficiency
    indices = np.array(match_indices)
    
    # Calculate the time difference (in match count) from the most recent match
    most_recent_idx = max(indices)
    match_age = most_recent_idx - indices
    
    # Calculate weights using exponential decay
    weights = np.exp(-np.log(2) * match_age / half_life)
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    return weights

def _prepare_regression_data(matches_df, team_encoder, include_home_advantage=True):
    """
    Prepare data for time-weighted regression.
    
    Parameters:
    -----------
    matches_df : pandas.DataFrame
        DataFrame containing match data
    team_encoder : dict
        Mapping of team names to indices
    include_home_advantage : bool, optional
        Whether to include a home advantage term (default: True)
    
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target vector
    """
    n_teams = len(team_encoder)
    n_matches = len(matches_df)
    
    # Initialize feature matrix
    # Each row represents a match, columns are team strengths
    if include_home_advantage:
        X = np.zeros((n_matches, 2 * n_teams + 1))  # +1 for home advantage
    else:
        X = np.zeros((n_matches, 2 * n_teams))
    
    # For each match, set the appropriate team strength indicators
    for i, (_, match) in enumerate(matches_df.iterrows()):
        # Home team attack and away team defense
        home_idx = team_encoder.get(match['home_team'], -1)
        away_idx = team_encoder.get(match['away_team'], -1)
        
        if home_idx >= 0 and away_idx >= 0:
            # Home team attack
            X[i, home_idx] = 1
            
            # Away team defense
            X[i, n_teams + away_idx] = 1
            
            # Home advantage (constant term)
            if include_home_advantage:
                X[i, -1] = 1
    
    # Target variable: goal difference
    y = matches_df['goal_diff'].values
    
    return X, y

def _prepare_logistic_data(matches_df, team_encoder, include_home_advantage=True):
    """
    Prepare data for time-weighted logistic regression.
    
    Parameters:
    -----------
    matches_df : pandas.DataFrame
        DataFrame containing match data
    team_encoder : dict
        Mapping of team names to indices
    include_home_advantage : bool, optional
        Whether to include a home advantage term (default: True)
    
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target vector
    """
    n_teams = len(team_encoder)
    n_matches = len(matches_df)
    
    # Initialize feature matrix
    # Each row represents a match, columns are team strengths
    if include_home_advantage:
        X = np.zeros((n_matches, 2 * n_teams + 1))  # +1 for home advantage
    else:
        X = np.zeros((n_matches, 2 * n_teams))
    
    # For each match, set the appropriate team strength indicators
    for i, (_, match) in enumerate(matches_df.iterrows()):
        # Home team and away team indicators
        home_idx = team_encoder.get(match['home_team'], -1)
        away_idx = team_encoder.get(match['away_team'], -1)
        
        if home_idx >= 0 and away_idx >= 0:
            # Home team strength
            X[i, home_idx] = 1
            
            # Away team strength (negative as they're playing against home team)
            X[i, n_teams + away_idx] = -1
            
            # Home advantage (constant term)
            if include_home_advantage:
                X[i, -1] = 1
    
    # Use full_time_result directly
    y = matches_df['full_time_result'].map({'H': 1, 'D': 0, 'A': -1}).values
    
    return X, y

def _fit_time_weighted_regression(X, y, weights):
    """
    Fit a time-weighted linear regression model.
    
    Parameters:
    -----------
    X : numpy.array
        Feature matrix
    y : numpy.array
        Target vector
    weights : numpy.array
        Sample weights (higher weights for more recent matches)
    
    Returns:
    --------
    numpy.array
        Regression coefficients
    """
    # Fit weighted linear regression
    model = LinearRegression(fit_intercept=False)  # No intercept as we include home advantage explicitly
    model.fit(X, y, sample_weight=weights)
    
    return model.coef_

def _scale_to_odds(team_strengths, min_odd=0.5, max_odd=5.0, center_val=1.0):
    """
    Scale team strengths to odds format.
    
    Parameters:
    -----------
    team_strengths : dict
        Dictionary of team strength values
    min_odd, max_odd : float
        Minimum and maximum allowed odds values
    center_val : float
        Value to center the odds around
        
    Returns:
    --------
    dict
        Dictionary mapping team names to odds-scaled strengths
    """
    # Convert to numpy array
    teams = list(team_strengths.keys())
    strengths = np.array(list(team_strengths.values()))
    
    # Avoid division by zero
    if len(strengths) == 0 or np.std(strengths) == 0:
        return {team: center_val for team in team_strengths}
    
    # Center and normalize
    centered = strengths - np.mean(strengths)
    normalized = centered / (2 * np.std(centered))
    
    # Scale to odds range
    odds_range = max_odd - min_odd
    scaled = center_val + normalized * (odds_range / 2)
    
    # Clip to ensure within bounds
    scaled = np.clip(scaled, min_odd, max_odd)
    
    # Convert back to dictionary
    return {team: scaled[i] for i, team in enumerate(teams)}

def add_time_weighted_ratings(file_path, num_recent_games=6, time_decay_half_life=10, output_file=None):
    """
    Add time-weighted regression model ratings to the football dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    num_recent_games : int, optional
        Number of recent games to consider for calculations (default: 6)
    time_decay_half_life : int, optional
        Number of matches after which the weight is halved (default: 10)
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_twr' to original filename)
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with time-weighted regression rating columns
    """
    # Set default output file name if not provided
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_twr_simplified.xlsx')
    
    # Load and process the data
    df = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize rating columns - simplified to just include team strengths and home advantage
    twr_columns = [
        'home_team_strength_twr',
        'away_team_strength_twr',
        'home_team_strength_twl',  # Time-weighted logistic
        'away_team_strength_twl',
        'home_advantage_twr'
    ]
    
    # Initialize all columns with default values
    for col in twr_columns:
        df[col] = 1.0  # Default strength is neutral (1.0)
    
    # Process each season separately
    seasons = sorted(df['season'].unique())
    for i, season in enumerate(seasons):
        print(f"\nProcessing season: {season}")
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Adjust minimum matches needed based on season position
        if i == 0:
            # First season - start calculating earlier
            min_matches_needed = num_recent_games-1
        else:
            # Subsequent seasons - standard threshold
            min_matches_needed = num_recent_games-1
        
        # Track team ratings for this season
        team_strengths_twr = {}  # For linear regression
        team_strengths_twl = {}  # For logistic regression
        home_advantage = 0.0
        
        # Process each match chronologically within the season
        for match_idx_position, match_idx in enumerate(season_indices):
            home_team = df.at[match_idx, 'home_team']
            away_team = df.at[match_idx, 'away_team']
            
            # Get all previous matches in this season up to this match
            prev_season_matches = season_df[season_df.index < match_idx]
            
            # Check if we have enough matches to fit the model
            if len(prev_season_matches) >= min_matches_needed:
                # Create team-specific match history maps
                team_match_history = {}
                
                # For each team, find their most recent matches
                for team in set(prev_season_matches['home_team'].unique()) | set(prev_season_matches['away_team'].unique()):
                    team_matches = prev_season_matches[(prev_season_matches['home_team'] == team) | 
                                                     (prev_season_matches['away_team'] == team)]
                    
                    # Sort by index (chronological order) and take the most recent matches
                    if len(team_matches) > 0:
                        team_match_history[team] = team_matches.sort_index(ascending=False).head(num_recent_games)
                
                # Create a set of unique match indices to include in the regression
                match_indices_to_include = set()
                for team, matches in team_match_history.items():
                    match_indices_to_include.update(matches.index)
                
                # Filter previous matches to only include the most recent ones for each team
                recent_matches = prev_season_matches[prev_season_matches.index.isin(match_indices_to_include)]
                
                # Get all teams that have played in recent matches
                teams = set(recent_matches['home_team'].unique()) | set(recent_matches['away_team'].unique())
                
                # Skip if recent matches don't contain both teams in the current match
                if home_team not in teams or away_team not in teams:
                    continue
                
                # Create team encoder (map team names to indices)
                team_encoder = {team: i for i, team in enumerate(teams)}
                
                # Prepare data for time-weighted regression
                X, y = _prepare_regression_data(recent_matches, team_encoder)
                
                # Calculate time weights
                weights = _calculate_time_weights(recent_matches.index, time_decay_half_life)
                
                # Fit time-weighted linear regression
                try:
                    coef = _fit_time_weighted_regression(X, y, weights)
                    
                    # Extract team strengths
                    n_teams = len(team_encoder)
                    attack_strengths = {team: coef[idx] for team, idx in team_encoder.items()}
                    defense_strengths = {team: -coef[idx + n_teams] for team, idx in team_encoder.items()}
                    
                    # Combine attack and defense into overall strength
                    team_strengths_twr = {team: (attack_strengths.get(team, 0) + defense_strengths.get(team, 0)) / 2 
                                         for team in teams}
                    
                    # Extract home advantage
                    raw_home_advantage = coef[-1] if len(coef) > 2 * n_teams else 0.0
                    
                    # Scale home advantage to 0-5 range
                    # In football, home advantage is typically 0.3-0.5 goals
                    # Map this to around 2.0-3.0 in our scaled range (above average)
                    if raw_home_advantage > 0:
                        # Positive home advantage (normal case)
                        # Map 0-2 goal advantage to 1.0-5.0 range
                        home_advantage = 1.0 + min(4.0, raw_home_advantage * 2)
                    else:
                        # Negative or zero home advantage (rare case)
                        # Map -2-0 goal disadvantage to 0.0-1.0 range
                        home_advantage = 1.0 + max(-1.0, raw_home_advantage * 2)
                        
                    # Ensure home advantage is within bounds
                    home_advantage = min(5.0, max(0.0, home_advantage))
                    
                except Exception as e:
                    print(f"  Error fitting linear regression: {e}")
                
                # Prepare data for logistic regression for team strength
                try:
                    # Prepare data for logistic regression
                    X_log, y_log = _prepare_logistic_data(recent_matches, team_encoder)
                    
                    # Fit model
                    log_model = LogisticRegression(fit_intercept=False, solver='lbfgs', max_iter=1000)
                    log_model.fit(X_log, y_log, sample_weight=weights)
                    
                    # Extract team strengths
                    log_coef = log_model.coef_[0]
                    team_strengths_log = {team: log_coef[idx] for team, idx in team_encoder.items()}
                    
                    # Convert to odds-style scaling
                    team_strengths_twl = _scale_to_odds(team_strengths_log)
                    
                except Exception as e:
                    print(f"  Error fitting logistic regression: {e}")
                
                # Scale linear regression strengths to odds format
                team_strengths_twr = _scale_to_odds(team_strengths_twr)
                
                # Print some debug info
                if match_idx_position % 50 == 0:
                    print(f"  Fitting model at match {match_idx_position+1}/{len(season_indices)}")
                    print(f"  Using {len(recent_matches)} recent matches (max {num_recent_games} per team)")
                    print(f"  Raw home advantage: {raw_home_advantage:.4f} goals, Scaled: {home_advantage:.2f}")
                    
                    if home_team in team_strengths_twr and away_team in team_strengths_twr:
                        print(f"  {home_team} (H) strength: {team_strengths_twr[home_team]:.2f} (TW-Linear), " + 
                              f"{team_strengths_twl.get(home_team, 1.0):.2f} (TW-Logistic)")
                        print(f"  {away_team} (A) strength: {team_strengths_twr[away_team]:.2f} (TW-Linear), " + 
                              f"{team_strengths_twl.get(away_team, 1.0):.2f} (TW-Logistic)")
            
            # Update ratings for current match
            if home_team in team_strengths_twr and away_team in team_strengths_twr:
                df.at[match_idx, 'home_team_strength_twr'] = round(team_strengths_twr[home_team], 2)
                df.at[match_idx, 'away_team_strength_twr'] = round(team_strengths_twr[away_team], 2)
                df.at[match_idx, 'home_advantage_twr'] = round(home_advantage, 2)
            
            if home_team in team_strengths_twl and away_team in team_strengths_twl:
                df.at[match_idx, 'home_team_strength_twl'] = round(team_strengths_twl[home_team], 2)
                df.at[match_idx, 'away_team_strength_twl'] = round(team_strengths_twl[away_team], 2)
    
    # Save the enhanced dataframe
    df.to_excel(output_file, index=False)
    print(f"\nData with time-weighted regression ratings saved to {output_file}")
    
    return df

# Example usage
df_with_twr = add_time_weighted_ratings(file_path='nfd_data.xlsx', num_recent_games=6, time_decay_half_life=10, output_file='regression.xlsx')