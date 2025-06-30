import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
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
        
        # Check for expected goals columns from odds
        if 'home_xg_from_odds' in df.columns:
            df.rename(columns={'home_xg_from_odds': 'home_xg_odds'}, inplace=True)
            print("Renamed 'home_xg_from_odds' to 'home_xg_odds'")
        
        if 'away_xg_from_odds' in df.columns:
            df.rename(columns={'away_xg_from_odds': 'away_xg_odds'}, inplace=True)
            print("Renamed 'away_xg_from_odds' to 'away_xg_odds'")
        
        print("\nProcessed data summary:")
        print(f"Rows: {len(df)}")
        if 'home_team' in df.columns:
            unique_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
            print(f"Unique teams: {len(unique_teams)}")
        if 'season' in df.columns:
            print(f"Seasons: {df['season'].unique()}")
        
        required_cols_regression = [
            'home_team_goals_scored_average', 
            'away_team_goals_conceded_average',
            'away_team_goals_scored_average',
            'home_team_goals_conceded_average',
            'home_xg_odds',
            'away_xg_odds'
        ]
        
        # Check if all required columns for regression are present
        missing_regression_cols = [col for col in required_cols_regression if col not in df.columns]
        if missing_regression_cols:
            print(f"Warning: Missing columns required for regression: {missing_regression_cols}")
            print("Available columns:")
            for col in df.columns:
                print(f"  - {col}")
        
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

def _calculate_time_weights(match_indices, half_life=6):
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

def _prepare_home_xg_regression_data(matches_df):
    """
    Prepare data for time-weighted regression for home expected goals.
    
    Parameters:
    -----------
    matches_df : pandas.DataFrame
        DataFrame containing match data with required columns
    
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target vector
    """
    # Select only matches with valid home_xg_odds (not 0)
    valid_matches = matches_df[matches_df['home_xg_odds'] > 0].copy()
    
    if len(valid_matches) == 0:
        return None, None
    
    # Prepare feature matrix X with home_team_goals_scored_average and away_team_goals_conceded_average
    X = valid_matches[['home_team_goals_scored_average', 'away_team_goals_conceded_average']].values
    
    # Target variable: home_xg_odds
    y = valid_matches['home_xg_odds'].values
    
    return X, y

def _prepare_away_xg_regression_data(matches_df):
    """
    Prepare data for time-weighted regression for away expected goals.
    
    Parameters:
    -----------
    matches_df : pandas.DataFrame
        DataFrame containing match data with required columns
    
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target vector
    """
    # Select only matches with valid away_xg_odds (not 0)
    valid_matches = matches_df[matches_df['away_xg_odds'] > 0].copy()
    
    if len(valid_matches) == 0:
        return None, None
    
    # Prepare feature matrix X with away_team_goals_scored_average and home_team_goals_conceded_average
    X = valid_matches[['away_team_goals_scored_average', 'home_team_goals_conceded_average']].values
    
    # Target variable: away_xg_odds
    y = valid_matches['away_xg_odds'].values
    
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
    sklearn.linear_model.LinearRegression
        Fitted regression model
    """
    # Fit weighted linear regression
    model = LinearRegression(fit_intercept=True)  # Include intercept
    model.fit(X, y, sample_weight=weights)
    
    return model

def predict_xg_with_model(model, home_goals_scored=None, away_goals_conceded=None, 
                          away_goals_scored=None, home_goals_conceded=None):
    """
    Predict expected goals using a fitted model.
    
    Parameters:
    -----------
    model : sklearn.linear_model.LinearRegression
        Fitted regression model
    home_goals_scored : float, optional
        Home team's average goals scored
    away_goals_conceded : float, optional
        Away team's average goals conceded
    away_goals_scored : float, optional
        Away team's average goals scored
    home_goals_conceded : float, optional
        Home team's average goals conceded
    
    Returns:
    --------
    float
        Predicted expected goals
    """
    if model is None:
        return 0.0
    
    # For home model
    if home_goals_scored is not None and away_goals_conceded is not None:
        X_pred = np.array([[home_goals_scored, away_goals_conceded]])
        return max(0, model.predict(X_pred)[0])
    
    # For away model
    elif away_goals_scored is not None and home_goals_conceded is not None:
        X_pred = np.array([[away_goals_scored, home_goals_conceded]])
        return max(0, model.predict(X_pred)[0])
    
    return 0.0

def add_time_weighted_xg_regression(file_path, num_recent_games=6, time_decay_half_life=6, output_file=None):
    """
    Add time-weighted regression model for expected goals to the football dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    num_recent_games : int, optional
        Number of recent games to consider for calculations (default: 6)
    time_decay_half_life : int, optional
        Number of matches after which the weight is halved (default: 10)
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_xg_twr' to original filename)
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with time-weighted regression expected goals columns
    """
    # Set default output file name if not provided
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_xg_twr.xlsx')
    
    # Load and process the data
    df = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize expected goals columns
    twr_columns = [
        'home_xg_twr',
        'away_xg_twr',
    ]
    
    # Initialize all columns with default values
    for col in twr_columns:
        df[col] = 0.0
    
    # Process each season separately
    seasons = sorted(df['season'].unique())
    for i, season in enumerate(seasons):
        print(f"\nProcessing season: {season}")
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Adjust minimum matches needed based on season position
        if i == 0:
            # First season - start calculating earlier
            min_matches_needed = max(5, num_recent_games-1)
        else:
            # Subsequent seasons - standard threshold
            min_matches_needed = max(5, num_recent_games-1)
        
        # Store models for this season
        home_xg_model = None
        away_xg_model = None
        
        # Process each match chronologically within the season
        for match_idx_position, match_idx in enumerate(season_indices):
            # Current match data
            home_team = df.at[match_idx, 'home_team']
            away_team = df.at[match_idx, 'away_team']
            home_goals_scored_avg = df.at[match_idx, 'home_team_goals_scored_average']
            away_goals_conceded_avg = df.at[match_idx, 'away_team_goals_conceded_average']
            away_goals_scored_avg = df.at[match_idx, 'away_team_goals_scored_average']
            home_goals_conceded_avg = df.at[match_idx, 'home_team_goals_conceded_average']
            
            # Get all previous matches in this season up to this match
            prev_season_matches = season_df[season_df.index < match_idx]
            
            # Check if we have enough matches to fit the model
            if len(prev_season_matches) >= min_matches_needed:
                # Only refit the model periodically to save computation time
                # if (match_idx_position % 10 == 0) or (home_xg_model is None or away_xg_model is None):
                if (match_idx_position % time_decay_half_life == 0) or (home_xg_model is None or away_xg_model is None):
                    # Calculate time weights for previous matches
                    weights = _calculate_time_weights(prev_season_matches.index, time_decay_half_life)
                    
                    # Prepare data for home expected goals regression
                    X_home, y_home = _prepare_home_xg_regression_data(prev_season_matches)
                    
                    # Fit home expected goals model if data is available
                    if X_home is not None and y_home is not None and len(X_home) >= min_matches_needed:
                        try:
                            home_xg_model = _fit_time_weighted_regression(X_home, y_home, weights[:len(X_home)])
                            # r2_home = home_xg_model.score(X_home, y_home)
                            # print(f"  Home xG model R² = {r2_home:.4f} (fitted on {len(X_home)} matches)")
                        except Exception as e:
                            print(f"  Error fitting home xG model: {e}")
                    
                    # Prepare data for away expected goals regression
                    X_away, y_away = _prepare_away_xg_regression_data(prev_season_matches)
                    
                    # Fit away expected goals model if data is available
                    if X_away is not None and y_away is not None and len(X_away) >= min_matches_needed:
                        try:
                            away_xg_model = _fit_time_weighted_regression(X_away, y_away, weights[:len(X_away)])
                            # r2_away = away_xg_model.score(X_away, y_away)
                            # print(f"  Away xG model R² = {r2_away:.4f} (fitted on {len(X_away)} matches)")
                        except Exception as e:
                            print(f"  Error fitting away xG model: {e}")
            
            # Predict expected goals using the models
            home_xg_pred = predict_xg_with_model(
                home_xg_model, 
                home_goals_scored=home_goals_scored_avg, 
                away_goals_conceded=away_goals_conceded_avg
            )
            
            away_xg_pred = predict_xg_with_model(
                away_xg_model, 
                away_goals_scored=away_goals_scored_avg, 
                home_goals_conceded=home_goals_conceded_avg
            )
            
            # Store predictions in the dataframe
            df.at[match_idx, 'home_xg_twr'] = np.round(home_xg_pred, 2)
            df.at[match_idx, 'away_xg_twr'] = np.round(away_xg_pred, 2)
            
            # Print progress periodically
            # if match_idx_position % 50 == 0 or match_idx_position == len(season_indices) - 1:
            #     print(f"  Processing match {match_idx_position+1}/{len(season_indices)}: {home_team} vs {away_team}")
            #     print(f"  Home xG: {home_xg_pred:.2f}, Away xG: {away_xg_pred:.2f}")
    
    # Save the enhanced dataframe
    df.to_excel(output_file, index=False)
    print(f"\nData with time-weighted regression expected goals saved to {output_file}")
    
    return df

# Example usage
df_with_xg_twr = add_time_weighted_xg_regression(
    file_path='nfd_data.xlsx', 
    num_recent_games=6, 
    time_decay_half_life=6, 
    output_file='nfd_data_xg_twr.xlsx'
)