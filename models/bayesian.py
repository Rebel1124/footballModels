import pandas as pd
import numpy as np
from scipy.stats import poisson
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
        
        # Calculate goal difference
        df['goal_diff'] = df['home_goals'] - df['away_goals']
        
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

def _get_prior_parameters(df, prior_strength=1.0):
    """
    Calculate prior parameters for the Bayesian model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing match data
    prior_strength : float, optional
        Relative strength of the prior (higher values mean stronger prior)
    
    Returns:
    --------
    tuple
        (prior_attack, prior_defense, prior_home_advantage, prior_precision)
    """
    # Calculate league averages
    home_goals = df['home_goals'].mean()
    away_goals = df['away_goals'].mean()
    
    # Default values if data is insufficient
    if pd.isna(home_goals) or pd.isna(away_goals) or home_goals <= 0 or away_goals <= 0:
        home_goals = 1.5
        away_goals = 1.1
    
    # Prior for team attack and defense - set to 1.5 (middle of 0-3 scale)
    prior_attack = 1.5
    prior_defense = 1.5
    
    # Prior for home advantage - set to 1.1
    prior_home_advantage = 1.1
    
    # Prior precision (inverse variance)
    # Lower values = less certain prior (more weight on the data)
    # Higher values = more certain prior (less weight on the data)
    prior_precision = prior_strength / (np.std([home_goals, away_goals]) + 0.1)
    
    return prior_attack, prior_defense, prior_home_advantage, prior_precision

def _update_bayesian_parameters(
    team, is_home, goals_for, goals_against,
    attack_params, defense_params, 
    opponent, home_advantage, prior_attack, prior_defense, prior_precision
):
    """
    Update Bayesian parameters for a team based on match results.
    
    Parameters:
    -----------
    team : str
        Team name
    is_home : bool
        Whether the team was playing at home
    goals_for : int
        Goals scored by the team
    goals_against : int
        Goals conceded by the team
    attack_params : dict
        Current attack parameters (alpha, beta) for all teams
    defense_params : dict
        Current defense parameters (alpha, beta) for all teams
    opponent : str
        Opponent team name
    home_advantage : float
        Home advantage factor
    prior_attack, prior_defense : float
        Prior means for attack and defense
    prior_precision : float
        Precision (inverse variance) of the prior
    
    Returns:
    --------
    tuple
        (updated_attack_params, updated_defense_params)
    """
    # Initialize parameters if not present
    if team not in attack_params:
        attack_params[team] = (prior_attack, prior_precision)
    if team not in defense_params:
        defense_params[team] = (prior_defense, prior_precision)
    if opponent not in attack_params:
        attack_params[opponent] = (prior_attack, prior_precision)
    if opponent not in defense_params:
        defense_params[opponent] = (prior_defense, prior_precision)
    
    # Get current parameters
    attack_alpha, attack_beta = attack_params[team]
    defense_alpha, defense_beta = defense_params[team]
    opp_attack_alpha, opp_attack_beta = attack_params[opponent]
    opp_defense_alpha, opp_defense_beta = defense_params[opponent]
    
    # Expected goals - Note the approach with defense parameter scaling
    if is_home:
        expected_goals_for = attack_alpha * ((3.0 - opp_defense_alpha) / 3.0) * home_advantage
        expected_goals_against = opp_attack_alpha * ((3.0 - defense_alpha) / 3.0)
    else:
        expected_goals_for = attack_alpha * ((3.0 - opp_defense_alpha) / 3.0)
        expected_goals_against = opp_attack_alpha * ((3.0 - defense_alpha) / 3.0) * home_advantage
    
    # Calculate likelihoods (simplification of Poisson likelihood)
    # Higher values indicate the current parameters explain the observed goals well
    attack_likelihood = poisson.pmf(goals_for, expected_goals_for) if expected_goals_for > 0 else 0.01
    defense_likelihood = poisson.pmf(goals_against, expected_goals_against) if expected_goals_against > 0 else 0.01
    
    # Update attack parameters using Bayesian update rule
    # Higher likelihood -> parameters move more toward observed data
    # Higher beta (prior precision) -> parameters stay closer to prior values
    attack_alpha_new = (attack_alpha * attack_beta + (goals_for / 2.0)) / (attack_beta + 0.5)
    attack_beta_new = attack_beta + attack_likelihood
    
    # Update defense parameters - for defense, higher values mean fewer goals conceded
    # We reward good defense with higher defensive ratings
    defense_adjustment = 3.0 / (goals_against + 1.0)  # Transforms goals against to a 0-3 scale value
    defense_alpha_new = (defense_alpha * defense_beta + defense_adjustment) / (defense_beta + 1)
    defense_beta_new = defense_beta + defense_likelihood
    
    # Update parameters with smoothing to prevent extreme values
    smooth_factor = 0.7  # Adjust for more/less responsiveness
    attack_alpha = (1 - smooth_factor) * attack_alpha + smooth_factor * attack_alpha_new
    attack_beta = (1 - smooth_factor) * attack_beta + smooth_factor * attack_beta_new
    defense_alpha = (1 - smooth_factor) * defense_alpha + smooth_factor * defense_alpha_new
    defense_beta = (1 - smooth_factor) * defense_beta + smooth_factor * defense_beta_new
    
    # Constrain parameters to 0-3 range for alpha (team strength)
    attack_alpha = min(max(attack_alpha, 0.0), 3.0)
    defense_alpha = min(max(defense_alpha, 0.0), 3.0)
    attack_beta = min(max(attack_beta, prior_precision * 0.5), prior_precision * 10)
    defense_beta = min(max(defense_beta, prior_precision * 0.5), prior_precision * 10)
    
    # Update parameter dictionaries
    attack_params[team] = (attack_alpha, attack_beta)
    defense_params[team] = (defense_alpha, defense_beta)
    
    return attack_params, defense_params

def _calculate_expected_goals(home_team, away_team, attack_params, defense_params, home_advantage):
    """
    Calculate expected goals for a match using Bayesian parameters.
    
    Parameters:
    -----------
    home_team, away_team : str
        Team names
    attack_params, defense_params : dict
        Dictionaries containing attack and defense parameters for all teams
    home_advantage : float
        Home advantage factor
    
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
    home_defense_factor = (3.0 - away_defense_alpha) / 3.0  # Transform to a 0-1 scale
    away_defense_factor = (3.0 - home_defense_alpha) / 3.0  # Transform to a 0-1 scale
    
    # Calculate expected goals using more direct scaling from the 0-3 values
    # This approach ensures that higher attack values correspond to more goals
    # and higher defense values correspond to fewer goals conceded
    expected_home_goals = home_attack_alpha * away_defense_factor * home_advantage
    expected_away_goals = away_attack_alpha * home_defense_factor
    
    # Ensure expected goals are within the 0-3 range
    expected_home_goals = min(max(expected_home_goals, 0.0), 3.0)
    expected_away_goals = min(max(expected_away_goals, 0.0), 3.0)
    
    return expected_home_goals, expected_away_goals

def _calculate_match_probabilities(expected_home_goals, expected_away_goals, max_goals=10):
    """
    Calculate match outcome probabilities based on expected goals.
    
    Parameters:
    -----------
    expected_home_goals, expected_away_goals : float
        Expected goals for home and away teams
    max_goals : int, optional
        Maximum number of goals to consider for each team
    
    Returns:
    --------
    tuple
        (home_win_prob, draw_prob, away_win_prob)
    """
    # Initialize counters
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0
    
    # Ensure positive expected goals
    expected_home_goals = max(expected_home_goals, 0.01)
    expected_away_goals = max(expected_away_goals, 0.01)
    
    # Calculate probabilities for each possible score
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            # Probability of this exact score
            home_pmf = poisson.pmf(home_goals, expected_home_goals)
            away_pmf = poisson.pmf(away_goals, expected_away_goals)
            score_prob = home_pmf * away_pmf
            
            # Add to the appropriate outcome
            if home_goals > away_goals:
                home_win_prob += score_prob
            elif home_goals == away_goals:
                draw_prob += score_prob
            else:
                away_win_prob += score_prob
    
    # Normalize probabilities
    total_prob = home_win_prob + draw_prob + away_win_prob
    if total_prob > 0:
        home_win_prob /= total_prob
        draw_prob /= total_prob
        away_win_prob /= total_prob
    else:
        home_win_prob = draw_prob = away_win_prob = 1/3
    
    return home_win_prob, draw_prob, away_win_prob

def _extract_team_strengths(attack_params, defense_params):
    """
    Extract team strengths from Bayesian parameters and ensure they're within 0-3 range.
    
    Parameters:
    -----------
    attack_params, defense_params : dict
        Dictionaries containing attack and defense parameters for all teams
    
    Returns:
    --------
    tuple
        (attack_strengths, defense_strengths)
    """
    attack_strengths = {}
    defense_strengths = {}
    
    # Extract alpha values directly
    for team, (alpha, _) in attack_params.items():
        attack_strengths[team] = alpha
    
    for team, (alpha, _) in defense_params.items():
        defense_strengths[team] = alpha
    
    # Apply final clipping to ensure all values are in 0-3 range
    attack_strengths = {team: min(max(strength, 0.0), 3.0) for team, strength in attack_strengths.items()}
    defense_strengths = {team: min(max(strength, 0.0), 3.0) for team, strength in defense_strengths.items()}
    
    return attack_strengths, defense_strengths

def add_bayesian_ratings(file_path, num_recent_games=6, output_file=None, prior_strength=0.7, update_interval=5):
    """
    Add Bayesian hierarchical model ratings to the football dataset.
    All ratings and expected goals are scaled to the 0-3 range.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    num_recent_games : int, optional
        Number of recent games to consider for calculations (default: 6)
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_bayesian' to original filename)
    prior_strength : float, optional
        Strength of the prior (default: 0.7, higher values give more weight to prior)
    update_interval : int, optional
        Number of matches after which to recalculate summaries (default: 5)
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with Bayesian model rating columns
    """
    # Set default output file name if not provided
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_bayesian_0_3.xlsx')
    
    # Load and process the data
    df = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize rating columns
    bayesian_columns = [
        'home_team_attack_bayesian',
        'home_team_defense_bayesian',
        'away_team_attack_bayesian',
        'away_team_defense_bayesian',
        'home_advantage_bayesian',
        'home_xg_bayesian',
        'away_xg_bayesian',
    ]
    
    # Initialize all columns with default values - set to middle of 0-3 range (1.5)
    for col in bayesian_columns:
        if col.startswith('expected_'):
            df[col] = 1.5  # Default expected goals is 1.5 (middle of 0-3 range)
        elif col == 'home_advantage_bayesian':
            df[col] = 1.1  # Default home advantage is 1.1
        else:
            df[col] = 1.5  # Default strength is 1.5 on the 0-3 scale
    
    # Process each season separately
    seasons = sorted(df['season'].unique())
    for season in seasons:
        print(f"\nProcessing season: {season}")
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Minimum matches needed before we start calculating
        min_matches_needed = 5
        
        # Initialize Bayesian parameters with default 1.5 strength (middle of 0-3 range)
        attack_params = {}  # (alpha, beta) for each team
        defense_params = {}  # (alpha, beta) for each team
        
        # Calculate prior parameters - using 1.1 for home advantage
        prior_attack, prior_defense, home_advantage, prior_precision = _get_prior_parameters(
            df, prior_strength=prior_strength
        )
        
        # Last update position for periodic recalculation
        last_update_position = -1
        
        # Track team strengths for this season
        attack_strengths = {}
        defense_strengths = {}
        
        # Process each match chronologically within the season
        for match_idx_position, match_idx in enumerate(season_indices):
            home_team = df.at[match_idx, 'home_team']
            away_team = df.at[match_idx, 'away_team']
            
            # Get all previous matches in this season up to this match
            prev_season_matches = season_df[season_df.index < match_idx]
            
            # Check if we have enough matches to fit the model
            if len(prev_season_matches) >= min_matches_needed:
                # For each team, find recent matches to update their parameters
                all_teams = set(prev_season_matches['home_team'].unique()) | set(prev_season_matches['away_team'].unique())
                
                # Skip if either team hasn't played yet
                if home_team not in all_teams or away_team not in all_teams:
                    continue
                
                # Filter recent matches for relevant teams
                team_recent_matches = {}
                
                for team in [home_team, away_team]:
                    team_matches = prev_season_matches[(prev_season_matches['home_team'] == team) | 
                                                     (prev_season_matches['away_team'] == team)]
                    
                    # Sort by index (chronological order) and take the most recent matches
                    if len(team_matches) > 0:
                        team_recent_matches[team] = team_matches.sort_index(ascending=False).head(num_recent_games)
                
                # Process each team's recent matches to update their parameters
                for team, matches in team_recent_matches.items():
                    for _, match in matches.iterrows():
                        match_home_team = match['home_team']
                        match_away_team = match['away_team']
                        
                        # Update parameters based on match outcome
                        if match_home_team == team:
                            attack_params, defense_params = _update_bayesian_parameters(
                                team, True, match['home_goals'], match['away_goals'],
                                attack_params, defense_params, 
                                match_away_team, home_advantage, 
                                prior_attack, prior_defense, prior_precision
                            )
                        else:  # Away team
                            attack_params, defense_params = _update_bayesian_parameters(
                                team, False, match['away_goals'], match['home_goals'],
                                attack_params, defense_params, 
                                match_home_team, home_advantage, 
                                prior_attack, prior_defense, prior_precision
                            )
                
                # Periodically recalculate team strengths from parameters
                if match_idx_position - last_update_position >= update_interval:
                    last_update_position = match_idx_position
                    attack_strengths, defense_strengths = _extract_team_strengths(attack_params, defense_params)
                    
                    # Print progress
                    if match_idx_position % 100 == 0:
                        print(f"  Updated model at match {match_idx_position+1}/{len(season_indices)}")
                
                # Calculate expected goals for this match
                expected_home_goals, expected_away_goals = _calculate_expected_goals(
                    home_team, away_team, attack_params, defense_params, home_advantage
                )
                
                # Update ratings for current match - all rounded to 2 decimal places
                df.at[match_idx, 'home_team_attack_bayesian'] = round(attack_strengths.get(home_team, 1.5), 2)
                df.at[match_idx, 'home_team_defense_bayesian'] = round(defense_strengths.get(home_team, 1.5), 2)
                df.at[match_idx, 'away_team_attack_bayesian'] = round(attack_strengths.get(away_team, 1.5), 2)
                df.at[match_idx, 'away_team_defense_bayesian'] = round(defense_strengths.get(away_team, 1.5), 2)
                df.at[match_idx, 'home_advantage_bayesian'] = round(home_advantage, 2)
                df.at[match_idx, 'home_xg_bayesian'] = round(expected_home_goals, 2)
                df.at[match_idx, 'away_xg_bayesian'] = round(expected_away_goals, 2)
    
    # Fill any remaining NaN values with default values
    for col in bayesian_columns:
        if col.startswith('expected_'):
            df[col] = df[col].fillna(1.5)
        elif col == 'home_advantage_bayesian':
            df[col] = df[col].fillna(1.1)
        else:
            df[col] = df[col].fillna(1.5)
        
        # Ensure all values are rounded to 2 decimal places
        if col in df.columns:
            df[col] = df[col].round(2)
        
        # Final safety check to ensure all values are within required 0-3 range
        if col != 'home_advantage_bayesian':  # Skip home advantage which is fixed at 1.1
            df[col] = df[col].apply(lambda x: min(max(x, 0.0), 3.0))
    
    # Save the enhanced dataframe
    df.to_excel(output_file, index=False)
    print(f"\nData with Bayesian hierarchical model ratings saved to {output_file}")
    print(f"All values are scaled to be between 0-3, and home advantage is set to 1.1")
    
    return df

# Example usage
df_with_bayesian = add_bayesian_ratings(
    file_path='nfd_data.xlsx', 
    num_recent_games=6, 
    output_file='bayesian.xlsx',
    prior_strength=0.6,
    update_interval=6
)