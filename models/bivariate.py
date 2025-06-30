import pandas as pd
import numpy as np
from scipy.optimize import minimize
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

def _get_team_matches(df, team):
    """Get all matches for a team (both home and away)"""
    return df[(df['home_team'] == team) | (df['away_team'] == team)]

def _bivariate_poisson_pmf(x, y, lambda1, lambda2, lambda3):
    """
    Bivariate Poisson probability mass function.
    
    Parameters:
    -----------
    x, y : int
        Observed counts (goals)
    lambda1, lambda2, lambda3 : float
        Parameters of the bivariate Poisson distribution
        lambda1: home team scoring rate
        lambda2: away team scoring rate
        lambda3: correlation parameter
    
    Returns:
    --------
    float
        Probability of observing (x, y)
    """
    # Regular Poisson probabilities
    p1 = poisson.pmf(x, lambda1)
    p2 = poisson.pmf(y, lambda2)
    
    # Handle the case where lambda3 is very small (independent case)
    if lambda3 < 1e-10:
        return p1 * p2
    
    # Full bivariate Poisson calculation
    prob = 0.0
    min_val = min(x, y)
    
    for k in range(min_val + 1):
        # Calculate combinatorial terms (binomial coefficients)
        c1 = np.math.comb(x, k)
        c2 = np.math.comb(y, k)
        
        # Calculate Poisson probabilities
        pk1 = poisson.pmf(x - k, lambda1)
        pk2 = poisson.pmf(y - k, lambda2)
        pk3 = poisson.pmf(k, lambda3)
        
        prob += c1 * c2 * pk1 * pk2 * pk3 * ((-1) ** k) / (np.math.factorial(k))
    
    return prob

def _bivariate_poisson_loglikelihood(params, match_data, teams_attack, teams_defense, num_teams):
    """
    Calculate the negative log-likelihood for the Bivariate Poisson model.
    
    Parameters:
    -----------
    params : array
        Array of parameters: home advantage and lambda3 (correlation parameter)
    match_data : list of tuples
        List of (home_idx, away_idx, home_goals, away_goals) tuples
    teams_attack : dict
        Mapping of team indices to their attack parameters
    teams_defense : dict
        Mapping of team indices to their defense parameters
    num_teams : int
        Number of teams
        
    Returns:
    --------
    float
        Negative log-likelihood
    """
    # Extract parameters
    home_advantage = params[0]
    lambda3 = params[1]  # Correlation parameter
    
    # Calculate negative log-likelihood for matches
    nll = 0.0
    
    for home_idx, away_idx, home_goals, away_goals in match_data:
        # Calculate expected goals (lambda parameters)
        lambda1 = np.exp(teams_attack[home_idx] + teams_defense[away_idx] + home_advantage)
        lambda2 = np.exp(teams_attack[away_idx] + teams_defense[home_idx])
        
        # Calculate probability using bivariate Poisson PMF
        prob = _bivariate_poisson_pmf(home_goals, away_goals, lambda1, lambda2, lambda3)
        
        # Add to negative log-likelihood
        if prob > 0:
            nll -= np.log(prob)
        else:
            # Avoid numerical issues
            nll += 10  # Penalize impossible scenarios
    
    # Add regularization to prevent extreme values
    regularization_strength = 0.1
    for team_idx in range(num_teams):
        nll += regularization_strength * (teams_attack[team_idx]**2 + teams_defense[team_idx]**2)
    
    return nll

def _fit_bivariate_poisson_model(match_data, teams, initial_values=None):
    """
    Fit the Bivariate Poisson model to match data.
    
    Parameters:
    -----------
    match_data : list of tuples
        List of (home_idx, away_idx, home_goals, away_goals) tuples
    teams : list
        List of team names
    initial_values : dict, optional
        Initial values for team parameters
        
    Returns:
    --------
    tuple
        (attack_strengths, defense_strengths, home_advantage, lambda3)
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
    
    # Initial values for global parameters (home advantage and lambda3)
    initial_params = np.array([0.3, 0.1])  # home advantage and lambda3
    
    # Define objective function for the global parameters optimization
    def objective(params):
        return _bivariate_poisson_loglikelihood(params, match_data, teams_attack, teams_defense, num_teams)
    
    # Optimize global parameters
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=[(0, 1), (0, 0.5)],  # home advantage [0,1], lambda3 [0,0.5]
        options={'maxiter': 50}
    )
    
    home_advantage = result.x[0]
    lambda3 = result.x[1]
    
    # Create datasets for estimating team parameters
    home_goals = np.array([match[2] for match in match_data])
    away_goals = np.array([match[3] for match in match_data])
    
    # For each team as home and away
    home_teams = np.array([match[0] for match in match_data])
    away_teams = np.array([match[1] for match in match_data])
    
    # Adjust for correlation in a simplified way
    adjusted_home_goals = np.maximum(0, home_goals - lambda3 * away_goals.mean() / away_goals.std())
    adjusted_away_goals = np.maximum(0, away_goals - lambda3 * home_goals.mean() / home_goals.std())
    
    # Simple estimation of attack and defense parameters
    # This is a simplified approach without using a full GLM
    team_avg_goals = {i: {'home_scored': 0, 'home_games': 0, 'away_scored': 0, 'away_games': 0,
                           'home_conceded': 0, 'away_conceded': 0} for i in range(num_teams)}
    
    # Collect goal statistics
    for i, (home_idx, away_idx, home_goal, away_goal) in enumerate(match_data):
        team_avg_goals[home_idx]['home_scored'] += adjusted_home_goals[i]
        team_avg_goals[home_idx]['home_games'] += 1
        team_avg_goals[home_idx]['home_conceded'] += adjusted_away_goals[i]
        
        team_avg_goals[away_idx]['away_scored'] += adjusted_away_goals[i]
        team_avg_goals[away_idx]['away_games'] += 1
        team_avg_goals[away_idx]['away_conceded'] += adjusted_home_goals[i]
    
    # Calculate average goals per game
    avg_home_goals = sum(adjusted_home_goals) / len(adjusted_home_goals) if len(adjusted_home_goals) > 0 else 1.0
    avg_away_goals = sum(adjusted_away_goals) / len(adjusted_away_goals) if len(adjusted_away_goals) > 0 else 0.7
    
    # Calculate team attack and defense strengths
    for i in range(num_teams):
        # Attack strength
        home_attack = 0.0
        if team_avg_goals[i]['home_games'] > 0:
            home_attack = team_avg_goals[i]['home_scored'] / team_avg_goals[i]['home_games'] / avg_home_goals
        
        away_attack = 0.0
        if team_avg_goals[i]['away_games'] > 0:
            away_attack = team_avg_goals[i]['away_scored'] / team_avg_goals[i]['away_games'] / avg_away_goals
        
        # Combined attack strength (weighted by number of games)
        home_games = max(1, team_avg_goals[i]['home_games'])
        away_games = max(1, team_avg_goals[i]['away_games'])
        attack_strength = (home_attack * home_games + away_attack * away_games) / (home_games + away_games)
        
        # Defense strength (inverted, lower is better)
        home_defense = 0.0
        if team_avg_goals[i]['home_games'] > 0:
            home_defense = team_avg_goals[i]['home_conceded'] / team_avg_goals[i]['home_games'] / avg_away_goals
        
        away_defense = 0.0
        if team_avg_goals[i]['away_games'] > 0:
            away_defense = team_avg_goals[i]['away_conceded'] / team_avg_goals[i]['away_games'] / avg_home_goals
        
        # Combined defense strength (weighted by number of games)
        defense_strength = (home_defense * home_games + away_defense * away_games) / (home_games + away_games)
        
        # Convert to log scale for the model
        teams_attack[i] = np.log(max(attack_strength, 0.01))
        teams_defense[i] = np.log(max(1/defense_strength, 0.01)) if defense_strength > 0 else 0.0
    
    # Ensure identifiability by centering the attack and defense parameters
    avg_attack = sum(teams_attack.values()) / num_teams
    avg_defense = sum(teams_defense.values()) / num_teams
    
    for i in range(num_teams):
        teams_attack[i] -= avg_attack
        teams_defense[i] -= avg_defense
    
    # Convert parameters to team ratings
    attack_strengths = {teams[i]: teams_attack[i] for i in range(num_teams)}
    defense_strengths = {teams[i]: teams_defense[i] for i in range(num_teams)}
    
    return attack_strengths, defense_strengths, home_advantage, lambda3

def add_bivariate_poisson_ratings(file_path, num_recent_games=6, output_file=None):
    """
    Add Bivariate Poisson model rating columns to the football dataset.
    Ratings are explicitly scaled to be positive and within the range 0-5.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    num_recent_games : int, optional
        Number of recent games to consider for calculations (default: 6)
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_bp' to original filename)
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with Bivariate Poisson rating columns
    """
    # Set default output file name if not provided
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_bp.xlsx')
    
    # Load and process the data
    df = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize Bivariate Poisson rating columns
    bp_columns = [
        'home_team_attack_bp',
        'home_team_defense_bp',
        'away_team_attack_bp',
        'away_team_defense_bp',
        'bp_home_advantage',
        'bp_lambda3'
    ]
    
    # Initialize all columns with default values
    for col in bp_columns:
        if col.endswith('_attack_bp') or col.endswith('_defense_bp'):
            df[col] = 0.5  # Default rating of 0.5
        else:
            df[col] = 0.0
    
    # Process each season separately
    seasons = sorted(df['season'].unique())
    for i, season in enumerate(seasons):
        print(f"Processing season: {season}")
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Bivariate Poisson model parameters for this season
        team_attack = {}  # Dictionary to store attack parameters
        team_defense = {}  # Dictionary to store defense parameters
        home_advantage = 0.3  # Default home advantage parameter
        lambda3 = 0.1  # Default correlation parameter
        
        # Initialize ratings for teams in this season
        teams_in_season = list(set(season_df['home_team'].unique()) | set(season_df['away_team'].unique()))
        for team in teams_in_season:
            team_attack[team] = 0.0
            team_defense[team] = 0.0
        
        # Track the model fitting frequency
        last_fit_index = -1
        
        # Adjust fit interval based on season length
        season_length = len(season_indices)
        fit_interval = max(3, season_length // 20)  # More frequent updates for shorter seasons
        
        # Adjust minimum matches needed based on season position
        if i == 0:
            # First season - start fitting earlier
            min_matches_needed = 6
        else:
            # Subsequent seasons - standard threshold
            min_matches_needed = 10
        
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
                    team_matches = _get_team_matches(prev_season_matches, team)
                    
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
                
                if len(match_data) >= 5 and len(teams_to_fit) >= 2:  # Reduced minimum matches for early fits
                    # Prepare initial values
                    initial_values = {}
                    for team in teams_to_fit:
                        if team in team_attack and team in team_defense:
                            initial_values[team] = {
                                'attack': team_attack[team],
                                'defense': team_defense[team]
                            }
                    
                    # Fit the Bivariate Poisson model
                    try:
                        attack, defense, home_advantage, lambda3 = _fit_bivariate_poisson_model(
                            match_data, teams_to_fit, initial_values
                        )
                        
                        # Update team parameters
                        for team, att in attack.items():
                            team_attack[team] = att
                        
                        for team, defs in defense.items():
                            team_defense[team] = defs
                        
                        print(f"  Model fit successful. Home advantage: {home_advantage:.2f}, Lambda3: {lambda3:.2f}")
                    except Exception as e:
                        print(f"  Error fitting Bivariate Poisson model: {e}")
                        # If optimization fails, keep existing ratings
            
            # Set current Bivariate Poisson ratings for this match
            current_home_attack = team_attack.get(home_team, 0.0)
            current_home_defense = team_defense.get(home_team, 0.0)
            current_away_attack = team_attack.get(away_team, 0.0)
            current_away_defense = team_defense.get(away_team, 0.0)
            
            # Scale ratings to 0-5 range
            # Get all current parameter values to determine the range
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
            else:
                # Not enough teams for scaling, use default value
                home_attack_scaled = 0.5
                home_defense_scaled = 0.5
                away_attack_scaled = 0.5
                away_defense_scaled = 0.5
            
            # Store ratings in the dataframe
            df.at[match_idx, 'home_team_attack_bp'] = home_attack_scaled
            df.at[match_idx, 'home_team_defense_bp'] = home_defense_scaled
            df.at[match_idx, 'away_team_attack_bp'] = away_attack_scaled
            df.at[match_idx, 'away_team_defense_bp'] = away_defense_scaled
            df.at[match_idx, 'bp_home_advantage'] = home_advantage
            df.at[match_idx, 'bp_lambda3'] = lambda3
    
    # Round all rating columns to 2 decimal places
    for col in bp_columns:
        df[col] = df[col].round(2)
    
    # Save the enhanced dataframe
    df.to_excel(output_file, index=False)
    print(f"Data with Bivariate Poisson ratings saved to {output_file}")
    
    return df

# Example usage
df_with_bp = add_bivariate_poisson_ratings(file_path='nfd_data.xlsx', num_recent_games=6, output_file='bivariate.xlsx')