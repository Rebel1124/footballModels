import pandas as pd
import numpy as np
from scipy.stats import skellam
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
                    missing_cols.remove('home_team')
                elif 'homeID' in df.columns:
                    df['home_team'] = df['homeID'].astype(str)
                    missing_cols.remove('home_team')
            
            if 'away_team' in missing_cols:
                away_team_candidates = [col for col in df.columns if 'away' in col.lower() and ('team' in col.lower() or 'name' in col.lower())]
                if away_team_candidates:
                    df['away_team'] = df[away_team_candidates[0]]
                    missing_cols.remove('away_team')
                elif 'awayID' in df.columns:
                    df['away_team'] = df['awayID'].astype(str)
                    missing_cols.remove('away_team')
            
            # If still missing required columns, raise error
            if missing_cols:
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
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def _calculate_expected_goals(match_df, attack_strengths, defense_strengths, home_advantage):
    """
    Calculate expected goals for matches using team strengths.
    
    Parameters:
    -----------
    match_df : pandas.DataFrame
        DataFrame containing match data
    attack_strengths : dict
        Dictionary of team attack strengths
    defense_strengths : dict
        Dictionary of team defense strengths
    home_advantage : float
        Home advantage factor
    
    Returns:
    --------
    tuple
        (home_expected, away_expected)
    """
    home_team = match_df['home_team']
    away_team = match_df['away_team']
    
    home_attack = attack_strengths.get(home_team, 1.0)
    home_defense = defense_strengths.get(home_team, 1.0)
    away_attack = attack_strengths.get(away_team, 1.0)
    away_defense = defense_strengths.get(away_team, 1.0)
    
    home_expected = home_attack * away_defense * home_advantage
    away_expected = away_attack * home_defense
    
    return home_expected, away_expected

def _calculate_skellam_probabilities(home_expected, away_expected, max_goal_diff=5):
    """
    Calculate win/draw/loss probabilities using the Skellam distribution.
    
    Parameters:
    -----------
    home_expected : float
        Expected goals for the home team
    away_expected : float
        Expected goals for the away team
    max_goal_diff : int
        Maximum goal difference to consider
    
    Returns:
    --------
    tuple
        (home_win_prob, draw_prob, away_win_prob)
    """
    # Ensure positive expected goals
    home_expected = max(home_expected, 0.01)
    away_expected = max(away_expected, 0.01)
    
    # Calculate probabilities for various goal differences
    home_win_prob = 0.0
    draw_prob = skellam.pmf(0, home_expected, away_expected)
    away_win_prob = 0.0
    
    # Use Skellam PMF function for goal differences
    # Optimize by calculating only the most likely outcomes
    for goal_diff in range(1, max_goal_diff + 1):
        # Home win probabilities (goal_diff > 0)
        home_win_prob += skellam.pmf(goal_diff, home_expected, away_expected)
        
        # Away win probabilities (goal_diff < 0)
        away_win_prob += skellam.pmf(-goal_diff, home_expected, away_expected)
    
    # Account for extreme goal differences beyond our calculation range
    remaining_prob = 1.0 - (home_win_prob + draw_prob + away_win_prob)
    if remaining_prob > 0:
        # Distribute remaining probability proportionally
        if home_expected > away_expected:
            home_win_prob += remaining_prob * 0.75
            away_win_prob += remaining_prob * 0.25
        elif away_expected > home_expected:
            home_win_prob += remaining_prob * 0.25
            away_win_prob += remaining_prob * 0.75
        else:
            home_win_prob += remaining_prob * 0.5
            away_win_prob += remaining_prob * 0.5
    
    # Normalize to ensure probabilities sum to 1
    total_prob = home_win_prob + draw_prob + away_win_prob
    if abs(total_prob - 1.0) > 1e-10:  # Only normalize if needed
        home_win_prob /= total_prob
        draw_prob /= total_prob
        away_win_prob /= total_prob
    
    return home_win_prob, draw_prob, away_win_prob

def _estimate_team_strengths(matches_df, league_avg_home=1.5, league_avg_away=1.0):
    """
    Estimate team strengths based on average goals.
    
    Parameters:
    -----------
    matches_df : pandas.DataFrame
        DataFrame containing match data
    league_avg_home : float, optional
        League average for home goals (default: 1.5)
    league_avg_away : float, optional
        League average for away goals (default: 1.0)
    
    Returns:
    --------
    tuple
        (attack_strengths, defense_strengths, home_advantage)
    """
    # Get all teams
    all_teams = set(matches_df['home_team'].unique()) | set(matches_df['away_team'].unique())
    
    # Initialize counters
    team_home_goals = {team: 0 for team in all_teams}
    team_away_goals = {team: 0 for team in all_teams}
    team_home_conceded = {team: 0 for team in all_teams}
    team_away_conceded = {team: 0 for team in all_teams}
    team_home_games = {team: 0 for team in all_teams}
    team_away_games = {team: 0 for team in all_teams}
    
    # Calculate totals
    for _, match in matches_df.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        home_goals = match['home_goals']
        away_goals = match['away_goals']
        
        # Update stats
        team_home_goals[home_team] += home_goals
        team_home_conceded[home_team] += away_goals
        team_home_games[home_team] += 1
        
        team_away_goals[away_team] += away_goals
        team_away_conceded[away_team] += home_goals
        team_away_games[away_team] += 1
    
    # Calculate averages
    total_home_goals = 0
    total_away_goals = 0
    total_home_games = 0
    total_away_games = 0
    
    for team in all_teams:
        if team_home_games[team] > 0:
            team_home_goals[team] /= team_home_games[team]
            team_home_conceded[team] /= team_home_games[team]
            total_home_goals += team_home_goals[team] * team_home_games[team]
            total_home_games += team_home_games[team]
        else:
            team_home_goals[team] = league_avg_home
            team_home_conceded[team] = league_avg_away
        
        if team_away_games[team] > 0:
            team_away_goals[team] /= team_away_games[team]
            team_away_conceded[team] /= team_away_games[team]
            total_away_goals += team_away_goals[team] * team_away_games[team]
            total_away_games += team_away_games[team]
        else:
            team_away_goals[team] = league_avg_away
            team_away_conceded[team] = league_avg_home
    
    # League averages
    if total_home_games > 0:
        league_avg_home = total_home_goals / total_home_games
    
    if total_away_games > 0:
        league_avg_away = total_away_goals / total_away_games
    
    # Home advantage
    home_advantage = league_avg_home / league_avg_away if league_avg_away > 0 else 1.3
    home_advantage = min(max(home_advantage, 1.0), 2.0)  # Constrain to reasonable range
    
    # Calculate attack and defense strengths
    attack_strengths = {}
    defense_strengths = {}
    
    for team in all_teams:
        # Attack strength (weighted average of home and away)
        home_weight = team_home_games[team] / max(team_home_games[team] + team_away_games[team], 1)
        away_weight = 1 - home_weight
        
        # Attack strength = weighted average of relative goal scoring rates
        home_attack = team_home_goals[team] / league_avg_home if league_avg_home > 0 else 1.0
        away_attack = team_away_goals[team] / league_avg_away if league_avg_away > 0 else 0.8
        attack_strengths[team] = home_weight * home_attack + away_weight * away_attack
        
        # Defense strength = weighted average of relative goal concession rates (inverted)
        home_defense = league_avg_away / team_home_conceded[team] if team_home_conceded[team] > 0 else 1.0
        away_defense = league_avg_home / team_away_conceded[team] if team_away_conceded[team] > 0 else 0.8
        defense_strengths[team] = home_weight * home_defense + away_weight * away_defense
    
    # Normalize strengths to have average of 1.0
    attack_avg = np.mean(list(attack_strengths.values()))
    if attack_avg > 0:
        attack_strengths = {t: s/attack_avg for t, s in attack_strengths.items()}
    
    defense_avg = np.mean(list(defense_strengths.values()))
    if defense_avg > 0:
        defense_strengths = {t: s/defense_avg for t, s in defense_strengths.items()}
    
    return attack_strengths, defense_strengths, home_advantage

def add_skellam_ratings(file_path, num_recent_games=6, output_file=None, update_interval=10):
    """
    Add Skellam model ratings to the football dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    num_recent_games : int, optional
        Number of recent games to consider for calculations (default: 6)
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_skellam' to original filename)
    update_interval : int, optional
        Number of matches after which to update the model (default: 10)
        Lower values give more precision but increase computation time
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with Skellam model rating columns
    """
    # Set default output file name if not provided
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_skellam_optimized.xlsx')
    
    # Load and process the data
    df = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize rating columns
    skellam_columns = [
        'home_team_attack_skellam',
        'home_team_defense_skellam',
        'away_team_attack_skellam',
        'away_team_defense_skellam',
        'home_advantage_skellam',
        'expected_home_goals_skellam',
        'expected_away_goals_skellam',
        # 'home_win_prob_skellam',
        # 'draw_prob_skellam',
        # 'away_win_prob_skellam'
    ]
    
    # Initialize all columns with default values
    for col in skellam_columns:
        if col.startswith('expected_') or col.endswith('_prob_skellam'):
            df[col] = 0.0  # Default expected goals and probabilities are 0.0
        elif col == 'home_advantage_skellam':
            df[col] = 1.3  # Default home advantage is 1.3
        else:
            df[col] = 1.0  # Default strength is neutral (1.0)
    
    # Process each season separately
    seasons = sorted(df['season'].unique())
    for season in seasons:
        print(f"\nProcessing season: {season}")
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Minimum matches needed before we start calculating
        min_matches_needed = max(10, 2 * update_interval)
        
        # Track last update position to avoid unnecessary recalculations
        last_update_position = -1
        
        # Track team ratings for this season
        attack_strengths = {}
        defense_strengths = {}
        home_advantage = 1.3
        
        # Process each match chronologically within the season
        for match_idx_position, match_idx in enumerate(season_indices):
            home_team = df.at[match_idx, 'home_team']
            away_team = df.at[match_idx, 'away_team']
            
            # Get all previous matches in this season up to this match
            prev_season_matches = season_df[season_df.index < match_idx]
            num_prev_matches = len(prev_season_matches)
            
            # Check if we have enough matches to fit the model
            if num_prev_matches >= min_matches_needed:
                # Only update model if we've processed update_interval more matches since last update
                if match_idx_position - last_update_position >= update_interval:
                    last_update_position = match_idx_position
                    
                    # Filter recent matches for each team
                    team_recent_matches = {}
                    all_recent_match_indices = set()
                    
                    all_teams = set(prev_season_matches['home_team'].unique()) | set(prev_season_matches['away_team'].unique())
                    
                    for team in all_teams:
                        team_matches = prev_season_matches[(prev_season_matches['home_team'] == team) | 
                                                         (prev_season_matches['away_team'] == team)]
                        
                        # Sort by index (chronological order) and take the most recent matches
                        if len(team_matches) > 0:
                            recent_matches = team_matches.sort_index(ascending=False).head(num_recent_games)
                            team_recent_matches[team] = recent_matches
                            all_recent_match_indices.update(recent_matches.index)
                    
                    # Get the combined set of recent matches
                    recent_matches = prev_season_matches[prev_season_matches.index.isin(all_recent_match_indices)]
                    
                    # Estimate team strengths based on performance in recent matches
                    attack_strengths, defense_strengths, home_advantage = _estimate_team_strengths(recent_matches)
                    
                    # Print progress update
                    print(f"  Updated model at match {match_idx_position+1}/{len(season_indices)} ({len(recent_matches)} recent matches)")
                
                # Only proceed if both teams have ratings
                if home_team in attack_strengths and away_team in attack_strengths:
                    # Calculate expected goals for this match
                    home_expected, away_expected = _calculate_expected_goals(
                        df.iloc[match_idx], attack_strengths, defense_strengths, home_advantage
                    )
                    
                    # Calculate match outcome probabilities
                    home_win_prob, draw_prob, away_win_prob = _calculate_skellam_probabilities(
                        home_expected, away_expected
                    )
                    
                    # Update ratings for current match - all rounded to 2 decimal places
                    df.at[match_idx, 'home_team_attack_skellam'] = round(attack_strengths[home_team], 2)
                    df.at[match_idx, 'home_team_defense_skellam'] = round(defense_strengths[home_team], 2)
                    df.at[match_idx, 'away_team_attack_skellam'] = round(attack_strengths[away_team], 2)
                    df.at[match_idx, 'away_team_defense_skellam'] = round(defense_strengths[away_team], 2)
                    df.at[match_idx, 'home_advantage_skellam'] = round(home_advantage, 2)
                    df.at[match_idx, 'expected_home_goals_skellam'] = round(home_expected, 2)
                    df.at[match_idx, 'expected_away_goals_skellam'] = round(away_expected, 2)
                    # df.at[match_idx, 'home_win_prob_skellam'] = round(home_win_prob, 2)
                    # df.at[match_idx, 'draw_prob_skellam'] = round(draw_prob, 2)
                    # df.at[match_idx, 'away_win_prob_skellam'] = round(away_win_prob, 2)
    
    # Fill any remaining NaN values with 0
    for col in skellam_columns:
        df[col] = df[col].fillna(0)
        # Ensure all values are rounded to 2 decimal places
        if col in df.columns:
            df[col] = df[col].round(2)
    
    # Save the enhanced dataframe
    df.to_excel(output_file, index=False)
    print(f"\nData with Skellam model ratings saved to {output_file}")
    
    return df

# Example usage
df_with_skellam = add_skellam_ratings(
    file_path='nfd_data.xlsx', 
    num_recent_games=6, 
    output_file='skellam_optimized.xlsx',
    update_interval=6  # Only update model every 6 matches
)