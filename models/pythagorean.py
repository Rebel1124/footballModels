import pandas as pd
import numpy as np
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

def _calculate_pythagorean_expectation(goals_scored, goals_conceded, exponent=1.83):
    """
    Calculate the Pythagorean expectation (expected win percentage) for a team.
    
    Parameters:
    -----------
    goals_scored : float
        Number of goals scored by the team
    goals_conceded : float
        Number of goals conceded by the team
    exponent : float
        Pythagorean exponent (default: 1.83, which is commonly used for soccer)
    
    Returns:
    --------
    float
        Expected win percentage (0-1)
    """
    # Avoid division by zero
    if goals_scored == 0 and goals_conceded == 0:
        return 0.5  # Neutral expectation
    elif goals_conceded == 0:
        return 1.0  # Perfect expectation
    
    # Calculate Pythagorean expectation
    expectation = goals_scored ** exponent / (goals_scored ** exponent + goals_conceded ** exponent)
    return expectation

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
    
    # Perform scaling
    scaled_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    
    # Ensure value is within new range
    return min(max(scaled_value, new_min), new_max)

def add_pythagorean_ratings(file_path, num_recent_games=6, exponent=1.83, output_file=None):
    """
    Add Pythagorean model ratings to the football dataset.
    All ratings are scaled to the 0.5-3 range.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    num_recent_games : int, optional
        Number of recent games to consider for calculations (default: 6)
    exponent : float, optional
        Pythagorean exponent (default: 1.83, common for soccer)
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_pyth' to original filename)
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with Pythagorean rating columns
    """
    # Set default output file name if not provided
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_pyth_0.5_3.xlsx')
    
    # Load and process the data
    df = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize Pythagorean model columns
    pyth_columns = [
        'home_team_strength_pyth',
        'away_team_strength_pyth',
        'home_team_luck_factor'
    ]
    
    # Initialize all columns with default values in 0.5-3 range
    for col in pyth_columns:
        if col == 'home_team_luck_factor':
            df[col] = 1.5  # Neutral luck factor (middle of 0.5-3 range)
        else:
            df[col] = 1.5  # Neutral team strength (middle of 0.5-3 range)
    
    # Process each season separately
    seasons = sorted(df['season'].unique())
    for i, season in enumerate(seasons):
        print(f"\nProcessing season: {season}")
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Track match counts to help diagnose the issue
        season_match_count = 0
        teams_with_ratings = set()
        team_match_counts = {}
        
        # Initialize team data for better early-season estimates
        teams_in_season = list(set(season_df['home_team'].unique()) | set(season_df['away_team'].unique()))
        for team in teams_in_season:
            team_match_counts[team] = 0
        
        # Lower minimum match threshold to ensure more teams get ratings earlier
        # min_matches_needed = 2  # Lowered from 3/5 to get more teams rated early
        min_matches_needed = num_recent_games  # Lowered from 3/5 to get more teams rated early: num_recent_games
        
        # Process each match chronologically within the season
        for match_idx_position, match_idx in enumerate(season_indices):
            home_team = df.at[match_idx, 'home_team']
            away_team = df.at[match_idx, 'away_team']
            season_match_count += 1
            
            # Get all previous matches in this season up to this match
            prev_season_matches = season_df[season_df.index < match_idx]
            
            # Get team-specific match histories
            home_team_matches = prev_season_matches[(prev_season_matches['home_team'] == home_team) | 
                                                    (prev_season_matches['away_team'] == home_team)]
            away_team_matches = prev_season_matches[(prev_season_matches['home_team'] == away_team) | 
                                                    (prev_season_matches['away_team'] == away_team)]
            
            # Update match counts
            team_match_counts[home_team] = len(home_team_matches)
            team_match_counts[away_team] = len(away_team_matches)
            
            # Calculate home team statistics directly from the match history
            if len(home_team_matches) > 0:
                # Initialize counters
                home_team_goals_scored = 0
                home_team_goals_conceded = 0
                home_team_wins = 0
                home_team_draws = 0
                
                # If very early in the season, we'll use all matches
                # Otherwise, use only the most recent matches
                if len(home_team_matches) <= min_matches_needed:
                    # Use all available matches
                    recent_home_matches = home_team_matches
                else:
                    # Get only the most recent matches
                    recent_home_matches = home_team_matches.tail(min(len(home_team_matches), num_recent_games))
                
                # Process each match
                for _, match in recent_home_matches.iterrows():
                    # Check if team played at home or away
                    if match['home_team'] == home_team:
                        # Team played at home
                        home_team_goals_scored += match['home_goals']
                        home_team_goals_conceded += match['away_goals']
                        
                        if match['full_time_result'] == 'H':
                            home_team_wins += 1
                        elif match['full_time_result'] == 'D':
                            home_team_draws += 1
                    else:
                        # Team played away
                        home_team_goals_scored += match['away_goals']
                        home_team_goals_conceded += match['home_goals']
                        
                        if match['full_time_result'] == 'A':
                            home_team_wins += 1
                        elif match['full_time_result'] == 'D':
                            home_team_draws += 1
                
                # Calculate actual win percentage (including draws as half wins)
                actual_win_pct = (home_team_wins + home_team_draws * 0.5) / len(recent_home_matches)
                
                # Calculate Pythagorean expectation
                expected_win_pct = _calculate_pythagorean_expectation(
                    home_team_goals_scored, home_team_goals_conceded, exponent
                )
                
                # Calculate luck factor
                home_team_luck = actual_win_pct - expected_win_pct
                
                # Scale expected win percentage to 0.5-3 range for team strength
                # The mapping is: win% 0.0 -> 0.5, win% 0.5 -> 1.75, win% 1.0 -> 3.0
                home_strength = _scale_to_range(expected_win_pct, 0.0, 1.0, 0.5, 3.0)
                
                # Scale luck factor to 0.5-3 range
                # A neutral luck factor (no luck) should be 1.75 (middle of 0.5-3)
                # Negative luck (actual < expected) -> 0.5-1.75
                # Positive luck (actual > expected) -> 1.75-3.0
                # Luck range is typically -0.3 to 0.3, so scale accordingly
                home_luck_factor = _scale_to_range(home_team_luck, -0.3, 0.3, 0.5, 3.0)
                
                # Print debug info for the first match for each team or periodically
                if home_team not in teams_with_ratings or match_idx_position % 100 == 0:
                    print(f"  Match {season_match_count}: {home_team} vs {away_team}")
                    print(f"    {home_team} matches so far: {len(home_team_matches)}")
                    print(f"    Used for calculation: {len(recent_home_matches)} matches")
                    print(f"    Goals: Scored={home_team_goals_scored}, Conceded={home_team_goals_conceded}")
                    print(f"    Expected Win%={expected_win_pct:.3f}, Actual Win%={actual_win_pct:.3f}")
                    print(f"    Strength={home_strength:.2f}, Luck={home_luck_factor:.2f}")
                
                # Update values even if below the old threshold, as long as team has played at least 1 match
                df.at[match_idx, 'home_team_strength_pyth'] = round(home_strength, 2)
                df.at[match_idx, 'home_team_luck_factor'] = round(home_luck_factor, 2)
                
                # Track teams that have received ratings
                teams_with_ratings.add(home_team)
            else:
                # No previous matches, keep default values or use league averages
                if match_idx_position % 100 == 0:
                    print(f"  Match {season_match_count}: {home_team} has no previous matches")
            
            # Same process for away team
            if len(away_team_matches) > 0:
                # Initialize counters
                away_team_goals_scored = 0
                away_team_goals_conceded = 0
                away_team_wins = 0
                away_team_draws = 0
                
                # If very early in the season, we'll use all matches
                # Otherwise, use only the most recent matches
                if len(away_team_matches) <= min_matches_needed:
                    # Use all available matches
                    recent_away_matches = away_team_matches
                else:
                    # Get only the most recent matches
                    recent_away_matches = away_team_matches.tail(min(len(away_team_matches), num_recent_games))
                
                # Process each match
                for _, match in recent_away_matches.iterrows():
                    # Check if team played at home or away
                    if match['home_team'] == away_team:
                        # Team played at home
                        away_team_goals_scored += match['home_goals']
                        away_team_goals_conceded += match['away_goals']
                        
                        if match['full_time_result'] == 'H':
                            away_team_wins += 1
                        elif match['full_time_result'] == 'D':
                            away_team_draws += 1
                    else:
                        # Team played away
                        away_team_goals_scored += match['away_goals']
                        away_team_goals_conceded += match['home_goals']
                        
                        if match['full_time_result'] == 'A':
                            away_team_wins += 1
                        elif match['full_time_result'] == 'D':
                            away_team_draws += 1
                
                # Calculate actual win percentage (including draws as half wins)
                actual_win_pct = (away_team_wins + away_team_draws * 0.5) / len(recent_away_matches)
                
                # Calculate Pythagorean expectation
                expected_win_pct = _calculate_pythagorean_expectation(
                    away_team_goals_scored, away_team_goals_conceded, exponent
                )
                
                # Scale expected win percentage to 0.5-3 range for team strength
                away_strength = _scale_to_range(expected_win_pct, 0.0, 1.0, 0.5, 3.0)
                
                # Print debug info for the first match for each team
                if away_team not in teams_with_ratings and (match_idx_position < 10 or match_idx_position % 100 == 0):
                    print(f"    {away_team} matches so far: {len(away_team_matches)}")
                    print(f"    Used for calculation: {len(recent_away_matches)} matches")
                    print(f"    Goals: Scored={away_team_goals_scored}, Conceded={away_team_goals_conceded}")
                    print(f"    Expected Win%={expected_win_pct:.3f}, Actual Win%={actual_win_pct:.3f}")
                    print(f"    Strength={away_strength:.2f}")
                
                # Update away team strength even if below old threshold
                df.at[match_idx, 'away_team_strength_pyth'] = round(away_strength, 2)
                
                # Track teams that have received ratings
                teams_with_ratings.add(away_team)
            else:
                # No previous matches
                if match_idx_position % 100 == 0 and match_idx_position < 20:
                    print(f"    {away_team} has no previous matches")
        
        # Print summary statistics for this season
        print(f"\nSeason {season} summary:")
        print(f"  Total matches in season: {season_match_count}")
        print(f"  Teams with ratings: {len(teams_with_ratings)} of {len(teams_in_season)}")
        
        # Print match counts for each team to diagnose slow rating assignments
        print("\nTeam match counts:")
        sorted_teams = sorted(team_match_counts.items(), key=lambda x: x[1], reverse=True)
        for team, count in sorted_teams:
            print(f"  {team}: {count} matches")
    
    # Final check to ensure all values are within the 0.5-3 range
    for col in pyth_columns:
        df[col] = df[col].apply(lambda x: min(max(x, 0.5), 3.0))
        df[col] = df[col].round(2)  # Round to 2 decimal places
    
    # Save the enhanced dataframe
    df.to_excel(output_file, index=False)
    print(f"\nData with Pythagorean ratings saved to {output_file}")
    print(f"All values are scaled to be between 0.5-3")
    
    return df

# Example usage
df_with_pyth = add_pythagorean_ratings(
    file_path='nfd_data.xlsx', 
    num_recent_games=6, 
    exponent=1.83, 
    output_file='pythagorean_0.5_3.xlsx'
)