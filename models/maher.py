import pandas as pd
import numpy as np

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
        print(f"Required columns found: {[col for col in required_cols if col in df.columns]}")
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

def add_maher_strength(file_path, num_recent_games=6, output_file=None):
    """
    Add Maher model team strength columns to the football dataset.
    All ratings and expected goals are scaled to the 0.5-3 range.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the football data
    num_recent_games : int, optional
        Number of recent games to consider for calculations (default: 6)
    output_file : str, optional
        Path to save the processed data (default: None, which appends '_maher' to original filename)
    
    Returns:
    --------
    pandas.DataFrame
        The enhanced dataframe with Maher model strength columns
    """
    # Set default output file name if not provided
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_maher_0.5_3.xlsx')
    
    # Load and process the data
    df = load_and_process_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return None
    
    # Create a sequential index for chronological ordering
    df = df.reset_index(drop=True)
    
    # Initialize Maher model strength columns
    strength_columns = [
        'home_team_strength',
        'away_team_strength',
        'home_advantage'
    ]
    
    # Initialize all columns with default values in the 0.5-3 range
    for col in strength_columns:
        if col == 'home_advantage':
            df[col] = 1.1  # Typical home advantage value
        else:
            df[col] = 1.75  # Neutral team strength (middle of 0.5-3 range)
    
    # Process each season separately
    for season in df['season'].unique():
        print(f"\nProcessing season: {season}")
        season_df = df[df['season'] == season].copy()
        season_indices = season_df.index.tolist()
        
        # Dictionary to store team strength metrics by team for this season
        team_attack_strength = {}
        team_defense_strength = {}
        league_avg_home_goals = None
        league_avg_away_goals = None
        home_advantage = None
        
        # Track teams with ratings for reporting
        teams_with_ratings = set()
        
        # Process each match chronologically within the season
        for match_idx_position, match_idx in enumerate(season_indices):
            home_team = df.at[match_idx, 'home_team']
            away_team = df.at[match_idx, 'away_team']
            
            # Get all previous matches in this season up to this match
            prev_season_matches = season_df[season_df.index < match_idx]
            
            # Calculate Maher model team strengths based on previous matches
            if len(prev_season_matches) >= num_recent_games:  # Require at least num_recent_games matches in the season so far
                # Calculate league averages for this subset of matches
                league_avg_home_goals = prev_season_matches['home_goals'].mean()
                league_avg_away_goals = prev_season_matches['away_goals'].mean()
                home_advantage = league_avg_home_goals / league_avg_away_goals if league_avg_away_goals > 0 else 1.1
                
                # Ensure home advantage is reasonable - scale to 0.5-3 range
                home_advantage = _scale_to_range(home_advantage, 0.5, 2.0, 0.5, 3.0)
                
                # Get all teams in the dataset so far
                teams = set(prev_season_matches['home_team'].unique()) | set(prev_season_matches['away_team'].unique())
                
                # Calculate attack and defense strengths for each team based on recent matches
                for team in teams:
                    # Get the most recent matches for this team
                    team_matches = _get_team_matches(prev_season_matches, team)
                    team_recent_matches = team_matches.tail(min(len(team_matches), num_recent_games))
                    
                    if len(team_recent_matches) >= 3:  # Need minimum matches for reasonable estimates
                        # Calculate attacking strength
                        home_goals_scored = team_recent_matches[team_recent_matches['home_team'] == team]['home_goals'].sum()
                        away_goals_scored = team_recent_matches[team_recent_matches['away_team'] == team]['away_goals'].sum()
                        home_matches_count = len(team_recent_matches[team_recent_matches['home_team'] == team])
                        away_matches_count = len(team_recent_matches[team_recent_matches['away_team'] == team])
                        
                        # Normalize by league averages and match counts
                        total_expected_goals = (home_matches_count * league_avg_home_goals) + (away_matches_count * league_avg_away_goals / home_advantage)
                        total_actual_goals = home_goals_scored + away_goals_scored
                        
                        attack_strength = total_actual_goals / total_expected_goals if total_expected_goals > 0 else 1.0
                        
                        # Calculate defensive strength (goals conceded)
                        home_goals_conceded = team_recent_matches[team_recent_matches['home_team'] == team]['away_goals'].sum()
                        away_goals_conceded = team_recent_matches[team_recent_matches['away_team'] == team]['home_goals'].sum()
                        
                        total_expected_conceded = (home_matches_count * league_avg_away_goals) + (away_matches_count * league_avg_home_goals * home_advantage)
                        total_actual_conceded = home_goals_conceded + away_goals_conceded
                        
                        defense_strength = total_expected_conceded / total_actual_conceded if total_actual_conceded > 0 else 1.0
                        
                        # Scale attack and defense strengths to 0.5-3 range
                        # For attack strength, typical range is 0.5-2.0, with 1.0 being average
                        # For defense strength, typical range is 0.5-2.0, with 1.0 being average
                        attack_strength_scaled = _scale_to_range(attack_strength, 0.3, 2.0, 0.5, 3.0)
                        defense_strength_scaled = _scale_to_range(defense_strength, 0.3, 2.0, 0.5, 3.0)
                        
                        # Store for this team
                        team_attack_strength[team] = attack_strength_scaled
                        team_defense_strength[team] = defense_strength_scaled
                        
                        # Track teams with ratings
                        teams_with_ratings.add(team)
                        
                        # Report progress for some teams
                        if match_idx_position % 100 == 0 and team in [home_team, away_team]:
                            print(f"  Match {match_idx_position+1}: {team} - Attack: {attack_strength_scaled:.2f}, Defense: {defense_strength_scaled:.2f}")
                    else:
                        # Not enough data, use neutral values in the 0.5-3 range
                        team_attack_strength[team] = 1.75  # Middle of 0.5-3 range
                        team_defense_strength[team] = 1.75  # Middle of 0.5-3 range
            
            # Set Maher model values for the current match
            if home_team in team_attack_strength and away_team in team_attack_strength:
                # Calculate overall strength as combination of attack and defense
                home_overall_strength = (team_attack_strength[home_team] + team_defense_strength[home_team]) / 2
                away_overall_strength = (team_attack_strength[away_team] + team_defense_strength[away_team]) / 2
                
                # Ensure overall strengths are in 0.5-3 range
                home_overall_strength = min(max(home_overall_strength, 0.5), 3.0)
                away_overall_strength = min(max(away_overall_strength, 0.5), 3.0)
                
                df.at[match_idx, 'home_team_strength'] = home_overall_strength
                df.at[match_idx, 'away_team_strength'] = away_overall_strength
                df.at[match_idx, 'home_advantage'] = home_advantage if home_advantage is not None else 1.1
        
        # Print summary for this season
        print(f"\nSeason {season} summary:")
        print(f"  Teams with ratings: {len(teams_with_ratings)}")
        print(f"  Home advantage: {home_advantage if home_advantage is not None else 1.1:.2f}")
    
    # Round all strength columns to 2 decimal places
    for col in strength_columns:
        df[col] = df[col].round(2)
    
    # Final safety check to ensure all values are in the 0.5-3 range
    for col in strength_columns:
        if col == 'home_advantage':
            df[col] = df[col].apply(lambda x: min(max(x, 0.5), 3.0))
        else:
            df[col] = df[col].apply(lambda x: min(max(x, 0.5), 3.0))
    
    # Save the enhanced dataframe
    df.to_excel(output_file, index=False)
    print(f"\nData with Maher model strength metrics saved to {output_file}")
    print(f"All values are scaled to be between 0.5-3")
    
    return df

# Example usage
df_with_strength = add_maher_strength(
    file_path='nfd_data.xlsx', 
    num_recent_games=6, 
    output_file='maher_0.5_3.xlsx'
)