def save_to_appwrite_storage(dataframe, bucket_id):
    """
    Save DataFrame directly as a file in Appwrite Storage
    Much faster than updating individual documents
    """
    start_time = time.time()
    
    # Make a shallow copy to avoid modifying the original
    df = dataframe.copy()
    print(f"Preparing to save {len(df)} records to storage")
    
    # Process data types just like in your original function
    # Process date columns
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Handle boolean columns
    if 'rolling_stats_valid' in df.columns:
        df['rolling_stats_valid'] = df['rolling_stats_valid'].fillna(False).astype(bool)
    
    # Process numeric columns to ensure proper JSON serialization
    integer_columns = [
        'homeID', 'awayID', 'winningTeam', 'result', 
        'home_goals', 'away_goals', 'goal_diff',
        'home_team_goals_scored_total', 
        'home_team_goals_conceded_total',
        'away_team_goals_scored_total', 
        'away_team_goals_conceded_total'
    ]
    
    float_columns = [
        'odds_ft_1', 'odds_ft_x', 'odds_ft_2',
        'home_xg_odds', 'away_xg_odds',
        'home_team_goals_scored_average', 'home_team_goals_conceded_average',
        'away_team_goals_scored_average', 'away_team_goals_conceded_average',
        'home_xg', 'away_xg', 'home_xg_elo', 'away_xg_elo',
        'home_xg_dc', 'away_xg_dc', 'home_xg_bt', 'away_xg_bt',
        'home_xg_pyth', 'away_xg_pyth', 'home_xg_bayesian', 'away_xg_bayesian',
        'home_xg_twr', 'away_xg_twr'
    ]
    
    # Clean up data for serialization
    for col in integer_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)
    
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"football_data_{timestamp}.json"
    
    # Convert DataFrame to JSON string
    # Use 'records' orientation for most compatible format
    json_str = df.to_json(orient='records', date_format='iso')
    
    # Create a file-like object
    from io import BytesIO
    file_obj = BytesIO(json_str.encode('utf-8'))
    
    try:
        # Upload to Appwrite Storage
        result = storage.create_file(
            bucket_id=bucket_id,
            file_id=f'unique()',  # Let Appwrite generate a unique ID
            file=file_obj,
            permissions=['read("role:all")']  # Set appropriate permissions
        )
        
        file_id = result['$id']
        
        # Also create a small metadata document for easy reference
        metadata = {
            'file_id': file_id,
            'file_name': file_name,
            'record_count': len(df),
            'created_at': timestamp,
            'columns': list(df.columns),
            'data_types': {
                'integer_columns': [col for col in integer_columns if col in df.columns],
                'float_columns': [col for col in float_columns if col in df.columns]
            }
        }
        
        # Store metadata in database for easy lookup
        try:
            databases.create_document(
                database_id=MODEL_DATABASE_ID,  # Your database ID
                collection_id='data_files',     # Create a collection for metadata
                document_id='unique()',
                data=metadata
            )
        except AppwriteException as e:
            print(f"Warning: Could not create metadata document: {str(e)[:100]}...")
            # Non-critical error, continue
        
        total_time = time.time() - start_time
        print(f"✅ Successfully saved {len(df)} records to storage in {total_time:.2f}s")
        print(f"File ID: {file_id}")
        
        return file_id
        
    except AppwriteException as e:
        total_time = time.time() - start_time
        print(f"❌ Error uploading to storage after {total_time:.2f}s: {str(e)}")
        return None
    



    def get_latest_data_from_appwrite():
    """Get the latest football data file from Appwrite storage"""
    try:
        # First check metadata to find the latest file
        response = databases.list_documents(
            database_id=MODEL_DATABASE_ID,
            collection_id='data_files',
            queries=[
                Query.orderDesc('created_at'),
                Query.limit(1)
            ]
        )
        
        if len(response['documents']) == 0:
            st.error("No data files found")
            return None
            
        # Get the file ID from metadata
        file_metadata = response['documents'][0]
        file_id = file_metadata['file_id']
        
        # Get file download URL
        file_url = storage.get_file_download(
            bucket_id=BUCKET_ID,
            file_id=file_id
        )
        
        # Download and process the file
        import requests
        import pandas as pd
        from io import StringIO
        
        # Get the file content
        r = requests.get(file_url)
        if r.status_code != 200:
            st.error(f"Failed to download file: {r.status_code}")
            return None
            
        # Parse JSON into DataFrame
        data = pd.read_json(StringIO(r.text), orient='records')
        
        # Show data info
        st.success(f"Loaded {len(data)} records from dataset updated on {file_metadata['created_at']}")
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None