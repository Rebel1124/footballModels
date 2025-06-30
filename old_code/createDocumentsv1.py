import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
from dotenv import load_dotenv
from appwrite.client import Client
from appwrite.id import ID
from appwrite.query import Query
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException

# Load environment variables
load_dotenv()

# Appwrite credentials from environment variables
API_ENDPOINT = 'https://cloud.appwrite.io/v1'
PROJECT_ID = os.getenv('APPWRITE_PROJECT_ID')
API_KEY = os.getenv('APPWRITE_API_KEY')

# Database and collection details
DATABASE_ID = os.getenv('APPWRITE_MODEL_DB_ID')

# Previous Seasons
# COLLECTION_ID = os.getenv('NFD_MODEL_PREVIOUS_SEASONS_COLLECTION_ID')

# Current Season
COLLECTION_ID = os.getenv('NFD_MODEL_CURRENT_SEASON_COLLECTION_ID')

# Excel file path
EXCEL_FILE = "nfd_data.xlsx"

# Function to initialize Appwrite client
def initialize_appwrite():
    client = Client()
    client.set_endpoint(API_ENDPOINT)
    client.set_project(PROJECT_ID)
    client.set_key(API_KEY)
    
    databases = Databases(client)
    return client, databases

# Function to create database and collection
def create_database_and_collection(databases):
    """
    Create and configure the database and collection with proper constraints
    """
    print("Setting up the database structure...")
    
    # Create attributes for the collection
    print("Creating attributes for the collection...")
    
    # Core attributes with better constraints
    create_string_attribute(databases, "match_id", required=True, size=100)
    create_string_attribute(databases, "date", size=10)  # YYYY-MM-DD format
    create_string_attribute(databases, "season", size=20)
    create_string_attribute(databases, "status", size=20)
    create_integer_attribute(databases, "homeID", min_value=0)
    create_string_attribute(databases, "home_team", size=100)
    create_integer_attribute(databases, "awayID", min_value=0)
    create_string_attribute(databases, "away_team", size=100)
    create_integer_attribute(databases, "winningTeam")
    create_string_attribute(databases, "winningTeamName", size=100)
    create_integer_attribute(databases, "result")
    create_string_attribute(databases, "ft_result", size=10)
    create_integer_attribute(databases, "home_goals", min_value=0)
    create_integer_attribute(databases, "away_goals", min_value=0)
    create_integer_attribute(databases, "goal_diff")
    
    # Odds attributes with sensible constraints
    create_float_attribute(databases, "odds_ft_1", min_value=1.0)
    create_float_attribute(databases, "odds_ft_x", min_value=1.0)
    create_float_attribute(databases, "odds_ft_2", min_value=1.0)
    
    # Expected goals attributes with sensible constraints
    create_float_attribute(databases, "home_xg_odds", min_value=0.0)
    create_float_attribute(databases, "away_xg_odds", min_value=0.0)
    
    # Team statistics with sensible constraints
    create_integer_attribute(databases, "home_team_goals_scored_total", min_value=0)
    create_integer_attribute(databases, "home_team_goals_conceded_total", min_value=0)
    create_integer_attribute(databases, "away_team_goals_scored_total", min_value=0)
    create_integer_attribute(databases, "away_team_goals_conceded_total", min_value=0)
    create_float_attribute(databases, "home_team_goals_scored_average", min_value=0.0)
    create_float_attribute(databases, "home_team_goals_conceded_average", min_value=0.0)
    create_float_attribute(databases, "away_team_goals_scored_average", min_value=0.0)
    create_float_attribute(databases, "away_team_goals_conceded_average", min_value=0.0)
    
    # Advanced xG attributes
    create_float_attribute(databases, "home_xg", min_value=0.0)
    create_float_attribute(databases, "away_xg", min_value=0.0)
    create_boolean_attribute(databases, "rolling_stats_valid")
    create_float_attribute(databases, "home_xg_elo", min_value=0.0)
    create_float_attribute(databases, "away_xg_elo", min_value=0.0)
    create_float_attribute(databases, "home_xg_dc", min_value=0.0)
    create_float_attribute(databases, "away_xg_dc", min_value=0.0)
    create_float_attribute(databases, "home_xg_bt", min_value=0.0)
    create_float_attribute(databases, "away_xg_bt", min_value=0.0)
    create_float_attribute(databases, "home_xg_pyth", min_value=0.0)
    create_float_attribute(databases, "away_xg_pyth", min_value=0.0)
    create_float_attribute(databases, "home_xg_bayesian", min_value=0.0)
    create_float_attribute(databases, "away_xg_bayesian", min_value=0.0)
    create_float_attribute(databases, "home_xg_twr", min_value=0.0)
    create_float_attribute(databases, "away_xg_twr", min_value=0.0)
    
    # Create better indexes with retry logic
    print("Creating indexes with proper constraints...")
    create_indexes(databases)
    
    print("Database setup complete!")
    return True

def create_indexes(databases, max_retries=3):
    """Create indexes with retry logic and proper constraints"""
    indexes_to_create = [
        {
            "key": "match_id_index",
            "type": "unique",  # Use unique constraint for match_id
            "attributes": ["match_id"],
            "description": "Primary unique index on match_id"
        },
        {
            "key": "season_date_index",
            "type": "key",
            "attributes": ["season", "date"],
            "description": "Index for season and date lookups"
        },
        {
            "key": "home_team_index",
            "type": "key",
            "attributes": ["homeID"],
            "description": "Index on homeID for team lookups"
        },
        {
            "key": "away_team_index",
            "type": "key",
            "attributes": ["awayID"],
            "description": "Index on awayID for team lookups"
        },
        {
            "key": "match_status_index",
            "type": "key", 
            "attributes": ["status"],
            "description": "Index on status for filtering"
        }
    ]
    
    for index_config in indexes_to_create:
        retry_count = 0
        success = False
        
        while not success and retry_count < max_retries:
            try:
                databases.create_index(
                    database_id=DATABASE_ID,
                    collection_id=COLLECTION_ID,
                    key=index_config["key"],
                    type=index_config["type"],
                    attributes=index_config["attributes"]
                )
                print(f"Created {index_config['type']} index: {index_config['key']} - {index_config['description']}")
                success = True
            except AppwriteException as e:
                retry_count += 1
                if 'duplicate' in str(e).lower():
                    print(f"Index '{index_config['key']}' already exists.")
                    success = True  # Consider this a success and move on
                elif retry_count >= max_retries:
                    print(f"Failed to create index '{index_config['key']}' after {max_retries} attempts: {e}")
                else:
                    print(f"Retrying index creation for '{index_config['key']}' (attempt {retry_count+1})")
                    time.sleep(1)  # Wait before retrying

# Improved attribute creation functions with better error handling
def create_string_attribute(databases, name, required=False, default=None, size=255):
    """Create string attribute with retry logic and better error handling"""
    try:
        databases.create_string_attribute(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            key=name,
            size=size,
            required=required,
            default=default
        )
        print(f"Created string attribute: {name} (size={size})")
    except AppwriteException as e:
        if 'duplicate' in str(e).lower():
            print(f"Attribute '{name}' already exists.")
        else:
            print(f"Error creating attribute '{name}': {e}")

def create_integer_attribute(databases, name, required=False, default=None, min_value=None, max_value=None):
    """Create integer attribute with improved validation"""
    try:
        kwargs = {
            "database_id": DATABASE_ID,
            "collection_id": COLLECTION_ID,
            "key": name,
            "required": required
        }
        
        if default is not None:
            kwargs["default"] = default
        
        if min_value is not None:
            kwargs["min"] = min_value
            
        if max_value is not None:
            kwargs["max"] = max_value
            
        databases.create_integer_attribute(**kwargs)
        
        # Log constraints for clarity
        constraints = []
        if min_value is not None:
            constraints.append(f"min={min_value}")
        if max_value is not None:
            constraints.append(f"max={max_value}")
        
        constraint_str = f" ({', '.join(constraints)})" if constraints else ""
        print(f"Created integer attribute: {name}{constraint_str}")
    except AppwriteException as e:
        if 'duplicate' in str(e).lower():
            print(f"Attribute '{name}' already exists.")
        else:
            print(f"Error creating attribute '{name}': {e}")

def create_float_attribute(databases, name, required=False, default=None, min_value=None, max_value=None):
    """Create float attribute with improved validation"""
    try:
        kwargs = {
            "database_id": DATABASE_ID,
            "collection_id": COLLECTION_ID,
            "key": name,
            "required": required
        }
        
        if default is not None:
            kwargs["default"] = default
        
        if min_value is not None:
            kwargs["min"] = min_value
            
        if max_value is not None:
            kwargs["max"] = max_value
            
        databases.create_float_attribute(**kwargs)
        
        # Log constraints for clarity
        constraints = []
        if min_value is not None:
            constraints.append(f"min={min_value}")
        if max_value is not None:
            constraints.append(f"max={max_value}")
        
        constraint_str = f" ({', '.join(constraints)})" if constraints else ""
        print(f"Created float attribute: {name}{constraint_str}")
    except AppwriteException as e:
        if 'duplicate' in str(e).lower():
            print(f"Attribute '{name}' already exists.")
        else:
            print(f"Error creating attribute '{name}': {e}")

def create_boolean_attribute(databases, name, required=False, default=None):
    """Create boolean attribute with improved error handling"""
    try:
        kwargs = {
            "database_id": DATABASE_ID,
            "collection_id": COLLECTION_ID,
            "key": name,
            "required": required
        }
        
        if default is not None:
            kwargs["default"] = default
            
        databases.create_boolean_attribute(**kwargs)
        print(f"Created boolean attribute: {name}")
    except AppwriteException as e:
        if 'duplicate' in str(e).lower():
            print(f"Attribute '{name}' already exists.")
        else:
            print(f"Error creating attribute '{name}': {e}")

# Function to load data from Excel file
def load_data_from_excel(file_path):
    """
    Load data from Excel file with improved validation and preprocessing
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_excel(file_path)
        original_count = len(df)
        
        # Validate data - remove rows with critical missing data
        if 'date' in df.columns and 'homeID' in df.columns and 'awayID' in df.columns:
            # Remove rows with missing critical data
            df = df.dropna(subset=['date', 'homeID', 'awayID'])
            if len(df) < original_count:
                print(f"WARNING: Removed {original_count - len(df)} rows with missing critical data (date, homeID, or awayID)")
        
        # Convert date to string format
        if 'date' in df.columns:
            # Handle potential date format issues
            try:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"WARNING: Error converting dates: {e}")
                # Fallback approach - try parsing as string
                df['date'] = df['date'].astype(str)
        
        # Convert team IDs to integers
        for id_col in ['homeID', 'awayID']:
            if id_col in df.columns:
                try:
                    df[id_col] = df[id_col].fillna(0).astype(int)
                except Exception as e:
                    print(f"WARNING: Error converting {id_col} to integer: {e}")
        
        # Convert rolling_stats_valid to boolean
        if 'rolling_stats_valid' in df.columns:
            df['rolling_stats_valid'] = df['rolling_stats_valid'].fillna(False).astype(bool)
        
        # Create a unique match_id
        if 'match_id' not in df.columns:
            try:
                df['match_id'] = df.apply(
                    lambda row: f"{row['date']}_{int(row['homeID'])}_{int(row['awayID'])}", 
                    axis=1
                )
            except Exception as e:
                print(f"WARNING: Error creating match_id: {e}")
                # Fallback - create IDs with string conversions
                df['match_id'] = df['date'].astype(str) + '_' + df['homeID'].astype(str) + '_' + df['awayID'].astype(str)
        
        # Check for duplicate match_ids
        duplicate_ids = df['match_id'].duplicated()
        if duplicate_ids.any():
            dup_count = duplicate_ids.sum()
            print(f"WARNING: Found {dup_count} duplicate match_ids in the source data")
            print("Keeping only the first occurrence of each match_id")
            df = df.drop_duplicates(subset=['match_id'], keep='first')
        
        print(f"Loaded {len(df)} valid records from Excel file.")
        return df
        
    except Exception as e:
        print(f"ERROR: Failed to load Excel file: {e}")
        return pd.DataFrame()  # Return empty dataframe on failure

# Function to check if a document exists
def document_exists(databases, match_id):
    try:
        # Query for documents with the given match_id
        response = databases.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            queries=[Query.equal("match_id", match_id)]
        )
        
        # If documents are found, return the first one's ID
        if response['total'] > 0:
            return response['documents'][0]['$id']
        return None
    except AppwriteException as e:
        print(f"Error checking if document exists: {e}")
        return None

# Function to update or create a document
def upsert_document(databases, document_data):
    try:
        match_id = document_data['match_id']
        
        # Check if document already exists
        doc_id = document_exists(databases, match_id)
        
        if doc_id:
            # Update existing document
            response = databases.update_document(
                database_id=DATABASE_ID,
                collection_id=COLLECTION_ID,
                document_id=doc_id,
                data=document_data
            )
            return "updated", response
        else:
            # Create new document
            response = databases.create_document(
                database_id=DATABASE_ID,
                collection_id=COLLECTION_ID,
                document_id=ID.unique(),
                data=document_data
            )
            return "created", response
            
    except AppwriteException as e:
        print(f"Error upserting document: {e}")
        return "error", None

# Main function to upload data to Appwrite
def upload_data_to_appwrite(df):
    """
    Upload data to Appwrite with improved batch processing and error handling
    """
    if df.empty:
        print("No valid data to upload. Aborting.")
        return False
        
    client, databases = initialize_appwrite()
    
    # Track statistics
    stats = {"created": 0, "updated": 0, "error": 0, "skipped": 0}
    
    # Use chunking for better memory management and progress tracking
    chunk_size = 50
    num_chunks = (len(df) + chunk_size - 1) // chunk_size  # Ceiling division
    
    print(f"Processing {len(df)} records in {num_chunks} chunks")
    
    # Process data in chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx]
        
        print(f"Processing chunk {chunk_idx+1}/{num_chunks} with {len(chunk_df)} records")
        
        # First, get existing match IDs for this chunk to determine create vs update
        match_ids = chunk_df['match_id'].tolist()
        
        try:
            # Query in batches to avoid query string length limits
            match_id_batch_size = 10
            existing_docs = {}
            
            for i in range(0, len(match_ids), match_id_batch_size):
                batch_ids = match_ids[i:i+match_id_batch_size]
                
                # Use an IN query for efficiency
                response = databases.list_documents(
                    database_id=DATABASE_ID,
                    collection_id=COLLECTION_ID,
                    queries=[
                        Query.equal("match_id", batch_ids),
                        Query.select(['$id', 'match_id'])
                    ]
                )
                
                for doc in response['documents']:
                    if 'match_id' in doc:
                        existing_docs[doc['match_id']] = doc['$id']
        except AppwriteException as e:
            print(f"Error checking existing documents: {e}")
            # Continue with partial information
            
        # Process each record in the chunk
        for _, row in chunk_df.iterrows():
            # Convert row to dictionary and clean data
            document_data = {}
            match_id = None
            
            for key, value in row.items():
                # Skip DataFrame index and other non-data columns
                if key == 'index' or key.startswith('Unnamed:'):
                    continue
                    
                # Store match_id for lookup
                if key == 'match_id':
                    match_id = value
                
                # Handle NaN values appropriately based on column type
                if pd.isna(value) or value is None:
                    # Use appropriate default values for different data types
                    if key in ['homeID', 'awayID', 'winningTeam', 'result', 
                              'home_goals', 'away_goals', 'goal_diff',
                              'home_team_goals_scored_total', 'home_team_goals_conceded_total',
                              'away_team_goals_scored_total', 'away_team_goals_conceded_total']:
                        document_data[key] = 0
                    elif key in ['odds_ft_1', 'odds_ft_x', 'odds_ft_2',
                                'home_xg_odds', 'away_xg_odds',
                                'home_team_goals_scored_average', 'home_team_goals_conceded_average',
                                'away_team_goals_scored_average', 'away_team_goals_conceded_average',
                                'home_xg', 'away_xg', 'home_xg_elo', 'away_xg_elo']:
                        document_data[key] = 0.0
                    elif key == 'rolling_stats_valid':
                        document_data[key] = False
                    else:
                        document_data[key] = ""
                else:
                    # Convert numpy/pandas types to native Python types
                    if isinstance(value, (np.integer, np.int64)):
                        document_data[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        document_data[key] = float(value)
                    elif isinstance(value, (np.bool_, bool)):
                        document_data[key] = bool(value)
                    else:
                        document_data[key] = value
            
            # Skip if no match_id
            if not match_id:
                stats["skipped"] += 1
                continue
            
            # Check if document exists and perform appropriate operation
            if match_id in existing_docs:
                # Update existing document
                try:
                    databases.update_document(
                        database_id=DATABASE_ID,
                        collection_id=COLLECTION_ID,
                        document_id=existing_docs[match_id],
                        data=document_data
                    )
                    stats["updated"] += 1
                except AppwriteException as e:
                    print(f"Error updating document {match_id}: {e}")
                    stats["error"] += 1
            else:
                # Create new document
                try:
                    databases.create_document(
                        database_id=DATABASE_ID,
                        collection_id=COLLECTION_ID,
                        document_id=ID.unique(),
                        data=document_data
                    )
                    stats["created"] += 1
                except AppwriteException as e:
                    if 'duplicate' in str(e).lower():
                        # Document was created between our check and now
                        # Try to get the ID and update instead
                        try:
                            find_query = [Query.equal("match_id", match_id)]
                            existing_doc = databases.list_documents(
                                database_id=DATABASE_ID,
                                collection_id=COLLECTION_ID,
                                queries=find_query
                            )
                            
                            if existing_doc['total'] > 0:
                                doc_id = existing_doc['documents'][0]['$id']
                                databases.update_document(
                                    database_id=DATABASE_ID,
                                    collection_id=COLLECTION_ID,
                                    document_id=doc_id,
                                    data=document_data
                                )
                                stats["updated"] += 1
                            else:
                                stats["error"] += 1
                        except AppwriteException:
                            stats["error"] += 1
                    else:
                        print(f"Error creating document {match_id}: {e}")
                        stats["error"] += 1
        
        # Print progress after each chunk
        print(f"Chunk {chunk_idx+1} complete. Running totals: Created: {stats['created']}, Updated: {stats['updated']}, Errors: {stats['error']}, Skipped: {stats['skipped']}")
    
    # Print summary
    print("\nUpload complete!")
    print(f"Documents created: {stats['created']}")
    print(f"Documents updated: {stats['updated']}")
    print(f"Errors: {stats['error']}")
    print(f"Skipped: {stats['skipped']}")
    
    return stats["created"] + stats["updated"] > 0  # Return success if any documents were processed

def main():
    # Initialize Appwrite client
    client, databases = initialize_appwrite()
    
    # Create database and collection if needed
    setup_successful = create_database_and_collection(databases)
    
    if not setup_successful:
        print("Database setup failed. Exiting...")
        return
    
    # Load data from Excel
    df = load_data_from_excel(EXCEL_FILE)
    
    # Upload data to Appwrite
    upload_data_to_appwrite(df)

main()

# if __name__ == "__main__":
#     main()