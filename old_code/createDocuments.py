import os
import pandas as pd
from tqdm import tqdm
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

# Previous Database and Models Update
# DATABASE_ID = os.getenv('APPWRITE_PREVIOUS_MODEL_DB_ID')
# COLLECTION_ID = os.getenv('NFD_MODEL_PREVIOUS_SEASONS_COLLECTION_ID')


# Current Database and Models Update
DATABASE_ID = os.getenv('APPWRITE_CURRENT_MODEL_DB_ID')
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
    # try:
    #     # Create database
    #     print(f"Creating database: {DATABASE_NAME}...")
    #     database = databases.create(
    #         database_id=DATABASE_ID,
    #         name=DATABASE_NAME
    #     )
    #     print(f"Database created with ID: {database['$id']}")
        
    # except AppwriteException as e:
    #     if 'duplicate' in str(e).lower():
    #         print(f"Database '{DATABASE_NAME}' already exists.")
    #     else:
    #         print(f"Error creating database: {e}")
    #         return False  # Exit if database creation fails
    
    # Wait a moment to ensure database is ready
    # import time
    # time.sleep(2)
    
    # try:
        # Create collection for NFD matches
        # print(f"Creating collection: {COLLECTION_NAME}...")
        # collection = databases.create_collection(
        #     database_id=DATABASE_ID,
        #     collection_id=COLLECTION_ID,
        #     name=COLLECTION_NAME,
        #     permissions=["read(\"any\")", "write(\"team:admin\")"]
        # )
        # print(f"Collection created with ID: {collection['$id']}")
        
    # except AppwriteException as e:
    #     if 'duplicate' in str(e).lower():
    #         print(f"Collection '{COLLECTION_NAME}' already exists.")
    #     else:
    #         print(f"Error creating collection: {e}")
    #         return False  # Exit if collection creation fails
    
    # # Wait a moment to ensure collection is ready
    # time.sleep(2)
    
    print("Creating attributes for the collection...")
    
    # Define attribute types based on pandas dtypes
    # These are the core attributes we need
    create_string_attribute(databases, "match_id", required=True)
    create_string_attribute(databases, "date")
    create_string_attribute(databases, "season")
    create_string_attribute(databases, "status")
    create_integer_attribute(databases, "homeID")
    create_string_attribute(databases, "home_team")
    create_integer_attribute(databases, "awayID")
    create_string_attribute(databases, "away_team")
    create_integer_attribute(databases, "winningTeam")
    create_string_attribute(databases, "winningTeamName")
    create_integer_attribute(databases, "result")
    create_string_attribute(databases, "ft_result")
    create_integer_attribute(databases, "home_goals")
    create_integer_attribute(databases, "away_goals")
    create_integer_attribute(databases, "goal_diff")
    
    # Odds attributes
    create_float_attribute(databases, "odds_ft_1")
    create_float_attribute(databases, "odds_ft_x")
    create_float_attribute(databases, "odds_ft_2")
    
    # Expected goals attributes
    create_float_attribute(databases, "home_xg_odds")
    create_float_attribute(databases, "away_xg_odds")
    
    # Team statistics
    create_integer_attribute(databases, "home_team_goals_scored_total")
    create_integer_attribute(databases, "home_team_goals_conceded_total")
    create_integer_attribute(databases, "away_team_goals_scored_total")
    create_integer_attribute(databases, "away_team_goals_conceded_total")
    create_float_attribute(databases, "home_team_goals_scored_average")
    create_float_attribute(databases, "home_team_goals_conceded_average")
    create_float_attribute(databases, "away_team_goals_scored_average")
    create_float_attribute(databases, "away_team_goals_conceded_average")
    
    # Advanced xG attributes
    create_float_attribute(databases, "home_xg")
    create_float_attribute(databases, "away_xg")
    create_boolean_attribute(databases, "rolling_stats_valid")
    create_float_attribute(databases, "home_xg_elo")
    create_float_attribute(databases, "away_xg_elo")
    create_float_attribute(databases, "home_xg_dc")
    create_float_attribute(databases, "away_xg_dc")
    create_float_attribute(databases, "home_xg_bt")
    create_float_attribute(databases, "away_xg_bt")
    create_float_attribute(databases, "home_xg_pyth")
    create_float_attribute(databases, "away_xg_pyth")
    create_float_attribute(databases, "home_xg_bayesian")
    create_float_attribute(databases, "away_xg_bayesian")
    create_float_attribute(databases, "home_xg_twr")
    create_float_attribute(databases, "away_xg_twr")
    
    # Create indexes for faster lookups
    print("Creating indexes...")
    
    try:
        # Primary index on match_id
        databases.create_index(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            key="match_id_index",  # using key instead of name
            type="key",
            attributes=["match_id"]
        )
        print("Created index on match_id")
        
        # Index for season and date lookups
        databases.create_index(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            key="season_date_index",  # using key instead of name
            type="key",
            attributes=["season", "date"]
        )
        print("Created index on season and date")
        
        # Index for team lookups
        databases.create_index(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            key="home_team_index",  # using key instead of name
            type="key",
            attributes=["homeID"]
        )
        print("Created index on homeID")
        
        databases.create_index(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            key="away_team_index",  # using key instead of name
            type="key",
            attributes=["awayID"]
        )
        print("Created index on awayID")
        
        print("Database setup complete!")
        return True
        
    except Exception as e:
        print(f"Error creating indexes: {e}")
        return False  # Exit if index creation fails

# Helper functions to create different attribute types
def create_string_attribute(databases, name, required=False, default=None, size=255):
    try:
        databases.create_string_attribute(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_ID,
            key=name,
            size=size,
            required=required,
            default=default
        )
        print(f"Created string attribute: {name}")
    except AppwriteException as e:
        if 'duplicate' in str(e).lower():
            print(f"Attribute '{name}' already exists.")
        else:
            print(f"Error creating attribute '{name}': {e}")

def create_integer_attribute(databases, name, required=False, default=None, min_value=None, max_value=None):
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
        print(f"Created integer attribute: {name}")
    except AppwriteException as e:
        if 'duplicate' in str(e).lower():
            print(f"Attribute '{name}' already exists.")
        else:
            print(f"Error creating attribute '{name}': {e}")

def create_float_attribute(databases, name, required=False, default=None, min_value=None, max_value=None):
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
        print(f"Created float attribute: {name}")
    except AppwriteException as e:
        if 'duplicate' in str(e).lower():
            print(f"Attribute '{name}' already exists.")
        else:
            print(f"Error creating attribute '{name}': {e}")

def create_boolean_attribute(databases, name, required=False, default=None):
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
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    # Convert date to string format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Convert rolling_stats_valid to boolean
    if 'rolling_stats_valid' in df.columns:
        df['rolling_stats_valid'] = df['rolling_stats_valid'].astype(bool)
    
    # Create a unique match_id
    df['match_id'] = df.apply(
        lambda row: f"{row['date']}_{row['homeID']}_{row['awayID']}", 
        axis=1
    )
    
    print(f"Loaded {len(df)} records from Excel file.")
    return df

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
    _, databases = initialize_appwrite()
    
    # Create a progress bar for the upload process
    progress_bar = tqdm(total=len(df), desc="Uploading data to Appwrite")
    
    # Track statistics
    stats = {"created": 0, "updated": 0, "error": 0}
    
    # Process each record in the dataframe
    for _, row in df.iterrows():
        # Convert row to dictionary and remove NaN values
        document_data = row.to_dict()
        
        # Clean up the data
        for key, value in document_data.copy().items():
            # Remove NaN, NaT, or None values
            if pd.isna(value) or value is None:
                # Replace with appropriate default values based on column type
                if isinstance(value, float):
                    document_data[key] = 0.0
                elif isinstance(value, int):
                    document_data[key] = 0
                elif key == 'rolling_stats_valid':
                    document_data[key] = False
                else:
                    document_data[key] = ""
        
        # Upload to Appwrite
        result, _ = upsert_document(databases, document_data)
        
        # Update statistics
        stats[result] += 1
        
        # Update progress bar
        progress_bar.update(1)
    
    # Close progress bar
    progress_bar.close()
    
    # Print summary
    print("\nUpload complete!")
    print(f"Documents created: {stats['created']}")
    print(f"Documents updated: {stats['updated']}")
    print(f"Errors: {stats['error']}")

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