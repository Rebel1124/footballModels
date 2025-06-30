from appwrite.client import Client
from appwrite.services.storage import Storage
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Appwrite credentials
API_ENDPOINT = 'https://cloud.appwrite.io/v1'
PROJECT_ID = os.getenv('APPWRITE_PROJECT_ID')
API_KEY = os.getenv('APPWRITE_API_KEY')
STORAGE_ID = os.getenv('NFD_MODEL_STORAGE')

print(f"API_ENDPOINT: {API_ENDPOINT}")
print(f"PROJECT_ID: {PROJECT_ID[:5]}..." if PROJECT_ID else "PROJECT_ID: Not found")
print(f"STORAGE_ID: {STORAGE_ID}")
print(f"API_KEY available: {'Yes' if API_KEY else 'No'}")

def get_football_data(file_id="nfdStatsAllModels"):
    """
    Simplified function to get football data from Appwrite storage
    """
    try:
        # Initialize Appwrite client directly - no parameter passing
        client = Client()
        client.set_endpoint(API_ENDPOINT)
        client.set_project(PROJECT_ID)
        client.set_key(API_KEY)
        
        # Initialize storage service
        storage = Storage(client)
        
        print(f"Getting file '{file_id}' from bucket '{STORAGE_ID}'...")
        
        # Use direct method which worked in diagnostics
        file_content = storage.get_file_view(bucket_id=STORAGE_ID, file_id=file_id)
        print(f"File downloaded: {len(file_content)} bytes")
        
        # Parse JSON
        json_data = json.loads(file_content.decode('utf-8'))
        
        # Extract data
        if 'data' in json_data:
            # Create DataFrame
            df = pd.DataFrame(json_data['data'])
            print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
            return df
        else:
            print("Error: No 'data' field found in the JSON")
            return None
        
    except Exception as e:
        print(f"Error retrieving data: {e}")
        import traceback
        traceback.print_exc()
        return None

# Get the data
football_df = get_football_data()

# Test if data was retrieved successfully
if football_df is not None:
    print("\nData retrieved successfully!")
    print(f"DataFrame shape: {football_df.shape}")
    print("\nFirst row:")
    print(football_df.head(1))
else:
    print("\nFailed to retrieve data from Appwrite storage.")