import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from appwrite.query import Query
from appwrite.client import Client
from appwrite.services.databases import Databases

load_dotenv()  # Note: Added parentheses here

# Replace these with your actual Appwrite credentials
API_ENDPOINT = 'https://cloud.appwrite.io/v1'
PROJECT_ID = os.getenv('APPWRITE_PROJECT_ID')
API_KEY = os.getenv('APPWRITE_API_KEY')
DATABASE_ID = os.getenv('APPWRITE_DB_ID')

# List of collection IDs to retrieve data from
COLLECTION_IDS = [
    os.getenv('SEASON_MATCHES_PSL20_21'),
    os.getenv('SEASON_MATCHES_PSL21_22'),  
    os.getenv('SEASON_MATCHES_PSL22_23'),
    os.getenv('SEASON_MATCHES_PSL23_24'),
    os.getenv('SEASON_MATCHES_PSL24_25'),  
]

OUTPUT_FILENAME = "all_psl_data.xlsx"

# Initialize Appwrite client
client = Client()
client.set_endpoint(API_ENDPOINT)
client.set_project(PROJECT_ID)
client.set_key(API_KEY)

# Initialize Databases service
databases = Databases(client)

# Function to retrieve all documents from a collection
def get_collection_documents(collection_id):
    all_documents = []
    limit = 600
    offset = 0
    
    while True:
        try:
            # Set up query options for offset-based pagination
            query_options = [
                Query.limit(limit),
                Query.offset(offset)
            ]
            
            # Get a batch of documents
            response = databases.list_documents(
                database_id=DATABASE_ID,
                collection_id=collection_id,
                queries=query_options
            )
            
            documents = response['documents']
            
            # If no documents were returned, we've reached the end
            if not documents:
                break
                
            # Add documents to our list
            all_documents.extend(documents)
            
            print(f"Retrieved {len(all_documents)} documents from collection {collection_id} so far...")
            
            # If we got fewer documents than the limit, we've reached the end
            if len(documents) < limit:
                break
                
            # Update offset for the next page
            offset += limit
            
        except Exception as e:
            print(f"Error retrieving documents from collection {collection_id}: {str(e)}")
            break
    
    return all_documents

# Create an empty list to store dataframes from each collection
all_dataframes = []

# Process each collection and create a dataframe
for collection_id in COLLECTION_IDS:
    if collection_id:  # Skip if collection ID is None or empty
        print(f"\nProcessing collection: {collection_id}")
        documents = get_collection_documents(collection_id)
        
        if documents:
            # Create a DataFrame for this collection
            df = pd.DataFrame(documents)
            
            # Add a column to identify which collection this data came from
            df['source_collection'] = collection_id
            
            # Add this dataframe to our list
            all_dataframes.append(df)
            print(f"Added {len(df)} rows from collection {collection_id}")
        else:
            print(f"No documents found in collection {collection_id}")

# Combine all dataframes into one
if all_dataframes:
    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\nCombined DataFrame has {len(combined_df)} rows")
    
    # Replace -1 with 0 only in numeric columns
    numeric_columns = combined_df.select_dtypes(include=['number']).columns
    combined_df[numeric_columns] = combined_df[numeric_columns].replace(-1, 0)
    
    # Create a copy of the DataFrame to defragment it
    combined_df = combined_df.copy()
    
    # Create a date column if date_unix exists
    if 'date_unix' in combined_df.columns:
        # Process all dates at once using vectorized operations
        combined_df['date_unix'] = pd.to_numeric(combined_df['date_unix'], errors='coerce')
        
        # Create a datetime column for sorting
        combined_df['datetime_temp'] = pd.to_datetime(combined_df['date_unix'], unit='s')
        
        # Create a formatted date string column
        combined_df['match_date'] = combined_df['datetime_temp'].dt.strftime('%Y-%m-%d')
        
        # Sort by the datetime column from earliest to latest
        combined_df = combined_df.sort_values(by='datetime_temp')
        
        # Drop the temporary datetime column (optional)
        combined_df = combined_df.drop(columns=['datetime_temp'])
    
    # Save to Excel
    combined_df.to_excel(OUTPUT_FILENAME, index=False)
    print(f"\nData saved to {OUTPUT_FILENAME}")
    
    # Print first few rows
    print("\nFirst 5 rows of the combined data:")
    print(combined_df.head())
else:
    print("No data found in any of the collections")