import pandas as pd
import numpy as np
import datetime as dt

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # 1. Basic cleaning - be more selective about dropping
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # 2. Date handling
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    
    # 3. Numeric conversion - handle errors more gracefully
    df['data_value'] = pd.to_numeric(df['data_value'], errors='coerce')
    
    # 4. Create measure_unit without dropping rows
    if 'measure_info' in df.columns:
        df['measure_unit'] = df['measure_info'].fillna('').str.lower()
    
    # 5. Extract year - don't drop if extraction fails
    df['year'] = df['time_period'].str.extract(r'(\d{4})', expand=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # 6. Clean string columns without dropping
    if 'name' in df.columns:
        df['indicator_name'] = df['name'].str.strip()
    if 'geo_place_name' in df.columns:
        df['location'] = df['geo_place_name'].str.strip()
    
    # 7. Extract season without dropping
    df['season'] = df['time_period'].str.extract(r'(Summer|Winter)', expand=False).fillna('N/A')
    
    # 8. Drop only specified columns that exist
    columns_to_drop = ['message', 'measure_info']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
    
    # Print shape for debugging
    print(f"Cleaned dataframe shape: {df.shape}")
    
    return df

def save_cleaned_data(df: pd.DataFrame, output_file: str = "data/cleaned_data.csv"):
    # Add error handling and verification
    try:
        if df.empty:
            print("Warning: DataFrame is empty!")
            return
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Saved {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def clean_data_main():
    try:
        # Read data with error handling
        data = pd.read_csv('data/air-data.csv')
        print(f"Original data shape: {data.shape}")
        
        # Clean data
        cleaned_df = clean_data(data)
        
        # Save data
        save_cleaned_data(cleaned_df)
        
    except Exception as e:
        print(f"Error in clean_data_main: {str(e)}")