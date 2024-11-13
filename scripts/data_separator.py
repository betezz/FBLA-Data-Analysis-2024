import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def group_by_name(df: pd.DataFrame) -> dict:
    # Group by name and create a dictionary of separate dataframes
    groups = {}
    for name, group in df.groupby('name'):
        groups[name] = group
    return groups

def save_grouped_data(type: str, grouped_dict: dict):
    # Create directory if it doesn't exist
    os.makedirs('data/grouped_data', exist_ok=True)
    
    # Save each group to a separate file
    for name, df in grouped_dict.items():
        # Clean filename by removing special characters
        clean_name = "".join(c if c.isalnum() else "_" for c in name)
        output_file = f"data/grouped_data/{clean_name}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved group '{name}' to {output_file}")

def main():
    # Load cleaned data
    data = pd.read_csv('data/cleaned_data.csv')
    
    # Group data
    grouped_data = group_by_name(data)
    
    # Save grouped data
    save_grouped_data("name", grouped_data)

if __name__ == "__main__":
    main()