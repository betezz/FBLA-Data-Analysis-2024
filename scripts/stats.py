import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import main_data
import libpysal
from libpysal.weights import Queen
from esda.moran import Moran
import geopandas as gpd

# Create visualizations directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Define the UHF to NTA mapping
UHF_TO_NTA = {
    'Astoria': ['Astoria'],
    'Battery Park/Tribeca': ['Battery Park City', 'Tribeca'],
    'Bay Ridge': ['Bay Ridge'],
    'Bayside': ['Bayside-Bayside Hills'],
    'Bedford Stuyvesant': ['Bedford'],
    'Bensonhurst': ['Bensonhurst East', 'Bensonhurst West'],
    'Borough Park': ['Borough Park'],
    'Canarsie': ['Canarsie'],
    'Central Harlem': ['Central Harlem North-Polo Grounds', 'Central Harlem South'],
    'Chelsea/Clinton': ['Chelsea - Clinton', 'Chelsea-Village'],
    'Coney Island': ['Coney Island - Sheepshead Bay'],
    'Crown Heights': ['Crown Heights North', 'Crown Heights South'],
    'East Flatbush': ['East Flatbush - Flatbush'],
    'East Harlem': ['East Harlem'],
    'East New York': ['East New York'],
    'Brownsville': ['Brownsville'],
    'Bushwick': ['Bushwick'],
    'Concourse/Highbridge': ['Concourse/Highbridge'],
    'Elmhurst/Corona': ['Elmhurst/Corona'],
    'Flatbush/Midwood': ['Flatbush/Midwood'],
    'Flushing': ['Flushing'],
    'Fort Greene/Brooklyn Hts': ['Fort Greene/Brooklyn Hts'],
    'Fresh Meadows/Briarwood': ['Fresh Meadows/Briarwood'],
    'Greenwich Village': ['Greenwich Village'],
    'Howard Beach': ['Howard Beach'],
    'Hunts Point': ['Hunts Point'],
    'Jackson Heights': ['Jackson Heights'],
    'Jamaica/St. Albans': ['Jamaica/St. Albans'],
    'Lower East Side': ['Lower East Side'],
    'Manhattanville': ['Manhattanville'],
    'Midtown Business District': ['Midtown Business District'],
    'Morrisania': ['Morrisania'],
    'Mott Haven': ['Mott Haven'],
    'Murray Hill/Stuyvesant': ['Murray Hill/Stuyvesant'],
    'Park Slope': ['Park Slope'],
    'Pelham Parkway': ['Pelham Parkway'],
    'Queens Village': ['Queens Village'],
    'Rego Park/Forest Hills': ['Rego Park/Forest Hills'],
    'Ridgewood/Glendale': ['Ridgewood/Glendale'],
    'Riverdale': ['Riverdale'],
    'Sheepshead Bay': ['Sheepshead Bay'],
    'South Beach': ['South Beach'],
    'St. George': ['St. George'],
    'Sunnyside/Woodside': ['Sunnyside/Woodside'],
    'Sunset Park': ['Sunset Park'],
    'The Rockaways': ['The Rockaways'],
    'Throgs Neck': ['Throgs Neck'],
    'Tottenville': ['Tottenville'],
    'Unionport/Soundview': ['Unionport/Soundview'],
    'University Heights': ['University Heights'],
    'Upper East Side': ['Upper East Side'],
    'Upper West Side': ['Upper West Side'],
    'Washington Heights': ['Washington Heights'],
    'Williamsbridge': ['Williamsbridge'],
    'Williamsburg/Greenpoint': ['Williamsburg/Greenpoint'],
    'Woodhaven': ['Woodhaven'],
}

# Load cleaned data
fine_particulate_matter = main_data.fine_particulate_matter
nitrogen_dioxide = main_data.nitrogen_dioxide  
ozone = main_data.ozone
benzene = main_data.benzene
formaldehyde = main_data.formaldehyde

boiler_nox = main_data.boiler_nox
boiler_pm25 = main_data.boiler_pm25
boiler_so2 = main_data.boiler_so2

vehicle_miles = main_data.vehicle_miles
car_miles = main_data.car_miles
truck_miles = main_data.truck_miles

deaths_pm25 = main_data.deaths_pm25
cardiac_deaths_ozone = main_data.cardiac_deaths_ozone
cardio_hosp_pm25 = main_data.cardio_hosp_pm25
resp_hosp_pm25 = main_data.resp_hosp_pm25

asthma_ed_pm25 = main_data.asthma_ed_pm25
asthma_ed_ozone = main_data.asthma_ed_ozone
asthma_hosp_ozone = main_data.asthma_hosp_ozone

# After loading poverty_data, clean it:
poverty_data = pd.read_csv('/Users/morabp27/FBLA-Data-Analysis-2024/data/grouped_data/poverty_rates.csv')
poverty_data['data_value'] = poverty_data['data_value'].str.rstrip('%').astype(float)

def calculate_morans_i(data):
    # Filter for only UHF data
    uhf_data = data[data['geo_type_name'].isin(['UHF34', 'UHF42'])]
    
    # First, ensure we have unique locations (average values if multiple readings per location)
    spatial_data = uhf_data.groupby('geo_place_name')['data_value'].mean().reset_index()
    
    # Load NYC geometry data
    nyc_geo = gpd.read_file('/Users/morabp27/FBLA-Data-Analysis-2024/data/geojsons/2010 Neighborhood Tabulation Areas (NTAs).geojson')
    
    # Expand spatial_data to include all matching NTA areas
    expanded_data = []
    for _, row in spatial_data.iterrows():
        if row['geo_place_name'] in UHF_TO_NTA:
            for nta_name in UHF_TO_NTA[row['geo_place_name']]:
                expanded_data.append({
                    'geo_place_name': row['geo_place_name'],
                    'nta_name': nta_name,
                    'data_value': row['data_value']
                })
    
    expanded_df = pd.DataFrame(expanded_data)
    
    # Merge with geometry
    gdf = nyc_geo.merge(expanded_df, left_on='ntaname', right_on='nta_name')
    
    # Create spatial weights matrix (Queen contiguity)
    weights = Queen.from_dataframe(gdf)
    weights.transform = 'r'  # Row-standardize the weights
    
    # Calculate Moran's I
    moran = Moran(gdf['data_value'], weights)
    
    print(f"Moran's I: {moran.I:.3f}")
    print(f"P-value: {moran.p_sim:.3f}")
    
    return moran

# Test it with fine particulate matter
moran_result = calculate_morans_i(fine_particulate_matter)

# Add this code temporarily to see what neighborhoods we need to map
print("UHF Neighborhoods in our data:")
uhf_neighborhoods = fine_particulate_matter[
    fine_particulate_matter['geo_type_name'].isin(['UHF34', 'UHF42'])
]['geo_place_name'].unique()
print(sorted(uhf_neighborhoods))

print("\nNTA Neighborhoods in geometry file:")
nyc_geo = gpd.read_file('/Users/morabp27/FBLA-Data-Analysis-2024/data/geojsons/2010 Neighborhood Tabulation Areas (NTAs).geojson')
print(sorted(nyc_geo['ntaname'].unique()))

def plot_triple_choropleth():
    """
    Creates three side-by-side choropleth maps showing:
    1. Air Quality (PM2.5)
    2. Health Outcomes (Asthma ED visits)
    3. Poverty Rates
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot each map
    plot_choropleth(fine_particulate_matter, 'PM2.5 Levels', ax=ax1)
    plot_choropleth(asthma_ed_pm25, 'Asthma ED Visits', ax=ax2)
    plot_choropleth(poverty_data, 'Poverty Rates (%)', ax=ax3)
    
    plt.suptitle('Environmental Justice: Air Quality, Health, and Poverty in NYC')
    plt.tight_layout()
    return fig

def plot_bubble_chart():
    """
    Creates a bubble plot where:
    - X-axis: Poverty Rate
    - Y-axis: Asthma ED Visits
    - Bubble size: PM2.5 Levels
    - Color: Borough
    """
    # Prepare data
    merged_data = pd.DataFrame({
        'poverty': poverty_data.groupby('geo_place_name')['data_value'].mean(),
        'health': asthma_ed_pm25.groupby('geo_place_name')['data_value'].mean(),
        'air': fine_particulate_matter.groupby('geo_place_name')['data_value'].mean()
    }).reset_index()
    
    # Create bubble plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = plt.scatter(merged_data['poverty'], 
                         merged_data['health'],
                         s=merged_data['air']*100,  # Scale bubble size
                         alpha=0.6)
    
    plt.xlabel('Poverty Rate (%)')
    plt.ylabel('Asthma ED Visits (per 10,000)')
    plt.title('Relationship between Poverty, Health, and Air Quality')
    
    # Add labels for notable points
    for idx, row in merged_data.iterrows():
        if row['poverty'] > 20 or row['health'] > 200:  # Adjust thresholds as needed
            plt.annotate(row['geo_place_name'], 
                        (row['poverty'], row['health']))
    
    return fig

def plot_neighborhood_comparison():
    """
    Creates a stacked bar chart comparing multiple metrics across neighborhoods
    """
    # Normalize all metrics to 0-1 scale for comparison
    metrics = {
        'Poverty Rate': poverty_data,
        'PM2.5 Levels': fine_particulate_matter,
        'Asthma ED Visits': asthma_ed_pm25
    }
    
    normalized_data = {}
    for name, data in metrics.items():
        values = data.groupby('geo_place_name')['data_value'].mean()
        normalized_data[name] = (values - values.min()) / (values.max() - values.min())
    
    # Create stacked bar chart
    df = pd.DataFrame(normalized_data)
    ax = df.plot(kind='bar', stacked=True, figsize=(15, 6))
    
    plt.title('Normalized Comparison of Environmental Justice Metrics')
    plt.xlabel('Neighborhood')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt.gcf()

# ... existing code ...

def plot_choropleth(data, title, ax=None):
    """
    Creates a choropleth map of NYC showing the data values by neighborhood.
    """
    # Load geometry data
    nyc_geo = gpd.read_file('/Users/morabp27/FBLA-Data-Analysis-2024/data/geojsons/2010 Neighborhood Tabulation Areas (NTAs).geojson')
    
    if 'geo_type_name' not in data.columns:
        # Assuming poverty data is already at neighborhood level
        expanded_df = pd.DataFrame({
            'ntaname': data['geo_place_name'],
            'data_value': data['data_value']
        })
    else:
        # Handle UHF data conversion
        uhf_data = data[data['geo_type_name'].isin(['UHF34', 'UHF42'])].copy()
        
        # Debug print
        print("\nDebug - UHF Data:")
        print(uhf_data.head())
        
        # Extract base UHF name by removing parentheses and their contents
        uhf_data['base_geo_place_name'] = uhf_data['geo_place_name'].apply(lambda x: x.split('(')[0].strip())
        
        # Debug print
        print("\nDebug - Base UHF Place Names:")
        print(uhf_data['base_geo_place_name'].unique())
        
        # Expand data to NTA level using existing UHF_TO_NTA mapping
        expanded_data = []
        for _, row in uhf_data.groupby('base_geo_place_name')['data_value'].mean().reset_index().iterrows():
            base_uhf_name = row['base_geo_place_name']
            if base_uhf_name in UHF_TO_NTA:
                for nta_name in UHF_TO_NTA[base_uhf_name]:
                    expanded_data.append({
                        'ntaname': nta_name,
                        'data_value': row['data_value']
                    })
            else:
                print(f"Warning: UHF name '{base_uhf_name}' not found in UHF_TO_NTA mapping.")
        
        expanded_df = pd.DataFrame(expanded_data)
        
        # Debug print
        print("\nDebug - Expanded data:")
        print(expanded_df.head())
        print(f"Debug - Expanded data shape: {expanded_df.shape}")
    
    # Ensure 'ntaname' exists in both DataFrames
    if 'ntaname' not in nyc_geo.columns:
        print("Error: 'ntaname' column not found in nyc_geo DataFrame.")
        return None
    if 'ntaname' not in expanded_df.columns:
        print("Error: 'ntaname' column not found in expanded_df DataFrame.")
        return None
    
    # Merge with geometry
    if expanded_df.empty:
        print("Error: expanded_df is empty. Cannot perform merge.")
        return None
    else:
        gdf = nyc_geo.merge(expanded_df, on='ntaname', how='left')
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(15, 10))
    
    gdf.plot(
        column='data_value', 
        ax=ax,
        legend=True,
        legend_kwds={'label': title},
        missing_kwds={'color': 'lightgrey'},
        cmap='YlOrRd'
    )
    
    ax.set_title(title)
    ax.axis('off')
    
    if ax is None:
        plt.tight_layout()
        return fig
    return ax

def plot_time_series(data, neighborhoods=None):
    """
    Creates a time series plot showing trends over time for selected neighborhoods
    """
    if neighborhoods is None:
        # Take top 5 neighborhoods by average value
        neighborhoods = (data.groupby('geo_place_name')['data_value']
                       .mean()
                       .sort_values(ascending=False)
                       .head(5)
                       .index)
    
    # Filter data for selected neighborhoods
    mask = data['geo_place_name'].isin(neighborhoods)
    plot_data = data[mask].copy()
    
    # Convert start_date to datetime
    plot_data['start_date'] = pd.to_datetime(plot_data['start_date'])
    
    # Create the plot
    fig, ax = plt.subplots(1, figsize=(15, 8))
    
    for name in neighborhoods:
        neighborhood_data = plot_data[plot_data['geo_place_name'] == name]
        ax.plot(neighborhood_data['start_date'], 
                neighborhood_data['data_value'],
                label=name,
                marker='o')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Time Series by Neighborhood')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_boxplot(data):
    """
    Creates a boxplot showing the distribution of values by neighborhood
    """
    # Prepare data
    plot_data = data[data['geo_type_name'].isin(['UHF34', 'UHF42'])].copy()
    
    # Create the plot
    fig, ax = plt.subplots(1, figsize=(15, 8))
    sns.boxplot(data=plot_data, 
                x='geo_place_name', 
                y='data_value',
                ax=ax)
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Neighborhood')
    plt.ylabel('Value')
    plt.title('Distribution of Values by Neighborhood')
    plt.tight_layout()
    return fig

def plot_health_choropleth(data, title):
    """
    Similar to plot_choropleth but specifically for health outcomes
    """
    return plot_choropleth(data, title)

def calculate_correlation_matrix(air_quality, health_data, poverty_data):
    """
    Calculates and visualizes correlations between air quality, health outcomes, and poverty
    """
    # Get average values by neighborhood for each dataset
    air_by_neighborhood = air_quality.groupby('geo_place_name')['data_value'].mean()
    health_by_neighborhood = health_data.groupby('geo_place_name')['data_value'].mean()
    poverty_by_neighborhood = poverty_data.groupby('geo_place_name')['data_value'].mean()
    
    # Merge the datasets
    correlation_df = pd.DataFrame({
        'PM2.5 Levels (μg/m³)': air_by_neighborhood,
        'Asthma ED Visits (per 10,000)': health_by_neighborhood,
        'Poverty Rate (%)': poverty_by_neighborhood
    })
    
    # Calculate correlation matrix
    corr_matrix = correlation_df.corr()
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdYlBu', 
                center=0,
                ax=ax)
    
    plt.title('Correlation between Air Quality, Health Outcomes, and Poverty')
    plt.tight_layout()
    return fig

def plot_scatter_matrix(air_quality, health_data, poverty_data):
    """
    Creates a scatter plot matrix showing relationships between variables
    """
    # Prepare data similar to correlation matrix
    air_by_neighborhood = air_quality.groupby('geo_place_name')['data_value'].mean()
    health_by_neighborhood = health_data.groupby('geo_place_name')['data_value'].mean()
    poverty_by_neighborhood = poverty_data.groupby('geo_place_name')['data_value'].mean()
    
    data_combined = pd.DataFrame({
        'Air Quality (PM2.5)': air_by_neighborhood,
        'Health Issues': health_by_neighborhood,
        'Poverty Rate': poverty_by_neighborhood
    })
    
    # Create scatter matrix
    fig = sns.pairplot(data_combined, 
                      diag_kind='kde',  # Kernel density plots on diagonal
                      plot_kws={'alpha': 0.6})
    
    fig.fig.suptitle('Relationships between Air Quality, Health, and Poverty', y=1.02)
    return fig

# Add this debug code temporarily
print("\nPoverty data neighborhoods:")
print(sorted(poverty_data['geo_place_name'].unique()))

print("\nGeometry file neighborhoods:")
print(sorted(nyc_geo['ntaname'].unique()))

# Example usage:
if __name__ == "__main__":
    # Create visualizations directory
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Generate all visualizations
    triple_map = plot_triple_choropleth()
    triple_map.savefig('visualizations/triple_choropleth.png')
    
    bubble = plot_bubble_chart()
    bubble.savefig('visualizations/bubble_chart.png')
    
    comparison = plot_neighborhood_comparison()
    comparison.savefig('visualizations/neighborhood_comparison.png')
    
