import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran

# Define the UHF to NTA mapping (keeping only essential mappings)
UHF_TO_NTA = {
    'Astoria': ['Astoria'],
    'Bayside - Little Neck': ['Bayside-Bayside Hills', 'Douglas Manor-Douglaston-Little Neck'],
    'Bedford Stuyvesant - Crown Heights': ['Bedford', 'Crown Heights North', 'Crown Heights South'],
    'Bensonhurst - Bay Ridge': ['Bay Ridge', 'Bensonhurst East', 'Bensonhurst West'],
    'Borough Park': ['Borough Park'],
    'Canarsie - Flatlands': ['Canarsie', 'Flatlands'],
    'Central Harlem - Morningside Heights': ['Central Harlem North-Polo Grounds', 'Central Harlem South', 'Morningside Heights'],
    'Chelsea - Clinton': ['Clinton', 'Hudson Yards-Chelsea-Flatiron-Union Square'],
    'Coney Island - Sheepshead Bay': ['Brighton Beach', 'Coney Island', 'Homecrest', 'Manhattan Beach', 'Sheepshead Bay'],
    'Downtown - Heights - Slope': ['DUMBO-Vinegar Hill-Downtown Brooklyn-Boerum Hill', 'Brooklyn Heights-Cobble Hill', 'Park Slope-Gowanus'],
    'East Flatbush - Flatbush': ['East Flatbush-Farragut', 'Erasmus', 'Flatbush'],
    'East Harlem': ['East Harlem North', 'East Harlem South'],
    'East New York': ['East New York', 'East New York (Pennsylvania Ave)', 'Starrett City'],
    'Flushing - Clearview': ['East Flushing', 'Flushing', 'Murray Hill'],
    'Fordham - Bronx Park': ['Belmont', 'Bronx Park South', 'East Tremont', 'Fordham South', 'Van Nest-Morris Park-Westchester Square'],
    'Gramercy Park - Murray Hill': ['Murray Hill-Kips Bay', 'Turtle Bay-East Midtown'],
    'Greenwich Village - SoHo': ['SoHo-TriBeCa-Civic Center-Little Italy', 'West Village'],
    'Greenpoint': ['Greenpoint'],
    'High Bridge - Morrisania': ['Highbridge', 'Melrose South-Mott Haven North', 'Morrisania-Melrose'],
    'Hunts Point - Mott Haven': ['Hunts Point', 'Longwood', 'Mott Haven-Port Morris'],
    'Jamaica': ['Jamaica', 'South Jamaica', 'St. Albans'],
    'Kingsbridge - Riverdale': ['Kingsbridge-Riverdale', 'Spuyten Duyvil-Kingsbridge'],
    'Lower Manhattan': ['Battery Park City-Lower Manhattan', 'Chinatown'],
    'Northeast Bronx': ['Eastchester-Edenwald-Baychester', 'Williamsbridge-Olinville'],
    'Pelham - Throgs Neck': ['Pelham Bay-Country Club-City Island', 'Schuylerville-Throgs Neck-Edgewater Park'],
    'Port Richmond': ['Port Richmond', 'Stapleton-Rosebank'],
    'Ridgewood - Forest Hills': ['Forest Hills', 'Glendale', 'Ridgewood'],
    'Southwest Queens': ['Howard Beach', 'Woodhaven'],
    'Sunset Park': ['Sunset Park East', 'Sunset Park West'],
    'Upper East Side': ['Lenox Hill-Roosevelt Island', 'Upper East Side-Carnegie Hill', 'Yorkville'],
    'Upper West Side': ['Lincoln Square', 'Upper West Side'],
    'Washington Heights': ['Washington Heights North', 'Washington Heights South'],
    'West Queens': ['Elmhurst', 'Jackson Heights', 'North Corona', 'South Corona'],
    'Williamsburg - Bushwick': ['Bushwick', 'East Williamsburg', 'Williamsburg']
}

def calculate_morans_i(data, variable_name):
    """
    Calculate Moran's I statistic for spatial autocorrelation
    
    Parameters:
    data (pd.DataFrame): DataFrame containing UHF-level data
    variable_name (str): Name of the variable being analyzed (for printing)
    
    Returns:
    moran: Moran's I result object
    """
    # Filter for only UHF data and get mean values per location
    uhf_data = data[data['geo_type_name'].isin(['UHF34', 'UHF42'])]
    spatial_data = uhf_data.groupby('geo_place_name')['data_value'].mean().reset_index()
    
    # Load NYC geometry data
    nyc_geo = gpd.read_file('data/geojsons/2010 Neighborhood Tabulation Areas (NTAs).geojson')
    
    # Create expanded dataset matching UHF areas to NTA areas
    expanded_data = []
    for _, row in spatial_data.iterrows():
        if row['geo_place_name'] in UHF_TO_NTA:
            for nta_name in UHF_TO_NTA[row['geo_place_name']]:
                expanded_data.append({
                    'ntaname': nta_name,
                    'data_value': row['data_value']
                })
    
    # Convert to DataFrame and merge with geometry
    expanded_df = pd.DataFrame(expanded_data)
    gdf = nyc_geo.merge(expanded_df, on='ntaname', how='inner')
    
    # Create spatial weights matrix
    weights = Queen.from_dataframe(gdf)
    weights.transform = 'r'  # Row-standardize weights
    
    # Calculate Moran's I
    moran = Moran(gdf['data_value'], weights)
    
    # Print results
    print(f"\nMoran's I Analysis for {variable_name}:")
    print(f"Statistic: {moran.I:.3f}")
    print(f"P-value: {moran.p_sim:.3f}")
    print(f"Number of observations: {len(gdf)}")
    
    return moran

if __name__ == "__main__":
    # Load your data (adjust path as needed)
    fine_particulate_matter = pd.read_csv('data/grouped_data/Fine_particles__PM_2_5_.csv')
    
    # Calculate Moran's I for PM2.5
    moran_result = calculate_morans_i(fine_particulate_matter, "PM2.5")
    
