import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV files
root = '/Users/morabp27/FBLA-Data-Analysis-2024/'

asthma_ozone = pd.read_csv(root + 'data/sitedata/NYC EH Data Portal - Asthma hospitalizations due to Ozone (full table).csv')
deaths_pm25 = pd.read_csv(root + 'data/sitedata/NYC EH Data Portal - Deaths due to PM2.5 (full table).csv')
cardio_deaths_ozone = pd.read_csv(root + 'data/sitedata/NYC EH Data Portal - Cardiac and respiratory deaths due to Ozone (full table).csv')
resp_hosp_pm25 = pd.read_csv(root + 'data/sitedata/NYC EH Data Portal - Respiratory hospitalizations due to PM2.5 (age 20+) (full table).csv')
cardio_hosp_pm25 = pd.read_csv(root + 'data/sitedata/NYC EH Data Portal - Cardiovascular hospitalizations due to PM2.5 (age 40+) (full table).csv')
poverty = pd.read_csv(root + 'data/sitedata/NYC EH Data Portal - Neighborhood poverty (full table).csv')
pm25_pollution = pd.read_csv(root + 'data/sitedata/NYC EH Data Portal - Fine particles (PM 2.5) (full table).csv')
# Filter for most recent time period (2017-2019) and UHF42 geography type
def filter_recent_uhf(df):
    return df[(df['TimePeriod'] == '2017-2019') & (df['GeoType'] == 'UHF42')]

# For poverty, we want the percent column
poverty_filtered = poverty[
    (poverty['TimePeriod'] == '2015-19') &  # Using 2015-19 period as it's most recent
    (poverty['GeoType'] == 'UHF42')
][['Geography', 'Percent']]

# Create dataframes with just the columns we need
metrics = {
    'asthma': filter_recent_uhf(asthma_ozone)[['Geography', 'Estimated annual rate (age 18+) per 100,000 adults']],
    'deaths_pm25': filter_recent_uhf(deaths_pm25)[['Geography', 'Estimated annual rate (age 30+) per 100,000 adults']],
    'cardio_deaths': filter_recent_uhf(cardio_deaths_ozone)[['Geography', 'Estimated annual rate per 100,000']],
    'resp_hosp': filter_recent_uhf(resp_hosp_pm25)[['Geography', 'Estimated annual rate per 100,000 adults']].rename(
        columns={'Estimated annual rate per 100,000 adults': 'resp_hosp_rate'}),
    'cardio_hosp': filter_recent_uhf(cardio_hosp_pm25)[['Geography', 'Estimated annual rate per 100,000 adults']].rename(
        columns={'Estimated annual rate per 100,000 adults': 'cardio_hosp_rate'}),
    'poverty': poverty_filtered
}

# Filter PM2.5 data for most recent time period and UHF42 geography
pm25_filtered = pm25_pollution[
    (pm25_pollution['TimePeriod'] == 'Annual Average 2019') & 
    (pm25_pollution['GeoType'] == 'UHF42')
][['Geography', 'Mean mcg/m3']]

# Add PM2.5 data to metrics dictionary
metrics['pm25'] = pm25_filtered.rename(columns={'Mean mcg/m3': 'pm25_level'})

# Merge all metrics on Geography
combined = metrics['asthma']
for name, df in metrics.items():
    if name != 'asthma':
        combined = combined.merge(df, on='Geography', how='inner')  # Use inner join

# Rename columns for clarity
combined = combined.rename(columns={
    'Estimated annual rate (age 18+) per 100,000 adults': 'asthma_rate',
    'Estimated annual rate (age 30+) per 100,000 adults': 'death_rate_pm25',
    'Estimated annual rate per 100,000': 'cardio_death_rate',
    'Percent': 'poverty_rate'
})

# Define weights for each health metric based on severity and impact
HEALTH_WEIGHTS = {
    'death_rate_pm25': 0.25,      # Deaths weighted highest
    'cardio_death_rate': 0.25,    # Cardiovascular deaths also heavily weighted
    'resp_hosp_rate': 0.20,       # Hospitalizations weighted medium
    'cardio_hosp_rate': 0.20,     # Cardiovascular hospitalizations weighted medium
    'asthma_rate': 0.10           # Asthma weighted lower as it's generally less severe
}

def calculate_health_score(df):
    """
    Calculate a weighted health score using min-max normalization and proper weighting.
    
    Parameters:
    df (DataFrame): DataFrame containing health metrics
    
    Returns:
    Series: Normalized health scores (0-100 scale, higher = better health)
    """
    # Create copy to avoid modifying original
    scores = pd.DataFrame()
    
    # Step 1: Min-max normalize each metric (0-1 scale)
    for metric in HEALTH_WEIGHTS.keys():
        # Invert the normalization since higher rates = worse health
        min_val = df[metric].min()
        max_val = df[metric].max()
        scores[f'{metric}_norm'] = 1 - ((df[metric] - min_val) / (max_val - min_val))
    
    # Step 2: Apply weights and combine
    weighted_score = 0
    for metric in HEALTH_WEIGHTS.keys():
        weighted_score += scores[f'{metric}_norm'] * HEALTH_WEIGHTS[metric]
    
    # Step 3: Scale to 0-100 for easier interpretation
    final_score = weighted_score * 100
    
    return final_score

# Replace the health score calculation in your code
combined['health_score'] = calculate_health_score(combined)

# Calculate correlations
poverty_correlation = combined['poverty_rate'].corr(combined['health_score'])
pm25_correlation = combined['pm25_level'].corr(combined['health_score'])

# Create two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Poverty plot
sns.regplot(data=combined, 
            x='poverty_rate', 
            y='health_score',
            scatter_kws={'alpha':0.5},
            line_kws={'color': 'red'},
            ax=ax1)

ax1.set_title(f'Poverty Rate vs Health Score\nCorrelation: {poverty_correlation:.3f}')
ax1.set_xlabel('Poverty Rate (%)')
ax1.set_ylabel('Health Score (Higher = Better Health)')

# PM2.5 plot
sns.regplot(data=combined, 
            x='pm25_level', 
            y='health_score',
            scatter_kws={'alpha':0.5},
            line_kws={'color': 'blue'},
            ax=ax2)

ax2.set_title(f'PM2.5 Level vs Health Score\nCorrelation: {pm25_correlation:.3f}')
ax2.set_xlabel('PM2.5 Level (mcg/m³)')
ax2.set_ylabel('Health Score (Higher = Better Health)')

plt.tight_layout()
plt.show()

# Print detailed correlation analysis
print("\nCorrelation Analysis:")
print(f"Poverty Rate vs Health Score:")
print(f"  R = {poverty_correlation:.3f}")
print(f"  R² = {poverty_correlation**2:.3f}")
print(f"\nPM2.5 Level vs Health Score:")
print(f"  R = {pm25_correlation:.3f}")
print(f"  R² = {pm25_correlation**2:.3f}")

# Determine stronger correlator
stronger = "Poverty" if abs(poverty_correlation) > abs(pm25_correlation) else "PM2.5"
print(f"\n{stronger} has a stronger correlation with health outcomes")

