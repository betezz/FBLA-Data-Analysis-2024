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

# Define weights for each health metric based on severity
HEALTH_WEIGHTS = {
    'death_rate_pm25': 0.25,      
    'cardio_death_rate': 0.25,   
    'resp_hosp_rate': 0.20,       
    'cardio_hosp_rate': 0.20,     
    'asthma_rate': 0.10          
}

def calculate_health_score(df):
    # Create copy to avoid modifying original
    scores = pd.DataFrame()
    
    # Step 1: Min-max normalize each metric (0-1 scale)
    for metric in HEALTH_WEIGHTS.keys():
        # Invert the normalization since higher rates = worse health
        min_val = df[metric].min()
        max_val = df[metric].max()
        inv = 1 - ((df[metric] - min_val) / (max_val - min_val))
        scores[f'{metric}_norm'] = inv
    
    # Step 2: Apply weights and combine
    weighted_score = 0
    for metric in HEALTH_WEIGHTS.keys():
        scale = HEALTH_WEIGHTS[metric]
        weighted_score += (scores[f'{metric}_norm'] * scale)
    
    # Step 3: Scale to 0-100 for easier interpretation
    final_score = weighted_score * 100
    
    return final_score

def calculate_health_score_2(df):
    # Initialize score to 100 (perfect health)
    score = 100
    
    # Subtract weighted values for each metric
    # Higher rates of bad health outcomes reduce the score
    for metric, weight in HEALTH_WEIGHTS.items():
        # Normalize the metric by its mean to make the scales comparable
        normalized_value = df[metric] / df[metric].mean()
        # Subtract the weighted normalized value from the score
        # Multiply by 10 to make the changes more noticeable
        score -= (normalized_value * weight * 10)
    
    return score

# Calculate health score and add it to the combined DataFrame
combined['health_score'] = calculate_health_score(combined)

# Calculate correlations
poverty_correlation = combined['poverty_rate'].corr(combined['health_score'])
pm25_correlation = combined['pm25_level'].corr(combined['health_score'])

# Set the style
plt.style.use('seaborn')
sns.set_palette("husl")

# Create figure with a specific background color and more space for labels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.patch.set_facecolor('#F0F2F6')

# Adjust the subplot parameters to match your desired spacing
plt.subplots_adjust(
    left=0.073,    # Distance from left edge
    right=1.0,     # Distance from right edge
    bottom=0.117,  # Distance from bottom edge
    top=0.839,     # Distance from top edge
    wspace=0.145,  # Width spacing between subplots
    hspace=0.2     # Height spacing between subplots (not used in this case since we have 1 row)
)

# Poverty plot
sns.regplot(data=combined, 
            x='poverty_rate', 
            y='health_score',
            scatter_kws={'alpha':0.6, 'color': '#FF6B6B', 's': 100},
            line_kws={'color': '#CC4455', 'linewidth': 2},
            ax=ax1)

ax1.set_title('Poverty Rate vs Health Score\n' + 
              f'Correlation: {poverty_correlation:.3f}',
              fontsize=14, pad=20, fontweight='bold')
ax1.set_xlabel('Poverty Rate (%)', fontsize=12, labelpad=10)
ax1.set_ylabel('Health Score\n(Higher = Better Health)', fontsize=12, labelpad=15)
ax1.grid(True, linestyle='--', alpha=0.7)

# PM2.5 plot
sns.regplot(data=combined, 
            x='pm25_level', 
            y='health_score',
            scatter_kws={'alpha':0.6, 'color': '#4ECDC4', 's': 100},
            line_kws={'color': '#45B7AF', 'linewidth': 2},
            ax=ax2)

ax2.set_title('PM2.5 Level vs Health Score\n' + 
              f'Correlation: {pm25_correlation:.3f}',
              fontsize=14, pad=20, fontweight='bold')
ax2.set_xlabel('PM2.5 Level (mcg/m³)', fontsize=12, labelpad=10)
ax2.set_ylabel('Health Score\n(Higher = Better Health)', fontsize=12, labelpad=15)
ax2.grid(True, linestyle='--', alpha=0.7)

# Add a super title with adjusted position
fig.suptitle('Relationship Between Environmental Factors and Public Health', 
             fontsize=16, y=0.95, fontweight='bold')

# Move the methodology note to match the new spacing
fig.text(0.073, 0.02, 
         'Note: Health Score is a weighted composite of mortality and hospitalization rates', 
         fontsize=10, style='italic', alpha=0.7)

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

