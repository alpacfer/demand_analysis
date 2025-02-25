import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import seaborn as sns

# Set pandas option for future behavior
pd.set_option('future.no_silent_downcasting', True)

# Set up consistent styling with seaborn
sns.set_style("white")  # Clean white background
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.figsize': [14, 8],
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'axes.grid': True,
    'grid.color': '#E5E5E5',
    'grid.linestyle': '-',
    'grid.alpha': 0.5,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})
colors = sns.color_palette("Set1")

# Set up color mapping for instruments
instrument_colors = {
    'MilkoScan™ FT3': colors[0],  # First color from Set1 (typically red)
    'BacSomatic™': colors[1]      # Second color from Set1 (typically blue)
}

# Create a function for consistent plot styling
def apply_consistent_style(ax, title, xlabel, ylabel, legend_title=None):
    """Apply consistent styling to all plots"""
    # Title styling
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    
    # Axis labels
    ax.set_xlabel(xlabel, labelpad=15, fontsize=12)
    ax.set_ylabel(ylabel, labelpad=15, fontsize=12)
    
    # Grid styling
    ax.grid(True, which='major', color='#E5E5E5', linestyle='-', alpha=0.8)
    ax.set_axisbelow(True)  # Ensure grid is behind the data
    
    # Background
    ax.set_facecolor('white')
    
    # Legend (check if legend exists and then style it using a more compatible approach)
    legend = ax.get_legend()
    if legend is not None:
        # Set legend properties in a version-compatible way
        try:
            # Try setting frame alpha property directly
            legend.get_frame().set_alpha(0.9)
        except:
            # If that fails, don't modify alpha
            pass
            
        # These properties should work across versions
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#E5E5E5')
        
        # Set legend title if provided
        if legend_title:
            legend.set_title(legend_title)
            legend.get_title().set_fontsize(12)
    
    # Add date axis formatting if the x-axis contains dates
    if isinstance(ax.get_xlim()[0], (np.float64, float)):
        try:
            dates = plt.matplotlib.dates.num2date(ax.get_xlim())
            set_consistent_date_axis(ax, ax.get_lines()[0].get_xdata())
        except:
            pass
    
    # Tight layout
    plt.tight_layout()
    
    return ax

def set_consistent_date_axis(ax, data_index):
    """Apply consistent date axis formatting"""
    # Set date limits to cover the entire dataset
    date_min = data_index.min()
    date_max = data_index.max()
    ax.set_xlim(date_min, date_max)
    
    # Set monthly ticks (changed interval from 2 to 1 to show all months)
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Create a directory to save plots if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Get current timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load the dataset
file_path = 'demand_data.csv'
data = pd.read_csv(file_path, encoding='Windows-1252')

# Ensure the OrderDate column is in datetime format
data['OrderDate'] = pd.to_datetime(data['OrderDate'], format='%d-%m-%y')

# Filter data for the specific instruments, including variations
filtered_data = data[data['Instrument'].str.contains('MilkoScan™ FT3|BacSomatic™', na=False, case=False)]

# Convert the 'Qty.' column to numeric, forcing non-numeric entries to NaN and replacing with 0
filtered_data.loc[:, 'Qty.'] = pd.to_numeric(filtered_data['Qty.'], errors='coerce').fillna(0)

# Group by Instrument and OrderDate, summing the quantities
grouped_data = filtered_data.groupby(['Instrument', 'OrderDate']).agg({'Qty.': 'sum'}).reset_index()

# Pivot data for analysis and plotting
pivot_data = grouped_data.pivot(index='OrderDate', columns='Instrument', values='Qty.')
pivot_data = pivot_data.fillna(0).infer_objects(copy=False).astype(float)

# Handle variations:
# Combine "MilkoScan™ FT3" and "MilkoScan™ FT3?"
if 'MilkoScan™ FT3?' in pivot_data.columns:
    pivot_data['MilkoScan™ FT3'] += pivot_data['MilkoScan™ FT3?']
    pivot_data = pivot_data.drop(columns=['MilkoScan™ FT3?'], errors='ignore')

# Integrate refurbished data into respective instruments
refurbished_instruments = [col for col in pivot_data.columns if 'Refurbished' in col]
valid_refurbished = [col for col in refurbished_instruments if col in pivot_data.columns]

for refurbished in valid_refurbished:
    instrument_base = refurbished.replace(" Refurbished", "")
    if instrument_base in pivot_data.columns:
        pivot_data[instrument_base] += pivot_data[refurbished]
    pivot_data = pivot_data.drop(columns=[refurbished], errors='ignore')

# Resample the data by week to smooth trends
weekly_data = pivot_data.resample('W').sum()

# First visualization (Weekly demand smoothed)
fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
for instrument in ['MilkoScan™ FT3', 'BacSomatic™']:
    sns.lineplot(ax=ax, data=weekly_data[instrument], label=instrument, 
                marker='o',  # Add circular markers
                markersize=6,  # Slightly smaller than the ratio plot
                linewidth=2.5, 
                color=instrument_colors[instrument])

set_consistent_date_axis(ax, weekly_data.index)
apply_consistent_style(ax, "Weekly Demand Trends\nMilkoScan™ FT3 and BacSomatic™", 
                      "Month", "Quantity Ordered", "Instrument")

plt.savefig(os.path.join(plots_dir, 'weekly_demand_smoothed.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Calculate ratio of MilkoScan FT3 to BacSomatic demand
weekly_data['Ratio_MilkoScan_to_BacSomatic'] = weekly_data['MilkoScan™ FT3'] / weekly_data['BacSomatic™']
weekly_data['Ratio_MilkoScan_to_BacSomatic'] = weekly_data['Ratio_MilkoScan_to_BacSomatic'].replace([np.inf, -np.inf], np.nan)

# Print relevant data
print("\n=== TOTAL ORDERS (entire dataset) ===")
print(weekly_data.sum())

print("\n=== AVERAGE WEEKLY DEMAND ===")
print(weekly_data.mean())

print("\n=== HEAD OF RATIO TIME SERIES ===")
print(weekly_data['Ratio_MilkoScan_to_BacSomatic'].head())

# Visualization 2: Ratio over time
fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
sns.lineplot(ax=ax, x=weekly_data.index, y=weekly_data['Ratio_MilkoScan_to_BacSomatic'], 
             color=instrument_colors['MilkoScan™ FT3'], marker='o', markersize=8, linewidth=2.5,
             label='MilkoScan™ FT3 / BacSomatic™')
ax.axhline(1, linestyle='--', color='gray', alpha=0.8, label='Equal Demand')

set_consistent_date_axis(ax, weekly_data.index)
apply_consistent_style(ax, "Demand Ratio Analysis\nMilkoScan™ FT3 to BacSomatic™", 
                      "Date", "Ratio")

plt.savefig(os.path.join(plots_dir, 'ratio_over_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Visualization 5: Cumulative demand over time
fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
cumulative_data = weekly_data[['MilkoScan™ FT3', 'BacSomatic™']].cumsum()
for instrument in ['MilkoScan™ FT3', 'BacSomatic™']:
    sns.lineplot(ax=ax, data=cumulative_data[instrument], label=instrument,
                marker='o',  # Add circular markers
                markersize=6,  # Slightly smaller than the ratio plot
                linewidth=2.5,
                color=instrument_colors[instrument])

set_consistent_date_axis(ax, weekly_data.index)
apply_consistent_style(ax, "Cumulative Demand Analysis\nMilkoScan™ FT3 and BacSomatic™", 
                      "Date", "Cumulative Quantity", "Instrument")

plt.savefig(os.path.join(plots_dir, 'cumulative_demand.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Enhanced plots saved to {os.path.abspath(plots_dir)} directory")
