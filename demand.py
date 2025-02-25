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
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
for instrument in ['MilkoScan™ FT3', 'BacSomatic™']:
    sns.lineplot(data=weekly_data[instrument], label=instrument, dashes=False, markers=True,
                linewidth=2.5, markersize=8, color=instrument_colors[instrument])

# Enforce monthly labels on the x-axis
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.gcf().autofmt_xdate()

plt.title("Weekly Demand Trends\nMilkoScan™ FT3 and BacSomatic™", pad=20)
plt.xlabel("Month", labelpad=15)
plt.ylabel("Quantity Ordered", labelpad=15)
plt.legend(title="Instrument", title_fontsize=12, fontsize=10, framealpha=0.9)

# Enhanced grid settings
ax.grid(True, which='major', color='#E5E5E5', linestyle='-', alpha=0.8)
ax.grid(True, which='minor', color='#E5E5E5', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)  # Ensure grid is behind the data

plt.tight_layout()
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

print("\n=== CORRELATION MATRIX ===")
print(weekly_data[['MilkoScan™ FT3', 'BacSomatic™']].corr())

print("\n=== HEAD OF RATIO TIME SERIES ===")
print(weekly_data['Ratio_MilkoScan_to_BacSomatic'].head())

# Visualization 2: Ratio over time
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
sns.lineplot(x=weekly_data.index, y=weekly_data['Ratio_MilkoScan_to_BacSomatic'], 
             color=instrument_colors['MilkoScan™ FT3'], marker='o', markersize=8, linewidth=2.5,
             label='MilkoScan™ FT3 / BacSomatic™')
plt.axhline(1, linestyle='--', color='gray', alpha=0.8, label='Equal Demand')
plt.title("Demand Ratio Analysis\nMilkoScan™ FT3 to BacSomatic™", pad=20)
plt.xlabel("Date", labelpad=15)
plt.ylabel("Ratio", labelpad=15)
plt.legend(fontsize=10, framealpha=0.9)
plt.grid(True, color='#E5E5E5', linestyle='-', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'ratio_over_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Visualization 3: Correlation heatmap
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
correlation = weekly_data[['MilkoScan™ FT3', 'BacSomatic™']].corr()
sns.heatmap(correlation, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, 
            annot_kws={"size": 12}, square=True, fmt=".2f",
            cbar_kws={"shrink": .8})
plt.title("Correlation Analysis", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Visualization 4: Cross-correlation (interaction analysis)
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
lags = np.arange(-8, 9)
cross_corr = [weekly_data['MilkoScan™ FT3'].corr(weekly_data['BacSomatic™'].shift(lag)) for lag in lags]
sns.barplot(x=lags, y=cross_corr, color=instrument_colors['MilkoScan™ FT3'], alpha=0.7)
plt.axhline(0, color='gray', linewidth=1, linestyle='--')
plt.title("Cross-correlation Analysis\nMilkoScan™ FT3 vs. BacSomatic™", pad=20)
plt.xlabel("Lag (Weeks)\nNegative: BacSomatic leads, Positive: MilkoScan leads", labelpad=15)
plt.ylabel("Correlation Coefficient", labelpad=15)
plt.grid(True, color='#E5E5E5', linestyle='-', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'cross_correlation.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Visualization 5: Cumulative demand over time
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
cumulative_data = weekly_data[['MilkoScan™ FT3', 'BacSomatic™']].cumsum()
for instrument in ['MilkoScan™ FT3', 'BacSomatic™']:
    sns.lineplot(data=cumulative_data[instrument], label=instrument, dashes=False, 
                markers=True, linewidth=2.5, markersize=8, color=instrument_colors[instrument])
plt.title("Cumulative Demand Analysis\nMilkoScan™ FT3 and BacSomatic™", pad=20)
plt.xlabel("Date", labelpad=15)
plt.ylabel("Cumulative Quantity", labelpad=15)
plt.legend(title="Instrument", title_fontsize=12, fontsize=10, framealpha=0.9)
plt.grid(True, color='#E5E5E5', linestyle='-', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'cumulative_demand.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Enhanced plots saved to {os.path.abspath(plots_dir)} directory")
