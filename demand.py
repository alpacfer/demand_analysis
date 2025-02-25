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
plots_dir = os.path.join('plots', 'demand_plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Get current timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load the dataset
file_path = 'demand_data.csv'
data = pd.read_csv(file_path, encoding='Windows-1252')

# Convert quantities to numeric right after loading
data['Qty.'] = pd.to_numeric(data['Qty.'], errors='coerce').fillna(0)

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

# Function to set consistent time axis formatting
def set_time_axis_format(ax):
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    ax.grid(True, which='major', color='#E5E5E5', linestyle='-', alpha=0.8)
    ax.grid(True, which='minor', color='#E5E5E5', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

# First visualization (Weekly demand smoothed)
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
for instrument in ['MilkoScan™ FT3', 'BacSomatic™']:
    sns.lineplot(data=weekly_data[instrument], label=instrument, marker='o',
                linewidth=2.5, markersize=8, color=instrument_colors[instrument])

set_time_axis_format(ax)
plt.title("Weekly Demand Trends\nMilkoScan™ FT3 and BacSomatic™", pad=20)
plt.xlabel("Month", labelpad=15)
plt.ylabel("Quantity Ordered", labelpad=15)
plt.legend(title="Instrument", title_fontsize=12, fontsize=10, framealpha=0.9)
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

set_time_axis_format(ax)
plt.title("Demand Ratio Analysis\nMilkoScan™ FT3 to BacSomatic™", pad=20)
plt.xlabel("Month", labelpad=15)
plt.ylabel("Ratio", labelpad=15)
plt.legend(fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'ratio_over_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Visualization 3: Cumulative demand over time
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
cumulative_data = weekly_data[['MilkoScan™ FT3', 'BacSomatic™']].cumsum()
for instrument in ['MilkoScan™ FT3', 'BacSomatic™']:
    sns.lineplot(data=cumulative_data[instrument], label=instrument, marker='o',
                linewidth=2.5, markersize=8, color=instrument_colors[instrument])

set_time_axis_format(ax)
plt.title("Cumulative Demand Analysis\nMilkoScan™ FT3 and BacSomatic™", pad=20)
plt.xlabel("Month", labelpad=15)
plt.ylabel("Cumulative Quantity", labelpad=15)
plt.legend(title="Instrument", title_fontsize=12, fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'cumulative_demand.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Calculate total units for all products
total_units = data.groupby('Instrument')['Qty.'].sum().reset_index()
total_units['Qty.'] = pd.to_numeric(total_units['Qty.'], errors='coerce').fillna(0)

# Get FT3 and BacSomatic totals, ensuring we handle NaN values
ft3_mask = total_units['Instrument'].str.contains('MilkoScan™ FT3', na=False, case=False)
bac_mask = total_units['Instrument'].str.contains('BacSomatic™', na=False, case=False)
other_mask = ~(ft3_mask | bac_mask)

ft3_total = total_units[ft3_mask]['Qty.'].sum()
bac_total = total_units[bac_mask]['Qty.'].sum()
others_total = total_units[other_mask]['Qty.'].sum()

# Ensure we have valid numbers
ft3_total = 0 if np.isnan(ft3_total) else ft3_total
bac_total = 0 if np.isnan(bac_total) else bac_total
others_total = 0 if np.isnan(others_total) else others_total

# Calculate percentages
total_all = ft3_total + bac_total + others_total
if total_all > 0:  # Prevent division by zero
    ft3_pct = (ft3_total / total_all) * 100
    bac_pct = (bac_total / total_all) * 100
    others_pct = (others_total / total_all) * 100
else:
    ft3_pct = bac_pct = others_pct = 0

# Create pie chart with enhanced styling
plt.figure(facecolor='white', figsize=(14, 8))  # Matched to other plots
ax = plt.gca()
ax.set_facecolor('white')

# Prepare data for pie chart
sizes = [ft3_total, bac_total, others_total]
instrument_names = ['MilkoScan™ FT3', 'BacSomatic™', 'Other Products']
percentages = [ft3_pct, bac_pct, others_pct]
colors = [instrument_colors['MilkoScan™ FT3'], 
          instrument_colors['BacSomatic™'], 
          '#999999']  # Gray for others

# Create custom labels for legend
legend_labels = [f'{name}\n{value:.0f} units ({pct:.1f}%)' 
                for name, value, pct in zip(instrument_names, sizes, percentages)]

# Create pie chart without labels on the pie itself
patches, _ = plt.pie(sizes, colors=colors, startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2})

# Add legend with consistent styling
plt.legend(patches, legend_labels,
          title="Instrument Distribution",
          title_fontsize=12,
          fontsize=10,
          loc="center left",
          bbox_to_anchor=(1.0, 0.5),
          framealpha=0.9)  # Match other plots' legend transparency

plt.title("Market Share Distribution\nMilkoScan™ FT3 and BacSomatic™", 
         pad=20, fontsize=14, fontweight='bold')  # Match other plots' title style

# Ensure the pie chart is circular
plt.axis('equal')

# Adjust layout to prevent legend cropping
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjusted for larger figure size
plt.savefig(os.path.join(plots_dir, 'market_share_pie.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n=== MARKET SHARE BREAKDOWN ===")
print(f"MilkoScan™ FT3: {ft3_total:.0f} units ({ft3_pct:.1f}%)")
print(f"BacSomatic™: {bac_total:.0f} units ({bac_pct:.1f}%)")
print(f"Other Products: {others_total:.0f} units ({others_pct:.1f}%)")
print(f"\nEnhanced plots saved to {os.path.abspath(plots_dir)} directory")
