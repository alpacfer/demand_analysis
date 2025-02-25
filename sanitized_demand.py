# sanitized_demand.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import seaborn as sns

# Set pandas option
pd.set_option('future.no_silent_downcasting', True)

# Set up styling
sns.set_style("white")
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

# Color mapping for instruments
instrument_colors = {
    'MilkoScan™ FT3': colors[0],
    'BacSomatic™': colors[1]
}

# Create directory for sanitized demand plots
sanitized_plots_dir = os.path.join('plots', 'sanitized_demand_plots')
if not os.path.exists(sanitized_plots_dir):
    os.makedirs(sanitized_plots_dir)

# Get timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load dataset
file_path = 'demand_data.csv'
data = pd.read_csv(file_path, encoding='Windows-1252')

data['Qty.'] = pd.to_numeric(data['Qty.'], errors='coerce').fillna(0)

data['OrderDate'] = pd.to_datetime(data['OrderDate'], format='%d-%m-%y')

# Filter data for specific instruments
filtered_data = data[data['Instrument'].str.contains('MilkoScan™ FT3|BacSomatic™', na=False, case=False)]
filtered_data.loc[:, 'Qty.'] = pd.to_numeric(filtered_data['Qty.'], errors='coerce').fillna(0)

# Group and pivot data
grouped_data = filtered_data.groupby(['Instrument', 'OrderDate']).agg({'Qty.': 'sum'}).reset_index()
pivot_data = grouped_data.pivot(index='OrderDate', columns='Instrument', values='Qty.')
pivot_data = pivot_data.fillna(0).astype(float)

# Handle variations and refurbished data
if 'MilkoScan™ FT3?' in pivot_data.columns:
    pivot_data['MilkoScan™ FT3'] += pivot_data['MilkoScan™ FT3?']
    pivot_data = pivot_data.drop(columns=['MilkoScan™ FT3?'], errors='ignore')

refurbished_instruments = [col for col in pivot_data.columns if 'Refurbished' in col]
for refurbished in refurbished_instruments:
    instrument_base = refurbished.replace(" Refurbished", "")
    if instrument_base in pivot_data.columns:
        pivot_data[instrument_base] += pivot_data[refurbished]
    pivot_data = pivot_data.drop(columns=[refurbished], errors='ignore')

# Resample weekly
weekly_data = pivot_data.resample('W').sum()

# Function for time axis formatting
def set_time_axis_format(ax):
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    ax.grid(True, which='major', color='#E5E5E5', linestyle='-', alpha=0.8)
    ax.grid(True, which='minor', color='#E5E5E5', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

# 1. Weekly Demand Trends (Normalized)
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# Normalize weekly demand per instrument
normalized_weekly = weekly_data.copy()
for instrument in ['MilkoScan™ FT3', 'BacSomatic™']:
    max_val = normalized_weekly[instrument].max()
    if max_val != 0:
        normalized_weekly[instrument] = normalized_weekly[instrument] / max_val
    else:
        normalized_weekly[instrument] = 0
    sns.lineplot(data=normalized_weekly[instrument], label=instrument, marker='o',
                 linewidth=2.5, markersize=8, color=instrument_colors[instrument])

set_time_axis_format(ax)
plt.title("Weekly Demand Trends (Normalized)", pad=20)
plt.xlabel("Month", labelpad=15)
plt.ylabel("Normalized Demand", labelpad=15)
plt.legend(title="Instrument", title_fontsize=12, fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(sanitized_plots_dir, 'weekly_demand_normalized.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 2. Ratio over Time (unchanged as ratio is non-sensitive)
weekly_data['Ratio_MilkoScan_to_BacSomatic'] = weekly_data['MilkoScan™ FT3'] / weekly_data['BacSomatic™']
weekly_data['Ratio_MilkoScan_to_BacSomatic'] = weekly_data['Ratio_MilkoScan_to_BacSomatic'].replace([np.inf, -np.inf], np.nan)

plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
sns.lineplot(x=weekly_data.index, y=weekly_data['Ratio_MilkoScan_to_BacSomatic'], 
             color=instrument_colors['MilkoScan™ FT3'], marker='o', markersize=8, linewidth=2.5,
             label='MilkoScan™ FT3 / BacSomatic™')
plt.axhline(1, linestyle='--', color='gray', alpha=0.8, label='Equal Demand')
set_time_axis_format(ax)
plt.title("Demand Ratio Analysis (Sanitized)", pad=20)
plt.xlabel("Month", labelpad=15)
plt.ylabel("Ratio", labelpad=15)
plt.legend(fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(sanitized_plots_dir, 'ratio_over_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 3. Cumulative Demand (as Percentage)
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

cumulative_data = weekly_data[['MilkoScan™ FT3', 'BacSomatic™']].cumsum()
# Normalize to percentage
cumulative_percentage = cumulative_data.copy()
for instrument in ['MilkoScan™ FT3', 'BacSomatic™']:
    total = cumulative_data[instrument].iloc[-1]
    if total != 0:
        cumulative_percentage[instrument] = (cumulative_data[instrument] / total) * 100
    else:
        cumulative_percentage[instrument] = 0

for instrument in ['MilkoScan™ FT3', 'BacSomatic™']:
    sns.lineplot(data=cumulative_percentage[instrument], label=instrument, marker='o',
                 linewidth=2.5, markersize=8, color=instrument_colors[instrument])

set_time_axis_format(ax)
plt.title("Cumulative Demand Analysis (Percentage)", pad=20)
plt.xlabel("Month", labelpad=15)
plt.ylabel("Cumulative Percentage (%)", labelpad=15)
plt.legend(title="Instrument", title_fontsize=12, fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(sanitized_plots_dir, 'cumulative_demand_percentage.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 4. Market Share Pie Chart (Using Percentages Only)
plt.figure(facecolor='white', figsize=(14, 8))
ax = plt.gca()
ax.set_facecolor('white')

# Calculate total units per instrument (but only use percentages)
total_units = data.groupby('Instrument')['Qty.'].sum().reset_index()
total_units['Qty.'] = pd.to_numeric(total_units['Qty.'], errors='coerce').fillna(0)

# Filter for our instruments
ft3_mask = total_units['Instrument'].str.contains('MilkoScan™ FT3', na=False, case=False)
bac_mask = total_units['Instrument'].str.contains('BacSomatic™', na=False, case=False)
others_mask = ~(ft3_mask | bac_mask)

ft3_total = total_units[ft3_mask]['Qty.'].sum()
bac_total = total_units[bac_mask]['Qty.'].sum()
others_total = total_units[others_mask]['Qty.'].sum()

total_all = ft3_total + bac_total + others_total
if total_all > 0:
    ft3_pct = (ft3_total / total_all) * 100
    bac_pct = (bac_total / total_all) * 100
    others_pct = (others_total / total_all) * 100
else:
    ft3_pct = bac_pct = others_pct = 0

sizes = [ft3_pct, bac_pct, others_pct]
instrument_names = ['MilkoScan™ FT3', 'BacSomatic™', 'Other Products']
# Use only percentage values in labels
legend_labels = [f'{name}\n{pct:.1f}%' for name, pct in zip(instrument_names, sizes)]
colors_pie = [instrument_colors['MilkoScan™ FT3'], 
              instrument_colors['BacSomatic™'], 
              '#999999']

patches, _ = plt.pie(sizes, colors=colors_pie, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
plt.legend(patches, legend_labels,
          title="Instrument Distribution",
          title_fontsize=12,
          fontsize=10,
          loc="center left",
          bbox_to_anchor=(1.0, 0.5),
          framealpha=0.9)
plt.title("Market Share Distribution (Sanitized)", pad=20, fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(os.path.join(sanitized_plots_dir, 'market_share_pie.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Sanitized demand plots saved to {os.path.abspath(sanitized_plots_dir)}")
