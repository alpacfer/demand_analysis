# sanitized_error_data.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os

# Set up font that supports Danish characters
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Set up consistent styling
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

# Read the CSV file with proper encoding
file_path = 'error_data.csv'
df = pd.read_csv(file_path, encoding='latin1', na_filter=True)
df['Dato:'] = pd.to_datetime(df['Dato:'], format='%d-%m-%y')

# Clean up special characters in DataFrame
def clean_text(text):
    if pd.isna(text):
        return text
    replacements = {
        chr(145): "'",
        chr(146): "'",
        '?': 'ø',
        "M'lk": 'Mælk',
        "m'lk": 'mælk',
        'L?sning': 'Løsning'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(clean_text)

# Combine error type columns
df['Combined Error Type'] = df.iloc[:, 4:6].fillna('').agg(lambda x: ' '.join(filter(None, x)), axis=1)
df['Combined Error Type'] = df['Combined Error Type'].str.strip()
df.loc[df['Combined Error Type'] == '', 'Combined Error Type'] = np.nan

# Consolidate similar error types
def clean_error_type(error):
    if pd.isna(error):
        return 'Other'
    error = str(error).lower()
    if 'trans' in error:
        return 'trans'
    elif 'fat' in error:
        return 'fat'
    elif 'protein' in error:
        return 'protein'
    elif 'id chip' in error:
        return 'id chip'
    elif 'connection' in error or 'connection problem' in error:
        return 'connection'
    elif 'human error' in error:
        return 'human error'
    elif 'adjustment' in error:
        return 'adjustment'
    elif 'waterdiff' in error or 'waterdrift' in error:
        return 'waterdiff - waterdrift'
    else:
        return error.title()

df['Error Category'] = df['Combined Error Type'].apply(clean_error_type)

# Create directory for sanitized error plots
sanitized_error_plots_dir = os.path.join('plots', 'sanitized_error_plots')
if not os.path.exists(sanitized_error_plots_dir):
    os.makedirs(sanitized_error_plots_dir)

# 1. Errors Over Time Histogram (Normalized to Percentage)
plt.figure(facecolor='white', figsize=(14, 8))
ax = plt.gca()
ax.set_facecolor('white')

# Define bins and range
bins = 30
hist_range = [df['Dato:'].min(), df['Dato:'].max()]

# For total errors, use weights to get percentages
weights_total = np.ones(len(df['Dato:'])) / len(df['Dato:']) * 100
plt.hist(df['Dato:'], bins=bins, range=hist_range, 
         color=sns.color_palette('Set1')[0], alpha=0.6,
         label='Total Errors', weights=weights_total)

# For unique instruments errors
unique_instruments = df.groupby(['Dato:', 'Serienummer:']).size().reset_index()
weights_unique = np.ones(len(unique_instruments['Dato:'])) / len(unique_instruments['Dato:']) * 100
plt.hist(unique_instruments['Dato:'], bins=bins, range=hist_range,
         color=sns.color_palette('Set1')[1], alpha=0.6,
         label='Unique Instruments (Errors)', weights=weights_unique)

plt.title('Distribution of Errors Over Time (Sanitized)', pad=20)
plt.xlabel('Date', labelpad=15)
plt.ylabel('Percentage of Errors (%)', labelpad=15)
plt.legend(loc='upper right')

ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.gcf().autofmt_xdate()
ax.grid(True, which='major', color='#E5E5E5', linestyle='-', alpha=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(sanitized_error_plots_dir, 'errors_over_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 2. Most Common Error Types (Showing Percentages)
plt.figure(facecolor='white', figsize=(14, 8))
ax = plt.gca()
ax.set_facecolor('white')

# Get top 10 error types
error_types = df['Combined Error Type'].value_counts()
error_types = error_types[error_types > 0].head(10)
# Convert counts to percentages
error_types_pct = (error_types / error_types.sum()) * 100
error_types_pct = error_types_pct.sort_values(ascending=True)

colors_bar = sns.color_palette('Set2', n_colors=len(error_types_pct))
bars = sns.barplot(x=error_types_pct.values, y=error_types_pct.index, 
                    palette=colors_bar, alpha=0.8)

plt.title('Top 10 Most Common Error Types (Sanitized)', pad=20)
plt.xlabel('Percentage of Occurrences (%)', labelpad=15)
plt.ylabel('Error Type', labelpad=15)

# Add percentage labels on bars
for i, v in enumerate(error_types_pct.values):
    ax.text(v, i, f' {v:.1f}%', va='center')

ax.grid(True, which='major', color='#E5E5E5', linestyle='-', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(sanitized_error_plots_dir, 'most_common_errors.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 3. Most Common Test Steps with Errors (Normalized Percentages)
plt.figure(facecolor='white', figsize=(14, 8))
ax = plt.gca()
ax.set_facecolor('white')

step_errors = df['Step'].value_counts()
step_errors = step_errors[step_errors > 0].head(10)
# Convert to percentages
step_errors_pct = (step_errors / step_errors.sum()) * 100
step_errors_pct = step_errors_pct.sort_values(ascending=True)

colors_steps = sns.color_palette('Set2', n_colors=len(step_errors_pct))
bars = sns.barplot(x=step_errors_pct.values, y=step_errors_pct.index, 
                    palette=colors_steps, alpha=0.8)

plt.title('Most Common Test Steps with Errors (Sanitized)', pad=20)
plt.xlabel('Percentage of Occurrences (%)', labelpad=15)
plt.ylabel('Test Step', labelpad=15)

for i, v in enumerate(step_errors_pct.values):
    ax.text(v, i, f' {v:.1f}%', va='center')

ax.grid(True, which='major', color='#E5E5E5', linestyle='-', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(sanitized_error_plots_dir, 'error_steps.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Sanitized error plots saved to {os.path.abspath(sanitized_error_plots_dir)}")
