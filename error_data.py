import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import matplotlib.font_manager as fm
import os

# Set up font that supports Danish characters
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Set up consistent styling with seaborn
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
df = pd.read_csv('error_data.csv', encoding='latin1', na_filter=True)
df['Dato:'] = pd.to_datetime(df['Dato:'], format='%d-%m-%y')

# Clean up special characters in DataFrame
def clean_text(text):
    if pd.isna(text):
        return text
    # Replace curly quotes and specific Danish characters
    replacements = {
        chr(145): "'",  # curly quote
        chr(146): "'",  # curly quote
        '?': 'ø',      # replace ? with ø for common Danish words
        'M\'lk': 'Mælk',  # fix specific case
        'm\'lk': 'mælk',  # fix specific case (lowercase)
        'L?sning': 'Løsning'  # fix specific case
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# Apply cleaning to all object columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(clean_text)

# Combine both "Type of error" columns without cleaning categories
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

# Create plots directory if it doesn't exist
plots_dir = os.path.join('plots', 'error_plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# 1. Errors over time histogram
plt.figure(facecolor='white', figsize=(14, 8))
ax = plt.gca()
ax.set_facecolor('white')

# Create bins for the histogram
bins = 30
hist_range = [df['Dato:'].min(), df['Dato:'].max()]

# Plot total errors histogram
plt.hist(df['Dato:'], bins=bins, range=hist_range, 
         color=sns.color_palette("Set1")[0], alpha=0.6,
         label='Total Errors')

# Create histogram for unique instruments
unique_instruments = df.groupby(['Dato:', 'Serienummer:']).size().reset_index()
plt.hist(unique_instruments['Dato:'], bins=bins, range=hist_range,
         color=sns.color_palette("Set1")[1], alpha=0.6,
         label='Unique Instruments with Errors')

# Update the plot titles and labels in English
plt.title("Distribution of Errors Over Time\nJanuary 2024 - February 2025", pad=20)
plt.xlabel("Date", labelpad=15)
plt.ylabel("Number of Errors", labelpad=15)

# Add legend
plt.legend(loc='upper right')

# Enforce monthly labels on the x-axis
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.gcf().autofmt_xdate()

# Enhanced grid settings
ax.grid(True, which='major', color='#E5E5E5', linestyle='-', alpha=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'errors_over_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 2. Most common error types visualization
plt.figure(facecolor='white', figsize=(14, 8))
ax = plt.gca()
ax.set_facecolor('white')

# Get top 10 error types and sort them (using original error types)
error_types = df['Combined Error Type'].value_counts()
error_types = error_types[error_types > 0].head(10)
error_types = error_types.sort_values(ascending=True)

# Create custom color palette
colors = sns.color_palette("Set2", n_colors=len(error_types))

# Create horizontal bar chart for error types
bars = sns.barplot(x=error_types.values, y=error_types.index, 
                  palette=colors, alpha=0.8, hue=error_types.index, legend=False)

# Update titles for second plot to English
plt.title("Top 10 Most Common Error Types", pad=20)
plt.xlabel("Number of Occurrences", labelpad=15)
plt.ylabel("Error Type", labelpad=15)

# Add value labels on the bars
for i, v in enumerate(error_types.values):
    ax.text(v, i, f' {v}', va='center')

# Enhanced grid settings
ax.grid(True, which='major', color='#E5E5E5', linestyle='-', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'most_common_errors.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 3. Test Steps visualization
plt.figure(facecolor='white', figsize=(14, 8))
ax = plt.gca()
ax.set_facecolor('white')

# Get top test steps where errors occur
step_errors = df['Step'].value_counts()
step_errors = step_errors[step_errors > 0].head(10)
step_errors = step_errors.sort_values(ascending=True)

# Create custom color palette for steps
colors_steps = sns.color_palette("Set2", n_colors=len(step_errors))

# Create horizontal bar chart for test steps
bars = sns.barplot(x=step_errors.values, y=step_errors.index, 
                  palette=colors_steps, alpha=0.8, hue=step_errors.index, legend=False)

# Update titles for third plot to English
plt.title("Most Common Test Steps with Errors", pad=20)
plt.xlabel("Number of Occurrences", labelpad=15)
plt.ylabel("Test Step", labelpad=15)

# Add value labels on the bars
for i, v in enumerate(step_errors.values):
    ax.text(v, i, f' {v}', va='center')

# Enhanced grid settings
ax.grid(True, which='major', color='#E5E5E5', linestyle='-', alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'error_steps.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Print enhanced statistics
print(f"=== ERROR ANALYSIS STATISTICS ===")
print(f"\nTotal number of errors: {len(df)}")
print(f"Total number of unique instruments with errors: {df['Serienummer:'].nunique()}")
print(f"Date range: from {df['Dato:'].min().strftime('%d %b %Y')} to {df['Dato:'].max().strftime('%d %b %Y')}")

# Calculate weekly statistics
weekly_total = df.groupby(pd.Grouper(key='Dato:', freq='W')).size()
weekly_unique = df.groupby([pd.Grouper(key='Dato:', freq='W'), 'Serienummer:']).size().reset_index().groupby('Dato:').size()

print(f"\nAverage total errors per week: {weekly_total.mean():.1f}")
print(f"Average unique instruments with errors per week: {weekly_unique.mean():.1f}")

# Most common error solutions with better formatting
print("\nTop 5 most common solutions:")
solutions = df['L?sning'].value_counts().head()
for solution, count in solutions.items():
    print(f"- {solution}: {count}")

# Add statistics about repeat errors
repeat_offenders = df['Serienummer:'].value_counts()
print(f"\nInstruments with most repeated errors:")
for sn, count in repeat_offenders.head().items():
    print(f"- Serial number {sn}: {count} errors")

# Print statistics about original error types
print(f"\nTop 10 Most Common Error Types:")
for error_type, count in error_types.items():
    print(f"- {error_type}: {count}")

# Add analysis of errors by test stage
print(f"\nErrors by Test Stage:")
stage_errors = df['Test'].value_counts().head()
for stage, count in stage_errors.items():
    print(f"- {stage}: {count}")

# Add statistics about test steps
print(f"\nMost Common Test Steps with Errors:")
for step, count in step_errors.items():
    print(f"- {step}: {count}")

# Get top 5 error types and their solutions
top_errors = df['Combined Error Type'].value_counts().head(5).index

# Print error-solution relationships
print("\nMost Common Solutions for Top Error Types:")
for error in top_errors:
    solutions = df[df['Combined Error Type'] == error]['L?sning'].value_counts().head(3)
    print(f"\n{error}:")
    for solution, count in solutions.items():
        if pd.notna(solution):
            print(f"  - {solution}: {count}")

# Get top 5 test steps
top_steps = df['Step'].value_counts().head(5).index

# Print error types in test steps
print("\nMost Common Error Types in Top Test Steps:")
for step in top_steps:
    errors = df[df['Step'] == step]['Combined Error Type'].value_counts().head(3)
    print(f"\n{step}:")
    for error, count in errors.items():
        if pd.notna(error):
            print(f"  - {error}: {count}")

print(f"Error analysis plots saved to {os.path.abspath(plots_dir)} directory")