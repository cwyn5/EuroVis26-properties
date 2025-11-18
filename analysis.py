import pandas as pd
import numpy as np
import sys


def normalize_column_name(col):
    try:
        num = float(col)
        # If it's an integer like 4.0 → store as int
        if num.is_integer():
            return int(num)
        return num
    except:
        # Not numeric → keep as string
        return str(col)
    
def normalize_value(v):
    try:
        num = float(v)
        if num.is_integer():
            return str(int(num))   # "4.0" → "4"
        return str(num)            # "4.5" → "4.5"
    except:
        return str(v).strip() 
    
r1 = pd.read_excel("Validation Study_Sophie.xlsx")
r1.set_index(r1.columns[0], inplace=True)

r2 = pd.read_excel("CN_processed.xlsx")
r2.set_index(r2.columns[0], inplace=True)

r1.columns = [normalize_column_name(c) for c in r1.columns]
r2.columns = [normalize_column_name(c) for c in r2.columns]


# Combine unique columns from both
all_cols = set(r1.columns).union(r2.columns)

df = pd.DataFrame(index=r1.index)  # start empty with same index

for col in all_cols:
    if col in r1.columns:
        if r1[col].astype(str).str.isnumeric().all():
            r1[col] = pd.to_numeric(r1[col])

    if col in r2.columns:
        if r2[col].astype(str).str.isnumeric().all():
            r2[col] = pd.to_numeric(r2[col])
    if col in r1.columns and col in r2.columns:
        # Average only where both are numeric
        df[col] = r1[col].combine(
            r2[col], lambda a, b: np.nanmean([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")])
            if pd.notna(a) or pd.notna(b)
            else np.nan
            )

    elif col in r1.columns:
        df[col] = r1[col]
    else:
        df[col] = r2[col]

#df.to_csv("full.csv")

df_groups = {letter: df[[c for c in df.columns if c.startswith(letter)]]
              for letter in df.columns.str[0].unique()}

desired_cols = ['1','1.5','2','2.5','3','3.5','4','4.5']

# Compute frequencies within each group
freqs_by_group = {}
for letter, subdf in df_groups.items():
    freq_df = (
        subdf.apply(
            lambda row: row.apply(normalize_value).value_counts(),
            axis=1
        )
        .fillna(0)
    )
    # Add missing desired columns
    for col in desired_cols:
        if col not in freq_df.columns:
            freq_df[col] = 0
    freq_df = freq_df[desired_cols] 
    freqs_by_group[letter] = freq_df
print("Frequencies for group V:")
print(freqs_by_group['V'])

freqs_by_group['V'].to_csv('vis_frequencies.csv')

print("Frequencies for group C:")
print(freqs_by_group['C'])

freqs_by_group['C'].to_csv('forum_frequencies.csv')

print("Frequencies for group W:")
print(freqs_by_group['W'])

freqs_by_group['W'].to_csv('blog_frequencies.csv')

print("Frequencies for group B:")
print(freqs_by_group['B'])

freqs_by_group['B'].to_csv('book_frequencies.csv')


rows_to_compare = freqs_by_group['V'].index  # Assuming all have the same rows

# Optional: combine into one dictionary for easy looping
dfs = {'VIS': freqs_by_group['V'], 'FORUM': freqs_by_group['C'], 'WEB': freqs_by_group['W'], 'BOOKS': freqs_by_group['B']}

normalized_dfs = {}
for name, df in dfs.items():
    df.columns = pd.to_numeric(df.columns, errors='coerce')

    df.dropna(axis=1)
    norm_df = df.div(df.sum(axis=1), axis=0)
    normalized_dfs[name] = norm_df.fillna(0)  # replace NaNs with 0

    print(f"{name} normalized, columns used: {df.columns.tolist()}")
    print(norm_df.head())


import matplotlib
matplotlib.use('Agg')  # avoid GUI issues
import matplotlib.pyplot as plt
import os


# Prepare output folder
output_folder = 'row_distributions'
os.makedirs(output_folder, exist_ok=True)

# Plot all rows
def numeric_columns(df):
    return [c for c in df.columns if isinstance(c, (int, float))]

output_folder = 'row_distributions'

# ---- PLOT FOR EACH ROW ----
for row_name in dfs["VIS"].index:  # assume same index across all dfs
    plt.figure(figsize=(8, 5))

    for label, df in dfs.items():
        num_cols = numeric_columns(df)
        plt.plot(num_cols, df.loc[row_name, num_cols], marker="o", label=label)

    plt.title(f"Scores for: {row_name}")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    safe_name = "".join(c for c in str(row_name) if c.isalnum() or c in (" ", "_")).rstrip()
    file_path = os.path.join(output_folder, f"{safe_name}.png")

    plt.savefig(file_path, dpi=150)
    plt.close()   # close figure to prevent memory buildup

    print(f"Saved: {file_path}")



# Example: freqs_by_group = {'VIS': df_vis, 'FORUM': df_forum, ...}

# Compute row averages per dataset
row_avg_by_dataset = {}
for name, df in freqs_by_group.items():
    # Ensure numeric columns only
    numeric_cols = [c for c in df.columns if isinstance(c, (int, float))]
    
    # Compute row-wise average
    row_avg = df[numeric_cols].multiply(df[numeric_cols].columns, axis=1).sum(axis=1) / df[numeric_cols].sum(axis=1)
    row_avg_by_dataset[name] = row_avg

# Combine into a single DataFrame
avg_df = pd.DataFrame(row_avg_by_dataset)

avg_df = avg_df.loc[(avg_df != 0).any(axis=1)]
avg_df = avg_df.dropna(how='all')
# Compute average across all datasets
avg_df['Overall'] = avg_df.mean(axis=1)

# Plot bar chart
plt.figure(figsize=(10, 6))
avg_df['Overall'].plot(kind='bar')
plt.ylabel("Average Rating")
plt.xlabel("Row / Item")
plt.title("Average Rating per Code Across All Guidelines")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.show()

plt.savefig('averages.png', dpi=150)