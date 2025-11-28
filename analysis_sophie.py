import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_transpose(filename, label):
    """
    Loads an Excel file and transposes it so Rows=Documents and Columns=Properties.
    Does NOT filter by intersection, preserving all data in the file.
    """
    print(f"Loading {label} data from {filename}...")
    try:
        df = pd.read_excel(filename)
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return None

    df_T = df.T
    df_T.columns = df_T.iloc[0]
    df_T = df_T[1:].reset_index()
    df_T.rename(columns={'index': 'DocID'}, inplace=True)
    
    return df_T

def get_pivot_data(df, target_col="Goal of articulation"):
    """
    Creates a count matrix (Source Type vs Goal).
    """
    if target_col not in df.columns:
        print(f"Warning: '{target_col}' not found in data.")
        return None

    df['SourceType'] = df['DocID'].astype(str).str[0].str.upper()

    name_map = {
        'V': 'VIS Papers',
        'B': 'Books',
        'C': 'Crowdsourced',
        'W': 'Web Blogs'
    }
    df['SourceName'] = df['SourceType'].map(name_map)

    pivot = pd.crosstab(df['SourceName'], df[target_col])
    
    return pivot

def generate_aligned_chart(df1, df2, name1="Sophie", name2="Cat"):
    print("Aligning data and generating chart...")

    pivot1 = get_pivot_data(df1)
    pivot2 = get_pivot_data(df2)

    if pivot1 is None or pivot2 is None:
        return

    all_goals = sorted(list(set(pivot1.columns) | set(pivot2.columns)))
    
    all_sources = ['VIS Papers', 'Books', 'Crowdsourced', 'Web Blogs']

    pivot1 = pivot1.reindex(index=all_sources, columns=all_goals, fill_value=0)
    pivot2 = pivot2.reindex(index=all_sources, columns=all_goals, fill_value=0)


    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    cmap = 'tab20' 

    # Plot 1: Sophie
    pivot1.plot(kind='bar', stacked=True, ax=axes[0], colormap=cmap, 
                edgecolor='black', linewidth=0.5, legend=False)
    axes[0].set_title(f"Coder: {name1}", fontsize=14)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Number of Documents", fontsize=12)
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)

    # Plot 2: Cat
    pivot2.plot(kind='bar', stacked=True, ax=axes[1], colormap=cmap, 
                edgecolor='black', linewidth=0.5, legend=False)
    axes[1].set_title(f"Coder: {name2}", fontsize=14)
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    
    fig.legend(handles, labels, title="Goal of Articulation", 
               loc='center right', bbox_to_anchor=(1.12, 0.5))

    plt.tight_layout()
    
    filename = "combined_goals_chart.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved chart to {filename}")

if __name__ == "__main__":
    df_sophie = load_and_transpose("Validation Study_Sophie - UPDATED.xlsx", "Sophie")
    df_cat = load_and_transpose("v3_CN_processed.xlsx", "Cat")
    
    if df_sophie is not None and df_cat is not None:
        generate_aligned_chart(df_sophie, df_cat)