import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_process_elements(filename, label):
    """
    Loads file, transposes, extracts source type, and calculates 
    the % presence of each element.
    """
    print(f"Processing {label}...")
    try:
        df = pd.read_excel(filename)
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return None

    # 1. Transpose
    df_T = df.T
    df_T.columns = df_T.iloc[0] # Header
    df_T = df_T[1:].reset_index()
    df_T.rename(columns={'index': 'DocID'}, inplace=True)
    
    # 2. Filter for the Binary Columns only
    target_cols = [
        "Example Present", 
        "Counter-example Present", 
        "Action Present", 
        "Slogan Present"
    ]
    
    # Check which columns actually exist in this file
    existing_cols = [c for c in target_cols if c in df_T.columns]
    if not existing_cols:
        print(f"Warning: No binary element columns found in {label}.")
        return None

    # 3. Convert Y/N to 1/0
    for col in existing_cols:
        df_T[col] = df_T[col].astype(str).str.strip().str.upper().map({
            'Y': 1, 'YES': 1, 'N': 0, 'NO': 0
        }).fillna(0)

    # 4. Extract Source Type
    df_T['SourceType'] = df_T['DocID'].astype(str).str[0].str.upper()
    name_map = {'V': 'VIS Papers', 'B': 'Books', 'C': 'Crowdsourced', 'W': 'Web Blogs'}
    df_T['SourceName'] = df_T['SourceType'].map(name_map)

    # 5. Group by Source and Calculate Mean (Percentage)
    # We use mean() to get a fraction (0.5 = 50%), then multiply by 100 later
    grouped = df_T.groupby('SourceName')[existing_cols].mean()
    
    # 6. Reindex to ensure all Sources appear (even if empty)
    all_sources = ['VIS Papers', 'Books', 'Crowdsourced', 'Web Blogs']
    grouped = grouped.reindex(all_sources, fill_value=0)
    
    # 7. Melt for Seaborn plotting (Wide -> Long format)
    melted = grouped.reset_index().melt(id_vars='SourceName', var_name='Element', value_name='Percentage')
    melted['Percentage'] = melted['Percentage'] * 100 # Convert to %
    
    return melted

def plot_side_by_side(df1, df2):
    print("Generating chart...")
    
    # Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Plot Parameters
    bar_order = ['VIS Papers', 'Books', 'Crowdsourced', 'Web Blogs']
    hue_order = ["Example Present", "Counter-example Present", "Action Present", "Slogan Present"]
    palette = "Set2" # Nice pastel colors

    # Plot 1: Sophie
    sns.barplot(data=df1, x='SourceName', y='Percentage', hue='Element', 
                order=bar_order, hue_order=hue_order, ax=axes[0], palette=palette, edgecolor='black')
    axes[0].set_title("Sophie's Coding", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].set_ylabel("% of Documents", fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    axes[0].legend_.remove() # Remove legend from first plot

    # Plot 2: CN (Cat)
    sns.barplot(data=df2, x='SourceName', y='Percentage', hue='Element', 
                order=bar_order, hue_order=hue_order, ax=axes[1], palette=palette, edgecolor='black')
    axes[1].set_title("Cat's Coding", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)
    
    # Handle Legend (Place it outside)
    axes[1].legend(title="Element", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("elements_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved to 'elements_comparison.png'")

if __name__ == "__main__":
    # Load Data
    # Note: Using the FIXED file for Cat to ensure IDs match if needed, 
    # though for this specific chart, IDs matter less than just the counts.
    sophie_data = load_and_process_elements("Validation Study_Sophie.xlsx", "Sophie")
    cat_data = load_and_process_elements("CN_processed_FIXED.xlsx", "Cat")
    
    if sophie_data is not None and cat_data is not None:
        plot_side_by_side(sophie_data, cat_data)