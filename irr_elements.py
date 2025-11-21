import pandas as pd
from sklearn.metrics import cohen_kappa_score

def load_and_preprocess(filename1, filename2):
    """
    Loads the two Excel files, finds common documents, and transposes them.
    """
    print("Loading data...")
    try:
        r1 = pd.read_excel(filename1)
        r2 = pd.read_excel(filename2)
    except FileNotFoundError:
        print("Error: Could not find one of the files.")
        return None, None

    common_docs = r1.columns.intersection(r2.columns)
    
    def prepare_df(df, common_cols):
        df_subset = df[common_cols]
        
        df_T = df_subset.T
        
        df_T.columns = df_T.iloc[0]
        
        df_T = df_T[1:]
        
        return df_T

    df1 = prepare_df(r1, common_docs)
    df2 = prepare_df(r2, common_docs)
    
    return df1, df2

def calculate_binary_kappa(df1, df2):
    print("\n--- Binary Cohen's Kappa (Yes/No Elements) ---")
    
    results = {}
    
    common_labels = df1.columns.intersection(df2.columns)

    for label in common_labels:
        if str(label).lower() in ['rater', 'id', 'nan'] or str(label).startswith("EX"):
            continue
            
        s1 = df1[label].astype(str).str.strip().str.upper()
        s2 = df2[label].astype(str).str.strip().str.upper()
        
        valid_values = {'Y', 'N', 'YES', 'NO'}
        unique_vals = set(s1.unique()) | set(s2.unique())
        
        if not unique_vals.intersection(valid_values):
            continue

        mapping = {'Y': 1, 'YES': 1, 'N': 0, 'NO': 0}
        
        clean_s1 = s1.map(mapping)
        clean_s2 = s2.map(mapping)
        
        temp_df = pd.DataFrame({'R1': clean_s1, 'R2': clean_s2}).dropna()
        
        if temp_df.empty:
            print(f"Skipping {label} (No valid Y/N pairs found)")
            continue
            
        try:
            k = cohen_kappa_score(temp_df['R1'], temp_df['R2'], weights=None)
            results[label] = k
        except Exception as e:
            print(f"Error calculating {label}: {e}")

    if not results:
        print("No Y/N columns found to analyze.")
    else:
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Cohen_Kappa'])
        print(result_df)
        result_df.to_csv("output_elements_kappa.csv")
        print("\nSaved results to 'output_elements_kappa.csv'")

if __name__ == "__main__":
    df_sophie, df_cat = load_and_preprocess("Validation Study_Sophie.xlsx", "CN_processed_FIXED.xlsx")
    
    if df_sophie is not None:
        calculate_binary_kappa(df_sophie, df_cat)