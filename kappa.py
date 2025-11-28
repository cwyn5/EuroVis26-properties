import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import pingouin as pg
import sys

def data_preprocess():
    r1 = pd.read_excel("Validation Study_ Sophie - UPDATED.xlsx")

    r2 = pd.read_excel("v3_CN_processed.xlsx")

    common_cols = r1.columns.intersection(r2.columns)

    # Subset both DataFrames to those columns
    df1_common = r1[common_cols]
    df2_common = r2[common_cols]

    df1_common.to_csv('sophie_shared.csv')
    df2_common.to_csv('cat_shared.csv')

    r1_file = df1_common.T

    r1_file.columns = r1_file.iloc[0]
    r1_file = r1_file[1:].reset_index(drop=True)




    #r1_file = r1_file[r1_file[first_col].astype(str).str.contains('GL', case=False, na=False)]
    r1_file['rater'] = 1
    r1_file['ID'] = r1_file.index


    r2_file = df2_common.T
    r2_file.columns = r2_file.iloc[0]
    r2_file = r2_file[1:].reset_index(drop=True)
    r2_file['rater'] = 2
    r2_file['ID'] = r2_file.index

    return r1_file, r2_file



def icc_analysis(df1_input, df2_input):
    r1_file = df1_input.copy()
    r2_file = df2_input.copy()

    results = {}

    numeric_cols = r1_file.select_dtypes(include='number').columns
    print(numeric_cols)

    for label in r1_file.columns:
        if label ==  'ID' or label.startswith("EX"):
            continue
        cols = [label, 'rater', 'ID']
        r1_file[label] = r1_file[label].fillna('missing').infer_objects()
        r2_file[label] = r2_file[label].fillna('missing').infer_objects()

        if r1_file[label].nunique() > 1 or r2_file[label].nunique() > 1:
            r1_file[label] = r1_file[label].replace({4: 5, '4': '5', 2: 1, '2': '1'})
            r2_file[label] = r2_file[label].replace({4: 5, '4': '5', 2: 1, '2': '1'})
            class_df = pd.concat([r1_file[cols], r2_file[cols]])
            class_df[label] = pd.to_numeric(class_df[label], errors='coerce')  # non-numbers become NaN

            if class_df[label].dropna().empty:
                print(f"Skipping '{label}' because it has no numeric values.")
                results[label] = float('nan')
            else:
                icc = pg.intraclass_corr(
                    data=class_df,
                    targets='ID',
                    raters='rater',
                    ratings=label
                )
                icc_value = icc.loc[icc['Type'] == 'ICC3', 'ICC'].values[0]
                results[label] = icc_value
        else:
            results[label] = float('nan')


    df = pd.DataFrame.from_dict(results, orient='index').transpose()

    # Save to CSV
    df.to_csv('output_icc_updated.csv', index=False)

def weighted_kappa_analysis(df1_input, df2_input):
    r1_file = df1_input.copy()
    r2_file = df2_input.copy()

    results = {}

    for label in r1_file.columns:
        if label == 'ID' or str(label).startswith("EX") or label == 'rater':
            continue

        # 1. Map Yes/No and ensure Numeric
        s1 = pd.to_numeric(r1_file[label], errors='coerce')
        s2 = pd.to_numeric(r2_file[label], errors='coerce')

        temp_df = pd.DataFrame({'R1': s1, 'R2': s2})
        temp_df = temp_df.dropna()

        if temp_df.empty:
            results[label] = float('nan')
            continue

        y1 = temp_df['R1'].astype(int)
        y2 = temp_df['R2'].astype(int)
        
        unique_values = np.unique(np.concatenate([y1, y2]))
        
        if len(unique_values) < 2:
             # Perfect agreement on a constant value (e.g. both voted 1 for everything)
             # Kappa is technically 1.0, but sklearn might return 0 or nan if vectors are constant.
             # Manual check:
            if np.array_equal(y1, y2):
                results[label] = 1.0
            else:
                results[label] = 0.0 # Disagreement on constants
            continue

        k = cohen_kappa_score(y1, y2, weights='quadratic')
            
        results[label] = k

    df = pd.DataFrame.from_dict(results, orient='index').transpose()

    # Save to CSV
    df.to_csv('output_kappa_updated.csv', index=False)

if __name__ == "__main__":
    # 1. Get Data
    df1, df2 = data_preprocess()
    
    # 2. Run Original Analysis (Modified only to include Y/N columns)
    icc_analysis(df1, df2)
    
    # 3. Run New Weighted Kappa Analysis
    weighted_kappa_analysis(df1, df2)