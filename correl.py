import pandas as pd

df = pd.read_csv('full.csv')

df = df.set_index(df.columns[0])
df = df.drop(columns=df.columns[0])
df = df.T

cols_to_drop = [c for c in df.columns if "EX:" in str(c) or "Other" in str(c) or "Goal" in str(c) or "Present" in str(c)]
df = df.drop(columns=cols_to_drop)


df.to_csv('filtered_df.csv')
#corr = df.corr()

#print(corr)

#corr.to_csv('correlations.csv')