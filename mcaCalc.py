import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

dataset = pd.read_csv('filtered_df.csv')
dataset.head()

ids = dataset.iloc[:, 0]          # first column
df_mca = dataset.iloc[:, 1:]    

import prince

mca = prince.MCA(
    n_components=3,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
)
mca = mca.fit(df_mca)

coords = mca.row_coordinates(df_mca)

# Get first letter of each ID
first_letters = ids.str[0]

# Turn letters into category → integer codes → colors
first_letters = ids.str[0]
cat = first_letters.astype("category")
codes = cat.cat.codes

# Scatterplot
plt.figure(figsize=(8,6))
scatter = plt.scatter(coords[0], coords[1], c=codes)

cmap = scatter.cmap
norm = scatter.norm

handles = []
for letter, code in zip(cat.cat.categories, range(len(cat.cat.categories))):
    color = cmap(norm(code))
    handles.append(
        mpatches.Patch(color=color, label=letter)
    )

plt.legend(handles=handles, title="First Letter")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.grid(True)
#plt.show()

#plt.savefig('mcaPlot.png')

column_loadings = mca.column_coordinates(df_mca)
print("Column Loadings (MCA):")
print(column_loadings)

column_loadings.to_csv('loadings.csv')


row_scores = mca.row_coordinates(df_mca)
print("\nRow Scores (MCA):")
print(row_scores)