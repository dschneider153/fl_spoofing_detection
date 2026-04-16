import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('features/features.csv', index_col=0)
df = df.drop('order_id', axis=1)
print(df.head())
print(df)
df["side"] = ((df["side"] == "B").astype(int))
corr_matrix = df.corr()

plt.figure(figsize=(12,12))
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
fig = heatmap.get_figure()
fig.savefig('features/corr_matrix.pdf')