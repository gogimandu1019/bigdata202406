import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv("data/mtcars.csv")
#print(df.head())

from sklearn.preprocessing import minmax_scale
df['qsec2'] = minmax_scale(df['qsec'])
#print(df.head())

cond = (df['qsec2'] > 0.5)
print(len(df[cond]))
