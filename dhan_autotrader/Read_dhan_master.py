import pandas as pd

df = pd.read_csv("dhan_master.csv")
df.columns = df.columns.str.strip().str.lower()
print(df.columns.tolist())