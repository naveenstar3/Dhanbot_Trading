import pandas as pd
import re

# Load CSV
df = pd.read_csv('/mnt/data/dhan_master.csv')
name_col = [c for c in df.columns if 'symbol' in c.lower()][0]
id_col = [c for c in df.columns if 'security_id' in c.lower()][0]

def normalize(name):
    n = name.upper().replace('LTD.', 'LIMITED').replace('LTD', 'LIMITED')
    n = re.sub(r'[\.\'\",&\-\(\)]', '', n)
    n = re.sub(r'\s+', ' ', n)
    n = n.strip()
    return n

# Show top 30 normalized name-ID pairs
norm_map = {}
for i, row in df.iterrows():
    norm = normalize(row[name_col])
    norm_map[norm] = int(row[id_col])
norm_map_list = list(norm_map.items())[:30]  # Only first 30 for brevity
norm_map_list
