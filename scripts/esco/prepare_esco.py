import pandas as pd
from pathlib import Path

# file path
input_path = Path("data/raw/skills_en.csv")
output_path = Path("data/processed/esco_skills_selected.csv")

# read csv
try:
    df = pd.read_csv(input_path)
except:
    df = pd.read_csv(input_path, sep=";")

print(df.columns)

# selected columns
selected_columns = ["preferredLabel", "altLabels", "description"]
df_selected = df[selected_columns].copy()

# cleaning
df_selected = df_selected.fillna("").astype(str)

# save
output_path.parent.mkdir(parents=True, exist_ok=True)
df_selected.to_csv(output_path, index=False)

print(df_selected.head(10))

