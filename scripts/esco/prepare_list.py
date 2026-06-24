import pandas as pd
from pathlib import Path
import re

# paths
input_path = Path("data/processed/esco_skills_selected.csv")
output_path = Path("data/processed/esco_all_skill_names_including_alts.csv")

# read data
df = pd.read_csv(input_path)

# preferred labels
preferred_series = df["preferredLabel"].astype(str).str.strip()

# alt labels: split by ; and \n, strip, filter out empty and "nan" values
# "Python; programming\nCoding" → ["Python", "programming", "Coding"]

alt_series = (
    df["altLabels"]
    .apply(lambda x: [i.strip() for i in re.split(r"[;\n]+", str(x)) if i.strip() and str(i).lower() != "nan"])
)

# expand alt labels into a single series
# ["Python", "programming"] → Python
#                             programming

alt_expanded = alt_series.explode()

# combine preferred and alt labels, clean, deduplicate, and sort
all_names = pd.concat([preferred_series, alt_expanded], ignore_index=True)

all_names = (
    all_names
    .astype(str)  
    .str.strip()
    .replace(["nan", "None", ""], pd.NA)
    .dropna()
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

# save to CSV
all_names.to_frame("skill_name").to_csv(output_path, index=False)

print(f"all skill names saved to: {output_path} ({len(all_names)} skills)")


