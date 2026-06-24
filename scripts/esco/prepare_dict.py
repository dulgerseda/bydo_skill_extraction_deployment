import pandas as pd
from pathlib import Path
import re
import json

# Paths
input_path = Path("data/processed/esco_skills_selected.csv")
output_path = Path("data/processed/esco_skill_dictionary.json")

# read data
df = pd.read_csv(input_path)

# Python; programming
# Coding
# Software development, seperated by ; or \n

skill_dict = {}

for _, row in df.iterrows():
    # clean
    preferred = str(row["preferredLabel"]).strip()
    if not preferred or preferred.lower() == "nan":
        continue
        
    # clean regex and split alternative names
    alt_raw = str(row["altLabels"])
    if alt_raw.lower() == "nan" or alt_raw.strip() == "":
        alts = []
    else:
        # distunguish between ; and \n as separators, and strip each alternative name
        alts = [x.strip() for x in re.split(r"[;\n]+", alt_raw) if x.strip()]
    
    # clean description
    description = str(row["description"]).strip()
    if description.lower() == "nan":
        description = ""

    # add to dictionary
    skill_dict[preferred] = {
        "alternative_names": alts,
        "description": description
    }

# save as JSON 
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(skill_dict, f, ensure_ascii=False, indent=2)

print(f"dictionary created: {len(skill_dict)} skills.")