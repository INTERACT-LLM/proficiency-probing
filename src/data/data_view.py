
# %%
import pandas as pd
print("Pandas version:", pd.__version__)

# %%
# Load datasets from HuggingFace (only learner corpora for now)
df_elle_et = pd.read_json("hf://datasets/UniversalCEFR/elle_et/elle.json")
df_icle500_en = pd.read_json("hf://datasets/UniversalCEFR/icle500_en/icle500.json")
df_merlin_cs = pd.read_json("hf://datasets/UniversalCEFR/merlin_cs/merlin-cs.json")
df_merlin_it = pd.read_json("hf://datasets/UniversalCEFR/merlin_it/merlin-it.json")
df_merling_de = pd.read_json("hf://datasets/UniversalCEFR/merlin_de/merlin-de.json")
df_cefr_asag_en = pd.read_json("hf://datasets/UniversalCEFR/cefr_asag_en/cefr_asag.json")
df_cople2_pt = pd.read_json("hf://datasets/UniversalCEFR/cople2_pt/cople2.json")
df_peapl2_pt = pd.read_json("hf://datasets/UniversalCEFR/peapl2_pt/peapl2.json")
df_zaebuc_ar = pd.read_json("hf://datasets/UniversalCEFR/zaebuc_ar/zaebuc-ar.json")
# %% 
# Cleaning functions
import re
def strip_title(text: str) -> str:
    # Remove a leading line starting with "Title:" (case-insensitive), plus one blank line after it if present
    return re.sub(r"(?im)^title:.*\n\s*\n?", "", text, count=1).strip()

# %%
## Manual: Inspect datasets and clean as needed
# Remove all df_elle_et rows where text starts with "I OSA."
df_elle_et = df_elle_et[~df_elle_et["text"].str.startswith("I OSA.")]
df_icle500_en["text"] = df_icle500_en["text"].apply(strip_title)

# %%
# Combine all datasets into a single DataFrame
df_all = pd.concat([
    df_elle_et,
    df_icle500_en,
    df_merlin_cs,
    df_merlin_it,
    df_merling_de,
    df_cefr_asag_en,
    df_cople2_pt,
    df_peapl2_pt,
    df_zaebuc_ar
], ignore_index=True)


# Show list of all unique CEFR levels in the combined DataFrame with counts
print("Unique CEFR levels in combined dataset:")
print(df_all["cefr_level"].value_counts().sort_index())

# Remove Empty, NA and unrated entries
df_all = df_all[~df_all["cefr_level"].isin(["NA", "unrated", "EMPTY"])]
# Convert B1+ and B2+ into B1 and B2 respectively
df_all["cefr_level"] = df_all["cefr_level"].replace({"B1+": "B1", "B2+": "B2"})
# Store cefr_level as an ordered categorical variable
cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
df_all["cefr_level"] = pd.Categorical(df_all["cefr_level"], categories=cefr_order, ordered=True)
# Remove rows where cefr_level is not in the defined categories (if any)
df_all = df_all[df_all["cefr_level"].notna()]

# Show updated list of unique CEFR levels in the cleaned combined DataFrame with counts
print("Unique CEFR levels in cleaned combined dataset:")
print(df_all["cefr_level"].value_counts().sort_index())

# %%
# Show count of each cefr_level for each language
print("CEFR level distribution by language:")
print(df_all.groupby("lang")["cefr_level"].value_counts().unstack().fillna(0))
# Show marignal distributions of language and cefr_level:
print("Language distribution:")
print(df_all["lang"].value_counts())
print("CEFR level distribution:")
print(df_all["cefr_level"].value_counts().sort_index())


# %%
# Make a dataframe where each row represents a source_name, with a column for each cefr_level, containing the count of each level for that source_name
source_cefr_counts = df_all.groupby("source_name")["cefr_level"].value_counts().unstack(fill_value=0)
print(source_cefr_counts)

# Make a dataframe where each row represents source name, with a column for language:
source_language = df_all.groupby("source_name")["lang"].first()
print(source_language)
# %%
# Complete the two queries in one dataframe:  
source_summary = pd.concat([source_cefr_counts, source_language], axis=1)
# move the language column to the second column:
source_summary = source_summary.reindex(columns=['lang'] + [col for col in source_summary.columns if col != 'lang'])
print(source_summary)

# %%
# Change all cefr_level C1 or C2 to C+:
def map_cefr_level(level):
    if level in ["C1", "C2"]:
        return "C+"
    else:
        return level

df_all["cefr_level"] = df_all["cefr_level"].apply(map_cefr_level)

# Save the cleaned combined DataFrame to a new CSV file
df_all.to_csv("data/combined_cefr_data.csv", index=False)





# %%
