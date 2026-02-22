# A python file for merging COWSL2H csv files into a single csv file for easier analysis and visualization.
# All flies is in the folder csv_corpus, and the output file will be called merged_cowsl2h.csv
# %%
import pandas as pd
import numpy as np
import glob
import os

print(f"working directory: {os.getcwd()}")

input_folder = "csv_corpus"
output_file = "merged_cowsl2h.csv"


# Get a list of all csv files in the input folder
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# Read each csv file into a pandas DataFrame and store them in a list
dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)


# Clean writing ability column by converting textual descriptions to numeric values
def convert_writing_ability(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "Extremely uncomfortable" or value == "1.0":
            return 1
        elif value == "Uncomfortable" or value == "2.0":
            return 2
        elif value == "Neither comfortable nor uncomfortable" or value == "3.0":
            return 3
        elif value == "Comfortable" or value == "4.0":
            return 4
        elif value == "Extremely comfortable" or value == "5.0":
            return 5
    try:
        return int(value)
    except ValueError:
        return np.nan  # Return NaN for any non-convertible values
    

merged_df["writing ability"] = merged_df["writing ability"].apply(convert_writing_ability)


# Save the merged DataFrame to a new csv file
merged_df.to_csv(output_file, index=False)
print(f"Merged {len(csv_files)} files into {output_file}")

# %%

merged_df[['id', 'prompt', 'listening comprehension', 'reading comprehension', 'speaking ability', 'writing ability', 'essay']]



# %%
