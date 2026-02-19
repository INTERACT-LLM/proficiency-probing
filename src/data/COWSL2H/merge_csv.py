# A python file for merging COWSL2H csv files into a single csv file for easier analysis and visualization.
# All flies is in the folder csv_corpus, and the output file will be called merged_cowsl2h.csv
# %%
import pandas as pd
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

# Save the merged DataFrame to a new csv file
merged_df.to_csv(output_file, index=False)
print(f"Merged {len(csv_files)} files into {output_file}")

# %%

merged_df[['id', 'prompt', 'listening comprehension', 'reading comprehension', 'speaking ability', 'writing ability', 'essay']]



# %%
