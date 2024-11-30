#%% imports
import pandas as pd
import re
from datetime import datetime
import csv


#%% Process data
chunk_size = 20000
data = pd.read_json("../data.json", lines=True, chunksize=chunk_size)

output_file_name = f'../data/pp-data-{datetime.now().timestamp()}.csv'
records_written = 0
with open(output_file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, )
    writer.writerow(['year', 'categories', 'title'])
    for chunk in data:
        for index, row in chunk.iterrows():
            # extract the first publication date
            date_string = row["versions"][0]["created"]
            date_object = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %Z")
            year = date_object.strftime("%Y")

            # lowercase all title words and abstract
            title_words = re.sub(r'[^\w\s\-]', '', row["title"].lower()) + re.sub(r'[^\w\s\-]', '', row["abstract"].lower())
            
            # extract categories
            categories = row["categories"]
            
            writer.writerow([year, categories, title_words])
            records_written += 1

        break
    print(f"Records written: {records_written}")

print(f"Total records written: {records_written}")

del data
df = pd.read_csv(output_file_name)
print(df.head())

#%% Check number of records
print(len(df))
# check for missing values
print(df.isnull().sum())

# %%
