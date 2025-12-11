import os
import pandas as pd

# creating sample dataframe
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
        }

df = pd.DataFrame(data)  # saving dataframe to csv file

data_dir = 'data'  # ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)  # saving dataframe to csv file

file_path = os.path.join(data_dir, 'sample.csv')

# saving dataframe to csv file and index is set to False to avoid writing row numbers
df.to_csv(file_path, index=False)

print(f"DataFrame saved to {file_path}")
