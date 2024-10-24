import os
import pandas as pd

CHUNKSIZE = 10**6
df = pd.read_csv("data/aisdk-2024-09-09.csv", chunksize=CHUNKSIZE)

os.makedirs("data", exist_ok=True)

for chunk in df: 
    chunk["# Timestamp"] = pd.to_datetime(chunk["# Timestamp"])  

    chunk["hour"] = chunk["# Timestamp"].dt.floor("h")  

    group: pd.DataFrame
    for hour, group in chunk.groupby("hour"):
        group.drop(columns="hour", inplace=True)
        file_path = f"data/aisdk-2024-09-09-hour-{hour.hour}.feather"
        if os.path.exists(file_path):
            file = pd.read_feather(file_path)
            combined_df = pd.concat([file, group])
            combined_df.to_feather(file_path)
        else: 
            group.to_feather(file_path)

