import asyncio
from io import StringIO
import os

import aiohttp
import numpy as np
import pandas as pd

from helpers import json_encode_iso, get_redis_connection, post_to_server

from ais_state import get_current_ais_data
import time

# PREDICTION_SERVER_IP = os.getenv("PREDICTION_SERVER_IP", "0.0.0.0")
# PREDICTION_SERVER_PORT = os.getenv("PREDICTION_SERVER_PORT", "8001")

PREDICTION_SERVER_IP="130.225.37.197"
PREDICTION_SERVER_PORT="8080"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

PREDICTIONS = pd.DataFrame()

BATCH_SIZE = 32
TRAJECTORY_TIME_THRESHOLD = 1 * 60  # 32 minutes

VESSEL_DATA = {}

async def process_and_send_batch(batch):
    # Combine all dataframes in the batch into one
    assert len(batch) == BATCH_SIZE
    combined_batch = pd.concat(batch, ignore_index=True)

    # Send the combined batch to the Redis queue
    redis = await get_redis_connection(REDIS_URL)
    await redis.rpush("prediction_queue", combined_batch.to_json())

    print(f"Sent a batch of {len(batch)} tasks to the prediction queue.")


async def interpolate_trajectory(data: pd.DataFrame) -> pd.DataFrame:
    # Create start and end time boundaries
    start_time = data["timestamp"].min().floor("min")
    end_time = data["timestamp"].max()

    # Generate complete range of timestamps at 1-minute intervals
    complete_range = pd.date_range(start=start_time, end=end_time, freq="1min")

    interpolated_df = pd.DataFrame(index=complete_range)

    # Interpolate the longitude and latitude values
    interpolated_df["longitude"] = np.interp(
        interpolated_df.index.astype("int64"),
        data["timestamp"].astype("int64"),
        data["longitude"],
    )
    interpolated_df["latitude"] = np.interp(
        interpolated_df.index.astype("int64"),
        data["timestamp"].astype("int64"),
        data["latitude"],
    )

    # Add the 'timestamp' as a column instead of index in the interpolated dataframe
    interpolated_df["timestamp"] = interpolated_df.index

    interpolated_df.reset_index(drop=True, inplace=True)

    return interpolated_df

async def filter_ais_data(data: pd.DataFrame):
    """
    Removes invalid data from the AIS data. (Invalid MMSI, too low speed, moored vessels, etc.)
    """

    data = data[(data["mmsi"] >= 201000000) & (data["mmsi"] <= 775999999)]
    data = data.loc[data["navigational status"] != "Moored"]
    data = data.loc[data["sog"] >= 1]
    data = data.loc[data["sog"] <= 50]
    data = data.drop_duplicates(subset=["timestamp", "mmsi"])
    data = data.dropna(subset=["longitude", "latitude", "sog", "cog"])
    data = data.replace([np.inf, -np.inf, np.nan], None)

    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")

    data.reset_index(drop=True, inplace=True)
    return data

async def consume_prediction_queue(redis):
    while True:
        # Blocking pop for a task
        task = await redis.blpop("prediction_queue")
        if task:
            _, task_data = task
            batch = pd.read_json(StringIO(task_data.decode("utf-8")))

            await get_ais_prediction(batch, redis)
            # After processing, remove each task (mmsi:timestamp) from the Redis set
            for mmsi, vessel in batch.groupby(
                "mmsi"
            ):  # Loop through the vessels in the batch
                mmsi = mmsi  # MMSI of the vessel
                timestamp = vessel["timestamp"].iloc[0]  # Timestamp of the vessel
                task_id = f"{mmsi}:{timestamp}"

                # Remove from the tracking set (processed_tasks)
                await redis.srem("processed_tasks", task_id)

            size = await redis.llen("prediction_queue")
            print(f"Queued tasks set size: {size}")

async def get_ais_prediction(batch: pd.DataFrame, redis):
    """
    Sends the AIS data from the prediction queue to the prediction server and receives the predictions.
    """
    async with aiohttp.ClientSession() as session:
        data = batch

        request_data = {"data": await json_encode_iso(data)}

        prediction_server_url = (
            f"http://{PREDICTION_SERVER_IP}:{PREDICTION_SERVER_PORT}/predict"
        )
        response = await post_to_server(request_data, session, prediction_server_url)

        if response:
            prediction = pd.DataFrame(response["prediction"])
            prediction["timestamp"] = pd.to_datetime(prediction["timestamp"])
            global PREDICTIONS
            if not PREDICTIONS.empty:
                PREDICTIONS = pd.concat([PREDICTIONS, prediction])
            else:
                PREDICTIONS = prediction
            await redis.set("predictions", prediction.to_json())

        else:
            print(f"No predictions received for batch: {batch}")

async def preprocess_ais_for_prediction(redis):
    """
    Preprocesses the AIS data by filtering out invalid data and grouping the data by MMSI.
    Also finds out which ships can be sent for prediction based on the time difference between the first and last record.
    """
    prev_timestamp = 0
    batch = []

    while True:
        print("Getting current AIS data")
        start_time = time.time()
        df, curr_timestamp = await get_current_ais_data(redis)
        end_time = time.time()
        print(f"Time taken to get current AIS data: {end_time - start_time} seconds")
        print("Got current AIS data")

        if df.empty:
            print(f"There was no ais data for current time: {curr_timestamp}")
            await asyncio.sleep(1)
            continue

        # We might be too fast and get data for the same timestamp again
        if prev_timestamp == curr_timestamp:
            await asyncio.sleep(1)
            continue
        
        print("Filtering AIS data")
        filtered_data = await filter_ais_data(df)

        grouped_data = filtered_data.groupby("mmsi")

        for name, group in grouped_data:
            if name in VESSEL_DATA:
                VESSEL_DATA[name] = pd.concat([VESSEL_DATA[name], group])
            else:
                VESSEL_DATA[name] = group

            vessel_df: pd.DataFrame = VESSEL_DATA[name].copy()
            diff = vessel_df["timestamp"].max() - vessel_df["timestamp"].min()

            if diff.total_seconds() >= TRAJECTORY_TIME_THRESHOLD:
                
                interpolated_df = await interpolate_trajectory(vessel_df)

                interpolated_df["mmsi"] = name

                expected_df_len = (TRAJECTORY_TIME_THRESHOLD / 60) + 1

                # This is pretty scuffed, but works. Might have to truncate data points before interpolation
                interpolated_df = interpolated_df.iloc[: int(expected_df_len)]

                # Check if this MMSI and timestamp pair has been processed
                timestamp = interpolated_df["timestamp"].iloc[0]
                task_id = f"{name}:{timestamp}"

                # Check if the MMSI and timestamp combination is already in Redis
                is_new = await redis.sadd("processed_tasks", task_id)
                if is_new:
                    # If it's a new task, add to the batch
                    assert len(interpolated_df) == expected_df_len

                    interpolated_df["trajectory_id"] = task_id
                    batch.append(interpolated_df)

                    # If batch size is met, process and send to Redis
                    if len(batch) == BATCH_SIZE:
                        await process_and_send_batch(batch)
                        batch = []

                # Remove the first minute of data for the vessel
                first_timestamp = VESSEL_DATA[name]["timestamp"].min()
                VESSEL_DATA[name] = VESSEL_DATA[name][
                    VESSEL_DATA[name]["timestamp"]
                    > first_timestamp + pd.Timedelta(minutes=1)
                ]

        prev_timestamp = curr_timestamp
        await asyncio.sleep(1)

if __name__ == "__main__":
    async def main():
        redis = await get_redis_connection(REDIS_URL)
        try:
            await redis.delete("prediction_queue")
            print("Deleted prediction queue")
        except Exception as e:
            print(f"Error deleting prediction queue: {e}")
        await asyncio.gather(
            preprocess_ais_for_prediction(redis), consume_prediction_queue(redis)
        )
    asyncio.run(main())