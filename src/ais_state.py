from asyncio import sleep
import asyncio
from asyncio.log import logger
from datetime import datetime
from io import StringIO
import json

import numpy as np
import pandas as pd

from helpers import get_redis_connection

TRAJECTORY_TIME_THRESHOLD = 1 * 60  # 32 minutes
BATCH_SIZE = 32
DATA_FILE_PATH = "data/"
REDIS_URL = "redis://localhost:6379"
AIS_STATE = {
    "data": pd.DataFrame(),
    "latest_vessel_states": pd.DataFrame(),
    "last_updated_hour": 0,
}


async def update_ais_state(redis):
    """
    Updates the current AIS_file_state with the data from the feather file for the current hour. This is done to simulate real-time data updates.

    Exceptions:
        FileNotFoundError: If the feather file for the current hour is not found.
        Exception: For any other errors encountered during the file reading process.
    """
    current_hour = datetime.now().hour
    AIS_STATE["last_updated_hour"] = current_hour
    try:
        AIS_STATE["data"] = pd.read_feather(
            DATA_FILE_PATH + "aisdk-2024-09-09-hour-" + str(current_hour) + ".feather"
        )
        AIS_STATE["data"] = AIS_STATE["data"].rename(
            lambda x: x.lower(), axis="columns"
        )
        AIS_STATE["data"] = AIS_STATE["data"].rename(
            columns={"# timestamp": "timestamp"}
        )

        await redis.set("ais_state", AIS_STATE["data"].to_json())
        print(f"Updated ais state. ({datetime.now().replace(microsecond=0)})")
    except FileNotFoundError:
        print(f"File not found for hour {current_hour}.")
    except Exception as e:
        print(f"Error reading file for hour {current_hour}: {e}")


async def update_latest_vessel_states():
    """
    Updates the latest vessel states by taking the last record of each vessel from the AIS data.
    """
    if AIS_STATE["data"] is None:
        return

    current_time = datetime.now().time()
    data_up_to_now = AIS_STATE["data"][
        AIS_STATE["data"]["timestamp"].dt.time <= current_time
    ]
    latest_vessel_states = data_up_to_now.groupby("mmsi").last().reset_index()
    AIS_STATE["latest_vessel_states"] = latest_vessel_states

    redis = await get_redis_connection(REDIS_URL)
    await redis.set("latest_vessel_states", latest_vessel_states.to_json())
    await sleep(60)


async def ais_state_updater(redis):
    while True:
        if (
            AIS_STATE["last_updated_hour"] != datetime.now().hour
            or AIS_STATE["data"] is None
        ):
            try:
                await update_ais_state(redis)
            except Exception as e:
                print(e)

        await update_latest_vessel_states()

        await sleep(1)


async def get_current_ais_data(redis):
    current_time = pd.Timestamp.now().time().replace(microsecond=0)
    state_json = await redis.get("ais_state")
    state = pd.read_json(StringIO(state_json.decode("utf-8")))
    timestamp = state["timestamp"].dt.time
    result: pd.DataFrame = state[timestamp == current_time]

    return result, current_time


if __name__ == "__main__":
    async def main():
        redis = await get_redis_connection(REDIS_URL)
        await ais_state_updater(redis)

    asyncio.run(main())
