import argparse
import asyncio
from io import StringIO
import os
import numpy as np
import uvicorn
import pandas as pd
import logging
import aiohttp
import aioredis
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from asyncio import sleep
from dotenv import load_dotenv
from datetime import datetime
from helpers import json_encode_iso, slice_query_validation, format_event
from multiprocessing import Process

# pandas show all columns
pd.set_option("display.max_columns", None)

load_dotenv(".env")
DATA_FILE_PATH = os.getenv("PATH_TO_DATA_FOLDER")
WORKERS = int(os.getenv("WORKERS", "1"))
SOURCE_IP = os.getenv("SOURCE_IP", "0.0.0.0")
SOURCE_PORT = int(os.getenv("SOURCE_PORT", "8000"))
PREDICTION_SERVER_IP = os.getenv("PREDICTION_SERVER_IP", "0.0.0.0")
PREDICTION_SERVER_PORT = int(os.getenv("PREDICTION_SERVER_PORT", "8001"))
CRI_SERVER_IP = os.getenv("CRI_SERVER_IP", "0.0.0.0")
CRI_SERVER_PORT = int(os.getenv("CRI_SERVER_PORT", "8002"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
TRAJECTORY_TIME_THRESHOLD = int(os.getenv("TRAJECTORY_TIME_THRESHOLD", 1920))

logger = logging.getLogger("uvicorn.error")

LOG_LEVEL = logging.INFO

AIS_STATE = {
    "data": pd.DataFrame(),
    "latest_vessel_states": pd.DataFrame(),
    "last_updated_hour": 0,
    "trajectory_predictions": pd.DataFrame(),
}


PREDICTIONS = pd.DataFrame()

CRI_for_vessels = pd.DataFrame()
Future_CRI_for_vessels = pd.DataFrame()

VESSEL_DATA = {}

BATCH_SIZE = 32


async def get_redis_connection():
    return await aioredis.from_url(REDIS_URL)


async def consume_prediction_queue():
    redis = consume_app.state.redis
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
                logger.debug(f"Removed task {task_id} from processed tasks set")

            size = await redis.llen("prediction_queue")
            logger.info(f"Queued tasks set size: {size}")
        await asyncio.sleep(1)


async def update_ais_state():
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

        logger.info(f"Updated ais state. ({datetime.now().replace(microsecond=0)})")
    except FileNotFoundError:
        logger.error(f"File not found for hour {current_hour}.")
    except Exception as e:
        logger.error(f"Error reading file for hour {current_hour}: {e}")


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

    await sleep(60)


async def update_trajectory_predictions():
    redis = app.state.redis
    pred = await redis.get("predictions")
    if pred:
        AIS_STATE["trajectory_predictions"] = pd.read_json(
            StringIO(pred.decode("utf-8"))
        )
    else:
        logger.warning("No predictions found in Redis.")

    await sleep(10)


async def ais_state_updater():
    while True:
        if (
            AIS_STATE["last_updated_hour"] != datetime.now().hour
            or AIS_STATE["data"] is None
        ):
            try:
                await update_ais_state()
            except Exception as e:
                logger.error(e)

        await update_latest_vessel_states()

        await update_trajectory_predictions()

        await sleep(1)


async def consumer_state_updater():
    while True:
        if (
            AIS_STATE["last_updated_hour"] != datetime.now().hour
            or AIS_STATE["data"] is None
        ):
            try:
                await update_ais_state()
            except Exception as e:
                logger.error(e)

        await sleep(1)


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


async def interpolate_ais_data(data: pd.DataFrame) -> pd.DataFrame:
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


async def preprocess_ais():
    """
    Preprocesses the AIS data by filtering out invalid data and grouping the data by MMSI.
    Also finds out which ships can be sent for prediction based on the time difference between the first and last record.
    """
    prev_timestamp = 0
    batch = []

    while True:
        df, curr_timestamp = await get_current_ais_data()

        if df.empty:
            logger.warning(f"There was no ais data for current time: {curr_timestamp}")
            await sleep(1)
            continue

        # We might be too fast and get data for the same timestamp again
        if prev_timestamp == curr_timestamp:
            await sleep(1)
            continue

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

                interpolated_df = await interpolate_ais_data(vessel_df)

                interpolated_df["mmsi"] = name

                expected_df_len = (TRAJECTORY_TIME_THRESHOLD / 60) + 1

                # This is pretty scuffed, but works. Might have to truncate data points before interpolation
                interpolated_df = interpolated_df.iloc[: int(expected_df_len)]

                # Check if this MMSI and timestamp pair has been processed
                timestamp = interpolated_df["timestamp"].iloc[0]
                task_id = f"{name}:{timestamp}"

                redis = app.state.redis

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
                    > first_timestamp + pd.Timedelta(minutes=2)
                ]

        prev_timestamp = curr_timestamp
        await sleep(1)


async def process_and_send_batch(batch):
    # Combine all dataframes in the batch into one
    assert len(batch) == BATCH_SIZE
    combined_batch = pd.concat(batch, ignore_index=True)

    # Send the combined batch to the Redis queue
    redis = app.state.redis
    await redis.rpush("prediction_queue", combined_batch.to_json())

    logger.debug(f"Sent a batch of {len(batch)} tasks to the prediction queue.")


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

            await redis.set("predictions", PREDICTIONS.to_json(orient="records"))
        else:
            logger.warning(f"No predictions received for batch: {batch}")

        await sleep(1)


async def get_current_CRI_and_clusters_for_vessels():
    """
    Gets the CRI and clusters for the vessels from the prediction server.
    """
    CRI_server_url = f"http://{CRI_SERVER_IP}:{CRI_SERVER_PORT}/clusters/current"

    async with aiohttp.ClientSession(trust_env=True) as session:
        while True:
            global CRI_for_vessels
            try:
                async with session.get(CRI_server_url) as response:
                    if response.status == 200:
                        json = await response.json()
                        CRI_for_vessels = pd.DataFrame(json["clusters"])
                        logger.debug(f"Received CRI and clusters: {CRI_for_vessels}")
                    else:
                        logger.warning(f"Failed to get CRI and clusters: {response}")
            except aiohttp.ClientError as e:
                logger.error(f"Network error while getting CRI and clusters: {e}")
            await sleep(5)


async def get_future_CRI_for_vessels():
    """
    Fetch future Collision Risk Index (CRI) for vessels in `CRI_for_vessels` dataframe.
    Only vessel pairs where both have predictions are sent in the request.
    """
    global Future_CRI_for_vessels

    async with aiohttp.ClientSession() as session:
        while True:
            predictions = AIS_STATE["trajectory_predictions"]
            if predictions.empty or CRI_for_vessels.empty:
                await sleep(5)
                continue

            # Filter CRI_for_vessels to include only pairs where both vessels have predictions
            valid_pairs = CRI_for_vessels[
                CRI_for_vessels.apply(
                    lambda row: row["vessel_1"] in predictions["mmsi"].values
                    and row["vessel_2"] in predictions["mmsi"].values,
                    axis=1,
                )
            ]

            if valid_pairs.empty:
                await sleep(5)
                continue

            # Prepare request data
            request_data = {
                "pairs": valid_pairs.to_dict(orient="records"),
                "predictions": predictions.to_dict(orient="records"),
            }
            # Prepare the JSON object with the required data
            future_cri_data = []

            for _, row in valid_pairs.iterrows():
                vessel_1_predictions = predictions[
                    predictions["mmsi"] == row["vessel_1"]
                ]
                vessel_2_predictions = predictions[
                    predictions["mmsi"] == row["vessel_2"]
                ]

                if not vessel_1_predictions.empty and not vessel_2_predictions.empty:
                    future_cri_data.append(
                        {
                            "vessel_1": row["vessel_1"],
                            "vessel_2": row["vessel_2"],
                            "vessel_1_speed": row["vessel_1_speed"],
                            "vessel_2_speed": row["vessel_2_speed"],
                            "vessel_1_longitude": vessel_1_predictions["lon"].tolist()[
                                -1
                            ],
                            "vessel_2_longitude": vessel_2_predictions["lon"].tolist()[
                                -1
                            ],
                            "vessel_1_latitude": vessel_1_predictions["lat"].tolist()[
                                -1
                            ],
                            "vessel_2_latitude": vessel_2_predictions["lat"].tolist()[
                                -1
                            ],
                            "vessel_1_course": row["vessel_1_course"],
                            "vessel_2_course": row["vessel_2_course"],
                        }
                    )

            request_data = {"future_cri_data": future_cri_data}
            future_cri_server_url = (
                f"http://{CRI_SERVER_IP}:{CRI_SERVER_PORT}/clusters/future"
            )

            try:
                response = await post_to_server(
                    request_data, session, future_cri_server_url
                )

                if response:
                    global Future_CRI_for_vessels
                    Future_CRI_for_vessels = pd.DataFrame(response["future_cri"])
                else:
                    logger.warning("No future CRI received for any pairs.")
            except Exception as e:
                logger.error(f"Error fetching future CRI: {e}")

            await sleep(5)


async def post_to_server(data: dict, client_session: aiohttp.ClientSession, url: str):
    try:
        logger.info(f"Posting data to {url}")
        async with client_session.post(url, json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.warning(f"Failed to post data to {url}: {response}")
    except Exception as e:
        logger.error(f"Network error while posting to {url}: {e}")
        return None


async def get_ais_cri():
    async with aiohttp.ClientSession() as session:
        while True:
            predictions = AIS_STATE["trajectory_predictions"]
            if predictions.empty:
                await sleep(1)
                continue

            data = predictions[["timestamp", "mmsi", "lon", "lat"]]
            data = await json_encode_iso(data)
            request_data = {"data": data}

            cri_calc_server_url = (
                f"http://{CRI_SERVER_IP}:{CRI_SERVER_PORT}/calculate_cri"
            )
            response = await post_to_server(request_data, session, cri_calc_server_url)

            if response:
                logger.debug(f"CRI received: {response}")
            else:
                logger.warning(f"No CRI received for data: {data}")

            await sleep(1)


async def ais_lat_long_slice_generator(
    latitude_range: str | None = None, longitude_range: str | None = None
):
    while True:
        data, _ = await get_current_ais_data()
        data = data[
            data["latitude"].between(latitude_range[0], latitude_range[1])
            & data["longitude"].between(longitude_range[0], longitude_range[1])
        ]
        data = await json_encode_iso(data)
        yield format_event("ais", data)
        await sleep(1)


async def ais_data_generator():
    while True:
        data, _ = await get_current_ais_data()
        data = await json_encode_iso(data)
        yield format_event("ais", data)
        await sleep(1)


async def dummy_prediction_generator():
    while True:
        data: pd.DataFrame = AIS_STATE["data"]
        current_time = pd.Timestamp.now()
        time_delta = (
            (current_time + pd.Timedelta(minutes=10)).time().replace(microsecond=0)
        )
        timestamp = data["timestamp"].dt.time

        data = data[(timestamp >= current_time.time()) & (timestamp <= time_delta)]
        data = data[["timestamp", "mmsi", "latitude", "longitude"]]
        data = await json_encode_iso(data)
        yield format_event("ais", data)
        await sleep(60)


async def predictions_generator(mmsi: int | None):
    """
    Asynchronous generator that yields prediction data for a given MMSI (Maritime Mobile Service Identity) or all MMSIs if none is provided.

    Args:
        mmsi (int | None): The MMSI to filter predictions by. If None, predictions for all MMSIs are returned.

    Yields:
        str: A formatted event string containing prediction data in JSON format.

    Notes:
        - The function continuously checks for new prediction data every 10 seconds.
        - If the predictions DataFrame is empty, an empty JSON array is returned.
        - The prediction data includes timestamp, mmsi, longitude, latitude, and future longitude and latitude columns.
    """
    predicted_cols = [f"lon(t+{i})" for i in range(1, 33)] + [
        f"lat(t+{i})" for i in range(1, 33)
    ]
    while True:
        predictions = AIS_STATE["trajectory_predictions"]
        if not predictions.empty:
            # If MMSI is provided in query, filter PREDICTIONS for that MMSI, else return all predictions
            data = (
                predictions
                if mmsi is None
                else predictions[predictions["mmsi"] == mmsi]
            )

            # Tranform predicted columns into rows for each timestamp
            # Melt only the future lat and lon columns
            df_melted = pd.melt(
                data,
                id_vars=["timestamp", "mmsi"],  # Keep these columns intact
                value_vars=predicted_cols,  # Only melt future columns
                var_name="coordinate_time",  # New column for melted keys
                value_name="value",  # New column for melted values
            )

            # Extract 'coordinate' (lon/lat) and 'time_step' (t+1, t+2, etc.)
            df_melted[["coordinate", "time_step"]] = df_melted[
                "coordinate_time"
            ].str.extract(r"(lon|lat)\(t\+(\d+)\)")

            # Convert 'time_step' to numeric
            df_melted["time_step"] = pd.to_numeric(
                df_melted["time_step"], errors="coerce"
            )

            # Drop rows with invalid extraction (i.e., when regex didn't match)
            df_melted = df_melted.dropna(subset=["coordinate", "time_step"])

            # Increment timestamp by timestep
            df_melted["timestamp"] = df_melted["timestamp"] + pd.to_timedelta(
                df_melted["time_step"], unit="m"
            )

            # Step 8: Pivot to separate lat and lon into different columns
            df_final = df_melted.pivot_table(
                index=["timestamp", "mmsi"],  # Group by timestamp and mmsi
                columns="coordinate",  # Separate lat and lon
                values="value",  # Use the melted values
                aggfunc="first",  # Resolve duplicates (if any)
            ).reset_index()

            df_final = df_final.rename(columns={"lat": "latitude", "lon": "longitude"})

            data = await json_encode_iso(df_final)

            yield format_event("prediction", data)

        await sleep(10)


async def CRI_generator():
    """
    Asynchronous generator that yields CRI data for all vessels.

    Yields:
        str: A formatted event string containing CRI data in JSON format.

    Notes:
        - The function continuously checks for new CRI data every 10 seconds.
        - If the CRI DataFrame is empty, an empty JSON array is returned.
        - The CRI data includes mmsi, cluster, and CRI columns.
    """
    while True:
        if not CRI_for_vessels.empty:
            data = await json_encode_iso(CRI_for_vessels)
        else:
            data = []

        yield format_event("cri", data)
        await sleep(10)


async def Future_CRI_generator():
    """
    Asynchronous generator that yields CRI data for all vessels.

    Yields:
        str: A formatted event string containing CRI data in JSON format.

    Notes:
        - The function continuously checks for new CRI data every 10 seconds.
        - If the CRI DataFrame is empty, an empty JSON array is returned.
        - The CRI data includes mmsi, cluster, and CRI columns.
    """
    while True:
        if not Future_CRI_for_vessels.empty:
            data = await json_encode_iso(Future_CRI_for_vessels)
        else:
            data = []

        yield format_event("future_cri", data)
        await sleep(10)


async def sse_data_generator(
    mmsi: int | None, latitude_range: tuple | None, longitude_range: tuple | None
):
    # Start both generators
    if latitude_range is None and longitude_range is None:
        ais_gen = ais_data_generator()
    else:
        ais_gen = ais_lat_long_slice_generator(latitude_range, longitude_range)

    predictions_gen = predictions_generator(mmsi)
    CRI_gen = CRI_generator()
    Future_CRI_gen = Future_CRI_generator()

    # Track the next events for each generator
    next_ais_event = asyncio.create_task(ais_gen.__anext__())
    next_predictions_event = asyncio.create_task(predictions_gen.__anext__())
    next_cri_event = asyncio.create_task(CRI_gen.__anext__())
    next_future_cri_event = asyncio.create_task(Future_CRI_gen.__anext__())

    while True:
        # Wait for any generator to produce an event
        done, _ = await asyncio.wait(
            [
                next_ais_event,
                next_predictions_event,
                next_cri_event,
                next_future_cri_event,
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            yield task.result()  # Yield the completed event

            # Restart the generator that completed
            if task == next_ais_event:
                next_ais_event = asyncio.create_task(ais_gen.__anext__())
            elif task == next_predictions_event:
                next_predictions_event = asyncio.create_task(
                    predictions_gen.__anext__()
                )
            elif task == next_cri_event:
                next_cri_event = asyncio.create_task(CRI_gen.__anext__())
            elif task == next_future_cri_event:
                next_future_cri_event = asyncio.create_task(Future_CRI_gen.__anext__())


async def get_current_ais_data():
    current_time = pd.Timestamp.now().time().replace(microsecond=0)
    timestamp = AIS_STATE["data"]["timestamp"].dt.time
    result: pd.DataFrame = AIS_STATE["data"][timestamp == current_time]

    return result, current_time


app = FastAPI(debug=True)
consume_app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def app_startup():
    app.state.start_time = datetime.now()
    app.state.redis = await get_redis_connection()

    try:
        await app.state.redis.delete("prediction_queue")
        logger.info("Deleted prediction queue")
    except Exception as e:
        logger.error(f"Error deleting prediction queue: {e}")

    asyncio.create_task(ais_state_updater())
    asyncio.create_task(preprocess_ais())
    asyncio.create_task(get_current_CRI_and_clusters_for_vessels())
    asyncio.create_task(get_future_CRI_for_vessels())
    # asyncio.create_task(consume_prediction_queue())


@app.get("/ais-state")
async def get_ais_state():
    return AIS_STATE["data"].to_json(orient="records", date_format="iso")


async def consume_app_startup():
    consume_app.state.redis = await get_redis_connection()
    asyncio.create_task(consume_prediction_queue())
    asyncio.create_task(consumer_state_updater())


app.add_event_handler("startup", app_startup)
consume_app.add_event_handler("startup", consume_app_startup)


def run_consume_app():
    uvicorn.run(
        "main:consume_app",
        host=SOURCE_IP,
        port=SOURCE_PORT + 1,  # Use a different port for the worker
        log_level=LOG_LEVEL,
        workers=WORKERS,
    )


@app.get("/uptime")
async def uptime():
    # Time elapsed since server startup
    uptime = str(datetime.now() - app.state.start_time)
    return {"uptime": uptime}


@app.get("/latest-vessel-states")
async def latest_vessel_states():
    return AIS_STATE["latest_vessel_states"].to_json(
        orient="records", date_format="iso"
    )


@app.get("/dummy-ais-data")
async def ais_data_fetch():
    generator = ais_data_generator()
    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/dummy-prediction")
async def dummy_prediction_fetch():
    generator = dummy_prediction_generator()
    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/dummy-CRI")
async def dummy_CRI_fetch():
    generator = CRI_generator()
    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/dummy-future-CRI")
async def dummy_CRI_fetch():
    generator = Future_CRI_generator()
    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/slice")
async def location_slice(
    latitude_range: str | None = None, longitude_range: str | None = None
):
    """
    Fetches AIS data based on the provided latitude and longitude ranges.
    """

    if latitude_range is None and longitude_range is None:
        generator = ais_data_generator()
    else:
        lat_range, long_range = await slice_query_validation(
            latitude_range, longitude_range
        )
        generator = ais_lat_long_slice_generator(lat_range, long_range)

    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/predictions")
async def prediction_fetch(
    mmsi: int | None = None,
    latitude_range: str | None = None,
    longitude_range: str | None = None,
):
    """
    Fetches prediction data  based on the provided parameters.

    This asynchronous function fetches prediction data for a given MMSI (Maritime Mobile Service Identity) and/or a specified latitude and longitude range. If latitude or longitude ranges are provided, they are validated and sliced accordingly.

    Args:
        mmsi (int | None): The Maritime Mobile Service Identity number. Defaults to None.
        latitude_range (str | None): The range of latitudes to filter the data. Defaults to None.
        longitude_range (str | None): The range of longitudes to filter the data. Defaults to None.

    Returns:
        StreamingResponse: A streaming response with the prediction data in "text/event-stream" format.
    """
    if latitude_range is not None or longitude_range is not None:
        latitude_range, longitude_range = await slice_query_validation(
            latitude_range, longitude_range
        )
    generator = sse_data_generator(mmsi, latitude_range, longitude_range)
    return StreamingResponse(generator, media_type="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if args.debug:
        LOG_LEVEL = logging.DEBUG

    # Run the consume app in a separate process
    consume_process = Process(target=run_consume_app)
    consume_process.start()

    uvicorn.run(
        "main:app",
        host=SOURCE_IP,
        port=SOURCE_PORT,
        log_level=LOG_LEVEL,
        workers=WORKERS,
    )

    consume_process.join()
