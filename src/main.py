import argparse
import asyncio
import os
import numpy as np
import uvicorn
import pandas as pd
import logging
import aiohttp
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from asyncio import sleep
from dotenv import load_dotenv
from datetime import datetime
from helpers import json_encode_iso, slice_query_validation, format_event

load_dotenv(".env")
DATA_FILE_PATH = os.getenv("PATH_TO_DATA_FOLDER")
WORKERS = int(os.getenv("WORKERS", "1"))
SOURCE_IP = os.getenv("SOURCE_IP", "0.0.0.0")
SOURCE_PORT = int(os.getenv("SOURCE_PORT", "8000"))
PREDICTION_SERVER_IP = os.getenv("PREDICTION_SERVER_IP", "0.0.0.0")
PREDICTION_SERVER_PORT = os.getenv("PREDICTION_SERVER_PORT", "8001")
CRI_SERVER_IP = os.getenv("CRI_SERVER_IP", "0.0.0.0")
CRI_SERVER_PORT = os.getenv("CRI_SERVER_PORT", "8002")

logger = logging.getLogger("uvicorn.error")

LOG_LEVEL = logging.INFO

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIS_STATE = {"data": pd.DataFrame(), "last_updated_hour": 0}

PREDICTION_QUEUE = asyncio.Queue()

TRAJECTORY_TIME_THRESHOLD = 900

PREDICTIONS = pd.DataFrame()

VESSEL_DATA = {}


async def update_ais_state():
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
        await sleep(1)


async def filter_ais_data(data: pd.DataFrame):
    # Remove invalid MMSI
    data = data[(data["mmsi"] >= 201000000) & (data["mmsi"] <= 775999999)]
    data = data.loc[data["navigational status"] != "Moored"]
    data = data.loc[data["sog"] != 0]
    data = data.loc[data["sog"] <= 50]
    data = data.drop_duplicates(subset=["timestamp", "mmsi"])
    data = data.dropna(subset=["longitude", "latitude", "sog", "cog"])
    data = data.replace([np.inf, -np.inf, np.nan], None)

    data.reset_index(drop=True, inplace=True)
    return data


async def preprocess_ais():
    prev_timestamp = 0

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

            diff = (
                VESSEL_DATA[name]["timestamp"].max()
                - VESSEL_DATA[name]["timestamp"].min()
            )

            vessel_df = VESSEL_DATA[name]
            if diff.total_seconds() >= TRAJECTORY_TIME_THRESHOLD:
                vessel_df.reset_index(drop=True, inplace=True)
                vessel_df = vessel_df.infer_objects(copy=False)
                interp_cols = ["longitude", "latitude"]
                vessel_df.loc[0, "timestamp"] = vessel_df.loc[0, "timestamp"].floor(
                    "min"
                )
                vessel_df = vessel_df.set_index("timestamp")
                vessel_df = vessel_df.resample("1min").asfreq()
                vessel_df[interp_cols] = vessel_df[interp_cols].interpolate(
                    method="linear"
                )
                vessel_df = vessel_df.ffill()
                vessel_df.reset_index(inplace=True)
                await PREDICTION_QUEUE.put(vessel_df)
                # Clear vessel data for mmsi
                VESSEL_DATA[name] = pd.DataFrame()

        prev_timestamp = curr_timestamp
        await sleep(1)


async def get_ais_prediction():
    async with aiohttp.ClientSession() as session:
        while True:
            if PREDICTION_QUEUE.empty():
                await sleep(0)
                continue

            trajectory = await PREDICTION_QUEUE.get()
            mmsi = str(trajectory["mmsi"].values[0])

            data = trajectory[["timestamp", "longitude", "latitude"]]

            request_data = {"data": await json_encode_iso(data)}

            prediction_server_url = (
                f"http://{PREDICTION_SERVER_IP}:{PREDICTION_SERVER_PORT}/predict"
            )
            response = await post_to_server(
                request_data, session, prediction_server_url
            )

            if response:
                prediction = pd.DataFrame(response["prediction"])

                logger.debug(f"Prediction received: {prediction}")

                global PREDICTIONS
                prediction["mmsi"] = mmsi
                prediction["timestamp"] = trajectory["timestamp"].values[1:]
                prediction["lon"] = trajectory["longitude"].values[1:]
                prediction["lat"] = trajectory["latitude"].values[1:]

                if not PREDICTIONS.empty:
                    PREDICTIONS = pd.concat([PREDICTIONS, prediction])
                else:
                    PREDICTIONS = prediction
            else:
                logger.warning(f"No prediction received for trajectory: {trajectory}")

            await sleep(1)


async def post_to_server(data: dict, client_session: aiohttp.ClientSession, url: str):
    try:
        async with client_session.post(url, json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.warning(f"Failed to post data to {url}: {response}")
    except aiohttp.ClientError as e:
        logger.error(f"Network error while posting to {url}: {e}")
        return None


async def get_ais_cri():
    async with aiohttp.ClientSession() as session:
        while True:
            if PREDICTIONS.empty:
                await sleep(1)
                continue

            data = PREDICTIONS[["timestamp", "mmsi", "lon", "lat"]]
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


async def startup():
    asyncio.create_task(ais_state_updater())
    asyncio.create_task(preprocess_ais())
    asyncio.create_task(get_ais_prediction())
    # asyncio.create_task(get_ais_cri())


app.add_event_handler("startup", startup)


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
    predicted_cols = [f"lon(t+{i})" for i in range(1, 33)] + [
        f"lat(t+{i})" for i in range(1, 33)
    ]
    while True:
        if not PREDICTIONS.empty:
            # If MMSI is provided in query, filter PREDICTIONS for that MMSI, else return all predictions
            PREDICTIONS.loc[:, "mmsi"] = (
                PREDICTIONS.loc[:, "mmsi"].astype(float).astype(int)
            )
            data = (
                PREDICTIONS
                if mmsi is None
                else PREDICTIONS[PREDICTIONS["mmsi"] == mmsi]
            )

            # Keep only last row in prediction since this is the most recent position for the vessel
            data = data[
                ["timestamp", "mmsi", "lon", "lat"] + predicted_cols
            ].drop_duplicates(subset=["mmsi"], keep="last")

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


async def sse_data_generator(
    mmsi: int | None, latitude_range: tuple | None, longitude_range: tuple | None
):
    # Start generators
    if latitude_range is None and longitude_range is None:
        ais_gen = ais_data_generator()
    else:
        ais_gen = ais_lat_long_slice_generator(latitude_range, longitude_range)

    predictions_gen = predictions_generator(mmsi)

    # Track the next events for each generator
    next_ais_event = asyncio.create_task(ais_gen.__anext__())
    next_predictions_event = asyncio.create_task(predictions_gen.__anext__())

    while True:
        # Wait for any generator to produce an event
        done, _ = await asyncio.wait(
            [next_ais_event, next_predictions_event],
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


async def get_current_ais_data():
    current_time = pd.Timestamp.now().time().replace(microsecond=0)
    timestamp = AIS_STATE["data"]["timestamp"].dt.time
    result: pd.DataFrame = AIS_STATE["data"][timestamp == current_time]

    return result, current_time


@app.get("/dummy-ais-data")
async def ais_data_fetch():
    generator = ais_data_generator()
    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/dummy-prediction")
async def dummy_prediction_fetch():
    generator = dummy_prediction_generator()
    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/slice")
async def location_slice(
    latitude_range: str | None = None, longitude_range: str | None = None
):

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

    uvicorn.run(
        "main:app",
        host=SOURCE_IP,
        port=SOURCE_PORT,
        log_level=LOG_LEVEL,
        workers=WORKERS,
    )
