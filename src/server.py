import argparse
import asyncio
from io import StringIO
import os
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
from helpers import json_encode_iso, slice_query_validation, format_event, get_redis_connection, post_to_server
from ais_state import get_current_ais_data


# pandas show all columns
pd.set_option("display.max_columns", None)

load_dotenv(".env")
DATA_FILE_PATH = os.getenv("PATH_TO_DATA_FOLDER")
WORKERS = int(os.getenv("WORKERS", "1"))
SOURCE_IP = os.getenv("SOURCE_IP", "0.0.0.0")
SOURCE_PORT = int(os.getenv("SOURCE_PORT", "8000"))
CRI_SERVER_IP = os.getenv("CRI_SERVER_IP", "0.0.0.0")
CRI_SERVER_PORT = os.getenv("CRI_SERVER_PORT", "8002")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

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

PREDICTIONS = pd.DataFrame()

CRI_for_vessels = pd.DataFrame()
Future_CRI_for_vessels = pd.DataFrame()

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
            if PREDICTIONS.empty or CRI_for_vessels.empty:
                await sleep(5)
                continue

            # Filter CRI_for_vessels to include only pairs where both vessels have predictions
            valid_pairs = CRI_for_vessels[
                CRI_for_vessels.apply(
                    lambda row: row["vessel_1"] in PREDICTIONS["mmsi"].values
                    and row["vessel_2"] in PREDICTIONS["mmsi"].values,
                    axis=1,
                )
            ]

            if valid_pairs.empty:
                await sleep(5)
                continue

            # Prepare request data
            request_data = {
                "pairs": valid_pairs.to_dict(orient="records"),
                "predictions": PREDICTIONS.to_dict(orient="records"),
            }
            # Prepare the JSON object with the required data
            future_cri_data = []

            for _, row in valid_pairs.iterrows():
                vessel_1_predictions = PREDICTIONS[
                    PREDICTIONS["mmsi"] == row["vessel_1"]
                ]
                vessel_2_predictions = PREDICTIONS[
                    PREDICTIONS["mmsi"] == row["vessel_2"]
                ]

                if not vessel_1_predictions.empty and not vessel_2_predictions.empty:
                    future_cri_data.append(
                        {
                            "vessel_1": row["vessel_1"],
                            "vessel_2": row["vessel_2"],
                            "vessel_1_speed": vessel_1_predictions[
                                "speed(t+32)"
                            ].tolist()[-1],
                            "vessel_2_speed": vessel_2_predictions[
                                "speed(t+32)"
                            ].tolist()[-1],
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
    app.state.start_time = datetime.now()
    app.state.redis = await get_redis_connection(REDIS_URL)
    
    # asyncio.create_task(get_current_CRI_and_clusters_for_vessels())
    # asyncio.create_task(get_future_CRI_for_vessels())


app.add_event_handler("startup", startup)


async def ais_lat_long_slice_generator(
    latitude_range: str | None = None, longitude_range: str | None = None
):
    while True:
        data, _ = await get_current_ais_data(app.state.redis)
        data = data[
            data["latitude"].between(latitude_range[0], latitude_range[1])
            & data["longitude"].between(longitude_range[0], longitude_range[1])
        ]
        data = await json_encode_iso(data)
        yield format_event("ais", data)
        await sleep(1)


async def ais_data_generator():
    while True:
        data, _ = await get_current_ais_data(app.state.redis)
        data = await json_encode_iso(data)
        yield format_event("ais", data)
        await sleep(1)


async def dummy_prediction_generator():
    while True:
        redis = app.state.redis
        data = await get_ais_state(redis)
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

async def get_ais_state(redis):
    print("Before")
    ais_state = await redis.get("ais_state")
    print("After")
    if ais_state:
        print("HERHER ")
        data = pd.read_json(StringIO(ais_state.decode("utf-8")))
        print("YEAH BUDDY")
    else:
        data = pd.DataFrame()
    return data


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
        redis = app.state.redis
        predictions = await redis.get("predictions")
        if predictions:
            PREDICTIONS = pd.read_json(StringIO(predictions.decode("utf-8")))
        else:
            PREDICTIONS = pd.DataFrame()
        if not PREDICTIONS.empty:
            # If MMSI is provided in query, filter PREDICTIONS for that MMSI, else return all predictions
            data = (
                PREDICTIONS
                if mmsi is None
                else PREDICTIONS[PREDICTIONS["mmsi"] == mmsi]
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




@app.get("/uptime")
async def uptime():
    # Time elapsed since server startup
    uptime = str(datetime.now() - app.state.start_time)
    return {"uptime": uptime}


@app.get("/latest-vessel-states")
async def latest_vessel_states():
    print("BEFORE")
    states_json = await get_ais_state(app.state.redis)
    print("AFTER")
    return states_json.to_json(
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
                    print("len", len(batch))
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

    uvicorn.run(
        "server:app",
        host=SOURCE_IP,
        port=SOURCE_PORT,
        log_level=LOG_LEVEL,
        workers=WORKERS,
    )
