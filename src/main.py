import argparse
import asyncio
import json
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
from helpers import json_encode_iso, slice_query_validation



load_dotenv('.env')
DATA_SUPPLIER_SERVER = os.getenv('PATH_TO_DATA_FOLDER')
DATA_FILE_PATH = os.getenv('PATH_TO_DATA_FOLDER')
WORKERS = int(os.getenv('WORKERS', '1'))
SOURCE_IP = os.getenv('SOURCE_IP')
SOURCE_PORT = int(os.getenv('SOURCE_PORT', '8000'))
PREDICTION_SERVER_IP = os.getenv('PREDICTION_SERVER_IP')
PREDICTION_SERVER_PORT = os.getenv('PREDICTION_SERVER_PORT')

logger = logging.getLogger('uvicorn.error')

LOG_LEVEL = logging.INFO

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ais_state = {
    "data": pd.DataFrame(),
    "last_updated_hour": 0
}

vessel_data = {}

trajectory_queue = asyncio.Queue()

predictions = {}

vessel_records_threshold = 10

async def update_ais_state():
    current_hour = datetime.now().hour
    ais_state["last_updated_hour"] = current_hour
    try:
        ais_state["data"] = pd.read_feather(DATA_FILE_PATH +'aisdk-2024-09-09-hour-' + str(current_hour) + '.feather')
        ais_state["data"] = ais_state["data"].rename(lambda x:x.lower(), axis="columns")
        ais_state["data"] = ais_state["data"].rename(columns={"# timestamp": "timestamp"} )
        logger.info(f"Updated ais state. ({datetime.now().replace(microsecond=0)})")
    except FileNotFoundError:
        logger.error(f"File not found for hour {current_hour}.")
    except Exception as e:
        logger.error(f"Error reading file for hour {current_hour}: {e}")

async def ais_state_updater():
    while True:
        if ais_state["last_updated_hour"] != datetime.now().hour or ais_state["data"] is None:
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
            logger.warning("There was no ais data for current time")
            await sleep(1)
            continue
    
        # We might be too fast and get data for the same timestamp again
        if prev_timestamp == curr_timestamp:
            await sleep(1)
            continue

        filtered_data = await filter_ais_data(df)

        grouped_data = filtered_data.groupby("mmsi")

        for name, group in grouped_data:
            if name in vessel_data:
                vessel_data[name] = pd.concat([vessel_data[name], group])
            else:
                vessel_data[name] = group

            if len(vessel_data[name]) == vessel_records_threshold:
                await trajectory_queue.put(vessel_data[name]) 
                # Clear vessel data for mmsi
                vessel_data[name] = pd.DataFrame()

        prev_timestamp = curr_timestamp
        await sleep(1)

async def get_ais_prediction():
    async with aiohttp.ClientSession() as session:
        while True:
            if trajectory_queue.empty():
                await sleep(0)
                continue

            trajectory = await trajectory_queue.get()
            mmsi = str(trajectory["mmsi"].values[0])

            data = trajectory[["timestamp", "longitude", "latitude"]]

            request_data = {"data": await json_encode_iso(data)}
            
            response = await post_to_prediction_server(request_data, session)

            if response:
                prediction = response["prediction"]

                logger.debug(f"Prediction received: {prediction}")

                predictions[mmsi] = prediction
            else:
                logger.warning(f"No prediction received for trajectory: {trajectory}")
            
            await sleep(1)

async def post_to_prediction_server(trajectory: dict, client_session: aiohttp.ClientSession):
    prediction_server_url = f"http://{PREDICTION_SERVER_IP}:{PREDICTION_SERVER_PORT}/predict"
    try:
        async with client_session.post(prediction_server_url, json=trajectory) as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.warning(f"Failed to post data to prediction server: {response}")
    except aiohttp.ClientError as e:
        logger.error(f"Network error while posting to prediction server: {e}")
        return None

async def startup():
    asyncio.create_task(ais_state_updater())
    asyncio.create_task(preprocess_ais())
    asyncio.create_task(get_ais_prediction())

app.add_event_handler("startup", startup)

async def ais_lat_long_slice_generator(latitude_range: tuple, longitude_range: tuple):
    while True:  
        data, _ = await get_current_ais_data()
        data = data[data["latitude"].between(latitude_range[0], latitude_range[1]) & data["longitude"].between(longitude_range[0], longitude_range[1])]
        data = await json_encode_iso(data) 
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(1)
        
async def ais_data_generator():
    while True: 
        data, _ = await get_current_ais_data()
        data = await json_encode_iso(data)
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(1)

async def dummy_prediction_generator():
    while True:
        data: pd.DataFrame = ais_state["data"]
        current_time = pd.Timestamp.now()
        time_delta = (current_time + pd.Timedelta(minutes=10)).time().replace(microsecond=0)
        timestamp = data["timestamp"].dt.time

        data = data[(timestamp >= current_time.time()) & 
                    (timestamp <= time_delta)]
        data = data[["timestamp", "mmsi", "latitude", "longitude"]]
        data = await json_encode_iso(data) 
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(60)

async def predictions_generator(mmsi: int | None):
    while True:
        # data = pd.DataFrame.from_dict(predictions)
        # data = await json_encode_iso(data)
        if mmsi:
            data = json.dumps(predictions.get(str(mmsi)))
        else:
            data = json.dumps(predictions)
        
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(10)

async def get_current_ais_data():
    current_time = pd.Timestamp.now().time().replace(microsecond=0)
    timestamp = ais_state["data"]["timestamp"].dt.time
    result: pd.DataFrame = ais_state["data"][timestamp == current_time]

    if result.empty:
        logger.warning(f"Current time: {current_time} \n AIS Timestamp: {timestamp}")
        logger.warning(f"AIS state: ", ais_state["data"])

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
async def location_slice(latitude_range: str | None = None, longitude_range: str | None = None):

    if latitude_range is None and longitude_range is None:
        generator = ais_data_generator()
    else:
        lat_range, long_range = await slice_query_validation(latitude_range, longitude_range)
        generator = ais_lat_long_slice_generator(lat_range, long_range)

    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/predictions")
async def prediction_fetch(mmsi: int | None = None):
    generator = predictions_generator(mmsi)
    return StreamingResponse(generator, media_type="text/event-stream")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if args.debug:
        LOG_LEVEL = logging.DEBUG

    uvicorn.run("main:app", host=SOURCE_IP, port=SOURCE_PORT, log_level=LOG_LEVEL, workers=WORKERS)
