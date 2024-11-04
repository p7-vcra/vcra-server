import asyncio
import os
import numpy as np
import uvicorn
import pandas as pd
import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from asyncio import sleep
from dotenv import load_dotenv 
from datetime import datetime 
from helpers import slice_query_validation

load_dotenv('.env')
DATA_SUPPLIER_SERVER = os.getenv('PATH_TO_DATA_FOLDER')
DATA_FILE_PATH = str(os.getenv('PATH_TO_DATA_FOLDER'))
WORKERS = int(os.getenv('WORKERS'))
SOURCE_IP = os.getenv('SOURCE_IP')
SOURCE_PORT = int(os.getenv('SOURCE_PORT'))

logger = logging.getLogger('uvicorn.error')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ais_state = {
    "data": None,
    "last_updated_hour": None
}

vessel_data = {}

trajectories = []

vessel_records_threshold = 10

async def update_ais_state():
    current_hour = datetime.now().hour
    ais_state["last_updated_hour"] = current_hour
    ais_state["data"] = pd.read_feather(DATA_FILE_PATH+'aisdk-2024-09-09-hour-' + str(current_hour) + '.feather')
    logger.info(f"Updated ais state. ({datetime.now().replace(microsecond=0)})")

async def ais_state_updater():
    while True:
        if ais_state["last_updated_hour"] != datetime.now().hour or ais_state["data"] is None:
            await update_ais_state()
        await asyncio.sleep(0)

async def filter_ais_data(data: pd.DataFrame):
    data = data.rename(lambda x:x.lower(), axis="columns")
    data = data.loc[data["navigational status"] != "Moored"]
    data = data.loc[data["type of mobile"] != "Base Station"]
    data = data.loc[data["sog"] != 0]
    data = data.loc[data["sog"] <= 50]
    data = data.drop_duplicates(subset=["# timestamp", "mmsi"])
    data = data.dropna(subset=["longitude", "latitude", "sog", "cog"])
    data = data.replace([np.inf, -np.inf, np.nan], None)

    return data

async def preprocess_ais():
    while True:
        data: pd.DataFrame = await get_current_ais_data()
        if data.empty:
            logger.warning("There was no ais data for current time")
            await sleep(0)
            continue

        filtered_data = await filter_ais_data(data)

        grouped_data = filtered_data.groupby("mmsi")

        for name, group in grouped_data:
            if name in vessel_data:
                vessel_data[name].extend(group.to_dict(orient="records"))
            else:
                vessel_data[name] = group.to_dict(orient="records")

            if len(vessel_data[name]) >= vessel_records_threshold:
                trajectories.append(vessel_data[name]) 
                vessel_data[name] = []  

        await sleep(1)

async def startup():
    asyncio.create_task(ais_state_updater())
    asyncio.create_task(preprocess_ais())

app.add_event_handler("startup", startup)

async def ais_lat_long_slice_generator(latitude_range: tuple, longitude_range: tuple):
    while True:  
        data: pd.DataFrame = await get_current_ais_data()
        data = data[data["Latitude"].between(latitude_range[0], latitude_range[1]) & data["Longitude"].between(longitude_range[0], longitude_range[1])]
        data = data.to_json(orient="records", date_format="iso")
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(1)
        
async def ais_data_generator():
    while True: 
        data: pd.DataFrame = await get_current_ais_data()
        data = data.to_json(orient='records', date_format="iso")
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(1)

async def dummy_prediction_generator():
    while True:
        data: pd.DataFrame = ais_state["data"]
        current_time = pd.Timestamp.now()
        time_delta = (current_time + pd.Timedelta(minutes=10)).time().replace(microsecond=0)
        timestamp = data["# Timestamp"].dt.time

        data = data[(timestamp >= current_time.time()) & 
                    (timestamp <= time_delta)]
        data = data.to_json(orient="records", date_format="iso")
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(60)

async def get_current_ais_data():
    current_time = pd.Timestamp.now().time().replace(microsecond=0)
    timestamp = ais_state["data"]["# Timestamp"].dt.time
    result = ais_state["data"][timestamp == current_time]

    return result

@app.get("/dummy-ais-data")
async def ais_data_fetch():
    generator = ais_data_generator()
    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/dummy-prediction")
async def prediction_fetch():
    generator = dummy_prediction_generator()
    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/slice")
async def location_slice(latitude_range: str | None = None, longitude_range: str | None = None):

    if latitude_range is None and longitude_range is None:
        generator = ais_data_generator()
    else:
        lat_range, long_range = slice_query_validation(latitude_range, longitude_range)
        generator = ais_lat_long_slice_generator(lat_range, long_range)

    return StreamingResponse(generator, media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host=SOURCE_IP, port=SOURCE_PORT, reload=True)
