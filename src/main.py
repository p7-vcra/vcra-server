import asyncio
import os
import numpy as np
import uvicorn
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from asyncio import sleep
from dotenv import load_dotenv 
from datetime import datetime

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

async def update_ais_state():
    current_hour = datetime.now().hour
    ais_state["last_updated_hour"] = current_hour
    ais_state["data"] = pd.read_feather(DATA_FILE_PATH+'aisdk-2024-09-09-hour-' + str(current_hour) + '.feather')
    ais_state["data"]["Time"] = ais_state["data"]["# Timestamp"].dt.time.astype(str)
    logger.info(f"Updated ais state. ({datetime.now().replace(microsecond=0)})")

async def ais_state_updater():
    while True:
        if ais_state["last_updated_hour"] != datetime.now().hour or ais_state["data"] is None:
            await update_ais_state()
        await asyncio.sleep(60)

async def preprocess_ais():
    while True:
        data: pd.DataFrame = await get_current_ais_data()
        if not data.empty:
            data = data.rename(lambda x:x.lower(), axis="columns")
            data = data.loc[data["navigational status"] != "Moored"]
            data = data.loc[data["type of mobile"] != "Base Station"]
            data = data.loc[data["sog"] != 0]
            data = data.loc[data["sog"] <= 50]
            data = data.drop_duplicates(subset=["mmsi", "# timestamp"])
            data = data.dropna(subset=["longitude", "latitude", "sog", "cog"])
            data = data.replace([np.inf, -np.inf, np.nan], None)
            
            grouped_data = data.groupby("mmsi")

            for name, group in grouped_data:
                if name in vessel_data:
                    vessel_data[name].extend(group.to_dict(orient="records"))
                else:
                    vessel_data[name] = group.to_dict(orient="records")
        else:
            logger.warning("There was no ais data for current time")
        
        await sleep(1)

async def startup():
    asyncio.create_task(ais_state_updater())
    asyncio.create_task(preprocess_ais())

app.add_event_handler("startup", startup)

async def ais_lat_long_slice_generator(latitude_range: tuple, longitude_range: tuple):
    while True:  
        data: pd.DataFrame = await get_current_ais_data()
        data = data[data["Latitude"].between(latitude_range[0], latitude_range[1]) & data["Longitude"].between(longitude_range[0], longitude_range[1])]
        data = data.to_json(orient="records")
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(1)
        
async def ais_data_generator():
    while True: 
        data: pd.DataFrame = await get_current_ais_data()
        data = data.to_json(orient='records')
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(1)

async def get_current_ais_data():
    current_time = datetime.now().time().strftime("%H:%M:%S")
    return ais_state["data"][ais_state["data"]["Time"] == current_time]

@app.get("/dummy-ais-data")
async def ais_data_fetch():
    generator = ais_data_generator()
    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/slice")
async def location_slice(latitude_range: str, longitude_range: str):
    latitude_range = latitude_range.split(",")
    longitude_range = longitude_range.split(",")

    if len(latitude_range) != 2:
        raise HTTPException(status_code=400, detail=f"Latitude range expected 2 arguments (min,max) got: {len(latitude_range)}")

    if len(longitude_range) != 2:
        raise HTTPException(status_code=400, detail=f"Longitude range expected 2 arguments (min,max) got: {len(longitude_range)}")

    try:
        lat_min = float(latitude_range[0])
        lat_max = float(latitude_range[1])
        long_min = float(longitude_range[0])
        long_max = float(longitude_range[1])
    except:
        raise HTTPException(status_code=400, detail="Latitude and Longitude must be numbers")
    
    if lat_min > lat_max:
        raise HTTPException(status_code=400, detail=f"Expected first argument to be smaller than second argument, got: latitude_min: {lat_min}, latitude_max: {lat_max}")

    if long_min > long_max:
        raise HTTPException(status_code=400, detail=f"Expected first argument to be smaller than second argument, got: longitude_min: {long_min}, longitude_max: {long_max}")

    generator = ais_lat_long_slice_generator((lat_min, lat_max), (long_min, long_max))

    return StreamingResponse(generator, media_type="text/event-stream")

@app.get("/test")
async def test():
    return vessel_data

if __name__ == "__main__":
    uvicorn.run("main:app", host=SOURCE_IP, port=SOURCE_PORT, reload=True)
