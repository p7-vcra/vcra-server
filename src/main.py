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
from helpers import json_encode_iso, wgs84_to_utm, timestamp_to_unix, slice_query_validation, calc_dt_dutmx_dutmy



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

vessel_records_threshold = 32

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

        # Convert latitude and longitude to utm coordinates
        utm_xs, utm_ys = wgs84_to_utm(filtered_data["longitude"].values, filtered_data["latitude"].values)
        filtered_data["utm_x"] = utm_xs
        filtered_data["utm_y"] = utm_ys
        filtered_data["t"] = await timestamp_to_unix(filtered_data["timestamp"])

        grouped_data = filtered_data.groupby("mmsi")

        for name, group in grouped_data:
            if name in vessel_data:
                vessel_data[name] = pd.concat([vessel_data[name], group])
            else:
                vessel_data[name] = group

            if len(vessel_data[name]) == vessel_records_threshold:
                dts, dutm_xs, dutm_ys = await calc_dt_dutmx_dutmy(vessel_data[name]["t"], vessel_data[name]["utm_x"], vessel_data[name]["utm_y"])
                vessel_data[name]["dt"] = dts
                vessel_data[name]["dutm_x"] = dutm_xs
                vessel_data[name]["dutm_y"] = dutm_ys

                # The first row has no previous row to calculate delta, so we get NaN
                vessel_data[name].dropna(subset=["dt", "dutm_x", "dutm_y"], inplace=True)
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

            logger.debug(f"Trajectory: \n {trajectory[['longitude', 'latitude']].head(4)}")
            logger.debug(f"Trajectory: \n {trajectory[['t', 'utm_x', 'utm_y']].head(4)}")

            data = trajectory[["dt", "dutm_x", "dutm_y"]]

            logger.debug(f"Trajectory before normalization: \n {data[['dt', 'dutm_x', 'dutm_y']].head(4)}")

            norm_params = pd.read_json("data/rd9_epoch100_h1n350_ffn150_norm_param_mean_std.json")

            norm_params = norm_params.transpose()

            norm_params.columns = ["d", "dlon", "dlat"]

            # normalize input
            norm_data = normalize_inputs(data, norm_params)

            logger.debug(f"Normalized trajectory: \n {norm_data.head(4)}")

            request_data = {"data": norm_data.values.reshape(1, norm_data.shape[0], 3).tolist()}
            
            response = await post_to_prediction_server(request_data, session)

            if response:
                prediction = response["prediction"]
                prediction = await post_process_prediction(prediction, trajectory, norm_params)
                predictions[mmsi] = prediction
            else:
                logger.warning(f"No prediction received for trajectory: {trajectory}")
            
            await sleep(1)

def normalize_inputs(df, norm_params):
    normalized_df = pd.DataFrame()
    for feature in norm_params.columns:
        feature_cols = [col for col in df.columns if col.startswith(feature)]
        for col in feature_cols:
            normalized_df[col] = (df[col] - norm_params.loc["sc_x_mean", feature]) / norm_params.loc["sc_x_std", feature]
    return normalized_df

async def post_process_prediction(prediction: list, trajectory: pd.DataFrame, norm_param):
    look_ahead_points = 32
    features_outputs = [f"dutm_x(t+{i})" for i in range(1, look_ahead_points + 1)] + \
                       [f"dutm_y(t+{i})" for i in range(1, look_ahead_points + 1)]

    normalized = pd.DataFrame(prediction[0], columns=features_outputs)

    # denormalize
    denormalized = pd.DataFrame()
    
    for feature in norm_param.columns:
        feature_cols = [col for col in normalized.columns if col.startswith(feature)]
        for col in feature_cols:
            denormalized[col] = (normalized[col] * norm_param.loc["sc_x_std", feature] + norm_param.loc["sc_x_mean", feature])

    logger.debug(f"Denormalized prediction: \n {denormalized.head(4)}")

    # convert to actual values
    for la in range(1, look_ahead_points + 1):
        denormalized[f"utm_x(t+{la})"] = denormalized[f"dutm_x(t+{la})"] + trajectory["utm_x"].iloc[-1]
        denormalized[f"utm_y(t+{la})"] = denormalized[f"dutm_y(t+{la})"] + trajectory["utm_y"].iloc[-1]
        denormalized = denormalized.copy()

    logger.debug(f"Prediction after conversion from delta values: \n {denormalized[['utm_x(t+1)', 'utm_x(t+2)', 'utm_y(t+1)', 'utm_y(t+2)']].head(4)}")

    # convert from utm to wgs84
    results = {}
    utm_columns = [(f"utm_x(t+{i})", f"utm_y(t+{i})") for i in range(1, look_ahead_points + 1)]

    for lon_col, lat_col in utm_columns:
        wgs_lon_col = lon_col.replace("utm_x", "lon")
        wgs_lat_col = lat_col.replace("utm_y", "lat")

        results[wgs_lon_col], results[wgs_lat_col] = zip(*denormalized.apply(lambda row: wgs84_to_utm(row[lon_col], row[lat_col], inverse=True), axis=1))

    df_results = pd.concat([denormalized, pd.DataFrame(results)], axis=1)
    
    logger.debug(f"Prediction after conversion to WGS84: \n {df_results[['lon(t+1)', 'lon(t+2)', 'lat(t+1)', 'lat(t+2)']].head(4)}")

    return df_results

async def post_to_prediction_server(trajectory: dict, client_session: aiohttp.ClientSession):
    prediction_server_url = f"http://{PREDICTION_SERVER_IP}:{PREDICTION_SERVER_PORT}/predict"
    async with client_session.post(prediction_server_url, json=trajectory) as response:
        try:
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
                    
        data = await json_encode_iso(data)    
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(60)

async def predictions_generator():
    while True:
        data = "Empty"
        if predictions:
            data = pd.concat(predictions.values())
            data = await json_encode_iso(data)
        yield 'event: ais\n' + 'data: ' + data + '\n\n'
        await sleep(60)

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
async def prediction_fetch():
    generator = predictions_generator()
    return StreamingResponse(generator, media_type="text/event-stream")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if args.debug:
        LOG_LEVEL = logging.DEBUG

    uvicorn.run("main:app", host=SOURCE_IP, port=SOURCE_PORT, reload=True, log_level=LOG_LEVEL, workers=WORKERS, access_log=True)
