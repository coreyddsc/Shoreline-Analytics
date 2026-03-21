from datetime import datetime, timedelta
from pathlib import Path
import PIL
import requests
import cv2
import numpy as np
import pandas as pd
import os
import sys
import math
import io
import time
import json
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import subprocess
import pywebcoos
import argparse

try:
    import webcoos.getTimexShoreline as gts
except ImportError:
    import getTimexShoreline as gts

class CONST:
    # Perform error handling to ensure the token is available
    if 'WebCOOS' not in os.environ:
        raise ValueError('WebCOOS token is required. Please set the WebCOOS environment variable.')
    else:
        webcoos_token = os.getenv('WebCOOS')
    image_dir = Path(os.getcwd()) / "images"
    station_list = ['oakisland_west', 'jennette_south', 'jennette_north', 'currituck_sailfish', 'currituck_hampton_inn']
    headers = {'Authorization': f'Token {webcoos_token}', 'Accept': 'application/json'}
    endpoint_url = 'https://app.webcoos.org/webcoos/api/v1/elements/'
    
    
def handle_time(time_str):
    """Convert a string to a datetime object."""
    if time_str is None:
        return None

    date = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

    current_time = datetime.now()
    if date > current_time:
        print(f"Error: The requested time {date.strftime('%Y-%m-%d %H:%M:%S')} is in the future.")
        print(f"Current time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return

    return date


# start date is required, end date is optional
def set_date_range(start: str, end: str = None):
    start_time = handle_time(start)
    if end is not None:
        end_time = handle_time(end)
    else:
        end_time = None
        print("End time not provided, using None.")
        
    if start_time and end_time and start_time >= end_time:
        raise ValueError("Start time must be before end time.")
    else:
        print(f"Date range set from {start_time} to {end_time}")
    return (start_time, end_time)
        

def get_webcoos_products():
    api = pywebcoos.API(CONST.webcoos_token, verbose=True)
    cams = api.get_cameras()
    unique_products = set()
    for _, row in cams.iterrows():
        cam = row["Camera Name"]
        products = api.get_products(cam)
        unique_products.update(products)
    print(f"WebCOOS Products: {unique_products}")
    products = list(unique_products)
    return products

# this should check the response status code and handle errors    
def check_station_availability(station: str = None):
    api = pywebcoos.API(CONST.webcoos_token, verbose=True)
    products = get_webcoos_products()
    cams = api.get_cameras()
    pd.set_option('display.max_columns', None)
    cams[products] = False
    for idx, row in cams.iterrows():
        cam = row["Camera Name"]
        available_products = api.get_products(cam)
        for product in available_products:
            if product in cams.columns:
                cams.at[idx, product] = True
    if station is None:
        print(f"All WebCOOS Cameras and Available Products:\n{cams}")
        return cams
    else:
        print(f"Checking WebCOOS station availability for camera namaes containing {station}...")
        # Get the row for this station
        station_row = cams[cams["Camera Name"].str.contains(station)]
        
        # Get only the product columns that are True for this station
        true_products = station_row.loc[:, products].iloc[0]
        true_products = true_products[true_products == True].index.tolist()
        
        # print(f"Station {station} has these products: {true_products}")
        return station_row[['Camera Name'] + true_products]
        

# this should handle multiple service options: stills, videos, annotated, etc.
def build_inventory(station: str = "oakisland_west", start: str = None, end: str = None, product: str = "one-minute-stills"):

    if station is None:
        raise ValueError("Station not set. Please set a station before fetching images.")
    print(f"Fetching still image for {station} from {start} to {end}...")

    all_results = []
    current_start_time = start
    
    while True:
        # Set parameters for element request.
        params = {
            'service': station + rf'-{product}-s3',
            'starting_after': current_start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            'starting_before': end.strftime("%Y-%m-%dT%H:%M:%S")
        }
        print(f"Request params: {params}")
        element_response = requests.get(CONST.endpoint_url, headers=CONST.headers, params=params)
        print(f"Element response status code: {element_response.status_code}")
        # Check for HTTP errors before trying to parse JSON
        if element_response.status_code != 200:
            print(f"API request failed with status {element_response.status_code}")
            print(f"Response content: {element_response.text[:200]}...")  # First 200 chars
            break  # Or implement retry logic here
        
        try:
            elements_data = element_response.json()
        except requests.exceptions.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response content: {element_response.text[:200]}...")
            break
        
        results = elements_data['results']
        if not results:
            break

        all_results.extend(results)

        # Check if the results list is less than 100, meaning it's the last page
        if len(results) < 100:
            break

        # Update the current_start_time to the last temporal_min in the current results
        current_start_time_str = results[-1]['data']['extents']['temporal']['min']
        current_start_time = datetime.strptime(current_start_time_str, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        print(f"Updated current_start_time: {current_start_time}")

        # Ensure the new start time does not exceed the end time
        if current_start_time >= end:
            break
        
    # This should be built within the preceding loop use set defaults
    results_dict = {}
    for element in all_results:
        url = element['data']['properties']['url']
        temporal_min = element['data']['extents']['temporal']['min']
        results_dict[temporal_min] = url
        # print(f"URL: {url}, Temporal Min: {temporal_min}")
            
    print(f"Total number of results: {len(all_results)}")
    datetime_inventory = results_dict
        
    return datetime_inventory


def get_video_metadata(url: str):
    """
    Get duration and frame count using JSON output.
    Returns tuple: (duration_seconds, total_frames)
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-show_entries', 'stream=nb_frames',
            '-select_streams', 'v:0',
            '-of', 'json',
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        
        duration = float(data.get('format', {}).get('duration', 0))
        
        streams = data.get('streams', [])
        frame_count = 0
        if streams:
            nb_frames = streams[0].get('nb_frames', 0)
            if nb_frames and nb_frames != 'N/A':
                frame_count = int(float(nb_frames))  # Safe conversion
        
        return duration, frame_count
        
    except Exception as e:
        print(f"Error getting metadata: {e}")
        return 0, 0


def process_video(station: str, date: str, url: str, min_duration: float = 450, target_samples: int = 100):
    img_dir = CONST.image_dir / rf"{station}" / "time_average" / rf"n={target_samples}-sample-frames"
    os.makedirs(img_dir, exist_ok=True)
    print(f"Date: {date}")
    safe_date = date.replace(':', '')  # Replace colons with hyphens
    formatted_date = safe_date.split('T')[0] + "_" + safe_date.split('T')[1][:4]
    print(f"Formatted Date: {formatted_date}")
    file_path = img_dir / rf"{station}-{formatted_date}.png"
    
    try:
        duration, total_frames = get_video_metadata(url)
        
        if duration < min_duration or total_frames == 0:
            print("Error: Could not get frame count or video does not meet minimum duration requirement")
            return None
        
        target_samples = target_samples
        sample_interval = max(1, total_frames // target_samples)
        
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print("Error: Could not open downloaded video")
            return None
        
        # Sample frames
        accumulated = None
        sample_count = 0
        
        for frame_idx in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame_float = frame.astype(np.float32)
                
                if accumulated is None:
                    accumulated = frame_float
                else:
                    accumulated += frame_float
                    
                sample_count += 1
                
        cap.release()

        # Create and save average image
        if sample_count > 0:
            average_image = (accumulated / sample_count).astype(np.uint8)
            cv2.imwrite(str(file_path), average_image)
            print(f"Saved time average image to:\n{file_path}")
            return sample_count
        else:
            print("No frames sampled")
            return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def process_video_worker(date, url, station, min_duration, target_samples):
    
    # Call your video processing function
    result = process_video(station, date, url, min_duration, target_samples)
    
    return {
        'url': url,
        'date': date,
        'frame_count': result,  # or whatever your function returns
        'station': station
    }


def parallel_processor(inventory, station, min_duration, target_samples):
    results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() // 2) as executor:
        # Submit each date-url pair
        future_to_pair = {
            executor.submit(process_video_worker, date, url, station, min_duration, target_samples): (date, url) 
            for date, url in inventory.items()
        }
        
        for future in as_completed(future_to_pair):
            date, url = future_to_pair[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"Completed video: {date}")
                else:
                    print(f"Video failed: {date}")
            except Exception as e:
                print(f"Video {date} failed with error: {e}")
    
    return results


def main(**kwargs):
    # dir = Path(kwargs.get("parent_dir"))
    station = kwargs.get("station")
    start = kwargs.get("start_time")
    end = kwargs.get("end_time")
    product = kwargs.get("product")
    min_duration = kwargs.get("min_vid_duration")
    target_samples = kwargs.get("target_samples")
    print(f"Number of target samples selected: {target_samples}")
    gtx_flag = kwargs.get("get_time_x")
    parallel_proc = kwargs.get("parallel_proc")
    
    dt_rng = set_date_range(start=start, end=end)
    inventory = build_inventory(station, dt_rng[0], dt_rng[1], product)
    inv_key_list = list(inventory.keys())
    
    
    if product == "video-archive" and not parallel_proc:
        i = 0
        for date, url in inventory.items():
            duration, frames = get_video_metadata(url)
            print(f"Video Date: {date} | Duration: {duration} | Frame Count: {frames}")
            if i < 3:
                print(f"Inventory DATE: {date}")
                print(f"Inventory URL:\n{url}")
                sample_count = process_video(station, date, url, min_duration, target_samples)
                print(f"Sample count used for image: {sample_count} ")
                if sample_count is not None:
                    i += 1
                
    if product == "video-archive" and parallel_proc:
        print(f"Parallel Processing with {multiprocessing.cpu_count() // 2} workers")
        results = parallel_processor(inventory, station, min_duration, target_samples)



def cli():
    parser = argparse.ArgumentParser()
    # parser.add_argument("parent_dir")
    parser.add_argument("station", type=str, help="WebCOOS Station Name")
    parser.add_argument("start_time", type=str, help="Start dateimte formatted as YYYY-MM-DD HH:MM:SS")
    parser.add_argument("end_time", type=str, help="End datetime formatted as YYYY-MM-DD HH:MM:SS")
    parser.add_argument('-p', "--product", type=str, help="WebCOOS Product Line", default="one-minute-stills")
    parser.add_argument('-mvd', "--min-vid-duration", type=float, help="Minimum required video duration", default=450.0)
    parser.add_argument('-t', '--target-samples', type=int, help="Number of sample frames to use in video to image processing", default=100)
    parser.add_argument('-gtx', "--get-time-x", action="store_true", help="Apply Static getTimexShoreline Extract Methods") # store default as false to not use static extract
    parser.add_argument('-ll', '--parallel-proc', action="store_true", help="Use Parallel Processing") # store default as true for parallel processing
    args = parser.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    sys.exit(cli())
    # python webcoos/webcoos.py "jennette_south" "2025-07-10 06:00:00" "2025-07-17 06:00:00" -p "video-archive"