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

def build_datetime_batches(datetime_inventory: dict, delta: timedelta | int):
    """Build inventory datetime batches wrt top of the hour"""
    if not datetime_inventory:
        raise ValueError("Datetime inventory is empty. Please build the inventory first.")
    if isinstance(delta, int):
        delta = timedelta(minutes=delta)
    print(f"Building datetime batches with delta: {delta}")

    # Build batches
    batches = []
    current_batch = []
    current_window_end = None
    
    for timestamp, url in datetime_inventory.items():
        # Convert string timestamp to datetime object if needed
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
            
        # Align to top of the hour for window calculation
        window_start = dt.replace(minute=0, second=0, microsecond=0)
        window_offset = (dt - window_start) // delta * delta
        window_start += window_offset
        window_end = window_start + delta
        
        # If we're in a new time window, start a new batch
        if current_window_end is None or dt >= current_window_end:
            if current_batch:  # Save previous batch if it exists
                batches.append(current_batch)
            current_batch = []
            current_window_end = window_end
        
        current_batch.append((timestamp, url))
    
    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)
    
    return batches


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
    
    
# not set up to handle videos yet
def get_webcoos_object(date: str, url: str):
    """
    Download image from URL and store as numpy array in memory.
    
    Args:
        date_key: datetime string (will be used as the key)
        url: URL string to download the image from
    
    Returns:
        tuple: (date_key, numpy_array) or (None, None) if error
    """
    try:
        # Download the image
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Convert to numpy array using PIL
        image = Image.open(io.BytesIO(response.content))
        image_array = np.array(image)
        # Use the provided date_key as the key
        return date, image_array
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None, None
    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return None, None
        
        
def process_batch(batch):
    """
    Process a batch of images.
    
    Args:
        batch: list of tuples (date_key, url)
    
    Returns:
        list: list of tuples (date_key, numpy_array)
    """
    processed_images = []
    for date_key, url in batch:
        date, image_array = get_webcoos_object(date=date_key, url=url)
        if image_array is not None:
            processed_images.append((date, image_array))
    
    return processed_images
    
    
def get_time_average(batch):
    """
    Compute the average image and weighted average timestamp for a batch.
    
    Args:
        batch: list of tuples (timestamp, image_array)

    Returns:
        tuple: (average_timestamp, average_image_array)
    """
    if not batch:
        return None, None
    
    # Extract timestamps and image arrays
    timestamps = []
    image_arrays = []
    
    
    for timestamp, url in batch:
        timestamp, image_array = get_webcoos_object(date=timestamp, url=url)

        if image_array is not None and hasattr(image_array, 'size') and image_array.size > 0:
            # Convert string timestamps to datetime objects for calculation
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            timestamps.append(dt)
            image_arrays.append(image_array)
    
    if not image_arrays:
        return None, None

    # Calculate weighted average timestamp
    total_seconds = sum(dt.timestamp() for dt in timestamps)
    average_timestamp_seconds = total_seconds / len(timestamps)
    average_timestamp = datetime.fromtimestamp(average_timestamp_seconds, tz=timestamps[0].tzinfo)

    # Remove microseconds at the source
    average_timestamp = average_timestamp.replace(microsecond=0)

    # Convert back to string if original was string, otherwise keep as datetime
    if isinstance(batch[0][0], str):
        average_timestamp = average_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Stack and average images
    stacked_images = np.stack(image_arrays, axis=0)
    average_image = np.mean(stacked_images, axis=0).astype(np.uint8)
    
    return average_timestamp, average_image



def batch_processing_worker(batch, station):
    """Worker function to process a single batch and save the time average image"""
    try:
        # Process the batch
        time_avg, image_avg = get_time_average(batch)
        
        if image_avg is not None and hasattr(image_avg, 'size') and image_avg.size > 0:
            # Handle datetime conversion for filename
            if isinstance(time_avg, str):
                time_avg_dt = datetime.strptime(time_avg, '%Y-%m-%d %H:%M:%S')
            else:
                time_avg_dt = time_avg  # It's already a datetime object
            
            # Format timestamp for filename
            formatted_ts = time_avg_dt.strftime("%Y-%m-%d_%H%M")
            filename = rf"{station}-{formatted_ts}.png"
            filepath = CONST.image_dir / "10min_avg" / filename
            os.makedirs(filepath, exist_ok=True)
            
            # Convert numpy array to PIL Image and save
            pil_image = Image.fromarray(image_avg)
            pil_image.save(filepath)
            
            # Convert back to string for the shoreline function
            time_avg_str = time_avg_dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Use a try-except around the shoreline function
            try:
                tranSL, fig_tranSL = gts.getTimexShoreline(
                    stationName=station, 
                    imgName=image_avg, 
                    imgPath=False, 
                    imgDate=time_avg_str  # Pass as string
                )
                shoreline_points = len(tranSL) if tranSL else 0
                return {"timestamp": time_avg_str, "shoreline_points": shoreline_points, "status": "success", "filename": filename}
            except ValueError as e:
                if "truth value" in str(e):
                    # The function worked but has this internal warning
                    return {"timestamp": time_avg_str, "shoreline_points": -1, "status": "success_with_warning", "filename": filename}
                else:
                    raise
        
        return {"timestamp": time_avg, "shoreline_points": 0, "status": "no_image", "filename": None}
        
    except Exception as e:
        print(f"Batch failed with error: {e}")
        import traceback
        traceback.print_exc()  # This will show the full traceback
        return {"timestamp": "unknown", "shoreline_points": 0, "status": "failed", "filename": None}
    

def average_image_batch_worker(batch, station):
    """Worker function to download images without shoreline processing"""
    try:
        time_avg, image_avg = get_time_average(batch)
        
        if image_avg is not None and hasattr(image_avg, 'size') and image_avg.size > 0:
            # Convert time_avg to datetime if it's a string
            if isinstance(time_avg, str):
                time_avg = datetime.strptime(time_avg, '%Y-%m-%d %H:%M:%S')
            
            # Convert numpy array to PIL Image and save
            formatted_ts = time_avg.strftime("%Y-%m-%d_%H%M")
            filename = rf"{station}-{formatted_ts}.png"
            filepath = CONST.image_dir / "10min_avg" / filename
            os.makedirs(filepath, exist_ok=True)
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image_avg)
            pil_image.save(filepath)
            
            return {"timestamp": time_avg, "status": "saved", "filename": filename}
        
        return {"timestamp": time_avg, "status": "no_image", "filename": None}
        
    except Exception as e:
        print(f"Batch failed with error: {e}")
        return {"timestamp": "unknown", "status": "failed", "filename": None}
    
    
# def parallel_processor(batches, station):
#     results = []
#     with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
#         future_to_batch = {executor.submit(average_image_batch_worker, (batch, station)): batch for batch in batches}
        
#         for future in as_completed(future_to_batch):
#             try:
#                 result = future.result()
#                 results.append(result)
#                 print(f"Completed batch: {result['timestamp']} with {result['shoreline_points']} shoreline points, saved as {result['filename']}")
#             except Exception as e:
#                 print(f"Batch failed with error: {e}")


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
    
    dtRng = set_date_range(start=start, end=end)
    print(f"Datetime Range: {dtRng}")
    inventory = build_inventory(station, dtRng[0], dtRng[1], product)
    inv_key_list = list(inventory.keys())
    print(f"Inventory Keys:\n{inv_key_list[:5]}")
    print(f"Inventory Length: {len(inv_key_list)}")
    
    i = 0
    for k, v in inventory.items():
        if i < 3:
            print(f"Inventory DATE: {k}")
            print(f"Inventory URL:\n{v}")
            i += 1
    
    if product == "video-archive" and not parallel_proc:
        i = 0
        for date, url in inventory.items():
            duration, frames = get_video_metadata(url)
            print(f"Video Date: {date} | Duration: {duration} | Frame Count: {frames}")
            if i < 3:
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