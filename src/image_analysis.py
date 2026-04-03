import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
import os
from skimage.measure import shannon_entropy
import subprocess
import sys



def make_dir_paths(pth: Path, station: str, endpt: str = "images"):
    if not isinstance(pth, Path):
        pth = Path(pth)
        
    if endpt == "images":
        out_pth = pth / "images" / rf"{station}" / "time_average"
    elif endpt == "stats":
        out_pth = pth / "data" / rf"{station}"
    elif endpt == "roi":
        out_pth = pth / "src" / "configs" / rf"{station}_roi.config.json"
    else:
        print("End point not recognized")
        
    return out_pth
        
        
def convert_filename_to_date(filename: str, station: str):
    date, time = filename.split(rf"{station}-")[1].split(".")[0].split("_")
    time = time[:2] + ":" + time[2:]
    dt = date + " " + time
    return dt

        
def make_image_paths(pth: Path, station: str, target_n: int = 10):
    img_pth = make_dir_paths(pth, station, "images")
    target_dir = img_pth / rf"n={target_n}-frames"
    if not target_dir.exists():
        print(f"Target Directory Not Found:\n{target_dir}")
    imgs = os.listdir(target_dir)
    datetimes = [convert_filename_to_date(f, station) for f in imgs]
    img_paths = {}
    for idx, img in enumerate(imgs):
        dt = datetimes[idx]
        img_paths.setdefault(dt, (target_dir / img))
    return img_paths
    
        
def get_images(pth: Path, station: str, target_n: int = 10):
    img_paths = make_image_paths(pth, station, target_n)
    imgs = {}
    [imgs.setdefault(dt, cv2.imread(img_pth, 1)) for dt, img_pth in img_paths.items()]
    return imgs
    
    
def get_process_stats(pth: Path, station: str):
    stats_pth = make_dir_paths(pth, station, "stats")
    
    
def get_region_of_interests(pth: Path, station: str):
    config_pth = make_dir_paths(pth, station, "roi")
    with open(config_pth, "r") as f:
        station_config = json.load(f)
    config_scale = float(station_config.get("Image Resize")) / 100
    roi = station_config.get("roi_points")
    if config_scale != 1.0:
        roi = np.array(roi, dtype=np.float32) * (1 / config_scale)
    return roi


def get_image_region(pth: Path, station: str, target_n: int = 10):
    # Load images and region of interests
    imgs = get_images(pth, station, target_n)
    roi = get_region_of_interests(pth, station)
    # Convert ROI points to numpy array
    roi_arr = np.array(roi, dtype=np.int32) # truncate roi points to integer values for cv2.fillPoly
    # Create Mask
    dt = list(imgs.keys())[0]
    sample_img = imgs[dt]
    mask = np.zeros(sample_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_arr], 255)
    # Apply Mask
    masked_imgs = {}
    for dt, img in imgs.items():
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        masked_imgs.setdefault(dt, masked_img)
    return masked_imgs


def make_image_segments(image: np.ndarray, size: int, crop: bool = False):
    m, n = image.shape[:2]
    if crop:
        m = (m // size) * size
        n = (n // size) * size
        image = image[: m, : n]
    segments = {}
    for i in range(0, m, size):
        for j in range(0, n, size):
            img_segment = image[i:i+size, j:j+size]
            segments[(i,j)] = {
                "std_dev": float(cv2.meanStdDev(img_segment)[1][0][0]),
                "sharpness": float(cv2.Laplacian(img_segment, cv2.CV_64F).var()),
                "brightness": float(cv2.mean(img_segment)[0]),
                "entropy": shannon_entropy(img_segment),
                "data": img_segment
                }
    return segments

def segment_images(images: dict, size: int, crop: bool = False):
    "key: datetime (str), value: np.ndarray"
    segmented_images = {}
    for dt, img in images.items():
        img_segments = make_image_segments(img, size, crop)
        segmented_images.setdefault(dt, img_segments)
    return segmented_images
    
    
def make_segments_table(segmented_images: dict):
    segment_stats = []
    sample_dt = list(segmented_images.keys())[0]
    sample_segment = segmented_images[sample_dt]
    pos_key = list(sample_segment.keys())[0]
    features = ["datetime", "y", "x"] + list(sample_segment[pos_key].keys())[:-1]
    for dt, segmented_image in segmented_images.items():
        for posit, feature in segmented_image.items():
            posit = list(posit)
            values = []
            for feat, val in feature.items():
                if feat != "data":
                    values.append(val)
            result = [dt] + posit + values
            segment_stats.append(result)
    out_df = pd.DataFrame(segment_stats, columns=features)
    return out_df
            
            
        
    
def segments_heatmap(segments_table: pd.DataFrame, feature: str = "brightness"):
    df = segments_table.copy()
    heatmap = df.groupby(['y', 'x'])[feature].mean().unstack().values
    plt.figure(figsize=(10,8))
    plt.imshow(heatmap, cmap='viridis', aspect='auto', origin='upper')
    plt.colorbar(label=f"Average {feature}")
    plt.show()
        
    
        

def main(**kwargs):
    pth = kwargs.get("parent_dir")
    if not isinstance(pth, Path):
        pth = Path(pth)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir")


if __name__ == "__main__":
    sys.exit(cli())
    