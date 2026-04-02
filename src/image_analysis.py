import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
from datetime import datetime, timedelta
import json
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
        
def get_images(pth: Path, station: str, target_n: int = 10):
    img_pth = make_dir_paths(pth, station, "images")
    target_dir = img_pth / rf"n={target_n}-frames"
    if not target_dir.exists():
        print(f"Target Directory Not Found:\n{target_dir}")
    imgs = os.listdir(target_dir)
    
def get_process_stats(pth: Path, station: str):
    stats_pth = make_dir_paths(pth, station, "stats")
    
def get_region_of_interests(pth: Path, station: str):
    roi_pth = make_dir_paths(pth, station, "roi")

def main(**kwargs):
    pth = kwargs.get("parent_dir")
    if not isinstance(pth, Path):
        pth = Path(pth)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir")

if __name__ == "__main__":
    sys.exit(cli())
    