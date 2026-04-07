import argparse
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.path as mpl
import numpy as np
import pandas as pd
from pathlib import Path
import os
from scipy.interpolate import interp1d, splprep, splev
import sys

try:
    from src.image_analysis import make_dir_paths
except ImportError:
    from image_analysis import make_dir_paths


class CONST:
    cwd = Path(os.getcwd())
    

def load_config(pth: Path, station: str):
    config_pth = make_dir_paths(pth, station, "config")
    with open(config_pth, "r") as f:
        config = json.load(f)
    return config


def load_annotations(pth: Path, station: str):
    ant_pth = make_dir_paths(pth, station, "annotations")
    ants = {}
    with open(ant_pth, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dt = data.get("timestamp")
            img_file = data.get("image_file")
            pts = data.get("points")
            ants[dt] = {
                "image_file": img_file,
                "points": pts
            }
    return ants


def resample_shoreline(annotated_dict: dict, n_pts: int = 100):
    "Builds a parameterized shoreline sequence."
    resampled = {}
    for dt, vals in annotated_dict.items():
        img_file = vals.get("image_file")
        pts = vals.get("points")
        t = np.linspace(0, 1, len(pts))
        new_t = np.linspace(0, 1, n_pts)
        interpolator = interp1d(t, np.array(pts), axis=0, kind='linear')
        resampled[dt] = {"image_file": img_file, "points": interpolator(new_t)}
    return resampled
        

def make_shoreline_crop_samples(
    pth: Path,
    station: str,
    resampled_dict: dict, 
    width: float = 100, 
    length: float = 200, 
    n_samples: int = 10, 
    _random: bool = False
):
    if not isinstance(pth, Path):
        pth = Path(pth)
        
    cropped_configs = pth / "data" / rf"{station}" / "crop_samples.config.json"
    cropped = {}
    
    for dt, vals in resampled_dict.items():
        # image paths
        image_file = vals.get("image_file")
        img_pth = make_dir_paths(pth, station, "images") / "n=100-frames" / image_file
        img = cv2.imread(img_pth, 1)
        
        # Compute Arcs
        pts = np.array(vals.get("points"))
        ds = np.diff(pts, axis=0) # point-to-point distance
        ds = np.pad(ds, ((1, 0), (0, 0)), 'constant', constant_values=0)  # pad at beginning
        s = np.linalg.norm(ds, axis=1) + 1e-8 # arc lengths
        arc_coord = np.cumsum(s, axis=0) # arc coordinates
        L = np.sum(s) # total arc length
        
        # Calculate the tangents and normals to the curve
        nx = ds[:, 1] / s # norm-x
        ny = -ds[:, 0] / s  # norm-y, we use (-ds[:,0] / s) because opencv uses a different coordinate system than matplotlab.
        norms = np.stack((nx, ny), axis=1)
        tangents = ds / s.reshape(-1, 1)  # unit tangent vectors (dx, dy) - same length as pts
        
        # crop location at n_sample equally spaced locations
        crop_location = np.array(range(1, n_samples+1)).reshape(-1,1) * (L / (n_samples + 1))
        # get random crop locations
        if _random:
            crop_location = np.array(sorted(np.random.uniform(width // 2, L - (width // 2), size=n_samples))).reshape(-1, 1)
        
        # get index for each arc coordinate nearest the crop location
        nearest_idx = [np.argmin(np.abs(arc_coord - loc)) for loc in crop_location.flatten()]
        crop_pts = pts[nearest_idx] # center point of the crop region
        crop_norms = norms[nearest_idx] # direction perpendicular to the shoreline crop region
        crop_tangents = tangents[nearest_idx] # direction along the shoreline crop region
        
        # crop region parameters
        half_length = length // 2 # pixels along tangent
        half_width = width // 2 # pixels along tangent
        
        samples = {}
        # crop corners
        for idx, pt in enumerate(crop_pts):
            t = crop_tangents[idx]
            n = crop_norms[idx]
            p1 = (pt + n * half_length) + (t * half_width)
            p2 = (pt + n * half_length) - (t * half_width)
            p3 = (pt - n * half_length) - (t * half_width)
            p4 = (pt - n * half_length) + (t * half_width)
            crop_corners = np.array([p1, p2, p3, p4])
            
            # extract crop region shoreline points with buffer multiplier
            # Expand by scaling relative to center
            center = pt
            extended_corners = center + (crop_corners - center) * 1.5
            polygon = mpl.Path(extended_corners)
            inside_mask = polygon.contains_points(pts)
            enclosed_pts = pts[inside_mask]
            
            samples[f"sample_{idx}"] = {
                "center": center.tolist(),
                "normal": n.tolist(),
                "tangent": t.tolist(),
                "region": crop_corners.tolist(),
                "enclosed_points": enclosed_pts.tolist()
            }
            
            # Mask and extract cropping region
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [crop_corners.astype(np.int32)], 255)
            image_crop = cv2.bitwise_and(img, img, mask=mask)
            # Get tighter bounding box
            x, y, w, h = cv2.boundingRect(crop_corners.astype(np.int32))
            image_crop = image_crop[y:y+h, x:x+w]
            
        cropped[dt] = {
            "image_file": image_file,
            "samples": samples
        }
            
    with open(cropped_configs, "w") as f:
        json.dump(cropped, f, indent=4)

    return cropped


def get_shoreline_crop_samples(pth: Path, station: str):
    if not isinstance(pth, Path):
        pth = Path(pth)
    crop_pth = pth / "data" / rf"{station}" / "crop_samples.config.json"
    with open(crop_pth, "r") as f:
        crops = json.load(f)
    
    converted = {}    
    for dt, vals in crops.items():
        image_file = vals.get("image_file")
        samples = vals.get("samples")
        sample_convert = {}
        for k, v in samples.items():
            sample_convert[k] = {
                "center": np.array(v.get("center")),
                "normal": np.array(v.get("normal")),
                "tangent": np.array(v.get("tangent")),
                "region": np.array(v.get("region")),
                "enclosed_points": np.array(v.get("enclosed_points"))
            }
        converted[dt] = {
            "image_file": image_file,
            "samples": sample_convert
        }
        
    return converted


def apply_shoreline_crop(pth: Path, station: str, image: np.ndarray | str, crop_corners: np.ndarray):
    img = image
    if not isinstance(image, np.ndarray):
        img_pth = make_dir_paths(pth, station, "images") / "n=100-frames" / image
        img = cv2.imread(img_pth, 1)
        
    # Mask and extract cropping region
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [crop_corners.astype(np.int32)], 255)
    image_crop = cv2.bitwise_and(img, img, mask=mask)
    # Get tighter bounding box
    x, y, w, h = cv2.boundingRect(crop_corners.astype(np.int32))
    image_crop = image_crop[y:y+h, x:x+w]
    
    return image_crop


def plot_shoreline_crop_sample(image: np.ndarray, points: np.ndarray, region: np.ndarray, plot_points: bool = False):
    # Draw the enclosed points on the image_crop
    image_crop = image.copy()
    if plot_points:
        enclosed_pts = points
        x, y, w, h = cv2.boundingRect(region.astype(np.int32))
        for point in enclosed_pts:
            # Adjust point coordinates relative to crop region
            pt_in_crop = (int(point[0] - x), int(point[1] - y))
            cv2.circle(image_crop, pt_in_crop, 2, (0, 0, 255), -1)  # red dots
    cv2.imshow("Sample Crop Region", image_crop)
    cv2.waitKey(0)
    
    
def plot_shoreline_normals(pth: Path, station: str, image_file: str, pts: np.ndarray, norms: np.ndarray):
    img_pth = make_dir_paths(pth, station, "images") / "n=100-frames" / image_file
    img = cv2.imread(img_pth, 1)

    # Scale factor for arrow size (since OpenCV draws in pixels)
    scale = 40  # adjust as needed

    # Use cv2.arrowedLine with endpoint adjustment for better visuals
    for i in range(0, len(pts), 2):
        x, y = int(pts[i, 0]), int(pts[i, 1])
        end_x = int(x + norms[i, 0] * scale)
        end_y = int(y + norms[i, 1] * scale)
        cv2.arrowedLine(img, (x, y), (end_x, end_y), (0, 255, 0), thickness=3, tipLength=0.3)

    # Draw the curve
    for i in range(len(pts) - 1):
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])), 
                (int(pts[i+1, 0]), int(pts[i+1, 1])), (0, 0, 255), 2)  # red curve

    # Display
    cv2.namedWindow("Shoreline with Normals", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Shoreline with Normals", 1920, 1080)
    cv2.imshow("Shoreline with Normals", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def main(**kwargs):
    pth = kwargs.get("parent_dir")
    if not isinstance(pth, Path):
        pth = Path(pth)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir")


if __name__ == "__main__":
    sys.exit(cli())
    