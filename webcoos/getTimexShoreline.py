# getTimexShoreline.py
import cv2 
from datetime import datetime 
from itertools import chain
import json 
# import math 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from PIL import Image, ImageDraw 
import re 
import scipy.signal as signal 
from skimage.filters import threshold_otsu 
from skimage.measure import profile_line 
from statsmodels.nonparametric.kde import KDEUnivariate

from pathlib import Path

# Add to your imports at the top of the file
from matplotlib.path import Path
import cv2


def getStationInfo(ssPath):
    # Loads json and converts data to NumPy arrays.
    with open(ssPath, 'r') as setupFile:
        stationInfo = json.load(setupFile)
    #if missing dune line interpolation, check dune line points
    if 'Dune Line Interpolation' in stationInfo['Dune Line Info']:
        stationInfo['Dune Line Info']['Dune Line Interpolation'] = np.asarray(stationInfo['Dune Line Info']['Dune Line Interpolation'])
    else:
        stationInfo['Dune Line Info']['Dune Line Points'] = np.asarray(stationInfo['Dune Line Info']['Dune Line Points'])
        
    stationInfo['Shoreline Transects']['x'] = np.asarray(stationInfo['Shoreline Transects']['x'])
    stationInfo['Shoreline Transects']['y'] = np.asarray(stationInfo['Shoreline Transects']['y'])
    return stationInfo

def mapROI(stationInfo, photo):
    """
    Creates a mask from pre-defined ROI points and extracts ROI from the image.
    Uses the roi_points directly for more reliable polygon construction.
    """
    # Input validation
    if 'roi_points' not in stationInfo:
        # raise ValueError("stationInfo must contain 'roi_points'")
        # Draws a mask on the region of interest and turns the other pixel values to nan.
        w, h = photo.shape[1], photo.shape[0]
        transects = stationInfo['Shoreline Transects']
        xt = np.asarray(transects['x'], dtype=int)
        yt = np.asarray(transects['y'], dtype=int)
        print(f"x-transect cords: {xt[:5]}")
        print(f"y-transect cords: {yt[:5]}") 
        cords = np.column_stack((xt[:, 1], yt[:, 1]))
        
        cords = np.vstack((cords, np.column_stack((xt[::-1, 0], yt[::-1, 0]))))
        cords = np.vstack((cords, cords[0]))  
        print(f"mapROI cords: {cords[:5]}")
        poly = list(chain.from_iterable(cords))
        print(f"poly values: {poly[:10]}")
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
        mask = np.array(img)
        maskedImg = photo.astype(np.float64)
        maskedImg[mask == 0] = np.nan
        maskedImg /= 255
        print(f"Masked image shape: {maskedImg.shape}")
        return maskedImg
    else:
        if not isinstance(photo, np.ndarray) or photo.ndim not in [2, 3]:
            raise ValueError("photo must be a 2D or 3D numpy array")

        h, w = photo.shape[:2]
        is_color = photo.ndim == 3

        # Get ROI points
        roi_points = np.array(stationInfo['roi_points'], dtype=float)
        
        # Debug: Print ROI info
        print(f"\nROI Points Info:")
        print(f"Number of points: {len(roi_points)}")
        print(f"First point: {roi_points[0]}")
        print(f"Last point: {roi_points[-1]}")
        print(f"Image dimensions: {w}x{h}")

        # Ensure ROI points are within image bounds
        roi_points[:, 0] = np.clip(roi_points[:, 0], 0, w-1)
        roi_points[:, 1] = np.clip(roi_points[:, 1], 0, h-1)

        # Close the polygon if not already closed
        if not np.array_equal(roi_points[0], roi_points[-1]):
            roi_points = np.vstack((roi_points, roi_points[0]))

        # Create mask
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        points = np.column_stack((x.ravel(), y.ravel()))
        path = Path(roi_points)
        mask = path.contains_points(points).reshape(h, w)

        # Apply slight dilation to include edge pixels
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, structure=np.ones((3, 3)))

        # Apply mask
        maskedImg = photo.astype(np.float64)
        if is_color:
            mask_3d = np.repeat(mask[:, :, np.newaxis], photo.shape[2], axis=2)
            maskedImg[~mask_3d] = np.nan
        else:
            maskedImg[~mask] = np.nan

        if maskedImg.max() > 1:
            maskedImg /= 255.0

        # # Enhanced visualization
        # fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # # Original image with ROI points
        # ax[0].imshow(photo, cmap='gray' if not is_color else None)
        # ax[0].plot(roi_points[:, 0], roi_points[:, 1], 'ro-', lw=1, markersize=3)
        # ax[0].set_title(f"Original Image with {len(roi_points)} ROI Points")

        # # Mask visualization
        # ax[1].imshow(mask, cmap='gray')
        # ax[1].plot(roi_points[:, 0], roi_points[:, 1], 'r-', lw=1)
        # ax[1].set_title("Generated Mask")

        # # Masked image
        # ax[2].imshow(maskedImg, cmap='gray' if not is_color else None)
        # ax[2].set_title("Masked Result")

        # plt.tight_layout()
        # plt.show()

        return maskedImg



def improfile(rmb, stationInfo):
    # Extract intensity profiles along shoreline transects.
    transects = stationInfo['Shoreline Transects']
    print(f"Transects: {transects}")
    xt = np.asarray(transects['x'], dtype=int)
    yt = np.asarray(transects['y'], dtype=int)
    print(f"(xt, yt) shapes: {xt.shape}, {yt.shape}")
    # round xt, yt to nearest integer
    print(f"Transect values --  xt: {xt[:5]}, yt: {yt[:5]}")
    n = len(xt)
    print(f"range for profile: {int(2*n/3-1)}, {int(2*n/3+1)}")
    print(f"Point Pair Transect Sets [(y1,x1), (y0,x0)]: \n{[[(yt[i,1], xt[i,1]), (yt[i,0], xt[i,0])] for i in range(0,n)][:5]}")
    print(f"Shape of RMB Array: {rmb.shape}")
    
    print(f"X range: {xt.min()} - {xt.max()}, image width: {rmb.shape[1]}")
    print(f"Y range: {yt.min()} - {yt.max()}, image height: {rmb.shape[0]}")
    # the input image has been rescaled in before this function is called in the getTimexShoreline function.
    # However, the new sloped (user-input) transect points have not been rescaled. This might be another source of error.
    
    # # Plot the original grayscale image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(rmb, cmap='gray')  # Display in grayscale

    # # Overlay transects as red lines (without modifying the original image)
    # for i in range(n):
    #     plt.plot(
    #         [xt[i, 1], xt[i, 0]],  # X-coordinates (x1, x0)
    #         [yt[i, 1], yt[i, 0]],  # Y-coordinates (y1, y0)
    #         'r-',  # Red line
    #         linewidth=1,
    #         alpha=0.7  # Slightly transparent
    #     )
        
    # plt.title('RMB Image with Transect Overlay')
    # plt.show()
    
    # Get Profile Line Intensity Values
    # imProf = [profile_line(rmb, (yt[i,1], xt[i,1]), (yt[i,0], xt[i,0]), mode='constant') for i in range(int(2*n/3-1), int(2*n/3+1))]
    imProf = [profile_line(rmb, (yt[i,1], xt[i,1]), (yt[i,0], xt[i,0]), mode='constant') for i in range(n)]
    # imProf = [profile_line(rmb, (yt[i,1], xt[i,1]), (yt[i,0], xt[i,0]), linewidth=1, order=1) for i in range(int(2*n/3-1), int(2*n/3+1))]
    # count the number of non-nan values in the profile
    print(f"imProf: {imProf}")
    # non_nan_count = np.sum(~np.isnan(imProf))
    # print(f"non_nan_count: {non_nan_count}")

    improfile = np.concatenate(imProf)[~np.isnan(np.concatenate(imProf))]
    print(f"improfile shape: {improfile.shape}")
    print(f"improfile: {improfile}")
    return improfile

def ksdensity(P, **kwargs):
    # Univariate kernel density estimation.
    x_grid = np.linspace(P.max(), P.min(), 1000) # Could cache this.
    kde = KDEUnivariate(P)
    kde.fit(**kwargs)
    pdf = kde.evaluate(x_grid)
    return (pdf, x_grid)

# This method needs to be updated to handle dynamic y-points because it previously assumed the y-points were fixed.
def extract(stationInfo, rmb, maskedImg, threshInfo):
    # Uses otsu's threshold to find shoreline points based on water orientation.
    stationname = stationInfo['Station Name']
    slTransects = stationInfo['Shoreline Transects']
    dtInfo = stationInfo['Datetime Info']
    date = dtInfo.date()
    xt = np.asarray(slTransects['x'])
    yt = np.asarray(slTransects['y'])
    print(f"Shoreline Transects (xt): {xt}")
    print(f"Shoreline Transects (yt): {yt}")
    orn = stationInfo['Orientation']
    thresh = threshInfo['Thresh']
    thresh_otsu = threshInfo['Otsu Threshold']
    thresh_weightings = threshInfo['Threshold Weightings']
    length = min(len(xt), len(yt))
    trsct = range(0, length)
    values = [0]*length
    revValues = [0]*length
    yList = [0]*length
    xList = [0]*length

    def find_first_exceeding_index(values, threshold):
        values = np.array(values)
        for i in range(1, len(values)):
            if (values[i-1] < threshold and values[i] >= threshold) or (values[i-1] >= threshold and values[i] < threshold):
                return i
        return None

    if orn == 0:
        for i in trsct:
            x = int(xt[i][0])
            if 'roi_points' not in stationInfo: 
                yMax = int(yt[i][0]) # JWL flipped these for new cocoabeach station config
                yMin = int(yt[i][1]) # JWL flipped these for new cocoabeach station config
            else:
                yMax = int(yt[i][1]) 
                yMin = int(yt[i][0])
            y = yMax - yMin
            # y = abs(y)
            yList[i] = np.zeros(shape=y)
            val = [0]*(yMax - yMin)
            for j in range(len(val)):
                k = yMin + j
                val[j] = rmb[k][x]
            val = np.array(val)
            values[i] = val

        idx = [0]*len(xt)
        xPt = [0]*len(xt)
        yPt = [0]*len(xt)
        for i in range(len(values)):
            idx[i] = find_first_exceeding_index(values[i], thresh_otsu)
            if idx[i] is None:
                yPt[i] = None
                xPt[i] = None
            else:
                yPt[i] = min(yt[i]) + idx[i]
                xPt[i] = int(xt[i][0])
        shoreline = np.vstack((xPt, yPt)).T
    # if orn == 3, then we need to find the first exceeding index in the opposite direction of orn == 0
    elif orn == 3:    
        for i in trsct:
            x = int(xt[i][0])
            if 'roi_points' not in stationInfo:
                yMax = int(yt[i][0]) # JWL flipped these for new cocoabeach station config
                yMin = int(yt[i][1]) # JWL flipped these for new cocoabeach station config
            else:
                yMax = int(yt[i][1]) # flipped for the Ferry Beach station config
                yMin = int(yt[i][0]) # flipped for the Ferry Beach station config
            y = yMax - yMin
            y = abs(y)
            print(f"shape of y: {y}")
            yList[i] = np.zeros(shape=y)
            val = [0]*y
            for j in range(len(val)):
                k = yMin + j
                val[j] = rmb[k][x]
            val = np.array(val)
            values[i] = val
        # reverse the values for orn == 3
        revValues = [val[::-1] for val in values]
        idx = [0]*len(xt)
        xPt = [0]*len(xt)
        yPt = [0]*len(xt)
        for i in range(len(revValues)):
            idx[i] = find_first_exceeding_index(revValues[i], thresh_otsu)
            if idx[i] is None:
                yPt[i] = None
                xPt[i] = None
            else:
                yPt[i] = max(yt[i]) - idx[i]
                xPt[i] = int(xt[i][0])
        shoreline = np.vstack((xPt, yPt)).T
    # for orn == 1 or 2
    else:
        for i in trsct:
            xMax = int(xt[i][1])  # JWL chnged this from 0 Jeanettes ok, still ok for Oak Island
            y = int(yt[i][0])
            yList[i] = np.full(shape=xMax, fill_value=y)
            xList[i] = np.arange(xMax)
            values[i] = rmb[y][0:xMax]
            revValues[i] = rmb[y][::-1]

        idx = [0]*len(yt)
        xPt = [0]*len(yt)
        yPt = [0]*len(yt)
        for i in range(len(revValues)):
            idx[i] = find_first_exceeding_index(values[i], thresh_otsu)
            xPt[i] = idx[i]
            yPt[i] = int(yt[i][0])
        shoreline = np.vstack((xPt, yPt)).T

    print(f"Find First Exceeding Index List: {idx}")
    print(f"Extracted Shoreline Points: {shoreline}")
    
    # print the y shoreline and yt values for comparison side by side in the same array
    print("Shoreline Y vs Transect Y:")
    for i in range(len(yt)):
        print(f"Shoreline Y: {shoreline[i][1] if i < len(shoreline) else None} | Transect Y: {yt[i][0] if i < len(yt) else None}")

    # Convert numpy data types to native Python types and handle None values in shoreline
    slVars = {
        'Station Name': stationname,
        'Date': str(date),
        'Time Info': str(dtInfo),
        'Thresh': float(thresh),
        'Otsu Threshold': float(thresh_otsu),
        'Shoreline Transects': {
            'x': xt.tolist(),
            'y': yt.tolist()
        },
        'Threshold Weightings': [float(w) for w in thresh_weightings],
        'Shoreline Points': [[float(item) if item is not None else None for item in point] for point in shoreline]
    }

    try:
        del slVars['Time Info']['DateTime Object (UTC)']
        del slVars['Time Info']['DateTime Object (LT)']
    except:
        pass

    if isinstance(slVars['Shoreline Transects']['x'], np.ndarray):
        slVars['Shoreline Transects']['x'] = slVars['Shoreline Transects']['x'].tolist()
        slVars['Shoreline Transects']['y'] = slVars['Shoreline Transects']['y'].tolist()

    # Create directories if they do not exist
    base_dir = os.path.join(os.getcwd(), 'transect_jsons', stationname)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        print(f"Directory exists: {base_dir}")

    # Save JSON file to the directory
    fname = os.path.join(base_dir, f'{stationname}-{datetime.strftime(dtInfo, "%Y-%m-%d_%H%M")}.avg.slVars.json')
    with open(fname, "w") as f:
        json.dump(slVars, f, indent=4)
    print(f"Saved JSON to: {fname}")
    
    return shoreline

# def extract(stationInfo, rmb, maskedImg, threshInfo):
#     # Uses otsu's threshold to find shoreline points based on water orientation.
#     stationname = stationInfo['Station Name']
#     slTransects = stationInfo['Shoreline Transects']
#     dtInfo = stationInfo['Datetime Info']
#     date = dtInfo.date()
#     xt = np.asarray(slTransects['x'])
#     yt = np.asarray(slTransects['y'])
#     print(f"Shoreline Transects (xt): {xt}")
#     print(f"Shoreline Transects (yt): {yt}")
#     orn = stationInfo['Orientation']
#     thresh = threshInfo['Thresh']
#     thresh_otsu = threshInfo['Otsu Threshold']
#     thresh_weightings = threshInfo['Threshold Weightings']
#     length = min(len(xt), len(yt))
#     trsct = range(0, length)
#     values = [0]*length

#     def find_first_exceeding_index(values, threshold):
#         values = np.array(values)
#         for i in range(1, len(values)):
#             if (values[i-1] < threshold and values[i] >= threshold) or (values[i-1] >= threshold and values[i] < threshold):
#                 return i
#         return None

#     # Common function to sample along sloped transects and find shoreline points
#     def sample_sloped_transects(reverse_direction=False):
#         idx = [0]*len(xt)
#         xPt = [0]*len(xt)
#         yPt = [0]*len(xt)
        
#         for i in trsct:
#             # Use both endpoints to define the sloped line
#             if reverse_direction:
#                 start_point = (yt[i,0], xt[i,0])  # (y0, x0) - reversed
#                 end_point = (yt[i,1], xt[i,1])    # (y1, x1) - reversed
#             else:
#                 start_point = (yt[i,1], xt[i,1])  # (y1, x1)
#                 end_point = (yt[i,0], xt[i,0])    # (y0, x0)
            
#             # Sample along the sloped transect
#             intensity_profile = profile_line(rmb, start_point, end_point, mode='constant')
#             values[i] = intensity_profile
            
#             # Find threshold crossing
#             idx[i] = find_first_exceeding_index(values[i], thresh_otsu)
            
#             if idx[i] is None:
#                 yPt[i] = None
#                 xPt[i] = None
#             else:
#                 # Calculate the actual coordinates along the sloped line
#                 line_length = len(values[i])
#                 if line_length > 0:
#                     # Interpolate position along the sloped line
#                     t = idx[i] / (line_length - 1) if line_length > 1 else 0
#                     xPt[i] = start_point[1] + t * (end_point[1] - start_point[1])
#                     yPt[i] = start_point[0] + t * (end_point[0] - start_point[0])
#                 else:
#                     yPt[i] = None
#                     xPt[i] = None
        
#         return np.vstack((xPt, yPt)).T, idx

#     if orn == 0:
#         # For orientation 0, sample from (y1,x1) to (y0,x0)
#         shoreline, idx = sample_sloped_transects(reverse_direction=False)
        
#     elif orn == 3:    
#         # For orientation 3, sample from (y0,x0) to (y1,x1) - reversed direction
#         shoreline, idx = sample_sloped_transects(reverse_direction=True)
        
#     else:  # orn == 1 or 2
#         # For horizontal orientations, use the same sloped approach
#         shoreline, idx = sample_sloped_transects(reverse_direction=True)

#     print(f"Find First Exceeding Index List: {idx}")
#     print(f"Extracted Shoreline Points: {shoreline}")
    
#     # print the y shoreline and yt values for comparison side by side in the same array
#     print("Shoreline Y vs Transect Y:")
#     for i in range(len(yt)):
#         print(f"Shoreline Y: {shoreline[i][1] if i < len(shoreline) else None} | Transect Y: {yt[i][0] if i < len(yt) else None}")

#     # Convert numpy data types to native Python types and handle None values in shoreline
#     slVars = {
#         'Station Name': stationname,
#         'Date': str(date),
#         'Time Info': str(dtInfo),
#         'Thresh': float(thresh),
#         'Otsu Threshold': float(thresh_otsu),
#         'Shoreline Transects': {
#             'x': xt.tolist(),
#             'y': yt.tolist()
#         },
#         'Threshold Weightings': [float(w) for w in thresh_weightings],
#         'Shoreline Points': [[float(item) if item is not None else None for item in point] for point in shoreline]
#     }

#     try:
#         del slVars['Time Info']['DateTime Object (UTC)']
#         del slVars['Time Info']['DateTime Object (LT)']
#     except:
#         pass

#     if isinstance(slVars['Shoreline Transects']['x'], np.ndarray):
#         slVars['Shoreline Transects']['x'] = slVars['Shoreline Transects']['x'].tolist()
#         slVars['Shoreline Transects']['y'] = slVars['Shoreline Transects']['y'].tolist()

#     # Create directories if they do not exist
#     base_dir = os.path.join(os.getcwd(), 'transect_jsons', stationname)
#     if not os.path.exists(base_dir):
#         os.makedirs(base_dir)
#         print(f"Created directory: {base_dir}")
#     else:
#         print(f"Directory exists: {base_dir}")

#     # Save JSON file to the directory
#     fname = os.path.join(base_dir, f'{stationname}-{datetime.strftime(dtInfo, "%Y-%m-%d_%H%M")}.avg.slVars.json')
#     with open(fname, "w") as f:
#         json.dump(slVars, f, indent=4)
#     print(f"Saved JSON to: {fname}")
    
#     return shoreline


def pltFig_tranSL(stationInfo, photo, tranSL):
    # Print the dimensions of the photo
    print(f"Photo dimensions: {photo.shape}")

    # Print the first few shoreline coordinates
    print(f"Shoreline coordinates (first 10): {tranSL}")
    # Creates shoreline product.
    stationname = stationInfo['Station Name']
    dtInfo = stationInfo['Datetime Info']
    date = str(dtInfo.date())
    time = str(dtInfo.hour).zfill(2) + str(dtInfo.minute).zfill(2)  # Ensure two digits for hour and minute
    Di = stationInfo['Dune Line Info']
    # duneInt = Di['Dune Line Interpolation']
    duneInt = Di['Dune Line Points']
    xi, py = duneInt[:,0], duneInt[:,1]
    
    
    # if orn == 0 or orn == 3 sort the tranSL points by x-coordinate
    # if stationInfo['Orientation'] == 0 or stationInfo['Orientation'] == 3:
    #     tranSL = tranSL[np.argsort(tranSL[:, 0])]
    # # if orn == 1 or orn == 2 sort the tranSL points by y-coordinate
    # elif stationInfo['Orientation'] == 1 or stationInfo['Orientation'] == 2:
    #     tranSL = tranSL[np.argsort(tranSL[:, 1])]
    # print(f"Sorted tranSL coordinates: {tranSL}")
    
    # Convert None values to np.nan and ensure float dtype
    tranSL = np.array(tranSL, dtype=np.float64)
    
    # Filter rows with NaN values
    valid_mask = ~np.isnan(tranSL).any(axis=1)
    tranSL = tranSL[valid_mask]
    
    # Sort based on orientation
    if len(tranSL) > 0:  # Only sort if we have valid points
        if stationInfo['Orientation'] in [0, 3]:
            tranSL = tranSL[np.argsort(tranSL[:, 0])]
        elif stationInfo['Orientation'] in [1, 2]:
            tranSL = tranSL[np.argsort(tranSL[:, 1])]
    else:
        print("Warning: No valid shoreline points after filtering")
    
    print(f"Sorted tranSL coordinates: {tranSL}")
    
    plt.ioff()
    fig_tranSL = plt.figure()
    plt.imshow(photo, interpolation='nearest')
    plt.xlabel("Image Width (pixels)", fontsize=10)
    plt.ylabel("Image Height (pixels)", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.plot(tranSL[:, 0], tranSL[:, 1], color='r', linewidth=2, label='Detected Shoreline')
    plt.plot(xi, py, color='blue', linewidth=2, label='Baseline', zorder=4)
    plt.title(('Transect Based Shoreline Detection (Time Averaged)\n' + stationname.capitalize() + 
            ' on ' + date + ' at ' + time[:2] + ':' + 
            time[2:] + ' UTC'), fontsize = 12)
    plt.legend(prop={'size': 9})
    plt.tight_layout()
    
    # Construct the save path for the figure
    base_dir = os.path.join(os.getcwd(), 'images', stationname, 'average')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        print(f"Directory exists: {base_dir}")

    print(f"Current Station Name from StationInfo: {stationname}")
    # print(f"Check Station Name from Input: {stationName}")
    saveName = os.path.join(base_dir, f'{stationname}-{date}_{time}.tranSL-avg.fix.jpeg')
    plt.savefig(saveName, bbox_inches='tight', dpi=400)
    plt.close()
    print(f"Saved fig_tranSL to: {saveName}")
    
    return fig_tranSL


# Current method only accepts image file paths.
# We need an option for passing numpy image arrays if images are preprocessed. 
def getTimexShoreline(stationName, imgName, imgPath=False, imgDate=None):
    # Main program.
    cwd = os.getcwd()
    stationPath = os.path.join(cwd, stationName + '_roi.config.json')
    # stationPath = Path(cwd) / 
    if not os.path.exists(stationPath):
        # Try the alternative path if the stationPath doesn't exist
        stationPath = os.path.join(f'.\configs\{stationName}_roi.config.json')
        # stationPath = r'shoreline\configs\{stationName}_roi.config.json'
        # stationPath = r'configs\{stationName}_roi.config.json'
    print(f"Station path: {stationPath}")
    stationInfo = getStationInfo(stationPath)
    if imgPath:
        dtObj = datetime.strptime(re.sub(r'\D', '', imgName), '%Y%m%d%H%M%S')
        stationInfo['Datetime Info'] = dtObj
        print(f"imgName: {imgName}")
        # print(f"imgName shape: {cv2.imread(imgName).shape}")
        print(f"imgName type: {type(cv2.imread(imgName))}")
        # Converts image color scale.
        photoAvg = cv2.cvtColor(cv2.imread(imgName), cv2.COLOR_BGR2RGB)
    else:
        stationInfo['Datetime Info'] = datetime.fromisoformat(imgDate)
        # stationInfo['Datetime Info'] = stationInfo['Datetime Info'].replace(microsecond=0)
        print(f"imgName type: {type(imgName)}")
        photoAvg = cv2.cvtColor(imgName[:,:,::-1], cv2.COLOR_BGR2RGB)
    
    # If "Image Resize" is in the stationInfo, resize the image, i.e., 30 -> 0.3
    if 'Image Resize' in stationInfo:
        resize_factor = stationInfo['Image Resize']
        # convert to float from string
        resize_factor = float(resize_factor) / 100
        # Check if resize_factor is a valid number between 0 and 1
        if isinstance(resize_factor, (int, float)) and 0 < resize_factor <= 1:
            # Resize the image
            new_size = (int(photoAvg.shape[1] * resize_factor), int(photoAvg.shape[0] * resize_factor))
        else:
            print(f"Invalid resize factor: {resize_factor}. Skipping resizing.")
    else:
        # Default to 30% if not specified
        print("No resize factor specified. Defaulting to 30%.") 
        # Resizes image to 30% of original size.
        new_size = (int(photoAvg.shape[1] * 0.3), int(photoAvg.shape[0] * 0.3))
        
    resized_image = cv2.resize(photoAvg, new_size, interpolation=cv2.INTER_AREA)
    
    # Creating an array version of image dimensions for plotting.
    h, w = resized_image.shape[:2]
    xgrid, ygrid = np.linspace(0, w, w, dtype=int), np.linspace(0, h, h, dtype=int)
    X, Y = np.meshgrid(xgrid, ygrid, indexing = 'xy')
    
    # Maps regions of interest on plot.
    maskedImg = mapROI(stationInfo, resized_image)
    
    # Computes rmb.
    rmb = maskedImg[:,:,0] - maskedImg[:,:,2]
    
    
    
    P = improfile(rmb, stationInfo).reshape(-1, 1)
    # plot the profile line
    # plt.figure(figsize=(10, 10))
    # plt.plot(P)
    # plt.title('Profile Line Intensity Values')
    # plt.xlabel('Pixel Index')
    # plt.ylabel('Intensity Value')
    # plt.grid()
    # plt.show()
    
    
    # Computing probability density function and finds threshold points.
    pdfVals, pdfLocs = ksdensity(P)
    thresh_weightings = [(1/3), (2/3)] # can we optimize these weightings based on controlling the variance/covariance to minimize detection of light scattering?
    peaks = signal.find_peaks(pdfVals)
    peakVals = np.asarray(pdfVals[peaks[0]])
    peakLocs = np.asarray(pdfLocs[peaks[0]])  

    thresh_otsu = threshold_otsu(P)
    print(f"Threshold Otsu: {thresh_otsu}")
    print(f"Shape of Thresh Otsu: {thresh_otsu.shape}")
    print(f"Shape of Peak Locs: {peakLocs.shape}")
    print(f"Shape of Peak Vals: {peakVals.shape}")
    print(f"Peak Locs: {peakLocs}")
    print(f"Peak Vals: {peakVals}")
    I1 = np.asarray(np.where(peakLocs < thresh_otsu))
    J1, = np.where(peakVals[:] == np.max(peakVals[I1]))
    I2 = np.asarray(np.where(peakLocs > thresh_otsu))
    J2, = np.where(peakVals[:] == np.max(peakVals[I2]))
    thresh = (thresh_weightings[0]*peakLocs[J1] +
            thresh_weightings[1]*peakLocs[J2])
    thresh = float(thresh)
    threshInfo = {
        'Thresh':thresh, 
        'Otsu Threshold':thresh_otsu,
        'Threshold Weightings':thresh_weightings
        }
    print(f"Threshold Info: {threshInfo}")

    # plot histogram of rmb
    # plt.figure(figsize=(10, 10))
    # plt.hist(rmb[~np.isnan(rmb)], bins=100, color='gray', alpha=0.7)
    # # add vertical line for thresh_otsu value
    # plt.axvline(thresh_otsu, color='red', linestyle='dashed', linewidth=1, label='Otsu Threshold')
    # plt.title('Histogram of RMB Values')
    # plt.xlabel('RMB Value')
    # plt.ylabel('Frequency')
    # plt.grid()
    # plt.show()

    # Generates final json and figure for shoreline products.
    tranSL = extract(stationInfo, rmb, maskedImg, threshInfo)
    fig_tranSL = pltFig_tranSL(stationInfo, resized_image, tranSL)
    
    return(tranSL, fig_tranSL)


###########################################################################


# stationName = 'oakisland_west'

# imgNames = ['timex.oakisland_west-2024-10-03-121832Z.jpg']

# for imgName in imgNames:
#     tranSL, fig_tranSL = getTimexShoreline(stationName, imgName)

# stationName = 'cocoabeach'
# imgName = "timex.cocoabeach-2024-07-07-120007Z.jpg"
# tranSL, fig_tranSL = getTimexShoreline(stationName, imgName)


# stationName = 'oakisland_west'
# stationName = 'oakisland_west_roi'
# imgName = "timex.oakisland_west-2024-10-03-121832Z.jpg"
# tranSL, fig_tranSL = getTimexShoreline(stationName, imgName)



# stationName = 'jennette_north'
# imgName = "timex.jennette_north-2024-06-13-121821Z.jpg"
# tranSL, fig_tranSL = getTimexShoreline(stationName, imgName)



# # stationName = 'ferrybeach_north'
# stationName = 'ferrybeach_north_roi'
# imgName = "ferrybeach_north-2025-04-03-102053Z.jpg"
# # imgName = "ferrybeach_north-2025-03-04-171049Z.jpg"
# tranSL, fig_tranSL = getTimexShoreline(stationName, imgName)

# stationName = 'unalakleet_camera'
# imgName = "unalakleet_camera-2025-05-12-201346Z.jpg"
# tranSL, fig_tranSL = getTimexShoreline(stationName, imgName)


# station_name = 'plum_island_point'
# # station_name = 'plum_island_point_roi'
# imgName = "plum_island_point-2025-05-13-143031Z.jpg"
# tranSL, fig_tranSL = getTimexShoreline(station_name, imgName)

# stationName = 'salisbury_beach_roi'
# imgName = "salisbury_beach-2025-05-13-142833Z.jpg"
# tranSL, fig_tranSL = getTimexShoreline(stationName, imgName)



# station_name = 'currituck_sailfish'
# imgName = "still.currituck_sailfish-2025-05-01-144550Z.jpg"
# tranSL, fig_tranSL = getTimexShoreline(station_name, imgName)


# station_name = 'madeira_beach_roi'
# imgName = "madeira_beach-2025-06-12-160835Z.jpg"
# tranSL, fig_tranSL = getTimexShoreline(station_name, imgName)

# station_name = 'jennette_south'
# station_name = 'jennette_north'
# station_name = 'cocoabeach'
# station_name = 'currituck_hampton_inn'
# station_name = 'oakisland_west'
# station_name = 'westerly'
# images\currituck_sailfish_roi\stills
# get the list of image files in the directory
# import glob
# image_files = glob.glob(os.path.join('images', station_name, 'stills', '*.jpg'))
# print(f"Found {len(image_files)} image files in {station_name} directory.")


# station_name = 'currituck_sailfish_roi'
# station_name = 'jennette_south_roi'
# station_name = 'jennette_north_roi'
# station_name = 'cocoabeach_roi'
# station_name = 'currituck_hampton_inn_roi'
# station_name = 'oakisland_west_roi'
# station_name = 'westerly_roi'
# for imgName in image_files[:]:
#     try:
#         print(f"Processing image: {imgName}")
#         tranSL, fig_tranSL = getTimexShoreline(station_name, imgName)
#         print(f"Processed {imgName} with {len(tranSL)} shoreline points.")
#     except Exception as e:
#         print(f"Error processing {imgName}: {e}")
#         continue