import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from skyfield_astro import *
except ImportError:
    from .skyfield_astro import *


class CONSTANTS:
    gtr = datetime(2024, 1, 1, 0, 0, 0)
    long = -75.594 # hardcoded for jennette south right now
    lat = 35.906

def load_config(station_name: str):
    """Load configuration from JSON file."""
    path = rf"configs\{station_name}.json"
    p = Path(path)
    with open(p, 'r') as f:
        config = json.load(f)
    # Load other necessary configurations
    print(f"Loaded config for {station_name}: {config.keys()}")
    return config

def load_data(file_path):
    """Load shoreline data from a CSV file."""
    return pd.read_csv(file_path)


def save_3d_arrays(filename, **arrays):
    """Save multiple 3D arrays with their names"""
    with h5py.File(filename, 'w') as f:
        for name, array in arrays.items():
            f.create_dataset(name, data=array)
            print(f"Saved {name} with shape {array.shape}")
            
            
def save_hdf5_datasets(filename, datasets):
    """Save multiple datasets to an HDF5 file."""
    with h5py.File(filename, 'a') as f:
        for dataset in datasets:
            if isinstance(dataset, dict):
                name = dataset['name']
                data = dataset['data']
                labels = dataset.get('labels')
            elif len(dataset) == 2:
                name, data = dataset
                labels = None
            else:
                name, data, labels = dataset
            
            if name in f:
                del f[name]
            dset = f.create_dataset(name, data=data)
            if labels:
                for i, label in enumerate(labels):
                    dset.dims[i].label = label

def load_3d_arrays(filename):
    """Load all 3D arrays from file"""
    arrays = {}
    with h5py.File(filename, 'r') as f:
        for name in f.keys():
            arrays[name] = f[name][:]
    return arrays


def manage_time_units(timestamps):
    """Convert timestamps to a consistent datetime format."""
    print(f"Managing time units for {timestamps.shape[0]} timestamps.")
    # get unique timestamps
    # unique_timestamps = pd.to_datetime(timestamps.unique())
    unique_timestamps = timestamps.unique()
    sorted_timestamps = np.sort(unique_timestamps)
    output = sorted_timestamps
    print(f"Unique timestamps found: {len(unique_timestamps)}")
    print(f"Sample timestamps: {output[:5]}")
    return output

# there has to be a better way to do this
def identify_continuous_segments(timestamps, max_gap_hours=2.0):
    """Split timestamps into continuous segments based on gap threshold"""
    # Convert to hours since first timestamp
    time_values = np.array([(ts - timestamps[0]).total_seconds() / 3600.0 
                        for ts in timestamps])
    
    dt = np.diff(time_values)
    large_gaps = dt > max_gap_hours
    
    # Find segment boundaries
    segment_starts = [0]
    segment_ends = []
    
    for i, is_large_gap in enumerate(large_gaps):
        if is_large_gap:
            segment_ends.append(i+1)  # End before gap
            segment_starts.append(i+1)  # Start after gap
    segment_ends.append(len(timestamps))
    
    segments = []
    for start, end in zip(segment_starts, segment_ends):
        if end - start >= 2:  # Need at least 2 points
            segments.append(slice(start, end))
    
    return segments, time_values


def build_time_features(timestamps, global_start=None):
    """Build time features from unique timestamps for tidal modeling."""
    
    # Get unique timestamps only
    unique_timestamps = sorted(set(timestamps))
    
    if global_start is None:
        global_start = min(unique_timestamps)
    
    time_features = []
    for timestamp in unique_timestamps:
        # Consistent across all days and stations
        t_hours = (timestamp - global_start).total_seconds() / 3600.0
        t_days = t_hours / 24.0
        
        # Basic tidal features
        t_mod_24 = t_hours % 24
        t_mod_12_4 = t_hours % 12.4
        sin_24 = np.sin(2 * np.pi * t_hours / 24)
        cos_24 = np.cos(2 * np.pi * t_hours / 24)
        sin_12_4 = np.sin(2 * np.pi * t_hours / 12.4)
        cos_12_4 = np.cos(2 * np.pi * t_hours / 12.4)
        
        # Lunar monthly cycle (29.53 days)
        t_mod_29_53 = t_days % 29.53
        sin_29_53 = np.sin(2 * np.pi * t_days / 29.53)
        cos_29_53 = np.cos(2 * np.pi * t_days / 29.53)
        
        # Seasonal cycle (365.25 days)
        t_mod_365_25 = t_days % 365.25
        sin_365_25 = np.sin(2 * np.pi * t_days / 365.25)
        cos_365_25 = np.cos(2 * np.pi * t_days / 365.25)
        
        time_data = {
            'timestamp': timestamp,  # Include the timestamp
            't_hours': t_hours,
            't_days': t_days,
            # Tidal cycles
            't_mod_24': t_mod_24,
            't_mod_12_4': t_mod_12_4, 
            'sin_24': sin_24,
            'cos_24': cos_24,
            'sin_12_4': sin_12_4,
            'cos_12_4': cos_12_4,
            # Monthly lunar cycle
            't_mod_29_53': t_mod_29_53,
            'sin_29_53': sin_29_53,
            'cos_29_53': cos_29_53,
            # Seasonal cycle  
            't_mod_365_25': t_mod_365_25,
            'sin_365_25': sin_365_25,
            'cos_365_25': cos_365_25,
        }
        
        time_features.append(time_data)
    
    df = pd.DataFrame(time_features)
    # df.set_index('timestamp', inplace=True)  # Set timestamp as index
    return df


def build_astro_features(timestamps, latitude, longitude):
    """Build astronomical features using Skyfield."""
    # Get unique timestamps only
    timestamps = sorted(set(timestamps))
    print(f"Building astronomical features for {len(timestamps)} timestamps at lat={latitude}, lon={longitude}.")
    astro_data = []
    for ts in timestamps:
        data = get_skyfield_positions(ts, latitude, longitude)
        astro_data.append(data)
    astro_df = pd.DataFrame(astro_data)
    return astro_df


def build_slpt_tensor(data):
    """Build shoreline position tensor from data."""
    # Placeholder implementation
    print("Building shoreline position tensor.")
    # the data format here is still given by detected_x, detected_y, transect_num, and timestamp per row with other metadata
    # we want to to pivot this into a 3D tensor: (timestamps, transects, positions)
    # where rows are timestamps, columns are transects, and depth is positions along the transect
    Qx = data.pivot_table(index='timestamps', columns='transect_num', values='detected_x')
    Qy = data.pivot_table(index='timestamps', columns='transect_num', values='detected_y')
    # ensure Qx,Qy are sorted by timestamp and transect_num
    Qx = Qx.sort_index().sort_index(axis=1)
    Qy = Qy.sort_index().sort_index(axis=1)
    
    # extract timestamps and transect numbers for reference lists
    timestamps = Qx.index.tolist()
    transects = Qx.columns.tolist()
    # build expanded time-array that matches the shape of Qx such that each transect has the same timestamps
    # convert timestamps to hourly deltas from a global reference time
    # use earliest timestamp as global reference time
    # convert each timestamp to datetime() if they are not already
    if not isinstance(timestamps[0], datetime):
        timestamps = pd.to_datetime(timestamps)
    global_time_reference = min(timestamps)
    print(f"Timestamps type: {type(timestamps[0])}, Global time reference: {global_time_reference}")
    Qt = (timestamps - global_time_reference).total_seconds() / 3600.0
    print(f"Qt type: {type(Qt.values)}, shape: {Qt.shape}")
    Qt = np.tile(Qt.values[:, np.newaxis], (1, len(transects)))  # shape (num_timestamps, num_transects)
    # stack values of Qx and Qy into 3D tensors along a new axis
    Qx_tensor = Qx.values[:, :, np.newaxis]  # shape (num_timestamps, num_transects, 1)
    Qy_tensor = Qy.values[:, :, np.newaxis]  # shape (num_timestamps, num_transects, 1)
    Qt_tensor = Qt[:, :, np.newaxis]  # shape (num_timestamps, num_transects, 1)
    Q = np.concatenate([Qx_tensor, Qy_tensor, Qt_tensor], axis=2)  # shape (num_timestamps, num_transects, 3)
    return Q, timestamps, transects


# Time should always be the last dimension in Q, if Q. Not all the tensors saved in the HDF5 will have time as last dimension.
def melt_tensor(tensor, var_name='Node', value_name='Value', time_index=None):
    """Melt a 3D tensor into a long-format DataFrame."""
    n_time, n_nodes, n_dims = tensor.shape
    records = []
    for t in range(n_time):
        for n in range(n_nodes):
            record = {
                'TimeIndex': time_index[t] if time_index is not None else t,
                var_name: n,
                value_name: tensor[t, n, :]
            }
            records.append(record)
    df = pd.DataFrame(records)
    # expand value_name column into multiple columns if n_dims > 1
    if n_dims > 1:
        value_df = pd.DataFrame(df[value_name].tolist(), columns=[f'{value_name}_{i}' for i in range(n_dims)])
        df = pd.concat([df.drop(columns=[value_name]), value_df], axis=1)
    return df
