import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
	from skyfield_astro import *
	from src.differential_geometry import *
except ImportError:
	from .skyfield_astro import *
	from .differential_geometry import *
	


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

def load_data(file_path: Path):
	"""Load shoreline data from a CSV file."""
	if isinstance(file_path, Path):
		pass
	else:
		file_path = Path(file_path)
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


def build_mldata(data: pd.DataFrame, kinematic_features: bool = True, astro_features: bool = True, time_features: bool = True):
	df = data.copy()
	print(f"Input Data Columns:\n{df.columns}")

	tnsr, ts, trx = build_slpt_tensor(df) # tensor, timestamps, transects
	print(f"Timestamps:\n{ts}")
	print(f"Transect Columns:\n{trx}")
	x_coords = tnsr[:,:,0]
	y_coords = tnsr[:,:,1]

	# Arc/Path Length Features
	dt, dQ, delQ = compute_differentials(tnsr) # tensor has dim(X,Y,Time)
	dxdt = dQ[:,:,0]
	dydt = dQ[:,:,1]
	delx = delQ[:,:,0]
	dely = delQ[:,:,1]
	dS, delS = compute_arc_lengths(dQ, delQ)
	Tau = compute_worldline_arc_length(dS)
	S = compute_arc_length_coordinate(delS)
	T = compute_tangent_vectors(delQ, delS)
	N = compute_normal_vectors(T)
	d_delS_dt = compute_arc_change_rates(delS, dt) # true shoreline strain rate
	_, d2_delS_dt2 = compute_shoreline_strain_acceleration(delS, dt)
	dQdS = compute_worldline_tangents(dQ, dS)
	kappa, dT_dS = compute_curvature(T, delS, S)
	kappa_Tau, dT_dTau = compute_worldline_curvature(dQdS, Tau)
	# Kinematic/Energy Features
	if kinematic_features:
		vel = compute_velocity_vectors(dQ, dt)
		speed = compute_speed(vel)
		KE = speed**2 // 2
		V_t, V_n, V_t_mag, V_n_mag = compute_velocity_components(T, N, vel)
		grad_v, strain_rate, vorticity, vorticity_vector = compute_velocity_gradient_tensor(vel, delS, dt)
		a_T = compute_tangential_acceleration(V_t_mag, dt)
		a_N = compute_normal_acceleration(kappa, V_n_mag) # similar to a kinetic energy equation
		A_Geometric, a_mag = compute_geometric_acceleration(a_T, a_N, T, N)
		g_ss, g_tt, g_st = compute_metrics(T, vel)
		g = compute_metric_tensor(T, vel)
		anistropy_ratio = compute_anistropy_ratio(g)
		eigvals, eigvecs = compute_metric_eigvals(g)
		
	tensors = {
		'xc': x_coords,
		'yc': y_coords,
		'dxdt': dxdt,
		'dydt': dydt,
		'delx': delx,
		'dely': dely,
		'dS': dS,
		'delS': delS,
		'Tau': Tau, # worldline coordinates
		'S': S,
		'T': T,
		'N': N,
		'dQdS': dQdS, # worldline tangents
		'v': speed,
		'V_t': V_t,
		'V_n': V_n,
		'V_t_mag': V_t_mag,
		'V_n_mag': V_n_mag,
		'grad_v': grad_v,
		'strain_rate': strain_rate,
		'Vorticity': vorticity,
		'Vorticity_Vector': vorticity_vector,
		'KE': KE,
		'a_T': a_T,
		'a_N': a_N,
		'A_Geometric': A_Geometric,
		'a_mag': a_mag,
		'd_delS_dt': d_delS_dt,
		'd2_delS_dt2': d2_delS_dt2,
		'Kappa': kappa,
		'dT_dS': dT_dS,
		'KappaTau': kappa_Tau,
		'dT_dTau': dT_dTau,
		'g_ss': g_ss, # spatial metric component
		'g_tt': g_tt, # temporal metric component
		'g_st': g_st, # mixed metric component
		'g': g, # metric tensor
		'anistropy_ratio': anistropy_ratio,
		'eigvals': eigvals,
		'eigvecs': eigvecs
	}

	melted_features = {}
	for feat, data in tensors.items():
		print(f"{feat} Data Shape: {data.ndim}")
		if data.ndim == 2:
			print(f"Feature: {feat} | Shape: {data.shape}")
			data = pd.DataFrame(data, index=ts, columns=trx)
			melt = data.melt(var_name='Node', value_name=feat, ignore_index=False).reset_index()
			melt.rename(columns={'index': 'timestamp'}, inplace=True)
			melted_features.setdefault(feat, melt)
			print(f"Feature: {feat} | Melted Shape: {melt.shape} | Melted Data Columns: {melt.columns}")

	print(f"Melted Features Keys:\n{melted_features.keys()}")
	# Start with the first dataframe
	merged_df = melted_features[list(melted_features.keys())[0]]

	# Merge the rest
	for feat in list(melted_features.keys())[1:]:
		merged_df = merged_df.merge(melted_features[feat], on=['timestamp', 'Node'])

	merged_df.info()
	print(f"Merged df timestamps check:\n{merged_df['timestamp']}")
	
	
	# Solar/Lunar Periodic Time Features
	if time_features:
		time_features = build_time_features(ts, CONSTANTS.gtr) # gtr = global time reference
		merged_df = merged_df.merge(time_features, on='timestamp', how='left')
	
	# Astro (Skyfield) Features
	if astro_features:
		astro_features = build_astro_features(ts, CONSTANTS.lat, CONSTANTS.long)
		merged_df = merged_df.merge(astro_features, on='timestamp', how='left')

	merged_df.info()
	return merged_df

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
