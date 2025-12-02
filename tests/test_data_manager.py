import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_manager import *
from src.differential_geometry import *

class BaseCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Common setup for all tests
        cls.sldata_path = rf"data\jennette_south_roi_model_118cf4cf71a10dce.csv"
        # filename without extension or path
        cls.base_filename = Path(cls.sldata_path).stem
        cls.data_dir = Path(cls.sldata_path).parent
        # test parameters
        cls.global_time_reference = datetime(2024, 1, 1, 0, 0, 0)
        cls.longitude = -75.594 # hardcoded for jennette south right now
        cls.latitude = 35.906


class TestLoadData(BaseCaseTest):
    def test_load_data(self):
        data = load_data(self.sldata_path)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        print("TestLoadData: Data loaded successfully and is a DataFrame.")
        data.info()
        
        
class TestSaveLoad3DArrays(BaseCaseTest):
    def test_save_load_3d_arrays(self):
        data = load_data(self.sldata_path)
        Q, _, _ = build_slpt_tensor(data)
        test_filename = f"{self.data_dir}/{self.base_filename}_test_arrays.h5"
        print(f"Testing save/load 3D arrays with file: {test_filename}")
        save_3d_arrays(test_filename, Q=Q)
        loaded_arrays = load_3d_arrays(test_filename)
        self.assertIn('Q', loaded_arrays)
        np.testing.assert_array_equal(Q, loaded_arrays['Q'])
        print("TestSaveLoad3DArrays: 3D arrays saved and loaded successfully.")
        
    def test_save_hdf5_dataset(self):
        data = load_data(self.sldata_path)
        Q, _, _ = build_slpt_tensor(data)
        t = Q[:,:,-1]
        dt, dQ, delQ = compute_differentials(Q, max_time_delta=1.0)
        dS, delS = compute_arc_lengths(dQ, delQ)
        s = compute_arc_length_coordinate(delS)
        tau = compute_worldline_arc_length(dS)
        velocity = compute_velocity_vectors(dQ, dt)
        speed = compute_speed(velocity)
        kinetic_energy = 0.5 * speed**2
        
        test_filename = f"{self.data_dir}/{self.base_filename}_test.h5"
        dataset_name = "Q"
        labels = ['x', 'y', 'time']
        print(f"Testing save HDF5 dataset with file: {test_filename}")
        datasets = [(dataset_name, Q, labels), ("time_hourly", t), ("dt", dt), ("dQ", dQ), ("delQ", delQ),
                    ("dS", dS), ("delS", delS), ("s", s), ("tau", tau),
                    ("Velocity", velocity), ("Speed", speed), ("Kinetic Energy", kinetic_energy)]
        save_hdf5_datasets(test_filename, datasets)
        with h5py.File(test_filename, 'r') as f:
            self.assertIn(dataset_name, f)
            dset = f[dataset_name]
            self.assertEqual(dset.shape, Q.shape)
            for i, label in enumerate(labels):
                self.assertEqual(dset.dims[i].label, label)
        print("TestSaveHDF5Dataset: HDF5 dataset saved and verified successfully.")

class TestManageTimeUnits(BaseCaseTest):
    def test_manage_time_units(self):
        data = load_data(self.sldata_path)
        timestamps = data['timestamps']
        # timestamps = pd.to_datetime(data['timestamp'])
        manage_time_units(timestamps)
        # print("TestManageTimeUnits: Time units managed successfully.")
        

class TestIdentifyContinuousSegments(BaseCaseTest):
    def test_identify_continuous_segments(self):
        data = load_data(self.sldata_path)
        timestamps = pd.to_datetime(data['timestamps'])
        segments = identify_continuous_segments(timestamps, max_gap_hours=2.0)
        print(f"Segments type: {type(segments)}")
        self.assertIsInstance(segments, tuple)
        self.assertGreater(len(segments), 0)
        # for segment in segments[:3]:
        #     print(f"Segment type: {type(segment)}, length: {len(segment)}")
        #     self.assertIsInstance(segment, np.ndarray)
            # self.assertGreater(len(segment), 0)
        # print(f"TestIdentifyContinuousSegments: Identified {len(segments)} continuous segments successfully.")
        
        
class TestBuildTimeFeatures(BaseCaseTest):
    def test_build_time_features(self):
        data = load_data(self.sldata_path)
        timestamps = pd.to_datetime(data['timestamps'])
        df = build_time_features(timestamps, self.global_time_reference)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        print(df.iloc[:30, :5])
        df.info()
        # print("TestBuildTimeFeatures: Time features built successfully.")


class TestBuildAstroFeatures(BaseCaseTest):
    def test_build_astro_features(self):
        data = load_data(self.sldata_path)
        timestamps = pd.to_datetime(data['timestamps'])
        # are the timestamps unique?
        df = build_astro_features(timestamps, self.latitude, self.longitude)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        print(df.iloc[:30, :5])
        df.info()
        # print("TestBuildAstroFeatures: Astronomical features built successfully.")
        
        
class TestBuildSLPTensor(BaseCaseTest):
    def test_build_slpt_tensor(self):
        data = load_data(self.sldata_path)
        slp_tensor, _, _ = build_slpt_tensor(data)
        self.assertIsInstance(slp_tensor, np.ndarray)
        self.assertGreater(slp_tensor.size, 0)
        print(f"SLP Tensor shape: {slp_tensor.shape}")
        # print("TestBuildSLPTensor: SLP tensor built successfully.")
        
        
class TestPivotTensorToDataFrame(BaseCaseTest):
    def test_pivot_tensor_to_dataframe(self):
        data = load_data(self.sldata_path)
        slp_tensor, timestamps, transects = build_slpt_tensor(data)
        df = pivot_tensor_to_dataframe(slp_tensor, timestamps, transects)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        print(df.head())
        df.info()
        # print("TestPivotTensorToDataFrame: Tensor pivoted to DataFrame successfully.")

if __name__ == '__main__':
    unittest.main()