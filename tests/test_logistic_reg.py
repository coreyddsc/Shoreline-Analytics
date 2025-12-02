import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_manager import *
from src.differential_geometry import *
from src.logistic_reg import *

class BaseCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Common setup for all tests
        cls.sldata_path = r"data\jennette_south_roi_model_118cf4cf71a10dce.csv"
        # filename without extension or path
        cls.base_filename = Path(cls.sldata_path).stem
        cls.data_dir = Path(cls.sldata_path).parent
        # test parameters
        cls.global_time_reference = datetime(2024, 1, 1, 0, 0, 0)
        cls.longitude = -75.594 # hardcoded for jennette south right now
        cls.latitude = 35.906
        
        cls.sldata = load_data(cls.sldata_path)
        cls.slp_tensor, cls.timestamps, cls.transects = build_slpt_tensor(cls.sldata)
        cls.dt, cls.dQ, cls.delQ = compute_differentials(cls.slp_tensor)
        cls.dS, cls.delS = compute_arc_lengths(cls.dQ, cls.delQ)
        S = compute_arc_length_coordinate(cls.delS)
        S = pd.DataFrame(S)
        # add timestamps to S dataframe as index
        S.index = pd.to_datetime(cls.timestamps)
        cls.S = S.copy()
        S_melt = S.melt(var_name='Node', value_name='ArcLength', ignore_index=False).reset_index()
        S_melt.rename(columns={'index':'timestamp'}, inplace=True)
        S_melt = S_melt.dropna(subset=['ArcLength'])
        cls.S_melt = S_melt.copy()
        
        
        cls.tau = compute_worldline_arc_length(cls.dS)


        cls.astro_features = build_astro_features(cls.timestamps, cls.latitude, cls.longitude)
        cls.time_features = build_time_features(cls.timestamps, cls.global_time_reference)
        
        print(cls.time_features[['timestamp', 't_mod_12_4']].head())
        print(f"Time Features Info:")
        cls.time_features.info()
        print(cls.S_melt.head())
        print(f"S_melt Info:")
        cls.S_melt.info()
        # Perform the one-to-many join
        merged_df = cls.S_melt.merge(
            # cls.time_features[['timestamp', 't_mod_12_4']],
            cls.time_features, 
            left_on='timestamp', 
            right_on='timestamp', 
            how='left'
        )

        # Drop the duplicate timestamp column if you don't need it
        merged_df = merged_df.drop('timestamp', axis=1)

        print(merged_df.head())
        print(f"Merged DataFrame Info:")
        merged_df.info()
        cls.merged_df = merged_df.copy()
        
        
class TestBuildFeatureSpace(BaseCaseTest):
    def test_build_feature_space(self):
        try:
            # X, y, encoder = build_feature_space(self.S_melt, nodes=[21, 22], feature_cols=['ArcLength'])
            X, y, encoder = build_feature_space(self.S_melt, nodes=[i for i in range(12, 38, 4)], feature_cols=['ArcLength'])
            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(encoder, OneHotEncoder)
        except Exception as e:
            self.fail(f"build_feature_space raised an exception: {e}")
            
            
class TestNodalLogisticReg(BaseCaseTest):
    def test_nodal_logistic_reg(self):
        try:
            # print(self.astro_features.columns)
            # self.astro_features.info()
            # X, y, encoder = build_feature_space(self.S_melt, nodes=[18, 22], feature_cols=['ArcLength'], one_hot_encode=True)
            # X, y, encoder = build_feature_space(self.merged_df, nodes=[18, 22], feature_cols=['ArcLength', 't_mod_12_4'], one_hot_encode=True)
            # X, y, encoder = build_feature_space(self.merged_df, nodes=[i for i in range(2, 38, 5)], feature_cols=['ArcLength', 't_mod_12_4'], one_hot_encode=False)
            # X, y, encoder = build_feature_space(self.merged_df, nodes=[i for i in range(17, 37, 4)], feature_cols=['ArcLength', 't_mod_12_4', 't_mod_29_53', 't_hours'], one_hot_encode=False)
            X, y, encoder = build_feature_space(self.merged_df, nodes=[i for i in range(17, 37, 4)], feature_cols=['ArcLength', 't_mod_12_4'], one_hot_encode=False)
            # X, y, encoder = build_feature_space(self.merged_df, feature_cols=['ArcLength', 't_mod_12_4'], one_hot_encode=False)
            # X, y, encoder = build_feature_space(self.S_melt, nodes=[i for i in range(2, 38, 5)], feature_cols=['ArcLength'])
            model, scaler = nodal_logistic_reg(X, y)
            # plot_binom_reg_2d(X, model, scaler)
            plot_polar_probability(X, model, scaler)
            plot_polar_nodal_regions(X, model, scaler)
            plot_polar_decision_boundaries(X, model, scaler)
            plot_polar_node_probability_contours(X, model, scaler, target_node=17)
            
            # For a single (s, t) point:
            s_value = 350.0
            t_value = 6.2  # tidal phase in hours

            # Create feature array and scale it
            X_point = np.array([[s_value, t_value]])
            X_point_scaled = scaler.transform(X_point)

            # Get probabilities for ALL nodes
            probabilities = model.predict_proba(X_point_scaled)
            # Returns: [p_node18, p_node22, p_node29, ...] for your 3 nodes

            print(f"Probabilities at s={s_value}, t={t_value}:")
            for i, node in enumerate(model.classes_):
                print(f"Node {node}: {probabilities[0,i]:.3f}")
            self.assertIsNotNone(model)
            self.assertIsNotNone(scaler)
            print("TestNodalLogisticReg: Model trained successfully.")
        except Exception as e:
            self.fail(f"nodal_logistic_reg raised an exception: {e}")
            
    def test_plot_binom_reg(self):
        try:
            X, y, encoder = build_feature_space(self.S_melt, nodes=[18, 22], feature_cols=['ArcLength'], one_hot_encode=True)
            model, scaler = nodal_logistic_reg(X, y)
            plot_binom_reg(X, model, scaler)
            print("TestPlotBinomReg: Plot generated successfully.")
        except Exception as e:
            self.fail(f"plot_binom_reg raised an exception: {e}")
        
