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
		cls.mldata = build_mldata(cls.sldata)
		
		
class TestBuildFeatureSpace(BaseCaseTest):
	def test_build_feature_space(self):
		try:
			# X, y, encoder = build_feature_space(self.S_melt, nodes=[21, 22], feature_cols=['S'])
			X, y, encoder = build_feature_space(self.mldata, nodes=[i for i in range(12, 38, 4)], feature_cols=['S', 't_mod_12_4'], one_hot_encode=False)
			self.assertIsInstance(X, np.ndarray)
			self.assertIsInstance(y, np.ndarray)
			# self.assertIsInstance(encoder, OneHotEncoder)
		except Exception as e:
			self.fail(f"build_feature_space raised an exception: {e}")
			
			
class TestNodalLogisticReg(BaseCaseTest):
	def test_nodal_logistic_reg(self):
		try:
			# print(self.astro_features.columns)
			# self.astro_features.info()
			# X, y, encoder = build_feature_space(self.S_melt, nodes=[18, 22], feature_cols=['S'], one_hot_encode=True)
			# X, y, encoder = build_feature_space(self.mldata, nodes=[18, 22], feature_cols=['S', 't_mod_12_4'], one_hot_encode=True)
			# X, y, encoder = build_feature_space(self.mldata, nodes=[i for i in range(2, 38, 5)], feature_cols=['S', 't_mod_12_4'], one_hot_encode=False)
			# X, y, encoder = build_feature_space(self.mldata, nodes=[i for i in range(17, 37, 4)], feature_cols=['S', 't_mod_12_4', 't_mod_29_53', 't_hours'], one_hot_encode=False)
			X, y, encoder = build_feature_space(self.mldata, nodes=[i for i in range(17, 37, 4)], feature_cols=['S', 't_mod_12_4'], one_hot_encode=False)
			# X, y, encoder = build_feature_space(self.mldata, feature_cols=['S', 't_mod_12_4'], one_hot_encode=False)
			model, scaler, _ = nodal_logistic_reg(X, y)
			# plot_binom_reg_2d(X, model, scaler)
			# plot_polar_probability(X, model, scaler)
			# plot_polar_nodal_regions(X, model, scaler)
			# plot_polar_decision_boundaries(X, model, scaler)
			# plot_polar_node_probability_contours(X, model, scaler, target_node=17)
			
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
			X, y, encoder = build_feature_space(self.mldata, nodes=[18, 22], feature_cols=['S'], one_hot_encode=True)
			model, scaler, _ = nodal_logistic_reg(X, y)
			plot_binom_reg(X, model, scaler)
			print("TestPlotBinomReg: Plot generated successfully.")
		except Exception as e:
			self.fail(f"plot_binom_reg raised an exception: {e}")
		
		
class TestEvaluateModel(BaseCaseTest):
	def test_evaluate_model(self):
		print(f"Features: {self.mldata.columns}")
		X, y, encoder = build_feature_space(self.mldata, nodes=[i for i in range(0,41,3)], feature_cols=['xc', 'yc', 'S', 'Tau', 'Kappa', 'KE', 'a_N', 't_mod_12_4'], one_hot_encode=False)
		# X, y, encoder = build_feature_space(self.mldata, nodes=[i for i in range(0,41,3)], feature_cols=['S', 't_mod_12_4'], one_hot_encode=False)
		# X, y, encoder = build_feature_space(self.mldata, nodes=[18, 22], feature_cols=['S', 't_mod_12_4'], one_hot_encode=False)
		# X, y, encoder = build_feature_space(self.mldata, nodes=[i for i in range(17, 37, 4)], feature_cols=['S', 't_mod_12_4'], one_hot_encode=False)
		model, scaler, XY_test = nodal_logistic_reg(X, y)
		evaluate_model(model, X_test=XY_test[0], y_test=XY_test[1])