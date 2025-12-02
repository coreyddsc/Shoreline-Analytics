import unittest
from matplotlib import axes
import pandas as pd
import numpy as np

from src.differential_geometry import *
from src.data_manager import *
from src.make_plots import *

class BaseCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Common setup for all tests
        cls.sldata_path = r"data\jennette_south_roi_model_118cf4cf71a10dce.csv"
        # test parameters
        cls.global_time_reference = datetime(2024, 1, 1, 0, 0, 0)
        cls.longitude = -75.594 # hardcoded for jennette south right now
        cls.latitude = 35.906
        cls.sldata = load_data(cls.sldata_path)
        cls.Q, cls.timestamps, cls.transects = build_slpt_tensor(cls.sldata)
        
        # time features from data manager
        # build_time_features(cls.timestamps, cls.global_time_reference)
        cls.timefeatures = build_time_features(cls.timestamps)
        
        cls.t = cls.Q[:,:,-1]
        cls.dt, cls.dQ, cls.delQ = compute_differentials(cls.Q)
        cls.dS, cls.delS = compute_arc_lengths(cls.dQ, cls.delQ)
        cls.tau = compute_worldline_arc_length(cls.dS)
        cls.s = compute_arc_length_coordinate(cls.delS)
        cls.T = compute_tangent_vectors(cls.delQ, cls.delS)
        cls.N = compute_normal_vectors(cls.T)
        cls.V = compute_velocity_vectors(cls.dQ, cls.dt)
        cls.speed = compute_speed(cls.V)
        cls.V_t, cls.V_n, cls.V_t_mag, cls.V_n_mag = compute_velocity_components(cls.T, cls.N, cls.V)
        cls.dQdS = compute_worldline_tangents(cls.dQ, cls.dS)

        # Reminder of Q structure for Shoreline data:
        # Q: 3D array (time-state, nodes, spatial dimensions)
        # X = Q[:,:,0]  # x-coordinates
        # Y = Q[:,:,1]  # y-coordinates
        # t = Q[:,:,2]  # parameterized time coordinate for a given record/particle
        # stop calling it a transect
        # get Q hourly average for testing by using the time dimension.
        # first record is time 0 and so on by hourly time since first record.
        
        
        
class TestPlotStatePoints(BaseCaseTest):
    def test_plot_state_points(self):
        try:
            plot_state_points(self.Q[::6,:,:])
            print("TestPlotStatePoints: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")
            
            
class TestPlotStateCurves(BaseCaseTest):
    def test_plot_state_curves(self):
        try:
            plot_state_curves(self.Q[:60,:,:])
            print("TestPlotStateCurves: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")


class TestPlotDifferentials(BaseCaseTest):
    def test_plot_differentials(self):
        try:
            plot_differentials(self.Q[:60,:,:], self.dQ[:60,:,:], self.delQ[:60,:,:], step=20)
            print("TestPlotDifferentials: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")

class TestPlotArcLengths(BaseCaseTest):
    def test_plot_arc_lengths(self):
        try:
            plot_arc_lengths(self.tau[:60,:], self.s[:60,:])
            print("TestPlotArcLength: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")
            
        
class TestPlotTangentField(BaseCaseTest):
    def test_plot_tangent_field(self):
        try:
            plot_tangent_field(self.Q[:60,:,:], self.T[:60,:,:], step=10)
            print("TestPlotTangentField: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")
            
            
class TestPlotNormalField(BaseCaseTest):
    def test_plot_normal_field(self):
        try:
            plot_normal_field(self.Q[:60,:,:], self.N[:60,:,:], step=10)
            print("TestPlotNormalField: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")
            
            
class TestPlotVelocityField(BaseCaseTest):
    def test_plot_velocity_field(self):
        try:
            plot_velocity_field(self.Q[:10,:,:], self.V[:10,:,:], step=2)
            print("TestPlotVelocityField: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")
            
            
class TestPlotSpeedTimeSeries(BaseCaseTest):
    def test_plot_speed_time_series(self):
        try:
            plot_speed_timeseries(self.t, self.s, nodes=range(20, 40, 2))
            print("TestPlotSpeedTimeSeries: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")


class TestPlotStatePointTimeSeries(BaseCaseTest):
    def test_plot_state_point_time_series(self):
        try:
            plot_statepoint_timeseries(self.t, self.Q, nodes=range(20, 30, 2))
            print("TestPlotStatePointTimeSeries: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")


class TestPlotTidalPhaseSpace(BaseCaseTest):
    def test_plot_tidal_phase_space(self):
        try:
            print(f"Time Features Columns: {self.timefeatures.columns}")
            print(f"t_mod_12_4 sample: {self.timefeatures['t_mod_12_4'].head()}")
            # plot_arc_length_tidal_phase_space(self.timefeatures['t_mod_12_4'].values, self.s)
            # plot_arc_length_tidal_phase_space_polar(self.timefeatures['t_mod_12_4'].values[:], self.s[:,::5])
            # plot_arc_length_tidal_phase_space_polar(self.timefeatures['t_mod_12_4'].values[:], (self.speed[:,::5]))
            # plot_arc_length_tidal_phase_space_polar(self.timefeatures['t_mod_12_4'].values[:], self.s[:,::5], tidal_period=24.8)
            plot_arc_length_tidal_phase_space_helix(self.timefeatures['t_mod_12_4'].values[:], self.s[:,::5], linear_time=self.timefeatures['t_hours'].values[:], world_lines=True)
            plot_arc_length_tidal_phase_space_helix(self.timefeatures['t_mod_12_4'].values[:], self.s[:,::5], linear_time=self.timefeatures['t_hours'].values[:])
            plot_arc_length_tidal_phase_space_helix(self.timefeatures['t_mod_12_4'].values[:], self.s[:,::5])
            print("TestPlotTidalPhaseSpace: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")


class TestPlotWorldlineTangents(BaseCaseTest):
    def test_plot_worldline_tangents(self):
        try:
            plot_worldline_tangents(self.Q[:60,:,:], self.dQdS[:60,:,:], step=5)
            print("TestPlotWorldlineTangents: Plot generated successfully.")
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")


class TestAddSubplots(BaseCaseTest):
    def test_add_subplots(self):
        """Build plot with flexible subplot arrangement."""
        try:
            # Create figure with 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Add plots to specific positions
            add_arc_lengths(fig, axes, self.tau[:60, :], self.s[:60, :], (0, 0))
            add_tangent_field(fig, axes, self.Q[:60, :, :], self.T[:60, :, :], (0, 1))
            add_arc_lengths(fig, axes, self.tau[:60, :], self.s[:60, :], (1, 0))  # Another instance
            add_tangent_field(fig, axes, self.Q[:60, :, :], self.T[:60, :, :], (1, 1))
            add_tangent_field(fig, axes, self.Q[:60, :, :], self.N[:60, :, :], (1, 1))
            # Invert y-axis once after all plotting is done
            axes[1, 1].invert_yaxis()

            plt.tight_layout()
            plt.show()
            print("TestAddSubplots: Subplot figure generated successfully.")
        except Exception as e:
            self.fail(f"Subplot generation failed with exception: {e}")


if __name__ == '__main__':
    unittest.main()