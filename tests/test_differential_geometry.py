import unittest
import pandas as pd
import numpy as np

from src.differential_geometry import *
from src.data_manager import *

class BaseCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Common setup for all tests
        cls.sldata_path = r"data\jennette_south_roi_model_118cf4cf71a10dce.csv"
        cls.sldata = load_data(cls.sldata_path)
        

        
class TestDifferentialGeometry(BaseCaseTest):
    pass


class TestComputeDifferentials(BaseCaseTest):
    def test_compute_differentials(self):
        # Create a simple test array
        Q, _, _ = build_slpt_tensor(self.sldata)
        dt, dQ, delQ = compute_differentials(Q)
        print("TestComputeDifferentials: Differentials computed correctly.")
    
if __name__ == '__main__':
    unittest.main()