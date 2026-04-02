import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime
from webcoos.webcoos import *

class BaseCaseTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		# Common setup for all tests
		cls.dir = os.getcwd()
		cls.image_dir = CONST.image_dir
		cls.station = CONST.station_list[0] # oakisland_west
		cls.start="2025-02-01 08:00:00" 
		cls.end="2025-02-01 12:30:00"

class TestCONST(BaseCaseTest):
	def test_CONST(self):
		print(f"Station List:\n{CONST.station_list}")
		print(f"Headers:\n{CONST.headers}")
		print(f"Endpoint URL:\n{CONST.endpoint_url}")
		print(f"Images Directory:\n{CONST.image_dir}")
		
		
class TestHandleTime(BaseCaseTest):
	def test_handle_time(self):
		start_dt = handle_time(self.start)
		end_dt = handle_time(self.end)
		print(f"Start: {start_dt} | End: {end_dt}")
	

class TestSetDateRange(BaseCaseTest):
	def test_set_date_range(self):
		dt_range = set_date_range(self.start, self.end)
		print(f"Datetime Range: {dt_range}")
	
	
class TestGetWebCOOSProducts(BaseCaseTest):
	def test_get_webcoos_products(self):
		products = get_webcoos_products()
		print(f"WebCOOS Products:\n{products}")
	
class TestCheckStationAvailability(BaseCaseTest):
	def test_check_station_availability(self):
		results = check_station_availability('Oak')
		print(results.head())
	
	
class TestBuildInventory(BaseCaseTest):
	def test_build_inventory_stills(self):
		dt_range = set_date_range(self.start, self.end)
		dt_inv = build_inventory(self.station, dt_range[0], dt_range[1])
		print(f"Datetime Inventory:\n{dt_inv.keys()}")

	def test_build_inventory_video_archive(self):
		dt_range = set_date_range(self.start, self.end)
		dt_inv = build_inventory(self.station, dt_range[0], dt_range[1], product = "video-archive")
		for dt, url in dt_inv.items():
			print(dt, url)
	

class TestFilterInventory(BaseCaseTest):
    def test_filter_inventory(self):
        pass
    
    
class TestSampleInventory(BaseCaseTest):
    def test_sample_inventory(self):
        start, end = "2025-07-10 06:00:00", "2025-07-10 14:00:00"
        dt_range = set_date_range(start, end)
        dt_inv = build_inventory("jennette_south", dt_range[0], dt_range[1], product="video-archive")
        sample_inv = sample_inventory(dt_inv)
        for date, url in sample_inv.items():
            print(date)