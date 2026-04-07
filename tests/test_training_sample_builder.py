import os
import unittest
from pathlib import Path

from src.training_sample_builder import *

class BaseCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_dir = Path(os.getcwd())
        cls.station = "jennette_south"
        
        
class TestLoadConfig(BaseCaseTest):
    def test_load_config(self):
        config = load_config(self.parent_dir, self.station)
        print(f"{self.station} config keys:\n{config.keys()}")
        assert type(config) == dict
        
        
class TestLoadAnnotations(BaseCaseTest):
    def test_load_annotations(self):
        ants = load_annotations(self.parent_dir, self.station)
        print(f"{self.station} annotation keys:\n{ants.keys()}")
        assert type(ants) == dict
        

class TestResampleShoreline(BaseCaseTest):
    def test_resample_shoreline(self):
        ants = load_annotations(self.parent_dir, self.station)
        resampled_ants = resample_shoreline(ants, 100)
        dt = list(ants.keys())[0]
        sample_pts = ants[dt].get("points")
        print(f"sample points:\n{sample_pts[:5]}")
        resampled_pts = resampled_ants[dt].get("points")
        print(f"resampled points:\n{resampled_pts[:5]}")
        img_file = ants[dt].get("image_file")
        img_pth = self.parent_dir / "images" / rf"{self.station}" / "time_average" / "n=100-frames" / img_file
        sample_image = cv2.imread(img_pth, 1)
        # Draw connected path for original points (red) and resampled points (blue)
        for i in range(len(sample_pts) - 1):
            cv2.line(sample_image, (int(sample_pts[i][0]), int(sample_pts[i][1])), 
                    (int(sample_pts[i+1][0]), int(sample_pts[i+1][1])), (0, 0, 255), 2)
        for i in range(len(resampled_pts) - 1):
            cv2.line(sample_image, (int(resampled_pts[i][0]), int(resampled_pts[i][1])), 
                    (int(resampled_pts[i+1][0]), int(resampled_pts[i+1][1])), (255, 0, 0), 2)
        cv2.namedWindow("Sample Overlay", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sample Overlay", 1920, 1080)
        cv2.imshow("Sample Overlay", sample_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
class TestMakeShorelineCropSamples(BaseCaseTest):
    def test_make_shoreline_crop_samples(self):
        ants = load_annotations(self.parent_dir, self.station)
        resampled_ants = resample_shoreline(ants, 300)
        cropped_samples = make_shoreline_crop_samples(self.parent_dir, self.station, resampled_ants, width=200, length=400, n_samples=3, _random = True)
        dt = list(cropped_samples.keys())[0]
        samples = cropped_samples[dt].get("samples")
        sample = samples["sample_0"]
        for k, v in sample.items():
            print(f"{k}:\n{v}")
            
class TestGetShorelineCropSamples(BaseCaseTest):
    def test_get_shoreline_crop_samples(self):
        cropped = get_shoreline_crop_samples(self.parent_dir, self.station)
        dt = list(cropped.keys())[0]
        samples = cropped[dt].get("samples")
        print(f"First Sample:\n{samples.get("sample_0")}")
        
        
class TestApplyShorelineCrop(BaseCaseTest):
    def test_apply_shoreline_crop(self):
        cropped = get_shoreline_crop_samples(self.parent_dir, self.station)
        dt = list(cropped.keys())[0]
        samples = cropped[dt].get("samples")
        image_file = cropped[dt]["image_file"]
        region = samples["sample_0"]["region"]
        enclosed_pts = samples["sample_0"]["enclosed_points"]
        cropped_img = apply_shoreline_crop(self.parent_dir, self.station, image_file, region)
        plot_shoreline_crop_sample(cropped_img, enclosed_pts, region, True)