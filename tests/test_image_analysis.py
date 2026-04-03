import os
import unittest
from pathlib import Path

from src.image_analysis import *

class BaseCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_dir = Path(os.getcwd())
        cls.station = "jennette_south"
        
        
class TestMakeDirPaths(BaseCaseTest):
    def test_make_dir_paths_images(self):
        outpth = make_dir_paths(self.parent_dir, self.station, "images")
        images_path = self.parent_dir / "images" / rf"{self.station}" / "time_average"
        assert images_path == outpth
        
    def test_make_dir_paths_data(self):
        outpth = make_dir_paths(self.parent_dir, self.station, "stats")
        data_path = self.parent_dir / "data" / rf"{self.station}"
        assert data_path == outpth
        
    def test_make_dir_paths_roi(self):
        outpth = make_dir_paths(self.parent_dir, self.station, "roi")
        roi_path = self.parent_dir / "src" / "configs" / rf"{self.station}_roi.config.json"
        assert roi_path == outpth
        
        
class TestConvertFilenameToDate(BaseCaseTest):
    def test_convert_filename_to_date(self):
        fname = "jennette_south-2025-07-10_0952.png"
        fdate = "2025-07-10 09:52"
        outdate = convert_filename_to_date(fname, self.station)
        assert fdate == outdate
        
        
class TestMakeImagePaths(BaseCaseTest):
    def test_make_image_paths(self):
        img_paths = make_image_paths(self.parent_dir, self.station, target_n=10)
        print(f"Image Datetime Sample: {list(img_paths.keys())[0]}")
        print(f"Image Path Sample:\n{list(img_paths.items())[0]}")
        assert type(img_paths) == dict

        
class TestGetImages(BaseCaseTest):
    def test_get_images(self):
        imgs = get_images(self.parent_dir, self.station, target_n=10)
        assert type(imgs) == dict
        dt = "2025-07-10 09:52"
        img = imgs[dt]
        cv2.imshow(rf"Sample Image {self.station} at {dt}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
class TestGetRegionOfInterests(BaseCaseTest):
    def test_get_region_of_interests(self):
        get_region_of_interests(self.parent_dir, self.station)
        
        
class TestGetImageRegion(BaseCaseTest):
    def test_get_image_regions(self):
        masked_imgs = get_image_region(self.parent_dir, self.station, target_n=10)
        assert type(masked_imgs) == dict
        dt = "2025-07-10 09:52"
        img = masked_imgs[dt]
        print(f"Image Type: {type(img)}")
        # cv2.imshow(rf"Sample Image Region {self.station} at {dt}", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
class TestMakeImageSegments(BaseCaseTest):
    def test_make_image_segments(self):
        masked_imgs = get_image_region(self.parent_dir, self.station, target_n=10)
        dt = "2025-07-10 09:52"
        img = masked_imgs[dt]
        size = 100
        segments = make_image_segments(img, size)
        j = 0
        position = list(segments.keys())[0]
        print(f"Test Position: {position}")
        img_seg = segments[position]
        print(f"Test Segment Shape: {img_seg["data"].shape}")
        assert size == img_seg["data"].shape[0]
        assert size == img_seg["data"].shape[1]
        
        
class TestSegmentImages(BaseCaseTest):
    def test_segment_images(self):
        masked_imgs = get_image_region(self.parent_dir, self.station, target_n=10)
        dt = list(masked_imgs.keys())[0]
        img = masked_imgs[dt]
        img_size = img.shape[:2]
        h, w = img_size[0], img_size[1]
        print(f"Image Size: {img_size}")
        size = 200
        import math
        h_segs = math.ceil(h / size)
        w_segs = math.ceil(w / size)
        print(f"Vertical Segments: {h_segs} | Horizontal Segments: {w_segs} | Total Segments: {h_segs*w_segs}")
        segmented_images = segment_images(masked_imgs, size, True)
        seg_img = segmented_images[dt]
        
        print(f"Number of Image Segments: {len(seg_img)}")
        print(f"Segment Position Keys:\n{seg_img.keys()}")
        for pos, vals in seg_img.items():
            i, j = pos[0], pos[1]
            if (1000 <= i <= 1600) and (1000 <= j <= 1600):
                print(f"Statistics for Position {pos}:")
                print(f"Standard Deviation: {vals["std_dev"]}")
                print(f"Sharpness: {vals["sharpness"]}")
                print(f"Brightness: {vals["brightness"]}")
                print(f"Entropy: {vals["entropy"]}")
                
                
class TestMakeSegmentsTable(BaseCaseTest):
    def test_make_segments_table(self):
        size = 100
        masked_imgs = get_image_region(self.parent_dir, self.station, target_n=100)
        segmented_images = segment_images(masked_imgs, size, True)
        df = make_segments_table(segmented_images)
        print(df)
        segments_heatmap(df, "sharpness")