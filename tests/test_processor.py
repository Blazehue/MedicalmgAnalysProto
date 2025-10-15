import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_processor import MedicalImageProcessor

class TestMedicalImageProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = MedicalImageProcessor()
        # Create test images
        self.test_xray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        self.test_mri = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    def test_xray_preprocessing(self):
        """Test X-ray preprocessing pipeline"""
        processed = self.processor.preprocess_xray(self.test_xray)
        
        self.assertEqual(processed.shape, self.test_xray.shape)
        self.assertEqual(processed.dtype, np.uint8)
        self.assertGreaterEqual(np.min(processed), 0)
        self.assertLessEqual(np.max(processed), 255)
    
    def test_mri_preprocessing(self):
        """Test MRI preprocessing pipeline"""
        processed = self.processor.preprocess_mri(self.test_mri)
        
        self.assertEqual(processed.shape, self.test_mri.shape)
        self.assertEqual(processed.dtype, np.uint8)
        self.assertGreaterEqual(np.min(processed), 0)
        self.assertLessEqual(np.max(processed), 255)
    
    def test_gamma_correction(self):
        """Test gamma correction function"""
        result = self.processor._gamma_correction(self.test_xray, gamma=1.5)
        
        self.assertEqual(result.shape, self.test_xray.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_unsharp_mask(self):
        """Test unsharp masking function"""
        result = self.processor._unsharp_mask(self.test_xray)
        
        self.assertEqual(result.shape, self.test_xray.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_intensity_normalization(self):
        """Test intensity normalization"""
        result = self.processor._intensity_normalization(self.test_mri)
        
        self.assertEqual(result.shape, self.test_mri.shape)
        # Check if normalization worked
        self.assertGreaterEqual(np.min(result), 0)
        self.assertLessEqual(np.max(result), 255)

if __name__ == '__main__':
    unittest.main()