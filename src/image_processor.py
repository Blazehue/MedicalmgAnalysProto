import cv2
import numpy as np
import pydicom
from skimage import exposure, filters, morphology
from scipy import ndimage

class MedicalImageProcessor:
    def __init__(self):
        self.supported_formats = ['.dcm', '.jpg', '.png', '.tiff', '.bmp']
    
    def load_image(self, filepath):
        """Load medical image from various formats including DICOM"""
        if filepath.lower().endswith('.dcm'):
            return self._load_dicom(filepath)
        else:
            return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    def _load_dicom(self, filepath):
        """Load DICOM image"""
        try:
            dicom_data = pydicom.dcmread(filepath)
            image = dicom_data.pixel_array.astype(np.float64)
            # Normalize to 0-255 range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            return image
        except Exception as e:
            raise ValueError(f"Error loading DICOM file: {str(e)}")
    
    def preprocess_xray(self, image):
        """Comprehensive X-ray preprocessing pipeline"""
        # 1. Noise reduction using Gaussian blur
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        
        # 2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Gamma correction for better visibility
        gamma_corrected = self._gamma_correction(enhanced, gamma=1.2)
        
        # 4. Edge enhancement using unsharp masking
        unsharp_masked = self._unsharp_mask(gamma_corrected)
        
        # 5. Background subtraction
        background_subtracted = self._subtract_background(unsharp_masked)
        
        return background_subtracted
    
    def preprocess_mri(self, image):
        """MRI-specific preprocessing pipeline"""
        # 1. N4 bias field correction (simplified version)
        bias_corrected = self._simple_bias_correction(image)
        
        # 2. Intensity normalization
        normalized = self._intensity_normalization(bias_corrected)
        
        # 3. Noise reduction using bilateral filter
        denoised = cv2.bilateralFilter(normalized, 9, 75, 75)
        
        # 4. Contrast enhancement
        enhanced = exposure.equalize_adapthist(denoised, clip_limit=0.03)
        enhanced = (enhanced * 255).astype(np.uint8)
        
        return enhanced
    
    def _gamma_correction(self, image, gamma=1.0):
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    def _unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Apply unsharp masking for edge enhancement"""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened
    
    def _subtract_background(self, image):
        """Remove background using morphological opening"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        result = cv2.subtract(image, background)
        return result
    
    def _simple_bias_correction(self, image):
        """Simplified bias field correction for MRI"""
        # Create a low-pass filtered version as bias field estimate
        kernel_size = max(image.shape) // 8
        if kernel_size % 2 == 0:
            kernel_size += 1
        bias_field = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Correct the bias
        corrected = image.astype(np.float32) / (bias_field.astype(np.float32) + 1e-8)
        corrected = (corrected * 255 / corrected.max()).astype(np.uint8)
        return corrected
    
    def _intensity_normalization(self, image):
        """Normalize image intensities to standard range"""
        # Use histogram-based normalization
        p2, p98 = np.percentile(image, (2, 98))
        normalized = exposure.rescale_intensity(image, in_range=(p2, p98))
        return normalized