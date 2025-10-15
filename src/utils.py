import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any

def validate_image(image: np.ndarray) -> bool:
    """Validate if image is suitable for medical analysis"""
    if image is None:
        return False
    
    if len(image.shape) not in [2, 3]:
        return False
    
    if image.size == 0:
        return False
    
    # Check if image has reasonable dimensions
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return False
    
    return True

def normalize_image(image: np.ndarray, target_range: Tuple[int, int] = (0, 255)) -> np.ndarray:
    """Normalize image to target range"""
    if image.dtype != np.uint8:
        image = image.astype(np.float64)
        image = (image - image.min()) / (image.max() - image.min())
        image = image * (target_range[1] - target_range[0]) + target_range[0]
        image = image.astype(np.uint8)
    
    return image

def create_circular_mask(image_shape: Tuple[int, int], center: Tuple[int, int], radius: int) -> np.ndarray:
    """Create circular mask for region of interest"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, (255,), -1)
    return mask

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union for bounding boxes"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Calculate intersection area
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def ensure_directory_exists(directory_path: str) -> None:
    """Ensure directory exists, create if not"""
    os.makedirs(directory_path, exist_ok=True)

def get_supported_formats() -> List[str]:
    """Get list of supported image formats"""
    return ['.dcm', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']

def batch_process_images(image_directory: str, output_directory: str, 
                        processor, detector, reporter) -> List[Dict[str, Any]]:
    """Process multiple images in batch"""
    results = []
    supported_formats = get_supported_formats()
    
    ensure_directory_exists(output_directory)
    
    image_files = [f for f in os.listdir(image_directory) 
                  if os.path.splitext(f.lower())[1] in supported_formats]
    
    for filename in image_files:
        filepath = os.path.join(image_directory, filename)
        
        try:
            # Load and process image
            original_image = processor.load_image(filepath)
            
            if not validate_image(original_image):
                results.append({
                    'filename': filename,
                    'status': 'error',
                    'error_message': 'Invalid image format or size'
                })
                continue
            
            # Determine image type based on filename or user input
            image_type = "x-ray" if "xray" in filename.lower() or "chest" in filename.lower() else "mri"
            
            if image_type == "x-ray":
                processed_image = processor.preprocess_xray(original_image)
                nodules = detector.detect_lung_nodules(processed_image)
                fractures = detector.detect_bone_fractures(processed_image)
                abnormalities = nodules + fractures
            else:
                processed_image = processor.preprocess_mri(original_image)
                abnormalities = detector.detect_brain_anomalies(processed_image)
            
            # Generate report
            report_result = reporter.generate_comprehensive_report(
                filepath, original_image, processed_image, abnormalities, image_type
            )
            
            results.append({
                'filename': filename,
                'status': 'success',
                'abnormalities_count': len(abnormalities),
                'report_path': report_result['pdf_path'],
                'report_data': report_result['report_data']
            })
            
        except Exception as e:
            results.append({
                'filename': filename,
                'status': 'error',
                'error_message': str(e)
            })
    
    return results

def convert_dicom_to_standard(dicom_path: str, output_path: str, format: str = 'png') -> bool:
    """Convert DICOM file to standard image format"""
    try:
        import pydicom
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array.astype(np.float64)
        
        # Normalize to 0-255 range
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Save as standard format
        success = cv2.imwrite(output_path, image)
        return success
        
    except Exception as e:
        print(f"Error converting DICOM: {str(e)}")
        return False

def analyze_image_quality(image: np.ndarray) -> Dict[str, float]:
    """Analyze image quality metrics"""
    # Calculate various quality metrics
    mean_intensity = float(np.mean(image))
    std_intensity = float(np.std(image))
    
    # Signal-to-noise ratio
    snr = mean_intensity / (std_intensity + 1e-7)
    
    # Contrast measure
    contrast = std_intensity / (mean_intensity + 1e-7)
    
    # Edge density (using Sobel operator)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_density = np.mean(edge_magnitude) / 255.0
    
    # Sharpness (variance of Laplacian)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    return {
        'snr': float(snr),
        'contrast': float(contrast),
        'edge_density': float(edge_density),
        'sharpness': float(sharpness),
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity)
    }