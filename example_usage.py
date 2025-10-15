#!/usr/bin/env python3
"""
Example usage of the Medical Image Analysis Tool
This script demonstrates how to use the tool programmatically
"""

import numpy as np
import cv2
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.image_processor import MedicalImageProcessor
from src.abnormality_detector import AbnormalityDetector
from src.statistical_reporter import StatisticalReporter

def create_synthetic_xray():
    """Create a synthetic X-ray image for demonstration"""
    # Create base image
    image = np.random.randint(20, 80, (512, 512), dtype=np.uint8)
    
    # Add lung regions (brighter areas)
    cv2.ellipse(image, (180, 200), (100, 120), 0, 0, 360, (120,), -1)
    cv2.ellipse(image, (330, 200), (100, 120), 0, 0, 360, (120,), -1)
    
    # Add some nodule-like features
    cv2.circle(image, (200, 180), 15, (200,), -1)
    cv2.circle(image, (310, 220), 12, (190,), -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def create_synthetic_mri():
    """Create a synthetic MRI image for demonstration"""
    # Create base brain-like structure
    image = np.zeros((256, 256), dtype=np.uint8)
    
    # Brain outline
    cv2.ellipse(image, (128, 128), (100, 110), 0, 0, 360, (100,), -1)
    
    # Add brain structures
    cv2.ellipse(image, (128, 128), (80, 90), 0, 0, 360, (150,), -1)
    cv2.ellipse(image, (128, 128), (60, 70), 0, 0, 360, (120,), -1)
    
    # Add some asymmetric features
    cv2.circle(image, (100, 100), 20, (200,), -1)  # Bright spot (potential anomaly)
    
    # Add noise
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def demonstrate_xray_analysis():
    """Demonstrate X-ray analysis workflow"""
    print("=== X-RAY ANALYSIS DEMONSTRATION ===")
    
    # Create synthetic X-ray
    xray_image = create_synthetic_xray()
    
    # Initialize components
    processor = MedicalImageProcessor()
    detector = AbnormalityDetector()
    reporter = StatisticalReporter("data/output")
    
    print("1. Processing X-ray image...")
    processed_image = processor.preprocess_xray(xray_image)
    
    print("2. Detecting lung nodules...")
    nodules = detector.detect_lung_nodules(processed_image)
    print(f"   Found {len(nodules)} potential nodules")
    
    print("3. Detecting bone fractures...")
    fractures = detector.detect_bone_fractures(processed_image)
    print(f"   Found {len(fractures)} potential fractures")
    
    # Combine abnormalities
    all_abnormalities = nodules + fractures
    
    print("4. Generating comprehensive report...")
    fake_path = "synthetic_xray.png"
    cv2.imwrite(os.path.join("data/output", fake_path), xray_image)
    
    report_result = reporter.generate_comprehensive_report(
        fake_path,
        xray_image,
        processed_image,
        all_abnormalities,
        "x-ray"
    )
    
    print(f"   Report saved to: {report_result['pdf_path']}")
    print(f"   CSV data saved to: {report_result['csv_path']}")
    
    return all_abnormalities

def demonstrate_mri_analysis():
    """Demonstrate MRI analysis workflow"""
    print("\n=== MRI ANALYSIS DEMONSTRATION ===")
    
    # Create synthetic MRI
    mri_image = create_synthetic_mri()
    
    # Initialize components
    processor = MedicalImageProcessor()
    detector = AbnormalityDetector()
    reporter = StatisticalReporter("data/output")
    
    print("1. Processing MRI image...")
    processed_image = processor.preprocess_mri(mri_image)
    
    print("2. Detecting brain anomalies...")
    anomalies = detector.detect_brain_anomalies(processed_image)
    print(f"   Found {len(anomalies)} potential anomalies")
    
    print("3. Generating comprehensive report...")
    fake_path = "synthetic_mri.png"
    cv2.imwrite(os.path.join("data/output", fake_path), mri_image)
    
    report_result = reporter.generate_comprehensive_report(
        fake_path,
        mri_image,
        processed_image,
        anomalies,
        "mri"
    )
    
    print(f"   Report saved to: {report_result['pdf_path']}")
    print(f"   CSV data saved to: {report_result['csv_path']}")
    
    return anomalies

def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n=== BATCH PROCESSING DEMONSTRATION ===")
    
    # Create multiple synthetic images
    images = [
        ("xray_1.png", create_synthetic_xray()),
        ("xray_2.png", create_synthetic_xray()),
        ("mri_1.png", create_synthetic_mri())
    ]
    
    # Save images to input directory
    input_dir = "data/sample_images"
    os.makedirs(input_dir, exist_ok=True)
    
    for filename, image in images:
        cv2.imwrite(os.path.join(input_dir, filename), image)
    
    print(f"Created {len(images)} sample images in {input_dir}")
    
    # Initialize components
    processor = MedicalImageProcessor()
    detector = AbnormalityDetector()
    reporter = StatisticalReporter("data/output")
    
    print("Processing images in batch...")
    
    from src.utils import batch_process_images
    results = batch_process_images(input_dir, "data/output", processor, detector, reporter)
    
    # Print results
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"Batch processing complete!")
    print(f"Successfully processed: {successful} images")
    print(f"Failed: {failed} images")
    
    # Generate batch summary
    batch_summary = reporter.generate_batch_summary(results)
    if batch_summary:
        print(f"Total abnormalities found: {batch_summary['total_abnormalities_found']}")
        print(f"Images with abnormalities: {batch_summary['images_with_abnormalities']}")

def main():
    """Main demonstration function"""
    print("Medical Image Analysis Tool - Example Usage")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs("data/output", exist_ok=True)
    
    try:
        # Run demonstrations
        xray_abnormalities = demonstrate_xray_analysis()
        mri_abnormalities = demonstrate_mri_analysis()
        demonstrate_batch_processing()
        
        print("\n=== SUMMARY ===")
        print(f"X-ray abnormalities found: {len(xray_abnormalities)}")
        print(f"MRI abnormalities found: {len(mri_abnormalities)}")
        print("Check the 'data/output' directory for generated reports and visualizations.")
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()