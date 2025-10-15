# Medical Image Analysis Tool - Quick Start Guide

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   python tests/run_tests.py
   ```
   All tests should pass.

## Usage

### GUI Application

**Start the GUI:**
```bash
python main.py
```

**Features:**
- Load DICOM or standard image formats (PNG, JPG, TIFF, BMP)
- Choose image type (X-ray or MRI)
- Process images with medical-specific preprocessing
- Detect abnormalities automatically
- Generate comprehensive PDF reports with visualizations

**Workflow:**
1. Click "Open" to load an image
2. Select image type (X-ray/MRI) from dropdown
3. Click "Process" to apply preprocessing
4. Click "Analyze" to detect abnormalities
5. Click "Report" to generate PDF report

### Command Line Batch Processing

**Batch process multiple images:**
```bash
python main.py --batch input_directory output_directory
```

### Programmatic Usage

See `example_usage.py` for complete examples:

```python
from src.image_processor import MedicalImageProcessor
from src.abnormality_detector import AbnormalityDetector  
from src.statistical_reporter import StatisticalReporter

# Initialize components
processor = MedicalImageProcessor()
detector = AbnormalityDetector()
reporter = StatisticalReporter("output_dir")

# Load and process image
image = processor.load_image("path/to/image.dcm")
processed = processor.preprocess_xray(image)  # or preprocess_mri()

# Detect abnormalities
nodules = detector.detect_lung_nodules(processed)  # for X-rays
fractures = detector.detect_bone_fractures(processed)  # for X-rays
# or detector.detect_brain_anomalies(processed)  # for MRI

# Generate report
abnormalities = nodules + fractures
report = reporter.generate_comprehensive_report(
    "image_path", image, processed, abnormalities, "x-ray"
)
```

## Supported Formats

- **DICOM** (.dcm) - Medical imaging standard
- **PNG** (.png) - Portable Network Graphics
- **JPEG** (.jpg, .jpeg) - Joint Photographic Experts Group
- **TIFF** (.tiff, .tif) - Tagged Image File Format
- **BMP** (.bmp) - Bitmap

## Image Types

### X-ray Analysis
- **Preprocessing**: Noise reduction, CLAHE, gamma correction, edge enhancement
- **Detection**: Lung nodules, bone fractures
- **Features**: Circular Hough transform, edge analysis, line detection

### MRI Analysis  
- **Preprocessing**: Bias field correction, intensity normalization, bilateral filtering
- **Detection**: Brain asymmetries, intensity anomalies, texture anomalies
- **Features**: Symmetry analysis, K-means clustering, Local Binary Pattern

## Output

The tool generates:
- **PDF Reports**: Comprehensive analysis with statistics and visualizations
- **CSV Data**: Detailed numerical data for further analysis
- **Visualization Images**: 
  - Original vs processed comparison
  - Histogram analysis
  - Abnormality overlays
  - Statistical summary plots

## Testing

Run individual test modules:
```bash
python -m unittest tests.test_processor
python -m unittest tests.test_detector  
python -m unittest tests.test_reporter
```

## Demo

Run the example with synthetic data:
```bash
python example_usage.py
```

This creates sample X-ray and MRI images and demonstrates the full analysis pipeline.

## Troubleshooting

**Import Errors:** Make sure all dependencies are installed with `pip install -r requirements.txt`

**GUI Issues:** The GUI requires a display. For headless systems, use batch processing mode.

**Memory Issues:** Large medical images may require significant RAM. Consider resizing images or processing in smaller batches.

**DICOM Issues:** Ensure pydicom is properly installed. Some DICOM files may require specific handling.

## Disclaimer

This tool is for research and educational purposes only. Medical analysis results should not be used for diagnosis without professional medical review.