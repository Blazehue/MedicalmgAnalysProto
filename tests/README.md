# Medical Image Analysis Tool - Test Suite

This directory contains the test suite for the Medical Image Analysis Tool.

## Running Tests

### Run all tests:
```bash
python tests/run_tests.py
```

### Run individual test modules:
```bash
python -m unittest tests.test_processor
python -m unittest tests.test_detector
python -m unittest tests.test_reporter
```

### Run specific test cases:
```bash
python -m unittest tests.test_processor.TestMedicalImageProcessor.test_xray_preprocessing
```

## Test Coverage

- **test_processor.py**: Tests for image preprocessing functionality
- **test_detector.py**: Tests for abnormality detection algorithms
- **test_reporter.py**: Tests for statistical reporting and visualization

## Notes

Some tests may require additional dependencies to be installed. If a test fails due to missing dependencies, it will either skip gracefully or provide a clear error message.

## Test Data

Tests use synthetic image data generated with NumPy. For more comprehensive testing with real medical images, place sample images in the `data/sample_images/` directory.