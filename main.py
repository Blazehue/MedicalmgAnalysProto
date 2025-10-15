#!/usr/bin/env python3
"""
Medical Image Analysis Tool
Main application entry point
"""

import tkinter as tk
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main application entry point"""
    try:
        # Import GUI after path setup
        from src.gui_interface import MedicalImageAnalyzerGUI
        
        root = tk.Tk()
        app = MedicalImageAnalyzerGUI(root)
        
        # Configure window
        try:
            if os.name == 'nt':  # Windows
                root.state('zoomed')
        except:
            pass  # Ignore if zoomed not supported
        
        root.minsize(800, 600)
        
        # Start the application
        root.mainloop()
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please install required dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

def run_batch_processing():
    """Run batch processing from command line"""
    try:
        from src.image_processor import MedicalImageProcessor
        from src.abnormality_detector import AbnormalityDetector
        from src.statistical_reporter import StatisticalReporter
        from src.utils import batch_process_images
        
        if len(sys.argv) < 3:
            print("Usage: python main.py --batch <input_directory> <output_directory>")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        
        if not os.path.exists(input_dir):
            print(f"Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        print("Initializing components...")
        processor = MedicalImageProcessor()
        detector = AbnormalityDetector()
        reporter = StatisticalReporter(output_dir)
        
        print(f"Processing images from {input_dir}...")
        results = batch_process_images(input_dir, output_dir, processor, detector, reporter)
        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        print(f"\nBatch processing complete!")
        print(f"Successfully processed: {successful} images")
        print(f"Failed: {failed} images")
        
        # Generate batch summary
        batch_summary = reporter.generate_batch_summary(results)
        if batch_summary:
            print(f"Total abnormalities found: {batch_summary['total_abnormalities_found']}")
            print(f"Images with abnormalities: {batch_summary['images_with_abnormalities']}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        run_batch_processing()
    else:
        main()