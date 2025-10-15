import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import os
import numpy as np
from .image_processor import MedicalImageProcessor
from .abnormality_detector import AbnormalityDetector
from .statistical_reporter import StatisticalReporter

class MedicalImageAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Analysis Tool")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.processor = MedicalImageProcessor()
        self.detector = AbnormalityDetector()
        self.reporter = StatisticalReporter()
        
        # Variables
        self.current_image = None
        self.processed_image = None
        self.current_image_path = ""
        self.analysis_results = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Create main frames
        self.setup_menu()
        self.setup_toolbar()
        self.setup_main_content()
        self.setup_status_bar()
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Export Report", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Process Image", command=self.process_image)
        analysis_menu.add_command(label="Detect Abnormalities", command=self.detect_abnormalities)
        analysis_menu.add_command(label="Generate Report", command=self.generate_report)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_toolbar(self):
        """Setup toolbar"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        ttk.Button(toolbar, text="Open", command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Process", command=self.process_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Analyze", command=self.detect_abnormalities).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Report", command=self.generate_report).pack(side=tk.LEFT, padx=2)
        
        # Image type selection
        ttk.Label(toolbar, text="Image Type:").pack(side=tk.LEFT, padx=(20, 5))
        self.image_type_var = tk.StringVar(value="x-ray")
        type_combo = ttk.Combobox(toolbar, textvariable=self.image_type_var, 
                                 values=["x-ray", "mri"], width=10)
        type_combo.pack(side=tk.LEFT, padx=2)
    
    def setup_main_content(self):
        """Setup main content area"""
        # Create paned window
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Images
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=2)
        
        # Image display area
        self.setup_image_display(left_frame)
        
        # Right panel - Analysis results
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Analysis results area
        self.setup_results_panel(right_frame)
    
    def setup_image_display(self, parent):
        """Setup image display area"""
        # Image tabs
        image_notebook = ttk.Notebook(parent)
        image_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Original image tab
        original_frame = ttk.Frame(image_notebook)
        image_notebook.add(original_frame, text="Original")
        
        self.original_canvas = tk.Canvas(original_frame, bg='black')
        orig_scrollbar_v = ttk.Scrollbar(original_frame, orient=tk.VERTICAL, command=self.original_canvas.yview)
        orig_scrollbar_h = ttk.Scrollbar(original_frame, orient=tk.HORIZONTAL, command=self.original_canvas.xview)
        self.original_canvas.configure(yscrollcommand=orig_scrollbar_v.set, xscrollcommand=orig_scrollbar_h.set)
        
        orig_scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        orig_scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Processed image tab
        processed_frame = ttk.Frame(image_notebook)
        image_notebook.add(processed_frame, text="Processed")
        
        self.processed_canvas = tk.Canvas(processed_frame, bg='black')
        proc_scrollbar_v = ttk.Scrollbar(processed_frame, orient=tk.VERTICAL, command=self.processed_canvas.yview)
        proc_scrollbar_h = ttk.Scrollbar(processed_frame, orient=tk.HORIZONTAL, command=self.processed_canvas.xview)
        self.processed_canvas.configure(yscrollcommand=proc_scrollbar_v.set, xscrollcommand=proc_scrollbar_h.set)
        
        proc_scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        proc_scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.processed_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def setup_results_panel(self, parent):
        """Setup analysis results panel"""
        results_notebook = ttk.Notebook(parent)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Statistics tab
        stats_frame = ttk.Frame(results_notebook)
        results_notebook.add(stats_frame, text="Statistics")
        
        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Abnormalities tab
        abnorm_frame = ttk.Frame(results_notebook)
        results_notebook.add(abnorm_frame, text="Abnormalities")
        
        # Abnormalities tree view
        columns = ('Type', 'Severity', 'Area', 'Confidence')
        self.abnorm_tree = ttk.Treeview(abnorm_frame, columns=columns, show='tree headings')
        
        for col in columns:
            self.abnorm_tree.heading(col, text=col)
            self.abnorm_tree.column(col, width=80)
        
        abnorm_scrollbar = ttk.Scrollbar(abnorm_frame, orient=tk.VERTICAL, command=self.abnorm_tree.yview)
        self.abnorm_tree.configure(yscrollcommand=abnorm_scrollbar.set)
        
        abnorm_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.abnorm_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def open_image(self):
        """Open and display medical image"""
        file_types = [
            ("All Supported", "*.dcm;*.jpg;*.jpeg;*.png;*.tiff;*.bmp"),
            ("DICOM files", "*.dcm"),
            ("Image files", "*.jpg;*.jpeg;*.png;*.tiff;*.bmp"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=file_types
        )
        
        if filename:
            try:
                self.current_image_path = filename
                self.current_image = self.processor.load_image(filename)
                self.processed_image = None
                self.analysis_results = None
                
                self.display_image(self.current_image, self.original_canvas)
                self.update_status(f"Loaded: {os.path.basename(filename)}")
                
                # Clear previous results
                self.clear_results()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def process_image(self):
        """Process the loaded image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        def process_thread():
            try:
                self.update_status("Processing image...")
                
                image_type = self.image_type_var.get()
                if image_type == "x-ray":
                    self.processed_image = self.processor.preprocess_xray(self.current_image)
                else:
                    self.processed_image = self.processor.preprocess_mri(self.current_image)
                
                # Update GUI in main thread
                self.root.after(0, self.on_processing_complete)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
                self.root.after(0, lambda: self.update_status("Processing failed"))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def on_processing_complete(self):
        """Handle completion of image processing"""
        self.display_image(self.processed_image, self.processed_canvas)
        self.update_status("Image processed successfully")
        self.display_image_statistics()
    
    def detect_abnormalities(self):
        """Detect abnormalities in the processed image"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "Please process the image first")
            return
        
        def detect_thread():
            try:
                self.update_status("Detecting abnormalities...")
                
                image_type = self.image_type_var.get()
                if image_type == "x-ray":
                    # Detect both nodules and fractures for X-rays
                    nodules = self.detector.detect_lung_nodules(self.processed_image)
                    fractures = self.detector.detect_bone_fractures(self.processed_image)
                    abnormalities = nodules + fractures
                else:
                    # Detect brain anomalies for MRI
                    abnormalities = self.detector.detect_brain_anomalies(self.processed_image)
                
                self.analysis_results = abnormalities
                
                # Update GUI in main thread
                self.root.after(0, self.on_detection_complete)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {str(e)}"))
                self.root.after(0, lambda: self.update_status("Detection failed"))
        
        threading.Thread(target=detect_thread, daemon=True).start()
    
    def on_detection_complete(self):
        """Handle completion of abnormality detection"""
        self.display_abnormalities()
        count = len(self.analysis_results) if self.analysis_results else 0
        self.update_status(f"Detection complete. Found {count} potential abnormalities")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        if self.current_image is None or self.processed_image is None:
            messagebox.showwarning("Warning", "Please load and process an image first")
            return
        
        def report_thread():
            try:
                self.update_status("Generating report...")
                
                abnormalities = self.analysis_results if self.analysis_results else []
                image_type = self.image_type_var.get()
                
                report_result = self.reporter.generate_comprehensive_report(
                    self.current_image_path,
                    self.current_image,
                    self.processed_image,
                    abnormalities,
                    image_type
                )
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.on_report_complete(report_result))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Report generation failed: {str(e)}"))
                self.root.after(0, lambda: self.update_status("Report generation failed"))
        
        threading.Thread(target=report_thread, daemon=True).start()
    
    def on_report_complete(self, report_result):
        """Handle completion of report generation"""
        pdf_path = report_result.get('pdf_path')
        self.update_status(f"Report generated: {os.path.basename(pdf_path)}")
        
        response = messagebox.askyesno("Report Generated", 
                                     f"Report saved to {pdf_path}\\nWould you like to open it?")
        if response:
            self.open_file(pdf_path)
    
    def display_image(self, image, canvas):
        """Display image on canvas"""
        if image is None:
            return
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Resize if too large
        canvas.update()  # Ensure canvas dimensions are updated
        canvas_width = max(canvas.winfo_width(), 400)
        canvas_height = max(canvas.winfo_height(), 300)
        
        img_width, img_height = pil_image.size
        
        # Calculate scaling factor
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, 
                           anchor=tk.CENTER, image=photo)
        
        # Keep a reference to prevent garbage collection
        canvas.image = photo
        
        # Update scroll region
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    def display_image_statistics(self):
        """Display image statistics in the stats panel"""
        if self.current_image is None:
            return
        
        stats_text = "IMAGE STATISTICS\\n" + "="*50 + "\\n\\n"
        
        # Original image statistics
        orig_stats = self.calculate_display_stats(self.current_image)
        stats_text += "ORIGINAL IMAGE:\\n"
        stats_text += f"Dimensions: {orig_stats['dimensions']}\\n"
        stats_text += f"Mean Intensity: {orig_stats['mean']:.2f}\\n"
        stats_text += f"Std Deviation: {orig_stats['std']:.2f}\\n"
        stats_text += f"Min Intensity: {orig_stats['min']}\\n"
        stats_text += f"Max Intensity: {orig_stats['max']}\\n"
        stats_text += f"Median: {orig_stats['median']:.2f}\\n\\n"
        
        # Processed image statistics (if available)
        if self.processed_image is not None:
            proc_stats = self.calculate_display_stats(self.processed_image)
            stats_text += "PROCESSED IMAGE:\\n"
            stats_text += f"Dimensions: {proc_stats['dimensions']}\\n"
            stats_text += f"Mean Intensity: {proc_stats['mean']:.2f}\\n"
            stats_text += f"Std Deviation: {proc_stats['std']:.2f}\\n"
            stats_text += f"Min Intensity: {proc_stats['min']}\\n"
            stats_text += f"Max Intensity: {proc_stats['max']}\\n"
            stats_text += f"Median: {proc_stats['median']:.2f}\\n\\n"
            
            # Enhancement metrics
            stats_text += "ENHANCEMENT METRICS:\\n"
            contrast_improvement = proc_stats['std'] / orig_stats['std'] if orig_stats['std'] > 0 else 1
            stats_text += f"Contrast Enhancement: {contrast_improvement:.2f}x\\n"
            stats_text += f"Dynamic Range: {proc_stats['max'] - proc_stats['min']}\\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def calculate_display_stats(self, image):
        """Calculate statistics for display"""
        return {
            'dimensions': f"{image.shape[1]} x {image.shape[0]}",
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'min': int(np.min(image)),
            'max': int(np.max(image)),
            'median': float(np.median(image))
        }
    
    def display_abnormalities(self):
        """Display detected abnormalities in the tree view"""
        # Clear existing items
        for item in self.abnorm_tree.get_children():
            self.abnorm_tree.delete(item)
        
        if not self.analysis_results:
            return
        
        for i, abnormality in enumerate(self.analysis_results):
            abnormality_type = abnormality.get('type', 'Unknown')
            severity = abnormality.get('severity', 'Unknown')
            area = abnormality.get('area', 0)
            confidence = abnormality.get('confidence', 0)
            
            self.abnorm_tree.insert('', 'end', 
                                  text=f'Abnormality {i+1}',
                                  values=(abnormality_type, severity, f'{area:.0f}', f'{confidence:.2f}'))
    
    def clear_results(self):
        """Clear all analysis results"""
        self.stats_text.delete(1.0, tk.END)
        for item in self.abnorm_tree.get_children():
            self.abnorm_tree.delete(item)
        
        # Clear processed image canvas
        self.processed_canvas.delete("all")
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
    
    def export_report(self):
        """Export analysis report"""
        if self.analysis_results is None:
            messagebox.showwarning("Warning", "No analysis results to export")
            return
        
        self.generate_report()
    
    def open_file(self, filepath):
        """Open file with default system application"""
        import subprocess
        import sys
        
        try:
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.call(('open', filepath))
            elif os.name == 'nt':  # Windows
                os.startfile(filepath)
            elif os.name == 'posix':  # Linux
                subprocess.call(('xdg-open', filepath))
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Medical Image Analysis Tool v1.0
        
A comprehensive tool for medical image preprocessing and abnormality detection.

Features:
• X-ray and MRI image preprocessing
• Automated abnormality detection
• Statistical analysis and reporting
• DICOM and standard format support

This tool is for research and educational purposes only.
Results should not be used for medical diagnosis without professional review.

© 2024 Medical Image Analysis Team"""
        
        messagebox.showinfo("About", about_text)