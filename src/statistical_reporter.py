import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
import cv2
from fpdf import FPDF

class StatisticalReporter:
    def __init__(self, output_dir="data/output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        
    def generate_comprehensive_report(self, image_path, original_image, processed_image, 
                                    abnormalities, image_type="x-ray"):
        """Generate comprehensive analysis report"""
        report_data = {
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_path': image_path,
            'image_type': image_type,
            'image_stats': self._calculate_image_statistics(original_image),
            'preprocessing_stats': self._calculate_image_statistics(processed_image),
            'abnormalities': abnormalities,
            'abnormality_summary': self._summarize_abnormalities(abnormalities)
        }
        
        # Generate visualizations
        self._create_visualization_plots(original_image, processed_image, abnormalities, image_path)
        
        # Generate PDF report
        pdf_path = self._generate_pdf_report(report_data, image_path)
        
        # Save detailed data
        csv_path = self._save_detailed_csv(report_data, image_path)
        
        return {
            'report_data': report_data,
            'pdf_path': pdf_path,
            'csv_path': csv_path,
            'visualizations_saved': True
        }
    
    def _calculate_image_statistics(self, image):
        """Calculate comprehensive image statistics"""
        return {
            'dimensions': image.shape,
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image)),
            'min_intensity': int(np.min(image)),
            'max_intensity': int(np.max(image)),
            'median_intensity': float(np.median(image)),
            'contrast': float(np.std(image) / np.mean(image)) if np.mean(image) > 0 else 0,
            'entropy': self._calculate_entropy(image),
            'histogram_stats': self._calculate_histogram_stats(image)
        }
    
    def _calculate_entropy(self, image):
        """Calculate image entropy"""
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        return float(entropy)
    
    def _calculate_histogram_stats(self, image):
        """Calculate histogram-based statistics"""
        hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))
        
        # Find peaks in histogram
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(hist, height=np.max(hist) * 0.1)
        except ImportError:
            # Fallback if scipy not available
            peaks = [np.argmax(hist)]
        
        return {
            'num_peaks': len(peaks),
            'dominant_intensity': int(bins[np.argmax(hist)]),
            'histogram_spread': float(np.average(bins[:-1], weights=hist))
        }
    
    def _summarize_abnormalities(self, abnormalities):
        """Create summary statistics of detected abnormalities"""
        if not abnormalities:
            return {
                'total_count': 0,
                'severity_distribution': {},
                'type_distribution': {},
                'total_affected_area': 0
            }
        
        severity_counts = {}
        type_counts = {}
        total_area = 0
        
        for abnormality in abnormalities:
            # Count by severity
            severity = abnormality.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by type
            abnormality_type = abnormality.get('type', 'unknown')
            type_counts[abnormality_type] = type_counts.get(abnormality_type, 0) + 1
            
            # Sum areas
            area = abnormality.get('area', 0)
            total_area += area
        
        return {
            'total_count': len(abnormalities),
            'severity_distribution': severity_counts,
            'type_distribution': type_counts,
            'total_affected_area': total_area,
            'average_area_per_abnormality': total_area / len(abnormalities) if abnormalities else 0
        }
    
    def _create_visualization_plots(self, original_image, processed_image, abnormalities, image_path):
        """Create comprehensive visualization plots"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 1. Original vs Processed comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(processed_image, cmap='gray')
        axes[1].set_title('Processed Image', fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{base_name}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Histogram comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].hist(original_image.ravel(), bins=256, alpha=0.7, color='blue', label='Original')
        axes[0].set_title('Original Image Histogram')
        axes[0].set_xlabel('Intensity')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        axes[1].hist(processed_image.ravel(), bins=256, alpha=0.7, color='red', label='Processed')
        axes[1].set_title('Processed Image Histogram')
        axes[1].set_xlabel('Intensity')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{base_name}_histograms.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Abnormalities visualization
        if abnormalities:
            self._visualize_abnormalities(processed_image, abnormalities, base_name)
        
        # 4. Statistical summary plots
        self._create_summary_plots(original_image, processed_image, abnormalities, base_name)
    
    def _visualize_abnormalities(self, image, abnormalities, base_name):
        """Create visualization of detected abnormalities"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(image, cmap='gray')
        
        colors = {'nodule': 'red', 'fracture': 'yellow', 'asymmetry': 'blue', 
                 'intensity': 'green', 'texture': 'orange', 'unknown': 'purple'}
        
        for i, abnormality in enumerate(abnormalities):
            abnormality_type = abnormality.get('type', 'unknown')
            color = colors.get(abnormality_type, 'white')
            
            if 'center' in abnormality:
                # Circular abnormalities (nodules)
                center = abnormality['center']
                radius = abnormality.get('radius', 10)
                circle = plt.Circle(center, radius, fill=False, color=color, linewidth=2)
                ax.add_patch(circle)
                ax.text(center[0] + radius, center[1], f'{abnormality_type}_{i+1}', 
                       color=color, fontsize=10, fontweight='bold')
            
            elif 'line' in abnormality:
                # Linear abnormalities (fractures)
                line = abnormality['line']
                ax.plot([line[0], line[2]], [line[1], line[3]], color=color, linewidth=3)
                ax.text((line[0] + line[2])/2, (line[1] + line[3])/2, f'{abnormality_type}_{i+1}', 
                       color=color, fontsize=10, fontweight='bold')
            
            elif 'contour' in abnormality:
                # Contour-based abnormalities
                contour = abnormality['contour']
                # Convert contour to matplotlib format
                contour_points = contour.reshape(-1, 2)
                ax.plot(contour_points[:, 0], contour_points[:, 1], color=color, linewidth=2)
                
                # Add text label at centroid
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    ax.text(cx, cy, f'{abnormality_type}_{i+1}', 
                           color=color, fontsize=10, fontweight='bold')
            
            elif 'region' in abnormality:
                # Rectangular regions
                region = abnormality['region']
                if len(region) == 4:  # (x, y, width, height) or (x1, y1, x2, y2)
                    x, y, w, h = region
                    rect = plt.Rectangle((x, y), w-x, h-y, fill=False, color=color, linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x, y-5, f'{abnormality_type}_{i+1}', 
                           color=color, fontsize=10, fontweight='bold')
        
        ax.set_title('Detected Abnormalities', fontsize=16)
        ax.axis('off')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=abn_type.title()) 
                          for abn_type, color in colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{base_name}_abnormalities.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_plots(self, original_image, processed_image, abnormalities, base_name):
        """Create statistical summary plots"""
        # Calculate statistics
        orig_stats = self._calculate_image_statistics(original_image)
        proc_stats = self._calculate_image_statistics(processed_image)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Intensity statistics comparison
        stats_comparison = {
            'Statistic': ['Mean', 'Std', 'Median', 'Contrast', 'Entropy'],
            'Original': [orig_stats['mean_intensity'], orig_stats['std_intensity'],
                        orig_stats['median_intensity'], orig_stats['contrast'],
                        orig_stats['entropy']],
            'Processed': [proc_stats['mean_intensity'], proc_stats['std_intensity'],
                         proc_stats['median_intensity'], proc_stats['contrast'],
                         proc_stats['entropy']]
        }
        
        df_stats = pd.DataFrame(stats_comparison)
        x = np.arange(len(df_stats['Statistic']))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, df_stats['Original'], width, label='Original', alpha=0.7)
        axes[0, 0].bar(x + width/2, df_stats['Processed'], width, label='Processed', alpha=0.7)
        axes[0, 0].set_xlabel('Statistics')
        axes[0, 0].set_ylabel('Values')
        axes[0, 0].set_title('Image Statistics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(df_stats['Statistic'], rotation=45)
        axes[0, 0].legend()
        
        # 2. Abnormality distribution by type
        if abnormalities:
            abnormality_summary = self._summarize_abnormalities(abnormalities)
            type_dist = abnormality_summary['type_distribution']
            
            if type_dist:
                axes[0, 1].pie(type_dist.values(), labels=type_dist.keys(), autopct='%1.1f%%')
                axes[0, 1].set_title('Abnormality Distribution by Type')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Abnormalities Detected', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Abnormality Distribution')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Abnormalities Detected', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Abnormality Distribution')
        
        # 3. Severity distribution
        if abnormalities:
            severity_dist = abnormality_summary['severity_distribution']
            if severity_dist:
                axes[1, 0].bar(severity_dist.keys(), severity_dist.values(), alpha=0.7)
                axes[1, 0].set_xlabel('Severity Level')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('Abnormality Distribution by Severity')
                axes[1, 0].tick_params(axis='x', rotation=45)
            else:
                axes[1, 0].text(0.5, 0.5, 'No Severity Data Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Severity Distribution')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Abnormalities Detected', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Severity Distribution')
        
        # 4. Image quality metrics
        quality_metrics = {
            'Metric': ['Signal-to-Noise Ratio', 'Contrast Enhancement', 'Edge Enhancement'],
            'Score': [
                proc_stats['mean_intensity'] / (proc_stats['std_intensity'] + 1e-7),
                proc_stats['contrast'] / (orig_stats['contrast'] + 1e-7),
                1.2  # Placeholder for edge enhancement metric
            ]
        }
        
        axes[1, 1].bar(quality_metrics['Metric'], quality_metrics['Score'], alpha=0.7)
        axes[1, 1].set_xlabel('Quality Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Image Processing Quality Metrics')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{base_name}_summary_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_pdf_report(self, report_data, image_path):
        """Generate comprehensive PDF report"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        pdf_path = os.path.join(self.output_dir, f'{base_name}_analysis_report.pdf')
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        
        # Title
        pdf.cell(0, 10, 'Medical Image Analysis Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Basic Information
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Analysis Information', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f'Date: {report_data["analysis_date"]}', 0, 1)
        pdf.cell(0, 6, f'Image Type: {report_data["image_type"].upper()}', 0, 1)
        pdf.cell(0, 6, f'Image Path: {os.path.basename(image_path)}', 0, 1)
        pdf.ln(5)
        
        # Image Statistics
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Image Statistics', 0, 1)
        pdf.set_font('Arial', '', 9)
        
        orig_stats = report_data['image_stats']
        proc_stats = report_data['preprocessing_stats']
        
        # Create a simple table
        stats_data = [
            ['Metric', 'Original', 'Processed'],
            ['Dimensions', f"{orig_stats['dimensions']}", f"{proc_stats['dimensions']}"],
            ['Mean Intensity', f"{orig_stats['mean_intensity']:.2f}", f"{proc_stats['mean_intensity']:.2f}"],
            ['Std Deviation', f"{orig_stats['std_intensity']:.2f}", f"{proc_stats['std_intensity']:.2f}"],
            ['Contrast', f"{orig_stats['contrast']:.3f}", f"{proc_stats['contrast']:.3f}"],
            ['Entropy', f"{orig_stats['entropy']:.2f}", f"{proc_stats['entropy']:.2f}"]
        ]
        
        for row in stats_data:
            for i, cell in enumerate(row):
                if i == 0:
                    pdf.set_font('Arial', 'B', 9)
                else:
                    pdf.set_font('Arial', '', 9)
                pdf.cell(63, 6, str(cell), 1, 0, 'C')
            pdf.ln()
        
        pdf.ln(10)
        
        # Abnormalities Section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Detected Abnormalities', 0, 1)
        
        abnormalities = report_data['abnormalities']
        abnormality_summary = report_data['abnormality_summary']
        
        if abnormalities:
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f'Total Abnormalities Detected: {abnormality_summary["total_count"]}', 0, 1)
            pdf.cell(0, 6, f'Total Affected Area: {abnormality_summary["total_affected_area"]:.0f} pixels', 0, 1)
            pdf.ln(5)
            
            # Abnormality details
            for i, abnormality in enumerate(abnormalities[:10]):  # Limit to first 10
                pdf.set_font('Arial', 'B', 9)
                pdf.cell(0, 5, f'Abnormality {i+1}:', 0, 1)
                pdf.set_font('Arial', '', 9)
                pdf.cell(0, 5, f'  Type: {abnormality.get("type", "Unknown")}', 0, 1)
                pdf.cell(0, 5, f'  Severity: {abnormality.get("severity", "Unknown")}', 0, 1)
                if 'area' in abnormality:
                    pdf.cell(0, 5, f'  Area: {abnormality["area"]:.0f} pixels', 0, 1)
                if 'confidence' in abnormality:
                    pdf.cell(0, 5, f'  Confidence: {abnormality["confidence"]:.2f}', 0, 1)
                pdf.ln(2)
        else:
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, 'No abnormalities detected in this image.', 0, 1)
        
        # Add disclaimer
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 8)
        pdf.multi_cell(0, 5, 'Disclaimer: This analysis is for research and educational purposes only. '
                            'Results should not be used for medical diagnosis without professional review.')
        
        pdf.output(pdf_path)
        return pdf_path
    
    def _save_detailed_csv(self, report_data, image_path):
        """Save detailed analysis data to CSV"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        csv_path = os.path.join(self.output_dir, f'{base_name}_detailed_data.csv')
        
        # Prepare data for CSV
        rows = []
        
        # Image statistics row
        orig_stats = report_data['image_stats']
        proc_stats = report_data['preprocessing_stats']
        
        base_row = {
            'analysis_date': report_data['analysis_date'],
            'image_path': os.path.basename(image_path),
            'image_type': report_data['image_type'],
            'original_mean_intensity': orig_stats['mean_intensity'],
            'original_std_intensity': orig_stats['std_intensity'],
            'original_contrast': orig_stats['contrast'],
            'original_entropy': orig_stats['entropy'],
            'processed_mean_intensity': proc_stats['mean_intensity'],
            'processed_std_intensity': proc_stats['std_intensity'],
            'processed_contrast': proc_stats['contrast'],
            'processed_entropy': proc_stats['entropy'],
            'total_abnormalities': report_data['abnormality_summary']['total_count']
        }
        
        # Add abnormality details
        abnormalities = report_data['abnormalities']
        if abnormalities:
            for i, abnormality in enumerate(abnormalities):
                row = base_row.copy()
                row.update({
                    'abnormality_id': i + 1,
                    'abnormality_type': abnormality.get('type', ''),
                    'abnormality_severity': abnormality.get('severity', ''),
                    'abnormality_area': abnormality.get('area', 0),
                    'abnormality_confidence': abnormality.get('confidence', 0)
                })
                rows.append(row)
        else:
            rows.append(base_row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        return csv_path

    def generate_batch_summary(self, analysis_results):
        """Generate summary report for batch processing"""
        if not analysis_results:
            return None
            
        summary_data = {
            'total_images_processed': len(analysis_results),
            'total_abnormalities_found': 0,
            'images_with_abnormalities': 0,
            'abnormality_types': {},
            'severity_distribution': {},
            'processing_success_rate': 0
        }
        
        successful_analyses = 0
        
        for result in analysis_results:
            if result.get('report_data'):
                successful_analyses += 1
                abnormalities = result['report_data'].get('abnormalities', [])
                
                if abnormalities:
                    summary_data['images_with_abnormalities'] += 1
                    summary_data['total_abnormalities_found'] += len(abnormalities)
                    
                    for abnormality in abnormalities:
                        # Count by type
                        abn_type = abnormality.get('type', 'unknown')
                        summary_data['abnormality_types'][abn_type] = \
                            summary_data['abnormality_types'].get(abn_type, 0) + 1
                        
                        # Count by severity
                        severity = abnormality.get('severity', 'unknown')
                        summary_data['severity_distribution'][severity] = \
                            summary_data['severity_distribution'].get(severity, 0) + 1
        
        summary_data['processing_success_rate'] = successful_analyses / len(analysis_results)
        
        # Save batch summary
        summary_path = os.path.join(self.output_dir, 'batch_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        return summary_data