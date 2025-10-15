import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.measure import label, regionprops

class AbnormalityDetector:
    def __init__(self):
        self.min_abnormality_area = 100
        self.max_abnormality_area = 10000
        
    def detect_lung_nodules(self, image):
        """Detect potential lung nodules in chest X-rays"""
        # 1. Lung segmentation
        lung_mask = self._segment_lungs(image)
        
        # 2. ROI extraction (focus on lung areas)
        lung_roi = cv2.bitwise_and(image, image, mask=lung_mask)
        
        # 3. Candidate nodule detection using circular Hough transform
        circles = cv2.HoughCircles(
            lung_roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )
        
        nodule_candidates = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Extract ROI around detected circle
                roi = self._extract_circular_roi(lung_roi, x, y, r)
                
                # Feature analysis
                features = self._extract_nodule_features(roi)
                
                # Simple classification based on intensity and texture
                if self._classify_nodule_candidate(features):
                    nodule_candidates.append({
                        'center': (x, y),
                        'radius': r,
                        'confidence': features['confidence'],
                        'area': np.pi * r * r,
                        'type': 'nodule',
                        'severity': self._assess_nodule_severity(features)
                    })
        
        return nodule_candidates
    
    def detect_bone_fractures(self, image):
        """Detect potential bone fractures using edge analysis"""
        # 1. Edge detection with multiple scales
        edges_canny = cv2.Canny(image, 50, 150)
        edges_sobel = self._sobel_edges(image)
        
        # 2. Combine edge information
        combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
        
        # 3. Line detection using Hough transform
        lines = cv2.HoughLinesP(
            combined_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=20,
            maxLineGap=10
        )
        
        # 4. Filter lines that might represent fractures
        fracture_candidates = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Filter based on fracture characteristics
                if self._is_potential_fracture(image, line[0], length, angle):
                    fracture_candidates.append({
                        'line': line[0],
                        'length': length,
                        'angle': angle,
                        'severity': self._assess_fracture_severity(image, line[0]),
                        'type': 'fracture',
                        'area': length * 2  # Approximate area
                    })
        
        return fracture_candidates
    
    def detect_brain_anomalies(self, image):
        """Detect potential brain anomalies in MRI scans"""
        # 1. Brain extraction (skull stripping)
        brain_mask = self._extract_brain_mask(image)
        brain_roi = cv2.bitwise_and(image, image, mask=brain_mask)
        
        # 2. Symmetry analysis
        asymmetry_map = self._analyze_brain_symmetry(brain_roi)
        
        # 3. Intensity anomaly detection
        intensity_anomalies = self._detect_intensity_anomalies(brain_roi)
        
        # 4. Texture analysis using LBP
        texture_anomalies = self._detect_texture_anomalies(brain_roi)
        
        # 5. Combine all anomaly types
        combined_anomalies = []
        
        # Process asymmetry regions
        asymmetry_regions = self._find_anomaly_regions(asymmetry_map, threshold=0.3)
        for region in asymmetry_regions:
            combined_anomalies.append({
                'type': 'asymmetry',
                'region': region,
                'severity': self._assess_asymmetry_severity(region['mean_intensity']),
                'area': region['area'],
                'contour': region['contour']
            })
        
        # Add other anomaly types
        combined_anomalies.extend(intensity_anomalies)
        combined_anomalies.extend(texture_anomalies)
        
        return combined_anomalies
    
    def _segment_lungs(self, image):
        """Segment lung regions from chest X-ray"""
        # Threshold-based segmentation
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find largest connected components (likely lungs)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        lung_mask = np.zeros_like(image)
        if len(contours) >= 2:
            # Draw the two largest contours (left and right lung)
            cv2.drawContours(lung_mask, contours[:2], -1, (255,), -1)
        
        return lung_mask
    
    def _extract_nodule_features(self, roi):
        """Extract features for nodule classification"""
        if roi.size == 0:
            return {'confidence': 0}
        
        # Intensity features
        mean_intensity = np.mean(roi)
        std_intensity = np.std(roi)
        
        # Texture features using Local Binary Pattern
        lbp = local_binary_pattern(roi, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist.astype(np.float32)
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # Shape features
        contours, _ = cv2.findContours(
            (roi > roi.mean()).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        else:
            area = 0
            circularity = 0
        
        # Calculate confidence score
        confidence = self._calculate_nodule_confidence(mean_intensity, std_intensity, circularity, area)
        
        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'texture': lbp_hist,
            'circularity': circularity,
            'area': area,
            'confidence': confidence
        }
    
    def _calculate_nodule_confidence(self, mean_intensity, std_intensity, circularity, area):
        """Calculate confidence score for nodule detection"""
        # Simple scoring based on typical nodule characteristics
        score = 0
        
        # Intensity criteria (nodules typically have higher intensity)
        if mean_intensity > 100:
            score += 0.3
        
        # Shape criteria (nodules are typically circular)
        if circularity > 0.7:
            score += 0.4
        
        # Size criteria
        if 50 < area < 1000:
            score += 0.3
        
        return min(score, 1.0)
    
    def _classify_nodule_candidate(self, features):
        """Simple rule-based classification for nodule candidates"""
        return features['confidence'] > 0.6
    
    def _assess_nodule_severity(self, features):
        """Assess nodule severity based on features"""
        area = features.get('area', 0)
        confidence = features.get('confidence', 0)
        
        if area > 500 and confidence > 0.8:
            return 'high'
        elif area > 200 and confidence > 0.7:
            return 'medium'
        else:
            return 'low'
    
    def _extract_circular_roi(self, image, x, y, radius):
        """Extract circular region of interest"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, (255,), -1)
        roi = cv2.bitwise_and(image, image, mask=mask)
        
        # Extract the bounding box of the circle
        x_start = max(0, x - radius)
        x_end = min(image.shape[1], x + radius)
        y_start = max(0, y - radius)
        y_end = min(image.shape[0], y + radius)
        
        return roi[y_start:y_end, x_start:x_end]
    
    def _sobel_edges(self, image):
        """Detect edges using Sobel operator"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        return (sobel_combined > 50).astype(np.uint8) * 255
    
    def _is_potential_fracture(self, image, line, length, angle):
        """Determine if a line might represent a fracture"""
        x1, y1, x2, y2 = line
        
        # Check line properties
        if length < 10 or length > 200:
            return False
        
        # Check intensity profile along the line
        num_points = int(length)
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)
        
        # Extract intensity values along the line
        intensities = []
        for x, y in zip(x_coords, y_coords):
            if 0 <= int(x) < image.shape[1] and 0 <= int(y) < image.shape[0]:
                intensities.append(image[int(y), int(x)])
        
        if not intensities:
            return False
        
        # Fractures typically show as dark lines (low intensity)
        mean_intensity = np.mean(intensities)
        return mean_intensity < 0.7 * np.mean(image)
    
    def _assess_fracture_severity(self, image, line):
        """Assess the severity of a potential fracture"""
        x1, y1, x2, y2 = line
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Simple severity assessment based on length and intensity contrast
        if length > 50:
            return 'high'
        elif length > 25:
            return 'medium'
        else:
            return 'low'
    
    def _extract_brain_mask(self, image):
        """Extract brain region from MRI scan (simplified skull stripping)"""
        # Threshold-based approach (simplified)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find largest connected component (brain)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            brain_mask = np.zeros_like(image)
            cv2.drawContours(brain_mask, [largest_contour], -1, (255,), -1)
            return brain_mask
        
        return np.ones_like(image) * 255
    
    def _analyze_brain_symmetry(self, brain_image):
        """Analyze brain symmetry to detect asymmetric anomalies"""
        h, w = brain_image.shape
        left_half = brain_image[:, :w//2]
        right_half = brain_image[:, w//2:]
        
        # Flip right half to compare with left
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize to ensure same dimensions
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        # Calculate absolute difference
        asymmetry = cv2.absdiff(left_half, right_half_flipped)
        
        # Create full asymmetry map
        asymmetry_map = np.zeros_like(brain_image)
        asymmetry_map[:, :min_width] = asymmetry
        
        return asymmetry_map
    
    def _detect_intensity_anomalies(self, image):
        """Detect regions with abnormal intensity patterns"""
        # Use K-means clustering to identify different tissue types
        data = image.reshape((-1, 1))
        data = data[data > 0]  # Remove background
        
        if len(data) == 0:
            return []
        
        # Reshape data to 2D for sklearn
        data = data.reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        centers = kmeans.cluster_centers_.flatten()
        
        # Find anomalous intensities (significantly different from main tissue types)
        anomalies = []
        threshold = 2 * np.std(data)
        
        for i, center in enumerate(centers):
            if np.abs(center - np.mean(centers)) > threshold:
                # This cluster represents an anomaly
                mask = (image == center).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:  # Filter small regions
                        anomalies.append({
                            'type': 'intensity',
                            'contour': contour,
                            'intensity': center,
                            'area': area,
                            'severity': 'high' if area > 500 else 'medium'
                        })
        
        return anomalies
    
    def _detect_texture_anomalies(self, image):
        """Detect texture-based anomalies using Local Binary Pattern"""
        # Calculate LBP
        lbp = local_binary_pattern(image, 8, 1, method='uniform')
        
        # Divide image into patches and analyze texture
        patch_size = 32
        anomalies = []
        
        for y in range(0, image.shape[0] - patch_size, patch_size):
            for x in range(0, image.shape[1] - patch_size, patch_size):
                patch_lbp = lbp[y:y+patch_size, x:x+patch_size]
                
                # Calculate texture uniformity
                hist, _ = np.histogram(patch_lbp.ravel(), bins=10, range=(0, 10))
                hist = hist.astype(np.float32)
                hist /= (hist.sum() + 1e-7)
                
                # Calculate entropy (measure of texture complexity)
                entropy = -np.sum(hist * np.log2(hist + 1e-7))
                
                # High entropy might indicate anomalous texture
                if entropy > 2.5:  # Threshold for anomalous texture
                    anomalies.append({
                        'type': 'texture',
                        'region': (x, y, x+patch_size, y+patch_size),
                        'entropy': entropy,
                        'area': patch_size * patch_size,
                        'severity': 'high' if entropy > 3.0 else 'medium'
                    })
        
        return anomalies
    
    def _find_anomaly_regions(self, anomaly_map, threshold=0.3):
        """Find connected regions in anomaly map"""
        binary = (anomaly_map > threshold * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small regions
                # Calculate region properties
                mask = np.zeros_like(binary)
                cv2.drawContours(mask, [contour], -1, (255,), -1)
                
                mean_intensity = np.mean(anomaly_map[mask > 0])
                
                regions.append({
                    'contour': contour,
                    'area': area,
                    'mean_intensity': mean_intensity,
                    'bbox': cv2.boundingRect(contour)
                })
        
        return regions
    
    def _assess_asymmetry_severity(self, mean_intensity):
        """Assess severity of brain asymmetry"""
        if mean_intensity > 100:
            return 'high'
        elif mean_intensity > 50:
            return 'medium'
        else:
            return 'low'