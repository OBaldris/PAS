import numpy as np
import cv2
import cv2.ximgproc as ximgproc
from scipy.ndimage import median_filter
from scipy.stats import gaussian_kde

class DepthCalculator:
    def __init__(self):
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.image_copy = None
        self.disparity = None
        self.P2 = None
        self.P3 = None
        self.window_name = 'Select Area for Depth Calculation'
        self.original_img = None
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
    def pixel_to_world(self, x, y, depth):
    	"""Convert pixel coordinates to world coordinates in meters with (0,0) at bottom-left."""
    	# Get camera parameters from P2 matrix
    	fx = self.P2[0, 0]  # Focal length x
    	fy = self.P2[1, 1]  # Focal length y
    	cx = self.P2[0, 2]  # Principal point x
    	cy = self.P2[1, 2]  # Principal point y
    
    	# Get image dimensions
    	img_height, img_width = self.original_img.shape[:2]
    
    	# Convert y coordinate to measure from bottom instead of top
    	y = img_height - y
    
    	# Convert to world coordinates
    	# X coordinate: measured from left edge
    	X = (x - cx) * depth / fx
    	# Make X positive by adding offset
    	X = X + (cx * depth / fx)
    
    	# Y coordinate: measured from bottom edge
    	Y = (y - cy) * depth / fy
    	# Make Y positive by adding offset
    	Y = Y + (cy * depth / fy)
    
    	Z = depth
    
    	return X, Y, Z

    def world_to_pixel(self, X, Y, Z):
        """Convert world coordinates in meters to pixel coordinates with (0,0) at top-left."""
        if Z == 0:
            Z += 0.001

        # Get camera parameters from P2 matrix
        fx = self.P2[0, 0]  # Focal length x
        fy = self.P2[1, 1]  # Focal length y
        cx = self.P2[0, 2]  # Principal point x
        cy = self.P2[1, 2]  # Principal point y

        #print("xyz: ", X, Y, Z)

        # Get image dimensions
        img_height, img_width = self.original_img.shape[:2]

        #print("img_height, img_width", img_height, img_width)

        # Project 3D point to image plane
        x = (fx * X / Z)# + cx
        #x = x - img_width/2

        # Convert Y to measure from top instead of bottom
        Y_from_top = Y - (cy * Z / fy)
        y = (fy * Y_from_top / Z) + cy
        y = img_height - y  # Invert Y coordinate

        # Round to nearest pixel
        x = int(round(x))
        y = int(round(y))

        # Ensure coordinates are within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))

        return x, y

        
    def detect_object_mask(self, region_img):
        hsv = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        fg_mask = self.object_detector.apply(region_img)
        combined_mask = cv2.bitwise_or(thresh, fg_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask

    def calculate_weighted_depth(self, top_left, bottom_right):
        """Calculate depth and world coordinates with corrected scaling factors."""
        try:
            if self.P2 is None or self.P3 is None:
                raise ValueError("Calibration matrices not initialized")
                
            focal_length = self.P2[0, 0]
            baseline = abs(self.P2[0, 3] - self.P3[0, 3]) / self.P2[0, 0]
            
            distance_correction = 1.8
            
            x1, y1 = min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])
            x2, y2 = max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])
            
            x1 = max(0, min(x1, self.disparity.shape[1] - 1))
            x2 = max(0, min(x2, self.disparity.shape[1] - 1))
            y1 = max(0, min(y1, self.disparity.shape[0] - 1))
            y2 = max(0, min(y2, self.disparity.shape[0] - 1))
            
            region_disparity = self.disparity[y1:y2 + 1, x1:x2 + 1]
            region_img = self.original_img[y1:y2 + 1, x1:x2 + 1]
            
            object_mask = self.detect_object_mask(region_img)
            object_mask = object_mask.astype(float) / 255.0
            
            valid_disparity = (region_disparity > 1.0) & (region_disparity < 96.0)

            if np.any(valid_disparity):
                height, width = region_disparity.shape
                y_coords, x_coords = np.ogrid[:height, :width]
                center_y, center_x = height / 2, width / 2
                
                size_factor = min(width, height) / 4
                dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                center_weights = np.exp(-0.5 * (dist_from_center / size_factor)**2)

                depths = np.zeros_like(region_disparity, dtype=float)
                valid_disparities = region_disparity[valid_disparity]
                
                depths[valid_disparity] = (focal_length * baseline * distance_correction) / valid_disparities
                depths *= 0.2
                
                median_depth = np.median(depths[depths > 0])
                depth_std = np.std(depths[depths > 0])
                
                range_factor = 0.8
                min_depth = max(1.0, median_depth - range_factor * depth_std)
                max_depth = median_depth + range_factor * depth_std
                
                valid_depths_mask = (depths > min_depth) & (depths < max_depth)
                valid_depths = depths[valid_depths_mask]

                if len(valid_depths) > 0:
                    proximity_weights = 1 / (depths + 0.5)**0.8
                    proximity_weights[~valid_depths_mask] = 0
                    
                    total_weights = center_weights * proximity_weights * (object_mask + 0.3)
                    total_weights = total_weights / (np.sum(total_weights) + 1e-10)
                    
                    weighted_depth = np.sum(depths * total_weights)

                    if len(valid_depths) > 5:
                        bandwidth = 0.15 * (np.max(valid_depths) - np.min(valid_depths))
                        kde = gaussian_kde(valid_depths, bw_method=bandwidth)
                        depth_range = np.linspace(np.min(valid_depths), np.max(valid_depths), 100)
                        mode_depth = depth_range[np.argmax(kde(depth_range))]
                    else:
                        mode_depth = np.median(valid_depths)
                    
                    confidence = min(1.0, len(valid_depths) / (width * height * 0.3))
                    final_depth = confidence * mode_depth + (1 - confidence) * weighted_depth
                    
                    # Calculate world coordinates for center point
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    X, Y, Z = self.pixel_to_world(center_x, center_y, final_depth)
                    
                    return ((X, Y, Z), mode_depth), (x1, y1), (x2, y2)
            
            return (None,None), (x1, y1), (x2, y2)
            
        except Exception as e:
            print(f"Error in depth calculation: {e}")
            return (None, None), top_left, bottom_right

    def compute_disparity(self, left_gray, right_gray):
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=96,
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity_left = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        stereo_right = ximgproc.createRightMatcher(stereo)
        disparity_right = stereo_right.compute(right_gray, left_gray).astype(np.float32) / 16.0

        wls_filter = ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(8000)
        wls_filter.setSigmaColor(1.5)
        filtered_disparity = wls_filter.filter(disparity_left, left_gray, None, disparity_right)

        filtered_disparity = median_filter(filtered_disparity, size=3)
        filtered_disparity = cv2.bilateralFilter(filtered_disparity, d=5, sigmaColor=50, sigmaSpace=50)
        
        disp_vis = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        
        return filtered_disparity, disp_vis

    def run(self, left_input, right_input, calib_file, input_is_array=False):
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if 'P_rect_02' in line:
                    self.P2 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
                elif 'P_rect_03' in line:
                    self.P3 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)

        if input_is_array:
            self.original_img = left_input
            right_img = right_input
        else:
            self.original_img = cv2.imread(left_input)
            right_img = cv2.imread(right_input)

        if self.original_img is None or right_img is None:
            raise ValueError("Could not load images")

        left_gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        left_gray = clahe.apply(left_gray)
        right_gray = clahe.apply(right_gray)

        self.disparity, self.disp_vis = self.compute_disparity(left_gray, right_gray)

if __name__ == "__main__":
    calculator = DepthCalculator()
    calculator.run(
        r'C:\Users\ivorb\Downloads\34759_final_project_rect\34759_final_project_rect\seq_01\image_02\data\000000.png',
        r'C:\Users\ivorb\Downloads\34759_final_project_rect\34759_final_project_rect\seq_01\image_03\data\000000.png',
        r'calib_cam_to_cam.txt'
    )