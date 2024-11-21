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

    def calculate_weighted_depth(self, top_left, bottom_right):
        """Calculate depth with more weight on central and nearer points."""
        focal_length = self.P2[0, 0]
        baseline = abs(self.P2[0, 3] - self.P3[0, 3]) / self.P2[0, 0]

        # Ensure correct order of coordinates
        x1, y1 = min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])
        x2, y2 = max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])

        # Bound checking
        x1 = max(0, min(x1, self.disparity.shape[1]-1))
        x2 = max(0, min(x2, self.disparity.shape[1]-1))
        y1 = max(0, min(y1, self.disparity.shape[0]-1))
        y2 = max(0, min(y2, self.disparity.shape[0]-1))

        region_disparity = self.disparity[y1:y2+1, x1:x2+1]
        valid_disparity = region_disparity > 0

        if np.any(valid_disparity):
            # Create center-weighted mask
            height, width = region_disparity.shape
            y_coords, x_coords = np.ogrid[:height, :width]
            center_y, center_x = height / 2, width / 2
            
            # Calculate distance from center for each pixel
            dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            # Create gaussian weight based on distance from center
            center_weights = np.exp(-0.5 * (dist_from_center / (max_dist/3))**2)

            # Calculate depths
            depths = np.zeros_like(region_disparity, dtype=float)
            depths[valid_disparity] = (focal_length * baseline) / region_disparity[valid_disparity]
            
            # Apply proximity weighting (closer objects get more weight)
            proximity_weights = 1 / (depths + 1e-6)  # Add small epsilon to avoid division by zero
            proximity_weights[~valid_disparity] = 0
            
            # Combine center and proximity weights
            total_weights = center_weights * proximity_weights
            
            # Normalize weights
            total_weights = total_weights / np.sum(total_weights)
            
            # Calculate weighted average depth
            weighted_depth = np.sum(depths * total_weights)
            
            # Find the most common depth range (mode)
            valid_depths = depths[valid_disparity]
            if len(valid_depths) > 0:
                kde = gaussian_kde(valid_depths)
                depth_range = np.linspace(np.min(valid_depths), np.max(valid_depths), 100)
                mode_depth = depth_range[np.argmax(kde(depth_range))]
                
                # Return both weighted average and mode
                return (weighted_depth, mode_depth), (x1,y1), (x2,y2)
            
            return (weighted_depth, None), (x1,y1), (x2,y2)
        return None, (x1,y1), (x2,y2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.image_copy = self.original_img.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_img = self.image_copy.copy()
                cv2.rectangle(temp_img, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # Calculate and display weighted depth
            depth_result, tl, br = self.calculate_weighted_depth(self.start_point, self.end_point)
            
            # Draw final rectangle
            cv2.rectangle(self.original_img, tl, br, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.original_img)
            
            if depth_result is not None:
                weighted_depth, mode_depth = depth_result
                print(f"Rectangle: Top-Left {tl}, Bottom-Right {br}")
                print(f"Weighted Average Depth: {weighted_depth:.2f} meters")
                if mode_depth is not None:
                    print(f"Most Common Depth: {mode_depth:.2f} meters")
            else:
                print(f"No valid depth in selected area: Top-Left {tl}, Bottom-Right {br}")

    def compute_disparity(self, left_gray, right_gray):
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=3,
            P1=8 * 3 * 3 ** 2,
            P2=32 * 3 * 3 ** 2,
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

        # Post-processing
        filtered_disparity = median_filter(filtered_disparity, size=3)
        disp_vis = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        
        return filtered_disparity, disp_vis

    def run(self, left_img_path, right_img_path, calib_file):
        # Load calibration
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if 'P_rect_02' in line:
                    self.P2 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
                elif 'P_rect_03' in line:
                    self.P3 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)

        # Load and preprocess images
        self.original_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        if self.original_img is None or right_img is None:
            raise ValueError("Could not load images")

        # Convert to grayscale and apply preprocessing
        left_gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        self.disparity, self.disp_vis = self.compute_disparity(left_gray, right_gray)

        # Set up window and mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Show original image initially
        cv2.imshow(self.window_name, self.original_img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break
            elif key == ord('d'):  # Press 'd' to toggle between disparity and original
                if cv2.getWindowImage(self.window_name) is self.original_img:
                    cv2.imshow(self.window_name, self.disp_vis)
                else:
                    cv2.imshow(self.window_name, self.original_img)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    calculator = DepthCalculator()
    calculator.run(
        r'C:\Users\ivorb\Downloads\34759_final_project_rect\34759_final_project_rect\seq_01\image_02\data\000000.png',
        r'C:\Users\ivorb\Downloads\34759_final_project_rect\34759_final_project_rect\seq_01\image_03\data\000000.png',
        'calib_cam_to_cam.txt'
    )