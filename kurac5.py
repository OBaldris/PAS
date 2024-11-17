import numpy as np
import cv2
import open3d as o3d
from scipy.ndimage import median_filter
import cv2.ximgproc as ximgproc

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        disparity = param['disparity']
        P2 = param['P2']
        P3 = param['P3']
        
        focal_length = P2[0, 0]
        baseline = abs(P2[0, 3] - P3[0, 3]) / P2[0, 0]
        
        if disparity[y, x] != 0:
            depth = (focal_length * baseline) / disparity[y, x]
            print(f"Coordinates (x,y): ({x},{y})")
            print(f"Depth: {depth:.2f} meters")
        else:
            print(f"No valid depth at coordinates (x,y): ({x},{y})")

def load_calib_data(calib_file):
    P2 = None
    P3 = None

    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if 'P_rect_02' in line:
                P2 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
            elif 'P_rect_03' in line:
                P3 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)

    return P2, P3

def create_point_cloud(disparity, left_img, P2, P3, max_depth=80, min_depth=1):
    focal_length = P2[0, 0]
    baseline = abs(P2[0, 3] - P3[0, 3]) / P2[0, 0]
    cx = P2[0, 2]
    cy = P2[1, 2]

    rows, cols = disparity.shape
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))

    valid_disparity = disparity != 0
    Z = np.zeros_like(disparity, dtype=np.float32)
    Z[valid_disparity] = (focal_length * baseline) / (disparity[valid_disparity])

    depth_mask = (Z > min_depth) & (Z < max_depth)
    valid_points = valid_disparity & depth_mask

    X = (xx[valid_points] - cx) * Z[valid_points] / focal_length
    Y = (yy[valid_points] - cy) * Z[valid_points] / focal_length

    points = np.column_stack((X, Y, Z[valid_points]))
    colors = left_img[valid_points] / 255.0

    return points, colors

def preprocess_images(left_img, right_img):
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    left_gray = clahe.apply(left_gray)
    right_gray = clahe.apply(right_gray)

    left_gray = cv2.bilateralFilter(left_gray, d=9, sigmaColor=75, sigmaSpace=75)
    right_gray = cv2.bilateralFilter(right_gray, d=9, sigmaColor=75, sigmaSpace=75)

    return left_gray, right_gray

def compute_disparity(left_gray, right_gray):
    min_disp = 0  # Set to 0 to avoid negative disparities
    max_num_disp = 128
    num_disp = 16 * ((max_num_disp // 16))  # Make sure it's divisible by 16

    window_size = 3  # Reduced window size for better edge detection

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
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

    lmbda = 8000
    sigma = 1.5
    wls_filter = ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disparity = wls_filter.filter(disparity_left, left_gray, None, disparity_right)

    # Fill the left border by propagating the nearest valid disparity values
    border_size = 20
    for y in range(filtered_disparity.shape[0]):
        valid_indices = np.where(filtered_disparity[y, :] > 0)[0]
        if len(valid_indices) > 0:
            first_valid = valid_indices[0]
            if first_valid > 0:
                filtered_disparity[y, :first_valid] = filtered_disparity[y, first_valid]

    # Apply median filter
    filtered_disparity = median_filter(filtered_disparity, size=3)

    disp_vis = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    return filtered_disparity, disp_vis

def main():
    try:
        P2, P3 = load_calib_data('calib_cam_to_cam.txt')

        if P2 is None or P3 is None:
            print("Error: Calibration data could not be loaded.")
            return

        left_img = cv2.imread(r'C:\Users\ivorb\Downloads\34759_final_project_rect\34759_final_project_rect\seq_01\image_02\data\000000.png')
        right_img = cv2.imread(r'C:\Users\ivorb\Downloads\34759_final_project_rect\34759_final_project_rect\seq_01\image_03\data\000000.png')

        if left_img is None or right_img is None:
            print("Error: Could not load images")
            return

        left_gray, right_gray = preprocess_images(left_img, right_img)

        disparity, disp_vis = compute_disparity(left_gray, right_gray)

        window_name = 'Click for Depth Values'
        cv2.namedWindow(window_name)
        param = {'disparity': disparity, 'P2': P2, 'P3': P3}
        cv2.setMouseCallback(window_name, on_mouse_click, param)

        while True:
            cv2.imshow(window_name, disp_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break

        cv2.destroyAllWindows()

        points, colors = create_point_cloud(disparity, left_img, P2, P3, max_depth=80, min_depth=1)

        if len(points) == 0:
            print("Error: No valid points generated")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud("output_cloud_high_detail.ply", pcd)
        print("Point cloud saved as 'output_cloud_high_detail.ply'")

        o3d.visualization.draw_geometries([pcd])

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()