import cv2
import numpy as np
import os

# Define the input folder containing the images
input_folder = "calib/image_02/data/"

# Manually defined regions with corresponding chessboard sizes
left_regions = [
    (120, 110, 190, 220, (11, 7)),   # Chessboard 1
    (370, 160, 110, 150, (11, 7)),   # Chessboard 2
    (450, 70, 190, 80, (7, 5)),     # Chessboard 3
    (500, 270, 100, 120, (5, 7)),   # Chessboard 4
    (500, 390, 180, 100, (7, 5)),   # Chessboard 5
    (680, 290, 90, 130, (5, 7)),    # Chessboard 6
    (800, 300, 100, 120, (5, 7)),   # Chessboard 7
    (790, 130, 130, 100, (7, 5)),   # Chessboard 8
    (1030, 60, 170, 100, (7, 5)),   # Chessboard 9
    (990, 380, 170, 90, (7, 5)),    # Chessboard 10
    (1080, 250, 120, 130, (5, 7)),  # Chessboard 11
    (1240, 160, 80, 300, (5, 15)),  # Chessboard 12
    (1320, 180, 70, 180, (5, 7)),   # Chessboard 13
]

right_regions = [
    (60, 140, 190, 200, (11, 7)),
    (290, 170, 130, 150, (11, 7)),
    (350, 80, 200, 90, (7, 5)),
    (430, 270, 110, 120, (5, 7)),
    (410, 380, 170, 100, (7, 5)),
    (590, 290, 110, 130, (5, 7)),
    (710, 290, 110, 130, (5, 7)),
    (700, 130, 140, 110, (7, 5)),
    (900, 60, 170, 90, (7, 5)),
    (880, 360, 170, 90, (7, 5)),
    (990, 240, 110, 130, (5, 7)),
    (1130, 160, 70, 280, (5, 15)),
    (1200, 170, 70, 180, (5, 7)),
]

# Prepare lists for calibration results
all_camera_matrices = []
all_dist_coeffs = []

# Process each image in the folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
total_images = len(image_files)

for idx, filename in enumerate(image_files):
    file_path = os.path.join(input_folder, filename)

    # Load the image
    img = cv2.imread(file_path)
    print(f"Processing image {idx + 1}/{total_images}: {filename}")

    # Copy the original image to draw regions
    output_img = img.copy()

    # Store per-image results
    camera_matrices = []
    dist_coeffs = []

    # Iterate through each region and process
    for i, (x, y, w, h, chessboard_size) in enumerate(left_regions):
        # Create a copy of the original image and blackout the rest
        masked_img = np.zeros_like(img)
        masked_img[y:y+h, x:x+w] = img[y:y+h, x:x+w]

        # Convert to grayscale
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        # Apply image enhancements
        sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_img = cv2.filter2D(gray, -1, sharpening_kernel)
        equalized_img = cv2.equalizeHist(sharpened_img)

        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(equalized_img, chessboard_size, None)

        # Draw the region and label it on the output image
        color = (0, 255, 0) if ret else (0, 0, 255)
        cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(output_img, f"Region {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if ret:
            corners_refined = cv2.cornerSubPix(equalized_img, corners, (11, 11), (-1, -1),
                                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

            _, camera_matrix, dist_coeff, _, _ = cv2.calibrateCamera(
                [objp], [corners_refined], equalized_img.shape[::-1], None, None
            )

            camera_matrices.append(camera_matrix)
            dist_coeffs.append(dist_coeff)

    # Display image
    cv2.imshow("Regions with Labels", output_img)
    cv2.waitKey(500)  # Wait for 500ms per image, then proceed automatically
    cv2.destroyAllWindows()

    # If at least one chessboard was detected in this image
    if camera_matrices and dist_coeffs:
        avg_camera_matrix = sum(camera_matrices) / len(camera_matrices)
        avg_dist_coeffs = sum(dist_coeffs) / len(dist_coeffs)

        print(f"Image: {filename}")
        print("Average Camera Matrix:")
        print(avg_camera_matrix)
        print("\nAverage Distortion Coefficients:")
        print(avg_dist_coeffs)

        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(avg_camera_matrix, avg_dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(img, avg_camera_matrix, avg_dist_coeffs, None, new_camera_matrix)

        # Validate ROI before cropping
        x, y, w, h = roi
        if w > 0 and h > 0:  # Check for valid ROI dimensions
            undistorted_img_cropped = undistorted_img[y:y+h, x:x+w]
            cv2.imshow("Undistorted Image", undistorted_img_cropped)
        else:
            print(f"Invalid ROI for undistorted image in {filename}. Displaying full undistorted image.")
            cv2.imshow("Undistorted Image", undistorted_img)

        cv2.waitKey(500)  # Wait for 500ms
        cv2.destroyAllWindows()


print("Processing complete.")
cv2.destroyAllWindows()
