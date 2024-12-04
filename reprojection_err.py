import cv2
import numpy as np


def calculate_reprojection_error_using_features(ground_truth_img, your_img):
    """
    Calculate reprojection error using feature detection and matching.
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute features in both images
    keypoints_gt, descriptors_gt = sift.detectAndCompute(ground_truth_img, None)
    keypoints_your, descriptors_your = sift.detectAndCompute(your_img, None)

    # Use BFMatcher for feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_gt, descriptors_your)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched points
    points_gt = np.float32([keypoints_gt[m.queryIdx].pt for m in matches])
    points_your = np.float32([keypoints_your[m.trainIdx].pt for m in matches])

    # Calculate reprojection error
    error = np.linalg.norm(points_gt - points_your, axis=1)
    mean_error = np.mean(error)

    return mean_error, points_gt, points_your, matches


if __name__ == "__main__":
    # Paths to the images
    #load another image
    ground_truth_path = '../34759_final_project_rect/seq_01/image_02/data/000000.png'
    your_image_path = "undistorted.png"

    # Load images
    ground_truth_img = cv2.imread(ground_truth_path)
    your_img = cv2.imread(your_image_path)

    if ground_truth_img is None or your_img is None:
        raise FileNotFoundError("One of the input images could not be loaded.")

    # Calculate reprojection error
    try:
        mean_error, points_gt, points_your, matches = calculate_reprojection_error_using_features(
            ground_truth_img, your_img
        )
        print(f"Mean Reprojection Error: {mean_error}")

        # Visualize matches
        match_img = cv2.drawMatches(
            ground_truth_img, 
            cv2.SIFT_create().detect(ground_truth_img, None),
            your_img, 
            cv2.SIFT_create().detect(your_img, None), 
            matches[:50],  # Display top 50 matches
            None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Show the matches
        cv2.imshow("Feature Matches", match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")