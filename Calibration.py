import cv2
import numpy as np
import glob



def rectify_image(img, calibration_file='calibration.npz'):
    # Load the calibration parameters
    with np.load(calibration_file) as data:
        mtx = data['mtx']
        dist = data['dist']
        newcameramtx = data['newcameramtx']
        roi = data['roi']

    # Undistort the original image
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the undistorted image so that it only shows the non-black pixels
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    return undistorted_img




class ImageRectifier:
    def __init__(self, calibration_file='calibration.npz'):
        # Load the calibration parameters
        with np.load(calibration_file) as data:
            self.mtx = data['mtx']
            self.dist = data['dist']
            self.newcameramtx = data['newcameramtx']
            self.roi = data['roi']

    def rectify(self, img):
        # Undistort the original image
        undistorted_img = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)

        # Crop the undistorted image so that it only shows the non-black pixels
        x, y, w, h = self.roi
        undistorted_img = undistorted_img[y:y+h, x:x+w]

        return undistorted_img






