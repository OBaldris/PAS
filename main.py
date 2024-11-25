import numpy as np
import cv2
import glob
import load_data
import kurac6
from ultralytics import YOLO
import Kalman


def within_margin(pred_x, pred_y, act_x, act_y):
    margin = 50
    if pred_x + margin >= act_x or pred_x - margin <= act_x and pred_y + margin >= act_y or pred_y - margin <= act_y:
        return True
    else:
        return False

def center(x1, y1, x2, y2):
    center_x = (x1+x2) / 2
    center_y = (y1+y2) / 2

    return (center_x, center_y)

def get_depth_from_bbox(depth_calculator, results):
    class_names = []
    confidences = []
    weighted_depths = []
    centers = []

    for result in results[0]:
        box = result.boxes.xyxy[0]
        x1, y1, x2, y2 = np.array(box).astype(int)

        center_coords = center(x1, y1, x2, y2)
        centers.append(center_coords)

        cls_idx = result.boxes.cls.item()
        class_name = model.names[cls_idx]
        class_names.append(class_name)
        confidence = result.boxes.conf.item()
        confidences.append(confidence)

        weighted_depth, _ = depth_calculator.calculate_weighted_depth((x1, y1), (x2, y2))[0]
        weighted_depths.append(weighted_depth)


        #print(f'object: {class_name}, confidence: {confidence} at {x, y, x2, y2}')
        #print(depth_calculator.calculate_weighted_depth((x, y), (x2, y2)))

    return class_names, confidences, weighted_depths, centers


if __name__ == "__main__":
    calibration_data_path = "C:/Users/gust0/OneDrive - Danmarks Tekniske Universitet/Perception_for_autonomous_sys/Final_project/34759_final_project_rect/34759_final_project_rect/calib_cam_to_cam.txt"


    # Example usage
    data = load_data.StereoVisionData()

    model = YOLO("yolo11n.pt")
    class_indexes = [0, 1, 2]   #Person, Bicycle, Car indexes in YOLO model :)

    KF_list = []
    previous_class_names = []

    # Load a specific sequence
    data.load_sequence(sequence_num=1)

    for i in range(0, 144):
        # Access a specific frame's data
        frame_data = data.get_frame_data(sequence_num=1, frame_index=i)
        results = model(frame_data["left_image"], classes=class_indexes)

        DepthObj = kurac6.DepthCalculator()
        DepthObj.run_without_fuckery(frame_data["left_image"], frame_data["right_image"], calibration_data_path)

        class_names, confidences, weighted_depths, centers = get_depth_from_bbox(DepthObj, results)

        if class_names is not None:
            new_KF_list = [None]*len(class_names)
            for j in range(len(class_names)):
                min_dist = 99999
                KF_match_index = -1
                KF_taken = []
                for q in range(len(KF_list)):
                    if KF_list[q].class_name != class_names[j]:
                        continue
                    if q in KF_taken:
                        continue
                    predicted_pos = KF_list[q].X_pred()
                    center = centers[j]

                    if not within_margin(predicted_pos[0], predicted_pos[1], center[0], center[1]):
                        continue

                    dist = np.sqrt((predicted_pos[0] - center[0])**2 + (predicted_pos[1] - center[1])**2 + (predicted_pos[2] - weighted_depths[j])**2)
                    if dist < min_dist:
                        min_dist = dist
                        KF_match_index = q

                if KF_match_index != -1:
                    new_KF_list[j] = KF_list[KF_match_index]
                    KF_taken.append(KF_match_index)
                else:
                    new_KF_list[j] = Kalman.KalmanFilter()
                    new_KF_list[j].class_name = class_names[j]

                KF_list = new_KF_list.copy()

        for w in range(len(new_KF_list)):
            print("KF LIST CLASS NAMES:", w, new_KF_list[w].class_name)






        previous_class_names = class_names

        print("CLASS NAMES: ", class_names)





    print("Hello, World!")


