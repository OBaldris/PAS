import numpy as np
import cv2
import glob
import load_data
import kurac6
from ultralytics import YOLO
import Kalman

def within_margin(pred_x, pred_y, act_x, act_y):
    margin = 20
    return (abs(pred_x - act_x) <= margin and abs(pred_y - act_y) <= margin)

def get_center(x1, y1, x2, y2):
    center_x = (x1+x2) / 2
    center_y = (y1+y2) / 2
    return (center_x, center_y)

def get_depth_from_bbox(depth_calculator, results):
    class_names = []
    confidences = []
    world_coords = []
    centers = []

    for result in results[0]:
        box = result.boxes.xyxy[0]
        x1, y1, x2, y2 = np.array(box).astype(int)

        center_coords = get_center(x1, y1, x2, y2)
        centers.append(center_coords)

        cls_idx = result.boxes.cls.item()
        class_name = model.names[cls_idx]
        class_names.append(class_name)
        confidence = result.boxes.conf.item()
        confidences.append(confidence)

        coords_data, _ = depth_calculator.calculate_weighted_depth((x1, y1), (x2, y2))[0]
        if coords_data is not None:
            world_coords.append(coords_data)
        else:
            world_coords.append((0, 0, 0))

    return class_names, confidences, world_coords, centers

if __name__ == "__main__":
    calibration_data_path = "C:\\Users\\ivorb\\Downloads\\34759_final_project_rect\\34759_final_project_rect\\calib_cam_to_cam.txt"

    data = load_data.StereoVisionData()
    model = YOLO("yolo11n.pt")
    class_indexes = [0]  # Person only

    KF_list = []
    previous_class_names = []

    data.load_sequence(sequence_num=1)

    for i in range(0, 144):
        frame_data = data.get_frame_data(sequence_num=1, frame_index=i)
        results = model(frame_data["left_image"], classes=class_indexes)

        DepthObj = kurac6.DepthCalculator()
        DepthObj.run(frame_data["left_image"], frame_data["right_image"], calibration_data_path, input_is_array=True)

        class_names, confidences, world_coords, centers = get_depth_from_bbox(DepthObj, results)

        if class_names is not None:
            new_KF_list = [None]*len(class_names)
            for j in range(len(class_names)):
                min_dist = 99999
                KF_match_index = -1
                KF_taken = []
                center = centers[j]
                world_coord = world_coords[j]

                for q in range(len(KF_list)):
                    if not KF_list:
                        continue
                    if KF_list[q].class_name != class_names[j]:
                        continue
                    if q in KF_taken:
                        continue
                    predicted_pos = [KF_list[q].X[0], KF_list[q].X[1], KF_list[q].X[2]]

                    if not within_margin(predicted_pos[0], predicted_pos[1], center[0], center[1]):
                        continue

                    dist = np.sqrt((predicted_pos[0] - world_coord[0])**2 + 
                                 (predicted_pos[1] - world_coord[1])**2 + 
                                 (predicted_pos[2] - world_coord[2])**2)
                    if dist < min_dist:
                        min_dist = dist
                        KF_match_index = q

                if KF_match_index != -1:
                    new_KF_list[j] = KF_list[KF_match_index]
                    KF_taken.append(KF_match_index)
                else:
                    new_KF_list[j] = Kalman.KalmanFilter()
                    new_KF_list[j].class_name = class_names[j]
                    new_KF_list[j].X = np.array([[world_coord[0]], 
                                               [world_coord[1]], 
                                               [world_coord[2]], 
                                               [0], [0], [0], 
                                               [0], [0], [0]])
                    print(f"New object: {class_names[j]} at X:{world_coord[0]:.2f}m Y:{world_coord[1]:.2f}m Z:{world_coord[2]:.2f}m")

            KF_list = new_KF_list.copy()

            for j in range(len(KF_list)):
                world_coord = world_coords[j]
                center = centers[j]
                
                # Draw detection
                cv2.circle(frame_data["left_image"], (int(center[0]),int(center[1])), 10, (0, 255, 0), 3)

                KF_list[j].Z = np.array([[world_coord[0]], [world_coord[1]], [world_coord[2]]])
                KF_list[j].update()
                KF_list[j].predict()

                circle_center_coords = (int(center[0]), int(center[1]))
                print(f"Object {j} - World coordinates: X:{world_coord[0]:.2f}m Y:{world_coord[1]:.2f}m Z:{world_coord[2]:.2f}m")
                
                # Draw prediction circle (size inversely proportional to Z distance)
                circle_radius = max(int(300/world_coord[2]), 5)
                cv2.circle(frame_data["left_image"], circle_center_coords, circle_radius, (0,0,255), 3)

            cv2.imshow("frame", frame_data["left_image"])
            cv2.waitKey(0)

        previous_class_names = class_names

    print("Processing complete!")