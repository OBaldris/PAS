import numpy as np
import cv2
import glob
import load_data
import kurac6
from ultralytics import YOLO
import Kalman


def within_margin(pred_x, pred_y, act_x, act_y):
    margin = 20
    if pred_x + margin >= act_x or pred_x - margin <= act_x and pred_y + margin >= act_y or pred_y - margin <= act_y:
        return True
    else:
        return False

def get_center(x1, y1, x2, y2):
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

        center_coords = get_center(x1, y1, x2, y2)
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
    calibration_data_path = "C:\\Users\\ivorb\\Downloads\\34759_final_project_rect\\34759_final_project_rect\\calib_cam_to_cam.txt"

    # Example usage
    data = load_data.StereoVisionData()

    model = YOLO("yolo11n.pt")
    #class_indexes = [0, 1, 2]   #Person, Bicycle, Car indexes in YOLO model :)
    class_indexes = [0]

    KF_list = []
    previous_class_names = []

    # Load a specific sequence
    data.load_sequence(sequence_num=1)

    for i in range(0, 144):
        # Access a specific frame's data
        frame_data = data.get_frame_data(sequence_num=1, frame_index=i)
        results = model(frame_data["left_image"], classes=class_indexes)

        DepthObj = kurac6.DepthCalculator()
        DepthObj.run(frame_data["left_image"], frame_data["right_image"], calibration_data_path, input_is_array=True)

        class_names, confidences, weighted_depths, centers = get_depth_from_bbox(DepthObj, results)
        #print("KF_LIST1:", KF_list)
        if class_names is not None:
            new_KF_list = [None]*len(class_names)
            for j in range(len(class_names)):

                min_dist = 99999
                KF_match_index = -1
                KF_taken = []
                center = centers[j]
                #print("KF_LIST2:", KF_list)
                for q in range(len(KF_list)):
                    #print("KF_LIST3:", KF_list)
                    if not KF_list:
                        continue
                    if KF_list[q].class_name != class_names[j]:
                        continue
                    if q in KF_taken:
                        continue
                    predicted_pos = [KF_list[q].X[0], KF_list[q].X[1], KF_list[q].X[2]]

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
                    new_KF_list[j].X = np.array([[center[0]], [center[1]], [weighted_depths[j]], [0], [0], [0], [0], [0], [0]])
                    print("new_KF_list[j].X", new_KF_list[j].X)

            KF_list = new_KF_list.copy()

            for j in range(len(KF_list)):
                center = centers[j]
                # draw detection
                cv2.circle(frame_data["left_image"], (int(center[0]),int(center[1])), 10, (0, 255, 0), 3)

                #print("Center:", center)
                KF_list[j].Z = np.array([[center[0]], [center[1]], [weighted_depths[j]]])
                #print("Shape of Z: ", np.shape(KF_list[j].Z), KF_list[j].Z)

                KF_list[j].update()
                KF_list[j].predict()

                #print("Output from KF: ",KF_list[j].X)

                circle_center_coords = (int(KF_list[j].X[0]), int(KF_list[j].X[1]))
                #print("circle center", circle_center_coords)
                print("Predicted distances", KF_list[j].X[2])
                print("Measured distances", weighted_depths[j])
                cv2.circle(frame_data["left_image"], circle_center_coords, max(int(300/KF_list[j].X[2]),5), (0,0,255), 3)

                #x = KF_list.Xpred
                #P = KF_list.Ppred
            cv2.imshow("frame",frame_data["left_image"])
            cv2.waitKey(0)


        #for w in range(len(new_KF_list)):
            #print("KF LIST CLASS NAMES:", w, new_KF_list[w].class_name)

        previous_class_names = class_names

        #print("CLASS NAMES: ", class_names)





    print("Hello, World!")

