import numpy as np
import cv2
import glob
import load_data
import kurac6
from ultralytics import YOLO
import Kalman
import matplotlib.pyplot as plt


def is_valid(Filter_list, Class_list, Taken_list, class_index, filter_index):
    if not Filter_list:
        return False
    if Filter_list[filter_index].class_name != Class_list[class_index]:
        return False
    if filter_index in Taken_list:
        return False
    return True

def within_margin(pred_x, pred_z, act_pos):
    margin = 0.5
    if np.sqrt((pred_x-act_pos[0])**2 + (pred_z-act_pos[2])**2) <= margin:
        return True
    return False

def within_borders(pixel_x, pixel_y, img):
    margin = 5
    height = img.shape[0]
    width = img.shape[1]
    print("img shape:", img.shape)
    if (pixel_x-margin) > 0 or (pixel_y-margin) > 0 or (pixel_x+margin) < width or (pixel_y+margin) < height:
        return True
    return False

def get_center(x1, y1, x2, y2):
    center_x = (x1+x2) / 2
    center_y = (y1+y2) / 2

    return (center_x, center_y)

def dist2D(pred_x, pred_z, act_pos):
    return np.sqrt((pred_x-act_pos[0])**2 + (pred_z-act_pos[2])**2)

def get_depth_from_bbox(depth_calculator, results):
    class_names = []
    confidences = []
    centers = []
    world_coords = []

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
    calibration_data_path = "C:/Users/gust0/OneDrive - Danmarks Tekniske Universitet/Perception_for_autonomous_sys/Final_project/34759_final_project_rect/34759_final_project_rect/calib_cam_to_cam.txt"


    # Example usage
    data = load_data.StereoVisionData()

    model = YOLO("yolo11n.pt")
    #class_indexes = [0, 1, 2]   #Person, Bicycle, Car indexes in YOLO model :)
    class_indexes = [0]

    KF_list = []
    previous_class_names = []

    pred_plot_list = []
    act_plot_list = []

    color11 = (0.1, 0.1, 0.4, 1.0)
    color12 = (0.2, 0.2, 0.8, 1.0)
    colormap1 = np.array([color11, color12])

    color21 = (0.4, 0.1, 0.1, 1.0)
    color22 = (0.8, 0.2, 0.2, 1.0)
    colormap2 = np.array([color21, color22])

    # Load a specific sequence
    data.load_sequence(sequence_num=1)

    for i in range(0, 144):
        # Access a specific frame's data
        frame_data = data.get_frame_data(sequence_num=1, frame_index=i)
        results = model(frame_data["left_image"], classes=class_indexes)

        DepthObj = kurac6.DepthCalculator()
        DepthObj.run(frame_data["left_image"], frame_data["right_image"], calibration_data_path, input_is_array=True)

        class_names, confidences, weighted_depths, centers = get_depth_from_bbox(DepthObj, results)
        #print("weighted_depths", weighted_depths)
        if class_names is not None:
            new_KF_list = [None]*len(class_names)


            KF_taken = []
            for j in range(len(class_names)):
                min_dist = 99999
                KF_match_index = -1
                center = centers[j]
                for q in range(len(KF_list)):
                    if not KF_list:
                        continue
                    if KF_list[q].class_name != class_names[j]:
                        continue
                    if q in KF_taken:
                        continue

                    predicted_pos = [KF_list[q].X[0], KF_list[q].X[1]]

                    #if not within_margin(predicted_pos[0], predicted_pos[1], weighted_depths[j]):
                        #print("KF NR ", q, " IS NOT WITHIN MARGIN?")
                        #continue

                    dist = dist2D(predicted_pos[0], predicted_pos[1], weighted_depths[j])
                    if dist < min_dist and dist < 10:
                        min_dist = dist
                        KF_match_index = q
                        #print("MATCH     j:", j, ", q:", q)

                if KF_match_index != -1:
                    new_KF_list[j] = KF_list[KF_match_index]
                    KF_taken.append(KF_match_index)

                else:
                    #new_KF_list[j] = Kalman.KalmanFilter3D()
                    new_KF_list[j] = Kalman.KalmanFilter2D()
                    new_KF_list[j].class_name = class_names[j]
                    new_KF_list[j].X = np.array([[weighted_depths[j][0]], [weighted_depths[j][2]], [0], [0], [0], [0]])
                    #print("new_KF_list[j].X", new_KF_list[j].X)

            KF_list = new_KF_list.copy()
            #print("len(KF_list)", len(KF_list))

            for j in range(len(KF_list)):
                center = centers[j]
                # draw detection
                cv2.circle(frame_data["left_image"], (int(center[0]),int(center[1])), 20, (0, 255, 0), 3)

                KF_list[j].Z = np.array([[weighted_depths[j][0]], [weighted_depths[j][2]]])

                #print("Predicted placement", KF_list[j].X[0], " ", KF_list[j].X[1])
                pixel_x, pixel_y = DepthObj.world_to_pixel(KF_list[j].X[0, 0], weighted_depths[j][1],
                                                           KF_list[j].X[1, 0])
                KF_list[j].predict()

                # MASSIVE DRIFT CAUSED BY DIFFERENCE IN UNITS!!!
                # WE USE PIXELS FOR x, y BUT METERS FOR z!!!

                #pixel_x, pixel_y = DepthObj.world_to_pixel(KF_list[j].X[0,0], weighted_depths[j][1], KF_list[j].X[1,0])

                pred_plot_list.append([float(KF_list[j].X[0, 0]), float(KF_list[j].X[1, 0])])
                act_plot_list.append([float(weighted_depths[j][0]), float(weighted_depths[j][2])])

                circle_center_coords = (pixel_x, pixel_y)#(int(KF_list[j].X[0]), int(center[1]))

                #print("Actual placement", weighted_depths[j][0], " ", weighted_depths[j][2])
                cv2.circle(frame_data["left_image"], circle_center_coords, max(int(50 / (KF_list[j].X[1] + 1)), 5), (0, 0, 255), 3)

                KF_list[j].update()

            #print("pred_plot_list", pred_plot_list)
            #print("pred_plot_list[:][0]", [x[0] for x in pred_plot_list])

            categories = [0]*len(pred_plot_list)
            print(categories)
            for g in range(len(KF_list)):
                categories[-g-1] = 1

            print(categories)

            plt.scatter([x[0] for x in pred_plot_list], [x[1] for x in pred_plot_list], c=colormap1[categories])
            plt.scatter([x[0] for x in act_plot_list], [x[1] for x in act_plot_list], c=colormap2[categories])
            plt.show()

            cv2.imshow("frame",frame_data["left_image"])
            cv2.waitKey(0)






    print("Hello, World!")


