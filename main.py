import numpy as np
import cv2
from ultralytics import YOLO
import load_data
import kurac6
import Kalman
import matplotlib.pyplot as plt

def get_center(x1, y1, x2, y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)

def dist3D(pred_x, pred_y, pred_z, act_pos):
    return np.sqrt((pred_x-act_pos[0])**2 + (pred_y-act_pos[1])**2 + (pred_z-act_pos[2])**2)

def within_borders(pixel_x, pixel_y, img):
    margin = 5
    height = img.shape[0]
    width = img.shape[1]
    if 0 < pixel_x < width and 0 < pixel_y < height:
        if (pixel_x-margin) > 0 and (pixel_y-margin) > 0 and (pixel_x+margin) < width and (pixel_y+margin) < height:
            return True
    return False

def get_depth_from_bbox(depth_calculator, results):
    class_names = []
    confidences = []
    centers = []
    world_coords = []

    if len(results) == 0 or len(results[0]) == 0:
        return None, None, None, None

    for result in results[0]:
        box = result.boxes.xyxy[0]
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)

        center_coords = get_center(x1, y1, x2, y2)
        centers.append(center_coords)

        cls_idx = int(result.boxes.cls.item())
        class_name = results[0].names[cls_idx]
        class_names.append(class_name)
        
        confidence = result.boxes.conf.item()
        confidences.append(confidence)

        new_x1 = round(((x1 + x2) / 2) * 0.9)
        new_x2 = round(((x1 + x2) / 2) * 1.1)
        new_y1 = round(((y1 + y2) / 2) * 0.75)
        new_y2 = round(((y1 + y2) / 2) * 1.33)

        coords_data = depth_calculator.calculate_weighted_depth((new_x1, new_y1), (new_x2, new_y2))
        
        if coords_data and coords_data[0] and coords_data[0][0] is not None:
            world_coords.append(coords_data[0][0])
        else:
            world_coords.append((0, 0, 0))

    return class_names, confidences, world_coords, centers

def main():
    calibration_data_path = "calib_cam_to_cam.txt"
    data = load_data.StereoVisionData()
    model = YOLO("yolov8n.pt")
    class_indexes = [0, 1, 2]  # Person, Bicycle, Car indexes
    margin = 1
    
    KF_list = []

    # Colors for visualization (BGR for OpenCV)
    colors = {
        'person': {'pred': (255, 0, 0), 'act': (255, 127, 127)},
        'bicycle': {'pred': (0, 0, 255), 'act': (127, 127, 255)},
        'car': {'pred': (0, 255, 0), 'act': (127, 255, 127)}
    }

    # Colors for matplotlib (RGB)
    plot_colors = {
        'person': {'pred': (0, 0, 1), 'act': (0.5, 0.5, 1)},
        'bicycle': {'pred': (1, 0, 0), 'act': (1, 0.5, 0.5)},
        'car': {'pred': (0, 1, 0), 'act': (0.5, 1, 0.5)}
    }

    plt.ion()
    data.load_sequence(sequence_num=3)

    for i in range(1, 208):
        frame_data = data.get_frame_data(sequence_num=3, frame_index=i)
        results = model(frame_data["left_image"], classes=class_indexes, conf=0.4)

        DepthObj = kurac6.DepthCalculator()
        DepthObj.run(frame_data["left_image"], frame_data["right_image"], 
                    calibration_data_path, input_is_array=True)

        class_names, confidences, weighted_depths, centers = get_depth_from_bbox(DepthObj, results)

        if class_names is not None:
            new_KF_list = [None] * len(class_names)
            KF_taken = []
            
            pred_plot_list = []
            act_plot_list = []
            pred_colors_list = []
            act_colors_list = []

            # Match detections with existing filters
            for j in range(len(class_names)):
                min_dist = float('inf')
                KF_match_index = -1
                
                # Find matching filter
                for q in range(len(KF_list)):
                    if not KF_list:
                        break
                    if q in KF_taken or KF_list[q].class_name != class_names[j]:
                        continue

                    predicted_pos = [KF_list[q].X[0, 0], KF_list[q].X[1, 0], KF_list[q].X[2, 0]]
                    predicted_pixel_x, predicted_pixel_y = DepthObj.world_to_pixel(
                        predicted_pos[0], predicted_pos[1], predicted_pos[2])

                    if not within_borders(predicted_pixel_x, predicted_pixel_y, frame_data["left_image"]):
                        continue

                    dist = dist3D(predicted_pos[0], predicted_pos[1], predicted_pos[2], weighted_depths[j])
                    if dist < min_dist and dist < margin:
                        min_dist = dist
                        KF_match_index = q

                # Create or update Kalman filter
                if KF_match_index != -1:
                    new_KF_list[j] = KF_list[KF_match_index]
                    KF_taken.append(KF_match_index)
                else:
                    new_KF_list[j] = Kalman.KalmanFilter3D()
                    new_KF_list[j].class_name = class_names[j]
                    new_KF_list[j].X = np.array([[weighted_depths[j][0]], 
                                               [weighted_depths[j][1]], 
                                               [weighted_depths[j][2]], 
                                               [0], [0], [0], [0], [0], [0]])

            KF_list = new_KF_list.copy()

            # Process each detection
            for j in range(len(KF_list)):
                if KF_list[j] is None:
                    continue
                    
                center = centers[j]
                class_name = KF_list[j].class_name.lower()
                
                # Draw detection
                cv2.circle(frame_data["left_image"], 
                         (int(center[0]), int(center[1])), 
                         20, 
                         colors[class_name]['act'], 
                         3)

                # Update Kalman filter
                KF_list[j].Z = np.array([[weighted_depths[j][0]], 
                                       [weighted_depths[j][1]], 
                                       [weighted_depths[j][2]]])
                
                KF_list[j].predict()

                # Draw prediction
                pixel_x, pixel_y = DepthObj.world_to_pixel(
                    KF_list[j].X[0, 0], KF_list[j].X[1, 0], KF_list[j].X[2, 0])
                
                circle_radius = max(int(50 / (KF_list[j].X[2, 0] + 1)), 5)
                cv2.circle(frame_data["left_image"], 
                         (int(pixel_x), int(pixel_y)), 
                         circle_radius,
                         colors[class_name]['pred'], 
                         3)

                # Store plotting data - using pixel x-coordinate instead of world x-coordinate
                pred_plot_list.append([float(pixel_x), float(KF_list[j].X[2, 0])])
                act_plot_list.append([float(center[0]), float(weighted_depths[j][2])])
                pred_colors_list.append(plot_colors[class_name]['pred'])
                act_colors_list.append(plot_colors[class_name]['act'])

                KF_list[j].update()

            # Update visualization
            plt.clf()
            
            # Add legend entries
            for class_name in ['person', 'bicycle', 'car']:
                plt.scatter([], [], c=[plot_colors[class_name]['pred']], 
                          label=f'{class_name} pred')
                plt.scatter([], [], c=[plot_colors[class_name]['act']], 
                          label=f'{class_name} actual')
            
            # Plot current frame data
            if pred_plot_list:
                plt.scatter([x[0] for x in pred_plot_list], [x[1] for x in pred_plot_list], 
                           c=pred_colors_list)
                plt.scatter([x[0] for x in act_plot_list], [x[1] for x in act_plot_list], 
                           c=act_colors_list)
            
            plt.xlabel('X position (pixels)')
            plt.ylabel('Z coordinate (meters)')
            plt.title('Object Tracking: Predicted vs Actual Positions')
            plt.legend()
            plt.grid(True)
            plt.pause(0.001)

            cv2.imshow("Detection", frame_data["left_image"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main()