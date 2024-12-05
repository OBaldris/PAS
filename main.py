import numpy as np
import cv2
from ultralytics import YOLO
import load_data
import kurac6
import Kalman
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def generate_unique_id():
    """Generator for unique track IDs."""
    current_id = 0
    while True:
        yield current_id
        current_id += 1

unique_id_generator = generate_unique_id()

def get_center(x1, y1, x2, y2):
    """Calculate the center coordinates of a bounding box."""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)

def within_borders(pixel_x, pixel_y, img, margin=5):
    """Check if the pixel coordinates are within the image borders with a margin."""
    height, width = img.shape[:2]
    return margin < pixel_x < (width - margin) and margin < pixel_y < (height - margin)

def get_depth_from_bbox(depth_calculator, results):
    """Extract class names, confidences, world coordinates, and centers from detection results."""
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

        # Adjusted bounding box for depth calculation
        new_x1 = round(((x1 + x2) / 2) * 0.9)
        new_x2 = round(((x1 + x2) / 2) * 1.1)
        new_y1 = round(((y1 + y2) / 2) * 0.75)
        new_y2 = round(((y1 + y2) / 2) * 1.33)

        coords_data = depth_calculator.calculate_weighted_depth((new_x1, new_y1), (new_x2, new_y2))
        
        if coords_data and coords_data[0] and coords_data[0][0] is not None:
            world_coords.append(np.array(coords_data[0][0]))
        else:
            world_coords.append(np.array([0, 0, 0]))

    return class_names, confidences, world_coords, centers

def associate_detections_to_tracks(kf_list, detections, matching_threshold):
    """Associate detections to existing tracks using the Hungarian algorithm."""
    if len(kf_list) == 0:
        return [], [], list(range(len(detections)))

    if len(detections) == 0:
        return [], list(range(len(kf_list))), []

    # Compute cost matrix (Euclidean distance between predicted and detected positions)
    cost_matrix = np.zeros((len(kf_list), len(detections)))
    for i, kf in enumerate(kf_list):
        predicted_pos = kf.X[:3].flatten()
        for j, det in enumerate(detections):
            cost_matrix[i, j] = np.linalg.norm(predicted_pos - det)

    # Perform Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(len(kf_list)))
    unmatched_detections = set(range(len(detections)))

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < matching_threshold:
            matches.append((i, j))
            unmatched_tracks.discard(i)
            unmatched_detections.discard(j)

    return matches, list(unmatched_tracks), list(unmatched_detections)

def main():
    # Initialize components
    calibration_data_path = "calib_cam_to_cam.txt"
    data = load_data.StereoVisionData()
    model = YOLO("yolo11n.pt")
    class_indexes = [0, 1, 2]  # Person, Bicycle, Car indexes
    KF_list = []
    matching_threshold = 3.0  # meters
    max_missed_frames = 5  # Number of frames a track can be missed before deletion

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

    # Initialize the depth calculator
    DepthObj = kurac6.DepthCalculator()

    # Track ID to class mapping for consistent visualization
    track_id_to_class = {}

    for i in range(1, 208):
        frame_data = data.get_frame_data(sequence_num=3, frame_index=i)
        results = model(frame_data["left_image"], classes=class_indexes, conf=0.4)

        DepthObj.run(frame_data["left_image"], frame_data["right_image"], 
                     calibration_data_path, input_is_array=True)

        class_names, confidences, weighted_depths, centers = get_depth_from_bbox(DepthObj, results)

        if class_names is not None:
            detections = np.array(weighted_depths)  # Shape: (N, 3)
        else:
            detections = np.array([]).reshape(0, 3)

        # Predict all tracks
        for kf in KF_list:
            kf.predict()

        # Associate detections to tracks
        matches, unmatched_tracks, unmatched_detections = associate_detections_to_tracks(KF_list, detections, matching_threshold)

        # Update matched tracks with assigned detections
        for (track_idx, det_idx) in matches:
            measurement = detections[det_idx] if len(detections) > det_idx else None
            if measurement is not None:
                KF_list[track_idx].update(measurement)
                if class_names and len(class_names) > det_idx:
                    track_id_to_class[KF_list[track_idx].id] = class_names[det_idx].lower()
            else:
                KF_list[track_idx].miss()

        # Handle unmatched detections: create new tracks
        for det_idx in unmatched_detections:
            if len(detections) > det_idx:
                class_name = class_names[det_idx].lower()
                new_kf = Kalman.KalmanFilter3D(class_name=class_name)
                new_kf.X[:3] = detections[det_idx].reshape(3, 1)
                new_kf.history.append(detections[det_idx])
                new_kf.id = next(unique_id_generator)
                KF_list.append(new_kf)
                track_id_to_class[new_kf.id] = class_name

        # Handle unmatched tracks: increase missed frames
        for track_idx in unmatched_tracks:
            KF_list[track_idx].miss()
            if KF_list[track_idx].missed_frames > max_missed_frames:
                KF_list[track_idx].delete()

        # Remove deleted tracks
        KF_list = [kf for kf in KF_list if not kf.is_deleted()]

        # Visualization and plotting
        pred_plot_list = []
        act_plot_list = []
        pred_colors_list = []
        act_colors_list = []

        for kf in KF_list:
            class_name = track_id_to_class.get(kf.id, 'car')  # Default to 'car' if not found
            color_pred = plot_colors[class_name]['pred']
            color_act = plot_colors[class_name]['act']
            
            # Draw predicted position
            predicted_pos = kf.X[:3].flatten()
            pixel_x, pixel_y = DepthObj.world_to_pixel(
                predicted_pos[0], predicted_pos[1], predicted_pos[2])
            
            # Scale circle size with depth
            circle_radius = max(int(50 / (predicted_pos[2] + 1)), 5)
            cv2.circle(frame_data["left_image"], 
                       (int(pixel_x), int(pixel_y)), 
                       circle_radius,
                       colors[class_name]['pred'], 
                       3)
            
            pred_plot_list.append([float(pixel_x), float(predicted_pos[2])])
            pred_colors_list.append(color_pred)
            
            # Draw actual detection if available
            if kf.confirmed and len(kf.history) > 0:
                last_measurement = kf.history[-1]
                actual_pixel_x, actual_pixel_y = DepthObj.world_to_pixel(
                    last_measurement[0], last_measurement[1], last_measurement[2])
                
                cv2.circle(frame_data["left_image"],
                           (int(actual_pixel_x), int(actual_pixel_y)),
                           20,
                           colors[class_name]['act'],
                           3)
                
                act_plot_list.append([float(actual_pixel_x), float(last_measurement[2])])
                act_colors_list.append(color_act)

        # Plotting
        plt.clf()
        
        # Add legend entries
        for class_name in ['person', 'bicycle', 'car']:
            plt.scatter([], [], c=[plot_colors[class_name]['pred']], 
                        label=f'{class_name.capitalize()} Predicted')
            plt.scatter([], [], c=[plot_colors[class_name]['act']], 
                        label=f'{class_name.capitalize()} Actual')
        
        plt.xlim(-5, 1200)
        plt.ylim(-5, 20)
        
        # Plot current frame data
        if pred_plot_list and act_plot_list:
            pred_x = [x[0] for x in pred_plot_list]
            pred_z = [x[1] for x in pred_plot_list]
            act_x = [x[0] for x in act_plot_list]
            act_z = [x[1] for x in act_plot_list]
            
            plt.scatter(pred_x, pred_z, c=pred_colors_list, label='Predicted')
            plt.scatter(act_x, act_z, c=act_colors_list, label='Actual')
        
        plt.xlabel('X position (pixels)')
        plt.ylabel('Z coordinate (meters)')
        plt.title(f'Object Tracking: Frame {i}')
        plt.legend()
        plt.grid(True)
        plt.pause(0.001)

        # Display the frame with detections and predictions
        cv2.imshow("Detection", frame_data["left_image"])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main()