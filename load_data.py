import cv2
import pandas as pd
import os

class StereoVisionData:
    def __init__(self):
        # Dictionary to store data paths for each sequence
        self.sequences = {}
        
    def load_sequence(self, sequence_num):
        # Define the sequence folder name with zero-padding if needed
        #"C:\Users\gust0\OneDrive - Danmarks Tekniske Universitet\Perception_for_autonomous_sys\Final_project\34759_final_project_rect\34759_final_project_rect\seq_01"
        sequence_folder = f"34759_final_project_rect/34759_final_project_rect/seq_{str(sequence_num).zfill(2)}"
        
        # Construct paths relative to the script's location
        left_image_path = f"{sequence_folder}/image_02/data/"
        right_image_path = f"{sequence_folder}/image_03/data/"
        left_timestamps_path = f"{sequence_folder}/image_02/timestamps.txt"
        right_timestamps_path = f"{sequence_folder}/image_03/timestamps.txt"
        labels_path = f"{sequence_folder}/labels.txt"
        
        # Debug print statements to verify paths
        #print("Loading sequence paths:")
        #print("Left Image Path:", left_image_path)
        #print("Right Image Path:", right_image_path)
        #print("Left Timestamps Path:", left_timestamps_path)
        #print("Right Timestamps Path:", right_timestamps_path)
        #print("Labels Path:", labels_path)

        # Load timestamps and ensure no duplicate rows are introduced
        left_timestamps = self._load_timestamps(left_timestamps_path)
        right_timestamps = self._load_timestamps(right_timestamps_path)

        # Load labels without duplicates
        labels = self._load_labels(labels_path) if self._file_exists(labels_path) else None

        # Store sequence data in a dictionary
        self.sequences[sequence_num] = {
            "left_image_path": left_image_path,
            "right_image_path": right_image_path,
            "left_timestamps": left_timestamps,
            "right_timestamps": right_timestamps,
            "labels": labels
        }
    
    def _load_timestamps(self, timestamps_path):
        # Load timestamps from a text file, ensuring each line is unique
        with open(timestamps_path, 'r') as file:
            timestamps = [line.strip() for line in file]
        print(f"Loaded {len(timestamps)} unique timestamps from {timestamps_path}")
        return timestamps

    def _load_labels(self, labels_path):
        # Load labels into a DataFrame with duplicate removal
        column_names = ["frame", "track_id", "type", "truncated", "occluded", "alpha",
                        "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
                        "dim_height", "dim_width", "dim_length",
                        "loc_x", "loc_y", "loc_z",
                        "rotation_y", "score"]
        labels_df = pd.read_csv(labels_path, delim_whitespace=True, header=None, names=column_names)

        # Remove duplicates if they exist in the DataFrame
        initial_rows = len(labels_df)
        labels_df.drop_duplicates(inplace=True)
        final_rows = len(labels_df)
        if initial_rows != final_rows:
            print(f"Removed {initial_rows - final_rows} duplicate rows from labels.")

        return labels_df

    def _file_exists(self, path):
        # Check if a file exists at the given path
        try:
            with open(path, 'r'):
                print(f"File exists: {path}")
                return True
        except FileNotFoundError:
            print(f"File not found: {path}")
            return False

    def get_frame_data(self, sequence_num, frame_index):
        # Retrieve specific frame data for a given sequence
        sequence = self.sequences.get(sequence_num)
        if not sequence:
            raise ValueError(f"Sequence {sequence_num} not loaded.")

        # Construct image paths for the specific frame using zero-padded format
        left_image_path = f"{sequence['left_image_path']}{str(frame_index).zfill(6)}.png"
        right_image_path = f"{sequence['right_image_path']}{str(frame_index).zfill(6)}.png"
        
        # Load left and right images
        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)
        left_timestamp = sequence["left_timestamps"][frame_index]
        right_timestamp = sequence["right_timestamps"][frame_index]
        
        # Retrieve labels for this frame if they exist
        frame_labels = None
        if sequence["labels"] is not None:
            frame_labels = sequence["labels"][sequence["labels"]["frame"] == frame_index]
        
        return {
            "left_image": left_image,
            "right_image": right_image,
            "left_timestamp": left_timestamp,
            "right_timestamp": right_timestamp,
            "labels": frame_labels
        }

# Example usage
data = StereoVisionData()

# Load a specific sequence
data.load_sequence(sequence_num=1)

# If you want to load multiple sequences
#sequence_numbers = [1, 2, 3]  # List of sequence numbers to load
#for seq_num in sequence_numbers:
#    data.load_sequence(seq_num)


# Access a specific frame's data
frame_data = data.get_frame_data(sequence_num=1, frame_index=0)

# Display frame details to verify that the output reflects labels and timestamps accurately
#print("Left Image File:", f"{data.sequences[1]['left_image_path']}{str(0).zfill(6)}.png")
#print("Right Image File:", f"{data.sequences[1]['right_image_path']}{str(0).zfill(6)}.png")
#print("Left Timestamp:", frame_data["left_timestamp"])
#print("Right Timestamp:", frame_data["right_timestamp"])
#print("Labels:", frame_data["labels"])

