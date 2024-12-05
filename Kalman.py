# Kalman.py
import numpy as np

class KalmanFilter3D:
    def __init__(self, class_name, dt=1/30.0):
        """
        Initialize the Kalman Filter with a constant velocity model.
        
        Parameters:
            class_name (str): The class name of the object (e.g., 'person', 'bicycle', 'car').
            dt (float): Time step between frames.
        """
        self.class_name = class_name
        self.dt = dt  # Time step
        self.init_arrays()
        
        # Track management attributes
        self.missed_frames = 0
        self.confirmed = False
        self.history = []
        self.max_missed = 3  # Increased from 5 to 10
        self.confirm_threshold = 2
        self.id = None  # To be set externally for unique identification

    def init_arrays(self):
        """
        Initialize the state vector, state transition matrix, measurement matrix,
        process noise covariance (Q), measurement noise covariance (R),
        and state covariance matrix (P).
        """
        # State vector: [x, y, z, vx, vy, vz]
        self.X = np.zeros((6, 1))
        
        # State transition matrix with constant velocity model
        self.F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix: we only measure positions [x, y, z]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        q_pos = 0.1  # Position process noise
        q_vel = 1.0  # Velocity process noise
        self.Q = np.zeros((6, 6))
        self.Q[:3, :3] = np.eye(3) * q_pos * self.dt**4 / 4
        self.Q[3:, 3:] = np.eye(3) * q_vel * self.dt**2
        
        # Measurement noise covariance
        self.R = np.eye(3)
        self.R[0, 0] = 0.05  # x measurement noise
        self.R[1, 1] = 0.05  # y measurement noise
        self.R[2, 2] = 0.1   # z measurement noise
        
        # Initial state covariance
        self.P = np.eye(6) * 10  # High uncertainty in initial state
        
        # Set margins based on class
        if self.class_name.lower() == 'person':
            self.margin = 1.5
        elif self.class_name.lower() == 'bicycle':
            self.margin = 2.0
        else:  # Default to 'car'
            self.margin = 2.5

    def predict(self):
        """
        Predict the next state and update the state covariance matrix.
        """
        # Predict state
        self.X = self.F @ self.X
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        Update the state vector and covariance matrix with a new measurement.
        
        Parameters:
            measurement (np.ndarray): The measurement vector [x, y, z].
        """
        if measurement is None:
            self.missed_frames += 1
            return
        
        measurement = measurement.reshape((3, 1))
        
        # Innovation
        y = measurement - self.H @ self.X
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.X = self.X + K @ y
        
        # Corrected covariance update using Joseph form
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        
        # Reset missed frames
        self.missed_frames = 0
        
        # Update history
        self.history.append(measurement.flatten())
        
        # Confirm the filter if enough measurements have been received
        if not self.confirmed and len(self.history) >= self.confirm_threshold:
            self.confirmed = True

    def miss(self):
        """
        Handle a missed detection (no measurement available).
        """
        self.missed_frames += 1

    def reset_miss_count(self):
        """Reset the missed frames counter to zero."""
        self.missed_frames = 0
    def delete(self):
        """Mark this track as deleted."""
        self.missed_frames = float('inf')

    def is_deleted(self):
        """
        Determine whether the track should be deleted based on missed frames.
        
        Returns:
            bool: True if the track should be deleted, False otherwise.
        """
        return self.missed_frames > self.max_missed
