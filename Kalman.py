import numpy as np


class KalmanFilter3D:
    def __init__(self):
        self.X = None
        self.F = None
        self.P = None
        self.u = None
        self.H = None
        self.R = None

        #self.init_arrays()

        self.Xpred = np.copy(self.X)
        self.Ppred = np.copy(self.P)
        self.Z = None
        self.error = None

        self.S = None
        self.K = None

        self.dt = None
        self.class_name = None
        self.margin = 0

    def init_arrays(self):
        self.dt = 0.2#0.25#0.04

        # The initial state (9x1).
        # x = [x, y, z, dx, dy, dz, ddx, ddy, ddz]
        # Initially everything is 0.
        self.X = np.zeros((9, 1))
        #print("Shape of x: ", np.shape(self.X), "(Should be 9x1)")

        # The initial uncertainty (9x9).
        #self.P = np.identity(9) * 0.5  # The "*1" is a factor. Tweak if needed.
        # self.P = np.identity(9)*1000
        if self.class_name == "person":
            #1.5 is chosen because 1.42 meters/second is roughly the average speed of a walking person.
            #This is then squared, because it is the covariance - which relies on the units of whatever you're measuring, but squared.
            self.P = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1.42**2, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1.42**2, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]])
            self.margin = 2
            # The measurement uncertainty.
            self.R = 0.5  # 74#75 #0.005#05  # I dunno

        if self.class_name == "bicycle":
            # 2.7 is chosen because 2.7 meters/second is about 10km/h
            # This is then squared, because it is the covariance - which relies on the units of whatever you're measuring, but squared.
            self.P = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 2.7**2, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 2.7**2, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0.1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0.1]])
            self.margin = 3
            # The measurement uncertainty.
            self.R = 0.5  # 74#75 #0.005#05  # I dunno

        if self.class_name == "car":
            # 2.7 is chosen because 2.7 meters/second is about 10km/h
            # This is then squared, because it is the covariance - which relies on the units of whatever you're measuring, but squared.
            self.P = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 2.7**2, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 2.7**2, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0.1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0.1]])
            self.margin = 3
            # The measurement uncertainty.
            self.R = 0.5  # 74#75 #0.005#05  # I dunno


        #print("Shape of P: ", np.shape(self.P), "(Should be 9x9)")

        # The external motion (9x1).
        self.u = np.transpose(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        #print("Shape of u: ", np.shape(self.u), "(Should be 9x1)")

        # The transition matrix (9x9).
        self.F = np.array([[1, 0, 0, self.dt, 0, 0, 0.5 * pow(self.dt, 2), 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, self.dt, 0, 0, 0.5 * pow(self.dt, 2)],
                      [0, 0, 0, 1, 0, 0, self.dt, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, self.dt],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        #print("Shape of F: ", np.shape(self.F), "(Should be 9x9)")

        # The observation matrix (3x9).
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        #print("Shape of H: ", np.shape(self.H), "(Should be 3x9)")


    def update(self):
        self.error = self.Z - self.H @ self.X
        #print("self.error", self.error)
        Ht = np.transpose(self.H)
        self.S = self.H @ self.P @ Ht + self.R
        #print("self.S", self.S)

        if np.linalg.det(self.S) != 0:
            self.K = self.P @ Ht @ np.linalg.inv(self.S)
        else:
            eps = 1e-10 * np.eye(self.S.shape[0])
            self.K = self.P @ Ht @ np.linalg.inv(self.S + eps)
        I = np.identity(9)  # 9 because our system is 9-dimensional

        #print("self.K", self.K)
        #print("udregningen:", self.K @ self.error)

        self.X = self.X + self.K @ self.error
        self.P = (I - self.K @ self.H) @ self.P

    def predict(self):
        #print("F", self.F)
        #print("u", self.u)
        self.X = self.F @ self.X + self.u
        self.P = self.F @ self.P @ np.transpose(self.F)

class KalmanFilter2D:
    def __init__(self):
        self.X = None
        self.F = None
        self.P = None
        self.u = None
        self.H = None
        self.R = None

        self.speed_x = 0
        self.speed_z = 0

        self.init_arrays()

        self.Xpred = np.copy(self.X)
        self.Ppred = np.copy(self.P)
        self.Z = None
        self.error = None

        self.S = None
        self.K = None

        self.dt = 1
        self.class_name = None

    def init_arrays(self):
        self.dt = 1

        # The initial state (6x1).
        # x = [x, z, dx, dz, ddx, ddz]
        # Initially everything is 0.
        #self.X = np.zeros((6, 1))
        self.H = np.array([[0], [0], [0], [0], [0], [0]])

        #print("Shape of x: ", np.shape(self.X), "(Should be 6x1)")

        # The initial uncertainty (6x6).
        # self.P = np.ones((6,6))*0.1#0.001
        self.P = np.identity(6) * 0.6


        # The external motion (6x1).
        self.u = np.transpose(np.array([[0, 0, 0, 0, 0, 0]]))
        #print("Shape of u: ", np.shape(self.u), "(Should be 6x1)")

        # The transition matrix (6x6).
        self.F = np.array([[1, 0, self.dt, 0, 0.5 * pow(self.dt, 2), 0],
                           [0, 1, 0, self.dt, 0, 0.5 * pow(self.dt, 2)],
                           [0, 0, 1, 0, self.dt, 0],
                           [0, 0, 0, 1, 0, self.dt],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        #print("Shape of F: ", np.shape(self.F), "(Should be 6x6)")

        # The observation matrix (2x6).
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        #print("Shape of H: ", np.shape(self.H), "(Should be 2x6)")

        # The measurement uncertainty.
        self.R = 0.01 #0.01  # I dunno

    def update(self):
        self.error = self.Z - self.H @ self.X
        print("self.Z", self.Z)
        Ht = np.transpose(self.H)
        self.S = self.H @ self.P @ Ht + self.R
        #print("self.S", self.S)

        if np.linalg.det(self.S) != 0:
            self.K = self.P @ Ht @ np.linalg.inv(self.S)
        else:
            eps = 1e-10 * np.eye(self.S.shape[0])
            self.K = self.P @ Ht @ np.linalg.inv(self.S + eps)
        I = np.identity(6)  # 6 because our system is 6-dimensional

        #print("self.K", self.K)
        #print("udregningen:", self.K @ self.error)

        self.X = self.X + self.K @ self.error
        self.P = (I - self.K @ self.H) @ self.P

    def predict(self):
        #print("F", self.F)
        #print("u", self.u)
        self.X = self.F @ self.X + self.u
        self.P = self.F @ self.P @ np.transpose(self.F)
        #print("X", self.X)

    def get_predict(self):
        #print("F", self.F)
        #print("u", self.u)
        X = self.F @ self.X + self.u

        return X



