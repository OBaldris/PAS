import numpy as np


class KalmanFilter:
    def __init__(self):
        self.X = None
        self.F = None
        self.P = None
        self.u = None
        self.H = None
        self.R = None

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
        # x = [x, y, z, dx, dy, dz, ddx, ddy, ddz]
        # Initially everything is 0.
        self.X = np.zeros((9, 1))
        print("Shape of x: ", np.shape(self.X), "(Should be 9x1)")

        # The initial uncertainty (9x9).
        self.P = np.identity(9) * 1  # The "*1" is a factor. Tweak if needed.
        print("Shape of P: ", np.shape(self.P), "(Should be 9x9)")

        # The external motion (9x1).
        self.u = np.transpose(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        print("Shape of u: ", np.shape(self.u), "(Should be 9x1)")

        # The transition matrix (9x9).
        self.F = np.array([[1, 0, 0, self.dt, 0, 0, 0.5 * pow(self.dt, 2), 0, 0],
                      [0, 1, 0, 0, self.dt, 0, 0, 0.5 * pow(self.dt, 2), 0],
                      [0, 0, 1, 0, 0, self.dt, 0, 0, 0.5 * pow(self.dt, 2)],
                      [0, 0, 0, 1, 0, 0, self.dt, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, self.dt, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, self.dt],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        print("Shape of F: ", np.shape(self.F), "(Should be 9x9)")

        # The observation matrix (3x9).
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        print("Shape of H: ", np.shape(self.H), "(Should be 3x9)")

        # The measurement uncertainty.
        self.R = 3.5  # I dunno

    def update(self):
        self.error = self.Z - self.H @ self.X
        Ht = np.transpose(self.H)
        self.S = self.H @ self.P @ Ht + self.R

        if np.linalg.det(self.S) != 0:
            self.K = self.P @ Ht @ np.linalg.inv(self.S)
        else:
            self.K = self.P @ Ht @ np.array([[0.000001, 0.000001], [0.000001, 0.000001]])
        I = np.identity(6)  # 6 because our system is 6-dimensional

        self.Xpred = self.X + self.K @ self.error
        self.Ppred = (I - self.K @ self.H) @ self.P

    def predict(self):
        self.Xpred = self.F @ self.X + self.u
        self.Ppred = self.F @ self.P @ np.transpose(self.F)



