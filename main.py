import numpy as np


class KalmanFilter:
    def __init__(self, X, F, P, u, H, R):
        self.X = X
        self.F = F
        self.P = P
        self.u = u
        self.H = H
        self.R = R

        self.Xpred = np.copy(self.X)
        self.Ppred = np.copy(self.P)
        self.Z = None
        self.error = None

        self.S = None
        self.K = None

        self.dt = 1

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

def init_arrays():

    dt = 1

    # The initial state (6x1).
    # x = [x, y, z, dx, dy, dz, ddx, ddy, ddz]
    # Initially everything is 0.
    x = np.zeros((9, 1))
    print("Shape of x: ", np.shape(x), "(Should be 9x1)")

    # The initial uncertainty (9x9).
    P = np.identity(9) * 1  # The "*1" is a factor. Tweak if needed.
    print("Shape of P: ", np.shape(P), "(Should be 9x9)")

    # The external motion (9x1).
    u = np.transpose(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    print("Shape of u: ", np.shape(u), "(Should be 9x1)")

    # The transition matrix (9x9).
    F = np.array([[1, 0, 0, dt,  0,  0, 0.5 * pow(dt, 2),                0,                0],
                  [0, 1, 0,  0, dt,  0,                0, 0.5 * pow(dt, 2),                0],
                  [0, 0, 1,  0,  0, dt,                0,                0, 0.5 * pow(dt, 2)],
                  [0, 0, 0, 1, 0, 0, dt,  0, 0],
                  [0, 0, 0, 0, 1, 0,  0, dt, 0],
                  [0, 0, 0, 0, 0, 1,  0,  0, dt],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]])
    print("Shape of F: ", np.shape(F), "(Should be 9x9)")

    # The observation matrix (3x9).
    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]])
    print("Shape of H: ", np.shape(H), "(Should be 3x9)")

    # The measurement uncertainty.
    R = 3.5  # I dunno




if __name__ == "__main__":
    print("Hello, World!")

    ### Initialize Kalman filter ###


    #Establish video, take frame by frame
    #Find object(s) of interest
    #Apply kalman filter(s?):

    ### If the ball is found, update the Kalman filter ###

    #Z = np.array([[circle_coords[0], 0, 0, 0, 0, 0], [0, circle_coords[1], 0, 0, 0, 0]])
    #x, P = update(x, P, Z, H, R)

    ### Predict the next state
    #x, P = predict(x, P, F, u)

    #Dont forget to draw results
    '''
    Draw the current tracked state and the predicted state on the image frame ###
    if np.shape(x) == (6,6):
       cv2.circle(frame, (int(x[0,0]), int(x[1,1])), radius, (255, 0, 0), 3)
    # Show the frame
    cv2.imshow('Frame', frame)
    '''













