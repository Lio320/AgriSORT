import numpy as np


class KalmanFilter():
    def __init__(self, dt, state, state_std, meas_std, color, id):
        self.dt = dt
        self.id = id

        self.last_seen = 0

        self.display = False

        self.age = 0
        self.temp = True

        self.color = color

        # Fix initial state
        self.x = np.array([state[0], state[1], 0, 0, state[2], state[3]])

        # State matrix
        self.A = np.array([[1, 0, self.dt, 0, 0, 0],
                           [0, 1, 0, self.dt, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        # No control matrix
        # self.B = 0
        # self.u = 0

        # Process noise (state prediction)
        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0, 0, 0],
                           [0, (self.dt**4)/4, 0, (self.dt**3)/2, 0, 0],
                           [(self.dt**3)/2, 0, self.dt**2, 0, 0, 0],
                           [0, (self.dt**3)/2, 0, self.dt**2, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]]) * state_std**2

        # State to measurements
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        # Measurement noise
        self.R = np.array([[meas_std**2, 0, 0, 0],
                           [0, meas_std**2, 0, 0],
                           [0, 0, meas_std**2, 0],
                           [0, 0, 0, meas_std**2]])
        # State covariance matrix
        self.P = np.array(np.eye(self.A.shape[1]))

    def predict(self, A=np.zeros((2, 3))):
        # self.x = np.dot(self.A, self.x)
        # self.x[0:2] += [A[0, 2], A[1, 2]]
        self.x[0] = A[0, 0]*self.x[0] + A[0, 1]*self.x[1] + A[0, 2]
        self.x[1] = A[1, 0]*self.x[0] + A[1, 1]*self.x[1] + A[1, 2]
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R))
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        self.P = np.dot((np.eye(self.H.shape[1]) - np.dot(K, self.H)), self.P)
        return self.x

    def get_state(self):
        return (self.x[0], self.x[1], self.x[4], self.x[5])


def convert_bbox_to_meas(det):
    height = int(det[3] - det[1])
    width = int(det[2] - det[0])
    x_c = int(det[0] + (width/2))
    y_c = int(det[1] + (height/2))
    return (x_c, y_c, width, height)


def convert_meas_to_bbox(meas):
    x1 = int(meas[0] - (meas[2]/2))
    y1 = int(meas[1] - (meas[3]/2))
    x2 = int(meas[0] + (meas[2]/2))
    y2 = int(meas[1] + (meas[3]/2))
    return (x1, y1, x2, y2)


def compute_iou(tracks, measurements):
    iou_matrix = np.zeros((len(measurements), len(tracks)))
    for i, track in enumerate(tracks):
        track_bbox = convert_meas_to_bbox(track.get_state())
        for j, meas in enumerate(measurements):
            xx1 = np.maximum(track_bbox[0], meas[0])
            yy1 = np.maximum(track_bbox[1], meas[1])
            xx2 = np.minimum(track_bbox[2], meas[2])
            yy2 = np.minimum(track_bbox[3], meas[3])
            if xx2 < xx1 or yy2 < yy1:
                intersection_area = 0
            else:
                intersection_area = (xx2-xx1) * (yy2-yy1)
            bb1_area = (track_bbox[2]-track_bbox[0]) * (track_bbox[3]-track_bbox[1])
            bb2_area = (meas[2]-meas[0]) * (meas[3]-meas[1])
            iou_matrix[j, i] = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou_matrix
