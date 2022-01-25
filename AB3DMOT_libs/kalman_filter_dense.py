# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
from filterpy.kalman import KalmanFilter


def map_angle_to_range(angle):
    if angle >= np.pi:
        angle -= np.pi * 2
    if angle < -np.pi:
        angle += np.pi * 2
    return angle


class KalmanBoxDenseTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox3D, confidence_score):
        """
        Initialises a tracker using initial bounding box.
        """

        self.dim_x = 10
        self.dim_z = 7

        # define constant velocity model
        self.kf = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)

        # state transition matrix
        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        # measurement function
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        # # with angular velocity
        # self.kf = KalmanFilter(dim_x=11, dim_z=7)
        # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
        #                       [0,1,0,0,0,0,0,0,1,0,0],
        #                       [0,0,1,0,0,0,0,0,0,1,0],
        #                       [0,0,0,1,0,0,0,0,0,0,1],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0],
        #                       [0,0,0,0,0,0,0,1,0,0,0],
        #                       [0,0,0,0,0,0,0,0,1,0,0],
        #                       [0,0,0,0,0,0,0,0,0,1,0],
        #                       [0,0,0,0,0,0,0,0,0,0,1]])

        # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
        #                       [0,1,0,0,0,0,0,0,0,0,0],
        #                       [0,0,1,0,0,0,0,0,0,0,0],
        #                       [0,0,0,1,0,0,0,0,0,0,0],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0]])

        # self.kf.R[0:,0:] *= 10.   # measurement uncertainty

        # Inital values
        self.kf.x[:7] = bbox3D.reshape((7, 1))  # initial condition
        self.kf.P = 10 * np.eye(self.dim_x)  #  initial covariances
        self.kf.P[7:, 7:] *= 100  # give high uncertainty to the unobservable initial velocities, covariance matrix

        self.kf.Q = 0.01 * np.eye(self.dim_x)  # process uncertainity
        self.kf.R = np.eye(self.dim_z)  # measurement uncertainty

        self.time_since_update = 0
        self.id = KalmanBoxDenseTracker.count
        KalmanBoxDenseTracker.count += 1
        self.history = []
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0

    def update(self, bbox3D, confidence_score):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

        # orientation correction
        self.kf.x[3] = map_angle_to_range(self.kf.x[3])
        bbox3D[3] = map_angle_to_range(bbox3D[3])
        # if the angle of two theta is not acute angle
        if abs(self.kf.x[3] - bbox3D[3]) > np.pi / 2.0 and abs(self.kf.x[3] - bbox3D[3]) < np.pi * 3 / 2.0:
            self.kf.x[3] += np.pi
            self.kf.x[3] = map_angle_to_range(self.kf.x[3])
        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(bbox3D[3] - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if bbox3D[3] > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        # Update kf with detections
        R = (1 / confidence_score**4) * self.kf.R
        self.kf.update(bbox3D, R=R)
        self.kf.x[3] = map_angle_to_range(self.kf.x[3])

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.kf.x[3] = map_angle_to_range(self.kf.x[3])

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7, ))


if __name__ == "__main__":

    bbox3D = np.array([10, 15, 20, 0, 1, 1, 1])
    info = None

    print("===== With GT =====")
    kbd = KalmanBoxDenseTracker(bbox3D, info, is_gt=True)
    print(kbd.kf.P)

    print("===== Without GT =====")
    kbd = KalmanBoxDenseTracker(bbox3D, info)
    print(kbd.kf.P)
