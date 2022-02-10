# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
import copy
# from sklearn.utils.linear_assignment_ import linear_assignment    # deprecated
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from AB3DMOT_libs.bbox_utils import convert_3dbox_to_8corner, iou3d, roty
from AB3DMOT_libs.kalman_filter_dense import KalmanBoxDenseTracker


DEBUG = False


def associate_radar_to_trackers(radar_dets, tracker_dets, distance_threshold=2, metric='to_corner'):
    """
    radar_dets: N x 3 [[x_sc, y_sc, z_sc], ....] -> numpy array in camera coordinates
    trackers: M x 6 [[bbox_x, bbox_y, bbox_z, theta, l, w, h], ...]
    """
    assert (radar_dets.shape[0] > 0) and (radar_dets.shape[1] == 3), str(radar_dets) + str(type(radar_dets))
    assert (tracker_dets.shape[0] > 0) and (tracker_dets.shape[1] == 7), str(tracker_dets)

    global DEBUG

    assert metric in ['to_center', 'to_corner'], f'{metric} is not an available metric'
    distance_matrix = np.ones((radar_dets.shape[0], tracker_dets.shape[0])) * np.inf
    if metric == 'to_corner':
        for idx in range(tracker_dets.shape[0]):
            width = tracker_dets[idx, 5]
            length = tracker_dets[idx, 4]
            x_corners = (length / 2) * np.asarray([-1, -1, -1, 1, 1, 1])
            z_corners = (width / 2) * np.asarray([1, 0, -1, -1, 0, 1])
            y_corners = np.zeros_like(x_corners)
            corners = np.dot(roty(tracker_dets[idx, 3]), np.vstack([x_corners, y_corners, z_corners]))
            corners += tracker_dets[idx, :3].reshape(3, 1)
            key_corners = np.transpose(corners)
            if DEBUG:
                print("########")
                print("Key corners")
                print(key_corners)
                print("Center", tracker_dets[idx, :3])
                print("lwh", tracker_dets[idx, -3:])
                print("Angle: ", tracker_dets[idx, 3])

            dist_to_radar = cdist(radar_dets, key_corners)
            min_dist_to_radar = np.min(dist_to_radar, axis=1)
            distance_vec = distance_matrix[:, idx]
            distance_vec = np.where(min_dist_to_radar < distance_vec, min_dist_to_radar, distance_vec)
            distance_matrix[:, idx] = distance_vec

    else:
        distance_matrix = cdist(radar_dets, tracker_dets[:, :3])

    if DEBUG:
        print("******")
        print("Num radar: ", radar_dets.shape[0])
        print("radar")
        print(radar_dets)
        print("Distance matrix")
        print(distance_matrix)
        print("******")

    radar_ind, tracker_ind = linear_sum_assignment(distance_matrix)
    assigned_idx = np.stack((radar_ind, tracker_ind), axis=1)

    unmatched_radar = np.setdiff1d(np.arange(radar_dets.shape[0]), radar_ind)
    unmatched_tracker = np.setdiff1d(np.arange(tracker_dets.shape[0]), tracker_ind)

    matches = np.array([])
    for idx in assigned_idx:
        if distance_matrix[idx[0], idx[1]] < distance_threshold:
            matches = np.append(matches, idx)
        else:
            unmatched_radar = np.append(unmatched_radar, idx[0])
            unmatched_tracker = np.append(unmatched_tracker, idx[1])

    return matches.reshape((-1, 2)).astype(int), unmatched_radar.astype(int), unmatched_tracker.astype(int)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.01):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    detections:  N x 8 x 3
    trackers:    M x 8 x 3


    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if np.linalg.norm(det-trk) < 1e-3:
                iou_matrix[d, t] = 0
            else:
                iou_matrix[d, t] = iou3d(det, trk)[0]             # det: 8 x 3, trk: 8 x 3
    # matched_indices = linear_assignment(-iou_matrix)      # hougarian algorithm, compatible to linear_assignment in sklearn.utils

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)      # hougarian algorithm
    matched_indices = np.stack((row_ind, col_ind), axis=1)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class AB3DMOT(object):
    def __init__(self, max_age=4, min_hits=5):      # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
        """
        TODO
        """
        self.max_age = max_age
        self.min_hits = min_hits

        self.frame_count = 0
        self.trackers = []
        self.trackers_over_time = []
        self.trajectories = {}

    def run_for_all_frames(self, dets, confidence_scores, radar_dets):
        """
        dets : (T, N, 7): T frames, N detections, [x, y, z, theta, l, w, h] in R^7 for bbox detections in camera coord.
        confidence_scores: (T, N): Confidence for detection
        radar_dets: (T, M, 5) Radar detection [x, y, z, r, v_r] in camera coord
        """
        assert (len(dets) == len(radar_dets)) and (len(dets) == len(confidence_scores)), f'Dets: {len(dets)}, ' \
            f'Radar: {len(radar_dets)}, Conf: {len(confidence_scores)}'
        num_frames = len(dets)
        self.trackers, self.trackers_over_time, trackers_over_time = [], [], []
        for frame in range(num_frames):

            global DEBUG
            if frame == 5:  #  and frame < 211:
                DEBUG = False 
            else:
                DEBUG = False

            # Prepare data
            det, conf, radar = [], [], []
            if isinstance(dets[frame], np.ndarray):
                det = dets[frame].reshape((-1, 7))
            if isinstance(confidence_scores[frame], np.ndarray):
                conf = confidence_scores[frame].reshape(-1)
            if isinstance(radar_dets[frame], np.ndarray):
                radar = radar_dets[frame].reshape((-1, 5))
                # Filter out radar detections without velocity
                velocity_mask = radar[:, 3] > 0
                radar_debug = radar[velocity_mask, :]
                radar = radar[velocity_mask, 0:3]
                if radar.size == 0:
                    radar = []
                if DEBUG:
                    print("Num radar 3", radar.shape)
                    print("Num radar debug 3", radar_debug.shape)

            # Run prediction and update step of Kalman Filter
            tracked_obj_in_frame = self.update(det, conf, radar)
            tracked_obj_in_frame_copy = copy.deepcopy(tracked_obj_in_frame)
            
            trackers_over_time.append(tracked_obj_in_frame_copy)

            if DEBUG:
                print(radar_debug)
                print("Num radar", radar_debug.shape)

        # Eliminate all tracked obj that were never assigned a radar detection
        for tracked_obj_in_frame in reversed(trackers_over_time):
            assigned_obj = []
            for obj in tracked_obj_in_frame:
                if obj.has_radar_assigned and (obj.id not in self.trajectories.keys()):
                    assigned_obj.append(obj)
                    self.trajectories[obj.id] = [obj]
                elif obj.has_radar_assigned and (obj.id in self.trajectories.keys()):
                    assigned_obj.append(obj)
                    self.trajectories[obj.id].insert(0, obj)
                elif (not obj.has_radar_assigned) and (obj.id in self.trajectories.keys()):
                    assigned_obj.append(obj)
                    self.trajectories[obj.id].insert(0, obj)
                else:
                    pass
            self.trackers_over_time.append(assigned_obj)
        self.trackers_over_time.reverse()

    def update(self, dets, confidence_scores, radar_dets=None):
        """
        Params:
          dets_all: dict
            dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],...]
            confidence_scores in [0, 1]: (N, ) vector describing confidence of detection -> used to scale covariances

        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        if isinstance(confidence_scores, np.ndarray):
            assert (confidence_scores <= 1).all() and (confidence_scores >= 0).all(), f' Confidence scores: {confidence_scores}'

        self.frame_count += 1  # Accept objects in first frames as tracked even if below min_hits

        # Run prediction for already found trackers
        trks = np.zeros((len(self.trackers), 7))  # N x 7 , # get predicted locations from existing trackers.
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Get corners to caclulate IOU for asscciation of detection with trackers
        dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []
        trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
        if len(trks_8corner) > 0:
            trks_8corner = np.stack(trks_8corner, axis=0)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner)

        # Update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0], confidence_scores[d].squeeze())

        # Assign radar detections
        if isinstance(radar_dets, np.ndarray) and self.trackers:
            # Check if radar detections can be assigned to tracker
            xyz_camera = []
            for idx, trk in enumerate(self.trackers):
                if trk.hits >= self.min_hits:
                    state = trk.get_state()
                    xyz_camera.append(state)
            if xyz_camera:
                xyz_camera = np.asarray(xyz_camera)
                if DEBUG:
                    print("Num radar 2: ", radar_dets.shape)
                matches, _, _, = associate_radar_to_trackers(radar_dets, xyz_camera, distance_threshold=1)
                for match in matches:
                    self.trackers[match[1]].assign_radar()

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxDenseTracker(dets[i, :], confidence_scores[i].squeeze())
            self.trackers.append(trk)

        # Eliminate dead trackers and create return
        good_trackers = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                # ret.append(np.concatenate((trk.get_state(), [trk.id + 1, 1 if trk.has_radar_assigned else 0])).reshape(1, -1))
            good_trackers.append(trk)
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)

        # Returns [[x, y, z, theta, l, w, h, has_radar],...]
        # if len(ret) > 0:
        #   return np.concatenate(ret)
        # else:
        #   return np.empty((0, 9))

        return good_trackers

    def get_num_of_frames(self):
        return len(self.trackers_over_time)

    def get_tracked_obj_in_frame(self, frame_idx):
        assert frame_idx < len(self.trackers_over_time)
        return self.trackers_over_time[frame_idx]

    def get_trajectory_by_id(self, obj_id):
        assert obj_id in self.trajectories.keys()
        return self.trajectories[obj_id]


if __name__ == "__main__":

    print("======= Test associate_radar_to_trackers =======")
    radar_dets = np.asarray([[5, 2, 3], [10, 10, 0], [20, 20, 10]])
    print("Radar dets:\n", radar_dets)
    tracker_dets = np.asarray([[10, 9, 1, 0, 2, 1, 1], [5, 2, 4, 0, 3, 1, 1], [19, 20, 11, 0, 2, 1, 1]])
    print("Tracker_dets\n", tracker_dets)
    matches, unmatched_radar, unmatched_tracker = associate_radar_to_trackers(radar_dets, tracker_dets, distance_threshold=10)
    print("Matches:\n",  matches)
    print("Unmatched radar:\n", unmatched_radar)
    print("Unmatched tracker:\n", unmatched_tracker)
    print("=====================")
