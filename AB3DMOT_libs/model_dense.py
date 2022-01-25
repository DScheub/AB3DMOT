# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment    # deprecated
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from AB3DMOT_libs.bbox_utils import convert_3dbox_to_8corner, iou3d
from AB3DMOT_libs.kalman_filter_dense import KalmanBoxDenseTracker


def associate_radar_to_trackers(radar_dets, tracker_dets, distance_threshold=2):
    """
    radar_dets: N x 3 [[x_sc, y_sc, z_sc], ....] -> numpy array in camera coordinates
    trackers: M x 3 [[bbox_x, bbox_y, bbox_z], ...]
    """
    assert (radar_dets.shape[0] > 0) and (radar_dets.shape[1] == 3), str(radar_dets)
    assert (tracker_dets.shape[0] > 0) and (tracker_dets.shape[1] == 3), str(tracker_dets)

    distance_matrix = cdist(radar_dets, tracker_dets)
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


class AB3DMOT(object):			  # A baseline of 3D multi-object tracking
    def __init__(self, max_age=2, min_hits=5):      # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]

    def update(self, dets, confidence_scores, radar_dets=None):
        """
        Params:
          dets_all: dict
            dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
            confidence_scores in [0, 1]: (N, ) vector describing confidence of detection -> used to scale covariances

        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # reorder the data to put x,y,z in front to be compatible with the state transition matrix
        # where the constant velocity model is defined in the first three rows of the matrix

        assert (confidence_scores <= 1).all() and (confidence_scores >= 0).all(), f' Confidence scores: {confidence_scores}'
        dets = dets[:, self.reorder]					# reorder the data to [[x,y,z,theta,l,w,h], ...]

        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 7))         # N x 7 , # get predicted locations from existing trackers.
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []
        trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
        if len(trks_8corner) > 0:
            trks_8corner = np.stack(trks_8corner, axis=0)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
                trk.update(dets[d, :][0], confidence_scores[d].squeeze())

        # Assign radar detections
        if (radar_dets is not None) and (len(self.trackers) > 0):
            # Check if radar detections can be assigned to tracker
            xyz_camera = np.zeros((len(self.trackers), 3))
            for idx, trk in enumerate(self.trackers):
                xyz_camera[idx, :] = trk.get_state()[:3]
            matches, _, _, = associate_radar_to_trackers(radar_dets, xyz_camera, distance_threshold=5)
            for match in matches:
                self.trackers[match[1]].assign_radar()

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:        # a scalar of index
            trk = KalmanBoxDenseTracker(dets[i, :], confidence_scores[i].squeeze())
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()  # bbox location
            d = d[self.reorder_back]  # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]

            if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1, 1 if trk.has_radar_assigned else 0])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1

            # remove dead tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)

        if (len(ret) > 0):
            ret_arr = np.concatenate(ret)  # h,w,l,x,y,z,theta, ID, other info, confidence
            return ret_arr
        else:
            return np.empty((0, 15))


if __name__ == "__main__":

    print("======= Test associate_radar_to_trackers =======")
    radar_dets = np.asarray([[5, 2, 3], [10, 10, 0], [20, 20, 10]])
    print("Radar dets:\n", radar_dets)
    tracker_dets = np.asarray([[10, 9, 1], [5, 2, 4], [19, 20, 11]])
    print("Tracker_dets\n", tracker_dets)
    matches, unmatched_radar, unmatched_tracker = associate_radar_to_trackers(radar_dets, tracker_dets, distance_threshold=10)
    print("Matches:\n",  matches)
    print("Unmatched radar:\n", unmatched_radar)
    print("Unmatched tracker:\n", unmatched_tracker)
    print("=====================")
    radar_dets = np.asarray([[5, 2, 3], [10, 10, 0]])
    print("Radar dets:\n", radar_dets)
    tracker_dets = np.asarray([[10, 9, 1], [5, 2, 4], [5, 2, 4.2], [19, 20, 11]])
    print("Tracker_dets\n", tracker_dets)
    matches, unmatched_radar, unmatched_tracker = associate_radar_to_trackers(radar_dets, tracker_dets, distance_threshold=1.1)
    print("Matches:\n",  matches)
    print("Unmatched radar:\n", unmatched_radar)
    print("Unmatched tracker:\n", unmatched_tracker)
    print("=====================")
    radar_dets = np.asarray([])
    print("Radar dets:\n", radar_dets)
    tracker_dets = np.asarray([[10, 9, 1], [5, 2, 4], [5, 2, 4.2], [19, 20, 11]])
    print("Tracker_dets\n", tracker_dets)
    matches, unmatched_radar, unmatched_tracker = associate_radar_to_trackers(radar_dets, tracker_dets, distance_threshold=1.1)
    print("Matches:\n",  matches)
    print("Unmatched radar:\n", unmatched_radar)
    print("Unmatched tracker:\n", unmatched_tracker)
    print("=====================")
