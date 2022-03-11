# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
import copy
# from sklearn.utils.linear_assignment_ import linear_assignment    # deprecated
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from AB3DMOT_libs.bbox_utils import convert_3dbox_to_8corner, iou3d, roty
from AB3DMOT_libs.kalman_filter_dense import KalmanBoxDenseTracker
from utils.radar_corrections import transform_radar_velocity

import matplotlib.pyplot as plt


def dist_to_rectangle(corners_rectangle, pts, debug=False):
    """
    Calculate distance of points to the edges of a BeV BBox
    corners_rectangle: (2, 4) in camera coordinates of bbox in BeV
    pts; (2, N) radar detections in camera coordinates
    """

    dims = np.asarray([0.0, 0.0])

    w = corners_rectangle[:, 0]
    u = (corners_rectangle[:, 1] - w)
    dims[0] = np.linalg.norm(u)
    v = (corners_rectangle[:, 3] - w)
    dims[1] = np.linalg.norm(v)
    if (abs(dims) < 1e-3).any():
        return 1000.0 * np.ones(pts.shape[1])
    v = v / dims[1]
    u = u / dims[0]
    if debug:
        print(f'{u=}')
        print(f'{v=}')
        print(f'{dims=}')

    T = np.zeros((2, 2))
    T[:, 0], T[:, 1] = u, v
    T_inv = np.linalg.inv(T)
    pts_rect = T_inv.dot(pts - w.reshape(2, 1))
    if debug:
        print(f'{pts=}')
        print(f'{pts_rect=}')

    zeros = np.zeros_like(pts_rect)
    du = np.max(np.vstack((zeros[0, :], pts_rect[0, :] - dims[0], -pts_rect[0, :])), axis=0)
    dv = np.max(np.vstack((zeros[1, :], pts_rect[1, :] - dims[1], -pts_rect[1, :])), axis=0)
    dist = np.sqrt(du**2 + dv**2)

    if debug:

        print(f'{du=}')
        print(f'{dv=}')
        print(f'{dist=}')

        plt.plot(np.hstack((corners_rectangle[0, :], corners_rectangle[0, 0])), np.hstack((corners_rectangle[1, :], corners_rectangle[1, 0])),
                 pts[0, :], pts[1, :], 'g^')
        plt.axis('square')
        plt.grid(True)
        plt.show()

    return dist


def associate_radar_to_trackers(radar_dets, tracker_dets, distance_threshold=2, metric='to_rect', debug=False):
    """
    radar_dets: N x 3 [[x_sc, y_sc, z_sc], ....] -> numpy array in camera coordinates
    trackers: M x 6 [[bbox_x, bbox_y, bbox_z, theta, l, w, h], ...]
    """
    assert (radar_dets.shape[0] > 0) and (radar_dets.shape[1] == 3), str(radar_dets) + str(type(radar_dets))
    assert (tracker_dets.shape[0] > 0) and (tracker_dets.shape[1] == 7), str(tracker_dets)

    assert metric in ['to_center', 'to_rect'], f'{metric} is not an available metric'
    distance_matrix = np.ones((radar_dets.shape[0], tracker_dets.shape[0])) * np.inf
    if metric == 'to_rect':
        for idx in range(tracker_dets.shape[0]):
            width = tracker_dets[idx, 5]
            length = tracker_dets[idx, 4]
            # Irgendwie um pi/2 gedreht ??
            x_corners = (length / 2) * np.asarray([-1, 1, 1, -1])
            z_corners = (width / 2) * np.asarray([1, 1, -1, -1])
            y_corners = np.zeros_like(x_corners)
            corners_rectangle = np.dot(roty(tracker_dets[idx, 3]), np.vstack([x_corners, y_corners, z_corners]))
            corners_2D = corners_rectangle[(0, 2), :]
            corners_2D += tracker_dets[idx, (0, 2)].reshape(2, 1)
            distance_matrix[:, idx] = dist_to_rectangle(corners_2D, radar_dets[:, (0, 2)].T, debug=debug)

    else:
        distance_matrix = cdist(radar_dets, tracker_dets[:, :3])

    radar_ind, tracker_ind = linear_sum_assignment(distance_matrix)
    assigned_idx = np.stack((radar_ind, tracker_ind), axis=1)
    if debug:
        print(f'{radar_dets=}')
        print(f'{tracker_dets=}')
        print(f'{distance_matrix=}')
        print(f'{radar_ind=}')
        print(f'{tracker_ind=}')

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
    def __init__(self, calib, max_age=4, min_hits=1):      # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
        """
        TODO
        """
        self.max_age = max_age
        self.min_hits = min_hits

        self.frame_count = 0
        self.trackers = []
        self.trackers_over_time = []
        self.trajectories = {}


        self.calib = calib

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

            # Prepare data
            det, conf, radar = [], [], []
            if isinstance(dets[frame], np.ndarray):
                det = dets[frame].reshape((-1, 7))
            if isinstance(confidence_scores[frame], np.ndarray):
                conf = confidence_scores[frame].reshape(-1)
            if isinstance(radar_dets[frame], np.ndarray):
                radar = radar_dets[frame]

            # Run prediction and update step of Kalman Filter
            tracked_obj_in_frame = self.update(det, conf, radar)
            tracked_obj_in_frame_copy = copy.deepcopy(tracked_obj_in_frame)
            trackers_over_time.append(tracked_obj_in_frame_copy)

        # Eliminate all tracked obj that were never assigned a radar detection
        for tracked_obj_in_frame in reversed(trackers_over_time):
            assigned_obj = []
            for obj in tracked_obj_in_frame:
                if obj.has_radar_assigned or obj.id in self.trajectories.keys():
                    assigned_obj.append(obj)
                    if obj.id not in self.trajectories.keys():
                        self.trajectories[obj.id] = copy.deepcopy(obj)
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

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxDenseTracker(dets[i, :], confidence_scores[i].squeeze(), global_time_idx=self.frame_count)
            self.trackers.append(trk)

        # Assign radar detections
        if isinstance(radar_dets, np.ndarray) and self.trackers:
            # Check if radar detections can be assigned to tracker
            xyz_camera = []
            for idx, trk in enumerate(self.trackers):
                # if trk.hits >= self.min_hits:
                state = trk.get_state()
                xyz_camera.append(state)
            if xyz_camera:
                xyz_camera = np.asarray(xyz_camera)
                matches, _, _, = associate_radar_to_trackers(radar_dets[:, :3], xyz_camera, distance_threshold=1)
                for match_row in range(matches.shape[0]):
                    radar_det = radar_dets[matches[match_row, 0], :].copy()
                    radar_det_camera = radar_det.copy()
                    radar_det_radar = self.calib.rect_to_radar(radar_det[:3].reshape(1, -1))
                    radar_det[:3] = radar_det_radar.reshape(-1)
                    orient3d = self.trackers[matches[match_row, 1]].get_state()[3]
                    yaw_ego = radar_det[5]
                    vel_ego = radar_det[6]
                    vel_tracker = transform_radar_velocity(radar_det, self.calib, orient3d, yaw_ego, vel_ego)
                    if abs(vel_tracker[0, 2]) > 10:
                        from utils.plot import plot_bev_radar, plot_bev_bbox_from_tracker
                        fix, ax = plt.subplots()
                        ax = plot_bev_radar(ax, radar_det_camera.reshape(1, -1))
                        ax = plot_bev_bbox_from_tracker(ax, self.trackers[matches[match_row, 1]].get_state())
                        plt.show()

                    radar_det_with_vel = np.zeros(6)
                    radar_det_with_vel[:3] = radar_det_camera[:3]
                    radar_det_with_vel[3:] = vel_tracker
                    self.trackers[matches[match_row, 1]].assign_radar(radar_det_with_vel, self.frame_count)

        # Eliminate dead trackers and create return
        good_trackers = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
            #   ret.append(np.concatenate((trk.get_state(), [trk.id + 1, 1 if trk.has_radar_assigned else 0])).reshape(1, -1))
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

        self.frame_count += 1  # Accept objects in first frames as tracked even if below min_hits

        return good_trackers

    def get_num_of_frames(self):
        return len(self.trackers_over_time)

    def get_tracked_obj_in_frame(self, frame_idx):
        assert frame_idx < len(self.trackers_over_time)
        return self.trackers_over_time[frame_idx]

    def plot_trajectories(self):
        plt.figure()
        print(self.trajectories)
        print("*****************")
        for obj in self.trajectories.values():
            if obj.id not in [0, 89]:
                continue

            t, x, t_rad, x_rad = obj.get_trajectory()
            print(f'############## {obj.id} ##############')
            print(f'{t=}')
            print(f'{x=}')
            print(f'{t_rad=}')
            print(f'{x_rad=}')
            label_filt = f'filtered for {obj.id}'
            label_rad = f'radar for {obj.id}'
            plt.subplot(221)
            plt.plot(t, x[:, 0], label=label_filt)
            plt.plot(t_rad, x_rad[:, 0], label=label_rad)
            plt.subplot(222)
            plt.plot(t, x[:, 2], label=label_filt)
            plt.plot(t_rad, x_rad[:, 2], label=label_rad)
            plt.subplot(223)
            plt.plot(t, x[:, 7] / 0.1, label=label_filt)
            plt.plot(t_rad, x_rad[:, 3], label=label_rad)
            plt.subplot(224)
            plt.plot(t, x[:, 9] / 0.1, label=label_filt)
            plt.plot(t_rad, x_rad[:, 5], label=label_rad)
        plt.legend()
        plt.subplot(221)
        plt.xlabel('time idx')
        plt.ylabel('x_camera [m]')
        plt.subplot(222)
        plt.xlabel('time idx')
        plt.ylabel('z_camera [m]')
        plt.subplot(223)
        plt.xlabel('time idx')
        plt.ylabel('vx')
        plt.subplot(224)
        plt.xlabel('time idx')
        plt.ylabel('vz')
        plt.show()


if __name__ == "__main__":

    # camera coord 2D [x, z]
    # center = np.asarray([1, 10])
    # angle_deg = 10
    # length, width, angle = 4, 2, angle_deg * np.pi / 180.0
    # x_corners = (width / 2) * np.asarray([-1, 1, 1, -1])
    # z_corners = (length / 2) * np.asarray([1, 1, -1, -1])
    # y_corners = np.zeros_like(x_corners)
    # corners_rectangle = np.dot(roty(angle), np.vstack([x_corners, y_corners, z_corners]))
    # corners_2D = corners_rectangle[(0, 2), :]
    # corners_2D += center.reshape(2, 1)

    # print(f'{corners_2D=}')
    # pts = np.zeros((2, 4))
    # pts[:, 0] = [1.4, 9.5]
    # pts[:, 1] = [3, 9.5]
    # pts[:, 2] = [1, 13]
    # pts[:, 3] = [-1, 10]
    # print(f'{pts=}')
    # dist = dist_to_rectangle(corners_2D, pts, debug=True)
    # print(f'{dist=}')

    print(" ========== Read label ==============")
    from utils.read_datastructure import generate_indexed_datastructure
    from utils.dense_transforms import Calibration
    from SeeingThroughFog.tools.DatasetViewer.lib.read import get_kitti_object_list, load_radar_points
    from utils.radar_corrections import transform_radar
    idx = 1001
    data_path = '/lhome/dscheub/ObjectDetection/data/external/SprayAugmentation/2021-07-28_18-21-02'
    stf_path = '/lhome/dscheub/ObjectDetection/SeeingThroughFog'
    calib = Calibration(stf_path=stf_path)
    dense_data = generate_indexed_datastructure(data_path, tracked_label_suffix='_pointrcnn_wet')
    frame = dense_data[idx]
    print(f'{frame=}')
    objects = get_kitti_object_list(frame['tracked_label'])
    trackers = np.zeros((len(objects), 7))
    for idx, obj in enumerate(objects):
        trackers[idx, :] = np.asarray([obj['posx'], obj['posy'], obj['posz'], obj['orient3d'], obj['length'], obj['width'], obj['height']])
    print(f'{trackers=}')
    radar_det = load_radar_points(frame['radar'])
    radar_det = radar_det[radar_det[:, 3] > 0]
    radar_camera, vel = transform_radar(radar_det, frame['can'], calib)
    matches, unmatched_radar, unmatched_tracker = associate_radar_to_trackers(radar_camera, trackers, debug=True)
    print("Matches:\n",  matches)
    print("Unmatched radar:\n", unmatched_radar)
    print("Unmatched tracker:\n", unmatched_tracker)

    # print("======= Test associate_radar_to_trackers =======")
    # # trackers: M x 6 [[bbox_x, bbox_y, bbox_z, theta, l, w, h], ...]
    # radar_dets = np.asarray([[0.5, 0, 10], [-0.8, 1, 5], [4.5, 1, 40]])
    # print("Radar dets:\n", radar_dets)
    # tracker_dets = np.asarray([[2, 1, 10, 0, 2, 1, 1], [-1, 2, 5, 0, 3, 1, 1], [5, 1, 40, 0, 2, 1, 1]])
    # print("Tracker_dets\n", tracker_dets)
    # matches, unmatched_radar, unmatched_tracker = associate_radar_to_trackers(radar_dets, tracker_dets, debug=True)
    # print("Matches:\n",  matches)
    # print("Unmatched radar:\n", unmatched_radar)
    # print("Unmatched tracker:\n", unmatched_tracker)
    # print("=====================")
