import numpy as np


def plot_bev_bbox(plt, labels):

    assert isinstance(labels, list)

    for obj in labels:
        corners = np.zeros((2, 4))
        corners[0, :] = (obj['length'] / 2) * np.asarray([-1, 1, 1, -1])
        corners[1, :] = (obj['width'] / 2) * np.asarray([1, 1, -1, -1])
        rad = obj['orient3d']
        rot = np.array([[np.cos(rad), np.sin(rad)],
                        [-np.sin(rad), np.cos(rad)]])
        corners_rot = np.dot(rot, corners)
        corners_rot += np.asarray([obj['posx'], obj['posz']]).reshape(2, 1)
        plt.plot(np.hstack((corners_rot[0, :], corners_rot[0, 0])),
                 np.hstack((corners_rot[1, :], corners_rot[1, 0])),
                 color=obj['color'] if 'color' in obj else (0, 0, 0))

    plt.axis('square')
    return plt


def plot_bev_bbox_from_tracker(plt, tracker):
    """
    tracker_state: [x,y,z,theta,l,w,h]
    """
    obj = dict(posx = tracker[0],
               posy = tracker[1],
               posz = tracker[2],
               orient3d = tracker[3],
               length = tracker[4],
               width = tracker[5],
               height = tracker[6])

    return plot_bev_bbox(plt, [obj])


def plot_bev_radar(plt, radar, radar_color=None, radar_label=None):

    assert isinstance(radar, np.ndarray)
    plt.plot(radar[:, 0], radar[:, 2],
             color=radar_color if radar_color else (0, 0, 0),
             marker="*",
             label=radar_label if radar_label else 'Radar')
    plt.legend()

    return plt


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import os
    from utils.read_datastructure import generate_indexed_datastructure
    from utils.dense_transforms import Calibration
    from SeeingThroughFog.tools.DatasetViewer.lib.read import get_kitti_object_list, load_radar_points
    from utils.radar_corrections import transform_radar, transform_radar_velocity, get_index_and_timestamp_from_file
    data_path = '/lhome/dscheub/ObjectDetection/data/external/ImmendingenSprayTestsv3/2021-07-26_19-15-15'
    stf_path = '/lhome/dscheub/ObjectDetection/SeeingThroughFog'
    calib = Calibration(stf_path=stf_path)
    dense_data = generate_indexed_datastructure(data_path,
                                                min_index=500,
                                                pred_label_suffix='_pointrcnn_wet',
                                                tracked_label_suffix='_pointrcnn_wet')
    idx = 10

    from utils.egomotion import Egomotion
    odom_path = f'{data_path}/odometry.csv'
    ego = Egomotion(odom_path)

    print(f"*****{idx}****")
    fig, ax = plt.subplots()
    frame = dense_data[idx]

    objects = get_kitti_object_list(frame['pred_label'])
    ax = plot_bev_bbox(ax, objects)

    radar_det = load_radar_points(frame['radar'])
    radar_det = radar_det[radar_det[:, 3] > 0]
    radar_camera = transform_radar(radar_det, calib)
    ax = plot_bev_radar(ax, radar_camera, radar_label='without egomotion')
    radar_det_not_trans = np.zeros_like(radar_det)
    radar_det_not_trans[:, 0] = -radar_det[:, 1]
    radar_det_not_trans[:, 2] = radar_det[:, 0]
    ax = plot_bev_radar(ax, radar_det_not_trans, radar_color='g', radar_label='untransformed')

    idx, radar_stmp = get_index_and_timestamp_from_file(frame['radar'])
    pc_idx, pc_stmp = get_index_and_timestamp_from_file(frame['pc'])
    assert idx == pc_idx
    delta = (radar_stmp - pc_stmp) / 1e6
    print(f'Idx {idx}: Radar: -> stamp: {radar_stmp}, PC: -> stamp: {pc_stmp}, Time Diff: {delta} ms')
    yaw_ego, vel_ego = ego.get_yaw_rate(pc_stmp), ego.get_velocity(pc_stmp)
    print(f'{yaw_ego=}, {vel_ego=}')
    radar_camera_egomotion = transform_radar(radar_det, calib, yaw_ego, vel_ego, delta / 1e3)
    ax = plot_bev_radar(ax, radar_camera_egomotion, radar_color='r', radar_label='egomotion')

    obj_asscociated = objects[1]
    radar_associated = radar_det[0, :]
    print(radar_associated.shape)
    vel_cam = transform_radar_velocity(radar_det, calib, obj_asscociated['orient3d'], yaw_ego, vel_ego, debug=True) 
    print(f'{vel_cam=}')

    print("######")

    plt.show()
