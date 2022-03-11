import numpy as np
import os

def get_index_and_timestamp_from_file(filepath):

    filename = os.path.split(filepath)[1]
    stamp = int(filename.split('_')[1].split('.')[0])
    idx = int(filename.split('_')[0])

    return idx, stamp


def calculate_egomotion_correction_velocity(radar_det, yaw_rate_ego, vel_ego):
    """
    radar_det: (N, 5) [x_sc, y_sc, z_sc, v_radial, r] in radar coordinates
    yaw_rate_ege: Scalar [rad/s]
    vel_ego:  Scalar [m/s]

    return: (N, 3) Correction velocity in [x, y, z] direction for egomotion of vehicle
    """
    pos_radar_coord = radar_det[:, :3]

    vel_correct = np.zeros_like(pos_radar_coord)  # (N x 3)
    vel_correct[:, 0] = vel_ego - pos_radar_coord[:, 1] * yaw_rate_ego
    vel_correct[:, 1] = pos_radar_coord[:, 0] * yaw_rate_ego

    return vel_correct


def transform_radar(radar_det, calib, yaw_rate_ego=0, vel_ego=0, delta_time=0):
    """
    radar_det: (N, 5) [x_sc, y_sc, z_sc, v_radial, r] in radar coordinates
    calib: Calibration object for transforms
    """
    pos_radar_coord = radar_det[:, :3].copy()

    # Egomotion correction
    vel_correct = calculate_egomotion_correction_velocity(radar_det, yaw_rate_ego, vel_ego)
    pos_radar_coord -= delta_time * vel_correct

    # Radar detections to camera coordinates
    pos_radar_cam = calib.radar_to_rect(pos_radar_coord)

    # Reappend velocity and radius
    radar_det_cam= np.zeros_like(radar_det)
    radar_det_cam[:, :3] = pos_radar_cam
    radar_det_cam[:, 3:] = radar_det[:, 3:].copy() 
    
    return radar_det_cam


def transform_radar_velocity(radar_det, calib, orient3D, yaw_rate_ego, vel_ego, debug=True):
    """
    radar_det: single (5, ) [x_sc, y_sc, z_sc, v_radial, r] in radar coordinates
    calib: Calibration object for transforms
    orient3D: Angle of corresponding bbox from kitti label (camera coordinates)
    yaw_rate_ege: Scalar [rad/s]
    vel_ego:  Scalar [m/s]
    """

    # Radar velocity is in radial direction relative to the ground
    vel_radial = radar_det[3]
    azimuth = np.arctan(radar_det[1] / radar_det[0])

    # Velocity in x-direction in object's coordinate frame
    orient3D_corr = orient3D + np.pi/2
    vel_x_obj = vel_radial * abs(np.cos(azimuth + orient3D_corr))
    # Use angle from bbox for estimating velocity vector in sensor coordinates
    vel_obj = vel_x_obj * np.asarray([np.cos(orient3D_corr),
                                      np.sin(orient3D_corr),
                                      0])

    if vel_radial > 0 and vel_obj[0] < 0:
        vel_obj *= -1.0

    # Subtract movement of the car
    vel_correct = calculate_egomotion_correction_velocity(radar_det.reshape(1, -1), yaw_rate_ego, vel_ego)
    vel_radar = vel_obj.reshape(1, -1) - vel_correct  # (1 x 3)

    vel_radar_hom = calib.cart_to_hom(vel_radar)
    R2C_without_trans = calib.R2C.copy()
    R2C_without_trans[:, 3] = np.zeros(3)  # Translation is constant and thus not considered
    vel_cam = np.dot(vel_radar_hom, np.dot(R2C_without_trans.T, calib.R0.T))

    if debug and abs(vel_cam[0, 2]) > 10:
        print('XXXXX Debug info transform_radar_velocity XXXXX')
        print(f'{radar_det=}')
        print(f'{vel_radial=}')
        print(f'{azimuth=}')
        print(f'{orient3D=}')
        print(f'{orient3D_corr=}')
        print(f'{vel_obj=}')
        print(f'{vel_correct=}')
        print(f'{vel_radar=}')
        print(f'{vel_cam=}')
        print('XXXXXX')

    return vel_cam
